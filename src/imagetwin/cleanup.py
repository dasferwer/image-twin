import asyncio
import logging
import signal
from contextlib import suppress
from uuid import uuid4

from sqlalchemy import text

from . import storage
from .config import settings
from .db import engine
from .service import heartbeat

logger = logging.getLogger("object_cleanup")


async def claim_due(image_id=None):
    async with engine.begin() as conn:
        candidates = (
            (
                await conn.execute(
                    text("""
                    SELECT * FROM object_deletions
                    WHERE next_attempt_at<=clock_timestamp()
                    AND (lease_until IS NULL OR lease_until<=clock_timestamp())
                    AND (CAST(:image_id AS uuid) IS NULL OR image_id=CAST(:image_id AS uuid))
                    ORDER BY next_attempt_at,image_id
                    LIMIT 1 FOR UPDATE SKIP LOCKED
                """),
                    {"image_id": image_id},
                )
            )
            .mappings()
            .all()
        )
        claimed = []
        for row in candidates:
            token = uuid4()
            saved = (
                (
                    await conn.execute(
                        text("""
                        UPDATE object_deletions SET attempts=attempts+1,lease_token=:token,
                        lease_until=clock_timestamp()+make_interval(secs=>:lease)
                        WHERE image_id=:id RETURNING *
                    """),
                        {
                            "id": row["image_id"],
                            "token": token,
                            "lease": settings.cleanup_lease_seconds,
                        },
                    )
                )
                .mappings()
                .one()
            )
            claimed.append(dict(saved))
        return claimed


def remove_object(key):
    storage.client().delete_object(Bucket=settings.s3_bucket, Key=key)


async def process_deletion(row):
    error = None
    try:
        await asyncio.to_thread(remove_object, row["object_key"])
    except Exception as exc:
        error = type(exc).__name__[:128]
    async with engine.begin() as conn:
        parameters = {"id": row["image_id"], "token": row["lease_token"]}
        if error:
            parameters.update(error=error, delay=min(2 ** min(row["attempts"], 9), 300))
            saved = await conn.scalar(
                text("""
                    UPDATE object_deletions SET last_error=:error,completed_at=NULL,
                    next_attempt_at=clock_timestamp()+make_interval(secs=>:delay),
                    lease_token=NULL,lease_until=NULL
                    WHERE image_id=:id AND lease_token=:token
                    RETURNING image_id
                """),
                parameters,
            )
            if saved:
                logger.warning("object_delete_retry image_id=%s kind=%s", row["image_id"], error)
            return False
        parameters["recheck"] = settings.cleanup_recheck_seconds
        saved = await conn.scalar(
            text("""
                UPDATE object_deletions SET completed_at=clock_timestamp(),last_error=NULL,
                next_attempt_at=clock_timestamp()+make_interval(secs=>:recheck),
                lease_token=NULL,lease_until=NULL
                WHERE image_id=:id AND lease_token=:token
                RETURNING image_id
            """),
            parameters,
        )
        return bool(saved)


async def cleanup_due(image_id=None):
    completed = 0
    for _ in range(1 if image_id else settings.cleanup_batch_size):
        rows = await claim_due(image_id)
        if not rows:
            break
        completed += bool(await process_deletion(rows[0]))
    return completed


async def deletion_state(conn, image_id):
    row = (
        (
            await conn.execute(
                text("""
                SELECT attempts,next_attempt_at,last_error,completed_at
                FROM object_deletions WHERE image_id=:id
            """),
                {"id": image_id},
            )
        )
        .mappings()
        .first()
    )
    if row is None:
        return {
            "status": "not_requested",
            "attempts": 0,
            "last_error": None,
            "next_attempt_at": None,
            "completed_at": None,
        }
    return {**row, "status": "removed" if row["completed_at"] else "pending"}


async def pulse(stop):
    while not stop.is_set():
        try:
            await heartbeat("cleanup")
        except Exception as exc:
            logger.warning("cleanup_heartbeat_unavailable kind=%s", type(exc).__name__)
        with suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), 2)


async def run(stop):
    heartbeat_stop = asyncio.Event()
    heartbeat_task = asyncio.create_task(pulse(heartbeat_stop))
    try:
        while not stop.is_set():
            try:
                await cleanup_due()
            except Exception as exc:
                logger.warning("object_cleanup_unavailable kind=%s", type(exc).__name__)
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), settings.worker_interval)
    finally:
        heartbeat_stop.set()
        await heartbeat_task


async def main():
    logging.basicConfig(level=logging.INFO)
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set)
    try:
        await run(stop)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
