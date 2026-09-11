import json
from uuid import uuid4

from sqlalchemy import text

from .config import settings
from .db import engine


async def event(conn, image_id, kind):
    await conn.execute(
        text("INSERT INTO image_events(image_id,type) VALUES (:id,:type)"),
        {"id": image_id, "type": kind},
    )


async def heartbeat(name):
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO worker_heartbeats(name) VALUES (:name) ON CONFLICT(name) DO UPDATE SET updated_at=clock_timestamp()"
            ),
            {"name": name},
        )


async def reserve_due(limit=20):
    result = []
    async with engine.begin() as conn:
        rows = (
            (
                await conn.execute(
                    text(
                        "SELECT * FROM images WHERE status='pending' OR (status IN ('queued','processing') AND lease_until<=clock_timestamp()) ORDER BY created_at LIMIT :limit FOR UPDATE SKIP LOCKED"
                    ),
                    {"limit": limit},
                )
            )
            .mappings()
            .all()
        )
        for row in rows:
            if row["attempts"] >= settings.max_attempts:
                await conn.execute(
                    text(
                        "UPDATE images SET status='failed',error='indexing_unavailable',lease_until=NULL WHERE id=:id"
                    ),
                    {"id": row["id"]},
                )
                await event(conn, row["id"], "failed")
                continue
            token = uuid4()
            await conn.execute(
                text(
                    "UPDATE images SET status='queued',generation=:token,lease_until=clock_timestamp()+make_interval(secs=>:lease),updated_at=clock_timestamp() WHERE id=:id"
                ),
                {"id": row["id"], "token": token, "lease": settings.lease_seconds},
            )
            await event(conn, row["id"], "queued" if row["status"] == "pending" else "recovered")
            result.append({"image_id": str(row["id"]), "generation": str(token)})
    return result


async def claim(image_id, generation):
    async with engine.begin() as conn:
        row = (
            (
                await conn.execute(
                    text(
                        "UPDATE images SET status='processing',attempts=attempts+1,lease_until=clock_timestamp()+make_interval(secs=>:lease),updated_at=clock_timestamp() WHERE id=:id AND generation=:token AND status='queued' RETURNING *"
                    ),
                    {"id": image_id, "token": generation, "lease": settings.lease_seconds},
                )
            )
            .mappings()
            .first()
        )
        if row:
            await event(conn, image_id, "processing")
            return dict(row)
    return None


async def complete(image_id, generation, features):
    async with engine.begin() as conn:
        # Результат старого процесса не должен вернуть удалённое изображение в поиск.
        saved = await conn.scalar(
            text(
                "UPDATE images SET status='ready',phash=CAST(CAST(:phash AS text) AS bit(64)),embedding=CAST(:vector AS vector),points=CAST(:points AS jsonb),descriptors=:descriptors,contrast=:contrast,encoder_version=:version,lease_until=NULL,updated_at=clock_timestamp() WHERE id=:id AND generation=:token AND status='processing' RETURNING id"
            ),
            {
                "id": image_id,
                "token": generation,
                "phash": features["phash"],
                "vector": json.dumps(features["vector"]),
                "points": json.dumps(features["points"]),
                "descriptors": features["descriptors"],
                "contrast": features["contrast"],
                "version": features["encoder_version"],
            },
        )
        if saved:
            await event(conn, image_id, "indexed")
        return bool(saved)
