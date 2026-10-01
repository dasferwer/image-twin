"""Проверяем повтор загрузки после отказа S3 и восстановление индексации после SIGKILL."""

import json
import os
import subprocess
import time
from uuid import uuid4

import smoke

smoke.BASE = "http://localhost:8150"


def compose(*args, delay="0"):
    subprocess.run(
        ["docker", "compose", *args],
        check=True,
        env={**os.environ, "INFERENCE_DELAY_SECONDS": delay},
    )


def main():
    token = smoke.login()
    cid = smoke.collection(token)
    data = (smoke.ROOT / "data/coffee.png").read_bytes()
    key = str(uuid4())
    try:
        compose("stop", "storage")
        smoke.upload(token, cid, data, key, expected=(503,))
        compose("up", "-d", "--no-deps", "--wait", "storage")
        restored = smoke.upload(token, cid, data, key)
        smoke.wait_image(token, restored["id"])
        compose("up", "-d", "--no-deps", "--force-recreate", "worker", delay="5")
        second = smoke.upload(token, cid, (smoke.ROOT / "data/eagle.png").read_bytes())
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if smoke.request("/images/" + second["id"], token)["status"] == "processing":
                break
            time.sleep(0.1)
        else:
            raise AssertionError("Worker did not claim the image")
        compose("kill", "-s", "SIGKILL", "worker", delay="5")
        compose("up", "-d", "--no-deps", "--force-recreate", "worker")
        result = smoke.wait_image(token, second["id"])
        events = smoke.request("/images/" + second["id"] + "/events", token)
        assert result["attempts"] == 2
        assert sum(row["type"] == "indexed" for row in events) == 1
        assert any(row["type"] == "recovered" for row in events)
        compose("stop", "storage")
        deleted = smoke.request("/images/" + second["id"], token, "DELETE")
        assert deleted["object_removed"] is False
        smoke.request("/images/" + second["id"], token, expected=(410,))
        compose("up", "-d", "--no-deps", "--wait", "storage")
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            cleanup = smoke.request("/images/" + second["id"] + "/deletion", token)
            if cleanup["status"] == "removed":
                break
            time.sleep(0.25)
        else:
            raise AssertionError("Cleanup did not remove the object after S3 recovery")
        # Проверяем физическое отсутствие, а не только сохранённый статус.
        compose(
            "exec",
            "-T",
            "api",
            "python",
            "-c",
            "import sys\nfrom imagetwin import storage\n"
            "from imagetwin.config import settings\n"
            "from botocore.exceptions import ClientError\n"
            "try:\n storage.client().head_object(Bucket=settings.s3_bucket,Key=sys.argv[1])\n"
            "except ClientError as error:\n assert error.response['Error']['Code'] in {'404','NoSuchKey'}\n"
            "else:\n raise AssertionError('Deleted object remains in S3')\n",
            f"{cid}/{second['id']}.png",
        )
        # Поздний PUT старой попытки должен исчезнуть без повторного DELETE клиента.
        compose(
            "exec",
            "-T",
            "api",
            "python",
            "-c",
            """
import asyncio, sys, time
from pathlib import Path
from sqlalchemy import text
from botocore.exceptions import ClientError
from imagetwin import storage
from imagetwin.config import settings
from imagetwin.db import engine

async def check():
    image_id, object_key = sys.argv[1:]
    async with engine.connect() as conn:
        before = await conn.scalar(text('SELECT attempts FROM object_deletions WHERE image_id=CAST(:id AS uuid)'), {'id':image_id})
    storage.put(object_key, Path('data/eagle.png').read_bytes())
    storage.client().head_object(Bucket=settings.s3_bucket, Key=object_key)
    async with engine.begin() as conn:
        await conn.execute(text('UPDATE object_deletions SET next_attempt_at=clock_timestamp() WHERE image_id=CAST(:id AS uuid)'), {'id':image_id})
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        async with engine.connect() as conn:
            row = (await conn.execute(text('SELECT attempts,completed_at FROM object_deletions WHERE image_id=CAST(:id AS uuid)'), {'id':image_id})).one()
        if row.attempts > before and row.completed_at:
            try:
                storage.client().head_object(Bucket=settings.s3_bucket, Key=object_key)
            except ClientError as exc:
                if exc.response['ResponseMetadata']['HTTPStatusCode'] == 404:
                    return
                raise
        await asyncio.sleep(0.1)
    raise AssertionError('Late PUT was not removed by background cleanup')

async def main():
    try:
        await check()
    finally:
        await engine.dispose()
asyncio.run(main())
""",
            second["id"],
            f"{cid}/{second['id']}.png",
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "s3_upload_retry": True,
                    "worker_sigkill_recovered": True,
                    "attempts": 2,
                    "index_commits": 1,
                    "s3_delete_recovered_without_client_retry": True,
                    "late_put_reaped_without_client_retry": True,
                    "image_id": second["id"],
                }
            )
        )
    finally:
        compose("up", "-d", "--no-deps", "storage", "worker")


if __name__ == "__main__":
    main()
