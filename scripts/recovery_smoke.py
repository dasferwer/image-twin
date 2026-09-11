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
        print(
            json.dumps(
                {
                    "ok": True,
                    "s3_upload_retry": True,
                    "worker_sigkill_recovered": True,
                    "attempts": 2,
                    "index_commits": 1,
                    "image_id": second["id"],
                }
            )
        )
    finally:
        compose("up", "-d", "--no-deps", "storage", "worker")


if __name__ == "__main__":
    main()
