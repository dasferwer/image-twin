"""Загружаем снимок и его копии, затем проверяем поиск через настоящий HTTP API."""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4

from fixtures import variants

BASE = os.environ.get("API_URL", "http://localhost:8000")
ROOT = Path(__file__).resolve().parents[1]


def request(path, token=None, method="GET", body=None, headers=None, expected=(200, 201, 202)):
    raw = isinstance(body, bytes)
    data = body if raw else json.dumps(body).encode() if body is not None else None
    supplied = {
        "Content-Type": "application/octet-stream" if raw else "application/json",
        **(headers or {}),
    }
    if token:
        supplied["Authorization"] = "Bearer " + token
    req = urllib.request.Request(BASE + path, data=data, headers=supplied, method=method)
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            status = response.status
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        status = error.code
        payload = json.load(error)
    assert status in expected, (path, status, payload)
    return payload


def login():
    return request(
        "/auth/login",
        method="POST",
        body={"email": "demo@example.com", "password": "ImageTwinDemo123!"},
    )["access_token"]


def collection(token):
    return request("/collections", token, "POST", {"name": "Smoke " + str(uuid4())})["id"]


def upload(token, cid, data, key=None, expected=(200, 202)):
    return request(
        "/collections/" + cid + "/images",
        token,
        "PUT",
        data,
        {
            "Idempotency-Key": key or str(uuid4()),
            "X-Content-SHA256": hashlib.sha256(data).hexdigest(),
        },
        expected,
    )


def wait_image(token, image_id, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = request("/images/" + image_id, token)
        if row["status"] == "ready":
            return row
        assert row["status"] not in {"failed", "deleted"}, row
        time.sleep(0.2)
    raise AssertionError("Image was not indexed: " + image_id)


def main():
    token = login()
    cid = collection(token)
    source = (ROOT / "data/coffee.png").read_bytes()
    key = str(uuid4())
    original = upload(token, cid, source, key)
    assert upload(token, cid, source, key)["id"] == original["id"]
    assert upload(token, cid, source)["id"] == original["id"]
    copies = {name: upload(token, cid, data)["id"] for name, data in variants(source).items()}
    unrelated = upload(token, cid, (ROOT / "data/eagle.png").read_bytes())
    for image_id in [original["id"], unrelated["id"], *copies.values()]:
        wait_image(token, image_id)
    for name, image_id in copies.items():
        result = request("/images/" + image_id + "/duplicates", token)
        ids = {row["image_id"] for row in result["matches"]}
        assert original["id"] in ids, (name, result)
        assert unrelated["id"] not in ids, (name, result)
    deleted = request("/images/" + copies["crop"], token, "DELETE")
    assert deleted["object_removed"]
    request("/images/" + copies["crop"], token, expected=(410,))
    final = request("/images/" + original["id"] + "/duplicates", token)
    assert copies["crop"] not in {row["image_id"] for row in final["matches"]}
    print(
        json.dumps(
            {
                "ok": True,
                "variants": list(copies),
                "exact_upload_deduplicated": True,
                "unrelated_image_rejected": True,
                "delete_removed_object_and_search_result": True,
                "collection_id": cid,
                "image_id": original["id"],
            }
        )
    )


if __name__ == "__main__":
    main()
