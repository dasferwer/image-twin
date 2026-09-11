import asyncio
import hashlib
import io
import runpy
from pathlib import Path
from uuid import UUID

import pytest
from PIL import Image
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from imagetwin import storage
from imagetwin.config import settings
from imagetwin.db import engine
from imagetwin.service import claim, complete, reserve_due
from imagetwin.vision import compare, encoder, normalize
from imagetwin.worker import handle, infer

variants = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/fixtures.py"))[
    "variants"
]
PHOTO = Path("data/coffee.png")


async def collection(client, identities, name="test", owner="alice"):
    response = await client.post(
        "/collections", headers=identities[owner]["headers"], json={"name": name}
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def upload(client, identities, cid, data=None, key="image-1", expected=(200, 202)):
    data = data if data is not None else PHOTO.read_bytes()
    response = await client.put(
        "/collections/" + cid + "/images",
        headers={
            **identities["alice"]["headers"],
            "Idempotency-Key": key,
            "X-Content-SHA256": hashlib.sha256(data).hexdigest(),
            "Content-Type": "application/octet-stream",
        },
        content=data,
    )
    assert response.status_code in expected, response.text
    return response


async def process():
    payloads = await reserve_due()
    for payload in payloads:
        await handle(payload)
    return payloads


async def test_exact_pixels_and_keys_do_not_create_extra_objects(client, identities):
    cid = await collection(client, identities)
    first = (await upload(client, identities, cid)).json()
    second = (await upload(client, identities, cid, key="different-key")).json()
    assert first["id"] == second["id"]
    async with engine.connect() as conn:
        assert await conn.scalar(text("SELECT count(*) FROM images")) == 1
        assert await conn.scalar(text("SELECT count(*) FROM uploads")) == 2
    assert len(storage.client().list_objects_v2(Bucket=settings.s3_bucket)["Contents"]) == 1


async def test_concurrent_uploads_share_one_image(client, identities):
    cid = await collection(client, identities)
    responses = await asyncio.gather(
        *[upload(client, identities, cid, expected=(200, 202, 409)) for _ in range(8)]
    )
    assert any(r.status_code == 202 for r in responses)
    async with engine.connect() as conn:
        assert await conn.scalar(text("SELECT count(*) FROM images")) == 1
    assert (await upload(client, identities, cid)).status_code == 200


async def test_different_file_rejects_reused_key(client, identities):
    cid = await collection(client, identities)
    await upload(client, identities, cid)
    assert (
        await upload(client, identities, cid, variants(PHOTO.read_bytes())["crop"], expected=(409,))
    ).status_code == 409


async def test_failed_object_upload_can_be_retried(client, identities, monkeypatch):
    cid = await collection(client, identities)
    with monkeypatch.context() as patch:
        patch.setattr(storage, "put", lambda *args: (_ for _ in ()).throw(OSError("storage down")))
        await upload(client, identities, cid, expected=(503,))
    async with engine.connect() as conn:
        assert await conn.scalar(text("SELECT status FROM images")) == "uploading"
    response = await upload(client, identities, cid)
    assert response.json()["status"] == "pending"
    assert await reserve_due()


async def test_upload_after_lost_ready_response_is_idempotent(client, identities):
    cid = await collection(client, identities)
    first = await upload(client, identities, cid)
    again = await upload(client, identities, cid)
    assert first.json()["id"] == again.json()["id"]
    async with engine.connect() as conn:
        assert (
            await conn.scalar(text("SELECT count(*) FROM image_events WHERE type='uploaded'")) == 1
        )


async def test_indexed_vector_and_duplicate_delivery(client, identities):
    cid = await collection(client, identities)
    row = (await upload(client, identities, cid)).json()
    payload = (await process())[0]
    assert not await handle(payload)
    async with engine.connect() as conn:
        assert await conn.scalar(text("SELECT vector_dims(embedding) FROM images")) == 1280
        assert (
            await conn.scalar(text("SELECT count(*) FROM image_events WHERE type='indexed'")) == 1
        )
    data = await client.get(
        "/images/" + row["id"] + "/content", headers=identities["alice"]["headers"]
    )
    assert data.status_code == 200 and data.headers["content-type"] == "image/png"


async def test_similarity_finds_crop_but_respects_collection(client, identities):
    cid = await collection(client, identities)
    original = (await upload(client, identities, cid)).json()
    crop = (
        await upload(client, identities, cid, variants(PHOTO.read_bytes())["crop"], key="crop")
    ).json()
    other = await collection(client, identities, name="other")
    hidden = (await upload(client, identities, other)).json()
    await process()
    response = await client.get(
        "/images/" + crop["id"] + "/duplicates", headers=identities["alice"]["headers"]
    )
    assert response.status_code == 200, response.text
    ids = [match["image_id"] for match in response.json()["matches"]]
    assert original["id"] in ids and hidden["id"] not in ids


async def test_other_user_cannot_read_content_search_or_delete(client, identities):
    cid = await collection(client, identities)
    row = (await upload(client, identities, cid)).json()
    for suffix in ["", "/content", "/duplicates", "/events"]:
        assert (
            await client.get("/images/" + row["id"] + suffix, headers=identities["bob"]["headers"])
        ).status_code == 404
    assert (
        await client.delete("/images/" + row["id"], headers=identities["bob"]["headers"])
    ).status_code == 404
    assert (
        await client.get("/collections/" + cid + "/images", headers=identities["bob"]["headers"])
    ).status_code == 404


async def test_old_worker_cannot_replace_new_result(client, identities):
    cid = await collection(client, identities)
    await upload(client, identities, cid)
    old = (await reserve_due())[0]
    claimed = await claim(UUID(old["image_id"]), UUID(old["generation"]))
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE images SET lease_until=clock_timestamp()-interval '1 second'")
        )
    new = (await reserve_due())[0]
    assert not await complete(UUID(old["image_id"]), UUID(old["generation"]), infer(claimed))
    assert await handle(new)


async def test_delete_fences_inflight_worker(client, identities):
    cid = await collection(client, identities)
    row = (await upload(client, identities, cid)).json()
    payload = (await reserve_due())[0]
    claimed = await claim(UUID(payload["image_id"]), UUID(payload["generation"]))
    saved_features = infer(claimed)
    deleted = await client.delete("/images/" + row["id"], headers=identities["alice"]["headers"])
    assert deleted.json()["object_removed"]
    assert not await complete(
        UUID(payload["image_id"]), UUID(payload["generation"]), saved_features
    )
    assert (
        await client.get("/images/" + row["id"], headers=identities["alice"]["headers"])
    ).status_code == 410
    assert not storage.client().list_objects_v2(Bucket=settings.s3_bucket).get("Contents")
    replacement = (await upload(client, identities, cid, key="new-after-delete")).json()
    assert replacement["id"] != row["id"]


async def test_index_and_event_are_atomic(client, identities, monkeypatch):
    cid = await collection(client, identities)
    await upload(client, identities, cid)
    payload = (await reserve_due())[0]
    row = await claim(UUID(payload["image_id"]), UUID(payload["generation"]))
    features = infer(row)
    original = AsyncConnection.execute

    async def fail(self, statement, *args, **kwargs):
        if "INSERT INTO image_events" in str(statement):
            raise RuntimeError("event failed")
        return await original(self, statement, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(AsyncConnection, "execute", fail)
        with pytest.raises(RuntimeError):
            await complete(UUID(payload["image_id"]), UUID(payload["generation"]), features)
    async with engine.connect() as conn:
        assert await conn.scalar(text("SELECT embedding IS NULL FROM images"))
        assert await conn.scalar(text("SELECT status FROM images")) == "processing"
    assert await complete(UUID(payload["image_id"]), UUID(payload["generation"]), features)


async def test_quota_checksum_and_invalid_content(client, identities, monkeypatch):
    cid = await collection(client, identities)
    monkeypatch.setattr(settings, "max_collection_images", 1)
    await upload(client, identities, cid)
    await upload(
        client, identities, cid, variants(PHOTO.read_bytes())["crop"], key="second", expected=(409,)
    )
    await upload(client, identities, cid, b"not an image", key="bad", expected=(422,))
    response = await client.put(
        "/collections/" + cid + "/images",
        headers={
            **identities["alice"]["headers"],
            "Idempotency-Key": "wrong-hash",
            "X-Content-SHA256": "0" * 64,
        },
        content=PHOTO.read_bytes(),
    )
    assert response.status_code == 422


async def test_tampered_object_is_not_indexed(client, identities):
    cid = await collection(client, identities)
    await upload(client, identities, cid)
    payload = (await reserve_due())[0]
    async with engine.connect() as conn:
        key = await conn.scalar(text("SELECT object_key FROM images"))
    storage.put(key, b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        await handle(payload)
    async with engine.connect() as conn:
        assert await conn.scalar(text("SELECT status FROM images")) == "processing"


async def test_reindex_preserves_image_identity(client, identities):
    cid = await collection(client, identities)
    row = (await upload(client, identities, cid)).json()
    await process()
    response = await client.post(
        "/images/" + row["id"] + "/reindex", headers=identities["alice"]["headers"]
    )
    assert response.status_code == 200
    await process()
    async with engine.connect() as conn:
        assert (
            await conn.scalar(text("SELECT count(*) FROM image_events WHERE type='indexed'")) == 2
        )
        assert await conn.scalar(text("SELECT count(*) FROM images")) == 1


@pytest.mark.parametrize("kind", ["jpeg", "resize", "crop", "watermark"])
def test_transformed_photo_matches(kind):
    source = PHOTO.read_bytes()
    a = encoder().extract(normalize(source)[0])
    b = encoder().extract(normalize(variants(source)[kind])[0])
    assert compare(a, b)["duplicate"]


def test_featureless_colors_are_not_automatically_duplicates():
    def feature(color):
        buffer = io.BytesIO()
        Image.new("RGB", (80, 80), color).save(buffer, format="PNG")
        return encoder().extract(buffer.getvalue())

    assert not compare(feature("red"), feature("blue"))["duplicate"]


def test_pixel_limit_and_animated_input(monkeypatch):
    monkeypatch.setattr(settings, "max_pixels", 100)
    with pytest.raises(ValueError, match="dimensions"):
        normalize(PHOTO.read_bytes())
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buffer, format="GIF")
    with pytest.raises(ValueError, match="PNG and JPEG"):
        normalize(buffer.getvalue())
