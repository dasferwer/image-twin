import asyncio
import hashlib
import json
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import text

from . import storage
from .auth import User, current_user, router
from .cleanup import cleanup_due, deletion_state
from .config import settings
from .db import engine
from .observability import instrument
from .service import event
from .vision import compare, normalize

app = FastAPI(
    title="ImageTwin",
    version="0.1.0",
    description="Find exact and modified copies of images inside a private collection.",
)
app.include_router(router)
instrument(app)
FIELDS = "id,collection_id,width,height,status,attempts,encoder_version,error,created_at,updated_at"


class CollectionInput(BaseModel):
    name: str = Field(min_length=1, max_length=100, pattern=r".*\S.*")


async def collection(conn, collection_id, user, lock=False):
    row = (
        (
            await conn.execute(
                text(
                    "SELECT * FROM collections WHERE id=:id AND user_id=:user"
                    + (" FOR UPDATE" if lock else "")
                ),
                {"id": collection_id, "user": user.id},
            )
        )
        .mappings()
        .first()
    )
    if not row:
        raise HTTPException(404, "Collection not found")
    return row


async def owned_image(conn, image_id, user, include_deleted=False):
    row = (
        (
            await conn.execute(
                text(
                    "SELECT i.*,i.phash::text AS hash_bits,i.embedding::text AS vector_text FROM images i JOIN collections c ON c.id=i.collection_id WHERE i.id=:id AND c.user_id=:user"
                ),
                {"id": image_id, "user": user.id},
            )
        )
        .mappings()
        .first()
    )
    if not row:
        raise HTTPException(404, "Image not found")
    if row["status"] == "deleted" and not include_deleted:
        raise HTTPException(410, "Image was deleted")
    return dict(row)


def public_image(row):
    return {key: row[key] for key in FIELDS.split(",")}


def features(row):
    return {
        "phash": row["hash_bits"],
        "vector": json.loads(row["vector_text"]),
        "points": row["points"],
        "descriptors": row["descriptors"],
        "contrast": row["contrast"],
    }


@app.get("/health", tags=["Operations"])
async def health():
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
        rows = (
            (
                await conn.execute(
                    text(
                        "SELECT name,extract(epoch FROM clock_timestamp()-updated_at) AS age FROM worker_heartbeats"
                    )
                )
            )
            .mappings()
            .all()
        )
    return {
        "status": "ok",
        "database": "ok",
        "workers": {row["name"]: float(row["age"]) for row in rows},
    }


@app.post("/collections", status_code=201, tags=["Collections"])
async def create_collection(data: CollectionInput, user: User = Depends(current_user)):
    if "\x00" in data.name:
        raise HTTPException(422, "Collection name must not contain NUL")
    async with engine.begin() as conn:
        row = (
            (
                await conn.execute(
                    text(
                        "INSERT INTO collections(id,user_id,name) VALUES (:id,:user,:name) ON CONFLICT(user_id,name) DO UPDATE SET name=EXCLUDED.name RETURNING id,name,created_at"
                    ),
                    {"id": uuid4(), "user": user.id, "name": data.name.strip()},
                )
            )
            .mappings()
            .one()
        )
    return dict(row)


@app.get("/collections", tags=["Collections"])
async def collections(user: User = Depends(current_user)):
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text(
                        "SELECT id,name,created_at FROM collections WHERE user_id=:user ORDER BY created_at"
                    ),
                    {"user": user.id},
                )
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


@app.get("/collections/{collection_id}/images", tags=["Images"])
async def images(
    collection_id: UUID,
    after: UUID | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    user: User = Depends(current_user),
):
    async with engine.connect() as conn:
        await collection(conn, collection_id, user)
        rows = (
            (
                await conn.execute(
                    text(
                        f"SELECT {FIELDS} FROM images WHERE collection_id=:collection AND status!='deleted' AND (CAST(:after AS uuid) IS NULL OR id>:after) ORDER BY id LIMIT :limit"
                    ),
                    {"collection": collection_id, "after": after, "limit": limit},
                )
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


@app.put(
    "/collections/{collection_id}/images",
    status_code=202,
    tags=["Images"],
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
            },
        }
    },
)
async def upload(
    collection_id: UUID,
    request: Request,
    response: Response,
    user: User = Depends(current_user),
    idempotency_key: str = Header(min_length=1, max_length=100),
    x_content_sha256: str = Header(pattern=r"^[0-9a-f]{64}$"),
):
    async with engine.connect() as conn:
        await collection(conn, collection_id, user)
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > settings.max_upload_bytes:
            raise HTTPException(413, "Image file is too large")
        raw.extend(chunk)
    if hashlib.sha256(raw).hexdigest() != x_content_sha256:
        raise HTTPException(422, "Upload checksum mismatch")
    try:
        content, pixel_sha, width, height = await asyncio.to_thread(normalize, bytes(raw))
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    source_sha = hashlib.sha256(content).hexdigest()
    async with engine.begin() as conn:
        await collection(conn, collection_id, user, lock=True)
        previous = (
            (
                await conn.execute(
                    text("SELECT * FROM uploads WHERE collection_id=:collection AND key=:key"),
                    {"collection": collection_id, "key": idempotency_key},
                )
            )
            .mappings()
            .first()
        )
        if previous and previous["request_sha256"] != x_content_sha256:
            raise HTTPException(409, "Idempotency key was used for another file")
        created = False
        if previous:
            row = await owned_image(conn, previous["image_id"], user)
        else:
            row = (
                (
                    await conn.execute(
                        text(
                            "SELECT * FROM images WHERE collection_id=:collection AND pixel_sha256=:sha AND status!='deleted'"
                        ),
                        {"collection": collection_id, "sha": pixel_sha},
                    )
                )
                .mappings()
                .first()
            )
            if row is None:
                count = await conn.scalar(
                    text(
                        "SELECT count(*) FROM images WHERE collection_id=:collection AND status!='deleted'"
                    ),
                    {"collection": collection_id},
                )
                if count >= settings.max_collection_images:
                    raise HTTPException(409, "Collection image limit reached")
                image_id = uuid4()
                row = (
                    (
                        await conn.execute(
                            text(
                                "INSERT INTO images(id,collection_id,pixel_sha256,source_sha256,object_key,width,height) VALUES (:id,:collection,:pixel,:source,:object,:width,:height) RETURNING *"
                            ),
                            {
                                "id": image_id,
                                "collection": collection_id,
                                "pixel": pixel_sha,
                                "source": source_sha,
                                "object": f"{collection_id}/{image_id}.png",
                                "width": width,
                                "height": height,
                            },
                        )
                    )
                    .mappings()
                    .one()
                )
                created = True
            await conn.execute(
                text(
                    "INSERT INTO uploads(collection_id,key,request_sha256,image_id) VALUES (:collection,:key,:sha,:image)"
                ),
                {
                    "collection": collection_id,
                    "key": idempotency_key,
                    "sha": x_content_sha256,
                    "image": row["id"],
                },
            )
        if row["status"] != "uploading":
            response.status_code = 200
            return public_image(row)
        now = await conn.scalar(text("SELECT clock_timestamp()"))
        if not created and row["lease_until"] and row["lease_until"] > now:
            raise HTTPException(409, "Upload is in progress; retry after its lease expires")
        token = uuid4()
        await conn.execute(
            text(
                "UPDATE images SET generation=:token,lease_until=clock_timestamp()+interval '60 seconds' WHERE id=:id"
            ),
            {"id": row["id"], "token": token},
        )
        await event(conn, row["id"], "upload_started")
    # Путь и намерение уже записаны в БД. Повтор загрузит те же пиксели по тому же ключу.
    try:
        await asyncio.to_thread(storage.put, row["object_key"], content)
    except Exception as error:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE images SET lease_until=clock_timestamp() WHERE id=:id AND generation=:token AND status='uploading'"
                ),
                {"id": row["id"], "token": token},
            )
        raise HTTPException(503, "Object storage is unavailable; retry the same upload") from error
    async with engine.begin() as conn:
        saved = await conn.scalar(
            text(
                "UPDATE images SET status='pending',lease_until=NULL,updated_at=clock_timestamp() WHERE id=:id AND generation=:token AND status='uploading' RETURNING id"
            ),
            {"id": row["id"], "token": token},
        )
        if not saved:
            raise HTTPException(409, "Upload lease was replaced; read the image status")
        await event(conn, row["id"], "uploaded")
        result = await owned_image(conn, row["id"], user)
    return public_image(result)


@app.get("/images/{image_id}", tags=["Images"])
async def image(image_id: UUID, user: User = Depends(current_user)):
    async with engine.connect() as conn:
        return public_image(await owned_image(conn, image_id, user))


@app.get("/images/{image_id}/content", tags=["Images"])
async def content(image_id: UUID, user: User = Depends(current_user)):
    async with engine.connect() as conn:
        row = await owned_image(conn, image_id, user)
    if row["status"] == "uploading":
        raise HTTPException(409, "Upload is not complete")
    try:
        data = await asyncio.to_thread(storage.get, row["object_key"], row["source_sha256"])
    except Exception as error:
        raise HTTPException(503, "Image content is temporarily unavailable") from error
    return Response(
        data,
        media_type="image/png",
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/images/{image_id}/events", tags=["Images"])
async def events(image_id: UUID, user: User = Depends(current_user)):
    async with engine.connect() as conn:
        await owned_image(conn, image_id, user)
        rows = (
            (
                await conn.execute(
                    text(
                        "SELECT id,type,created_at FROM image_events WHERE image_id=:id ORDER BY id"
                    ),
                    {"id": image_id},
                )
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


@app.get("/images/{image_id}/duplicates", tags=["Search"])
async def duplicates(
    image_id: UUID, limit: int = Query(default=20, ge=1, le=80), user: User = Depends(current_user)
):
    async with engine.connect() as conn:
        source = await owned_image(conn, image_id, user)
        if source["status"] != "ready":
            raise HTTPException(409, "Image is not indexed yet")
        # Каждый поиск ограничен коллекцией владельца. Две выборки помогают не терять обрезанные копии.
        rows = (
            (
                await conn.execute(
                    text("""WITH vectors AS (
        SELECT id FROM images WHERE collection_id=:collection AND status='ready' AND id!=:id AND encoder_version=:version
        ORDER BY embedding <=> CAST(:vector AS vector),id LIMIT 40
        ), hashes AS (
        SELECT id FROM images WHERE collection_id=:collection AND status='ready' AND id!=:id AND encoder_version=:version
        ORDER BY bit_count(phash # CAST(CAST(:hash AS text) AS bit(64))),id LIMIT 40
        ) SELECT i.*,phash::text AS hash_bits,embedding::text AS vector_text FROM images i
        WHERE i.id IN (SELECT id FROM vectors UNION SELECT id FROM hashes)"""),
                    {
                        "collection": source["collection_id"],
                        "id": image_id,
                        "version": source["encoder_version"],
                        "vector": source["vector_text"],
                        "hash": source["hash_bits"],
                    },
                )
            )
            .mappings()
            .all()
        )

    def rank():
        results = []
        for candidate in rows:
            evidence = compare(features(source), features(candidate))
            if evidence["duplicate"]:
                results.append({"image_id": candidate["id"], **evidence})
        results.sort(
            key=lambda item: (
                -item["geometric_inliers"],
                -item["cosine_similarity"],
                str(item["image_id"]),
            )
        )
        return results[:limit]

    matches = await asyncio.to_thread(rank)
    return {
        "image_id": image_id,
        "encoder_version": source["encoder_version"],
        "candidates_considered": len(rows),
        "candidate_limit": 80,
        "matches": matches,
    }


@app.post("/images/{image_id}/reindex", tags=["Images"])
async def reindex(image_id: UUID, user: User = Depends(current_user)):
    async with engine.begin() as conn:
        row = await owned_image(conn, image_id, user)
        changed = await conn.scalar(
            text(
                "UPDATE images SET status='pending',attempts=0,error=NULL,embedding=NULL,phash=NULL,points=NULL,descriptors=NULL WHERE id=:id AND status IN ('ready','failed') RETURNING id"
            ),
            {"id": image_id},
        )
        if not changed:
            raise HTTPException(409, "Indexing is already in progress")
        await event(conn, image_id, "reindex_requested")
    return {"id": row["id"], "status": "pending"}


@app.delete("/images/{image_id}", tags=["Images"])
async def delete(image_id: UUID, user: User = Depends(current_user)):
    async with engine.begin() as conn:
        row = await owned_image(conn, image_id, user, include_deleted=True)
        if row["status"] == "uploading":
            raise HTTPException(409, "Finish the upload before deleting it")
        # Сначала убираем запись из поиска. Даже при ошибке S3 она больше не выдаётся клиентам.
        changed = await conn.scalar(
            text(
                "UPDATE images SET status='deleted',generation=NULL,lease_until=NULL,embedding=NULL,phash=NULL,points=NULL,descriptors=NULL WHERE id=:id AND status!='deleted' RETURNING id"
            ),
            {"id": image_id},
        )
        if changed:
            await event(conn, image_id, "deleted")
        await conn.execute(
            text(
                "INSERT INTO object_deletions(image_id,object_key) VALUES (:id,:key) "
                "ON CONFLICT(image_id) DO NOTHING"
            ),
            {"id": image_id, "key": row["object_key"]},
        )
    await cleanup_due(image_id)
    async with engine.connect() as conn:
        state = await deletion_state(conn, image_id)
    return {
        "id": image_id,
        "status": "deleted",
        "object_removed": state["status"] == "removed",
        "cleanup": state,
    }


@app.get("/images/{image_id}/deletion", tags=["Images"])
async def read_deletion(image_id: UUID, user: User = Depends(current_user)):
    async with engine.connect() as conn:
        await owned_image(conn, image_id, user, include_deleted=True)
        return {"id": image_id, **await deletion_state(conn, image_id)}
