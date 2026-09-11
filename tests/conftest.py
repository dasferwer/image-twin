from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import jwt
import pytest
from sqlalchemy import text

from imagetwin.config import settings
from imagetwin.db import engine
from imagetwin.main import app

assert settings.testing and settings.database_url.endswith("_test"), (
    "Tests require an isolated *_test database"
)


@pytest.fixture(autouse=True)
async def clean():
    from imagetwin import storage

    assert settings.s3_bucket == "imagetwin-test", "Tests require their own S3 bucket"
    storage.ensure_bucket()
    objects = storage.client().list_objects_v2(Bucket=settings.s3_bucket).get("Contents", [])
    if objects:
        storage.client().delete_objects(
            Bucket=settings.s3_bucket, Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]}
        )
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "TRUNCATE users,collections,images,uploads,image_events,worker_heartbeats RESTART IDENTITY CASCADE"
            )
        )
    yield


@pytest.fixture
async def client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as session:
        yield session


@pytest.fixture
async def identities():
    result = {}
    async with engine.begin() as conn:
        for name in ["admin", "alice", "bob"]:
            uid = uuid4()
            await conn.execute(
                text(
                    "INSERT INTO users(id,email,password_hash,role) VALUES (:id,:email,:hash,:role)"
                ),
                {
                    "id": uid,
                    "email": name + "@example.com",
                    "hash": "unused",
                    "role": "admin" if name == "admin" else "user",
                },
            )
            now = datetime.now(UTC)
            token = jwt.encode(
                {
                    "sub": str(uid),
                    "iat": now,
                    "exp": now + timedelta(hours=1),
                    "iss": "imagetwin",
                    "aud": "imagetwin",
                },
                settings.jwt_secret,
                algorithm="HS256",
            )
            result[name] = {"id": uid, "headers": {"Authorization": f"Bearer {token}"}}
    return result


@pytest.fixture(scope="session", autouse=True)
async def dispose_pool():
    yield
    await engine.dispose()
