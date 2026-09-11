import asyncio
from uuid import UUID

from sqlalchemy import text

from imagetwin.auth import hasher
from imagetwin.db import engine
from imagetwin.storage import ensure_bucket
from imagetwin.vision import encoder


async def seed():
    await asyncio.to_thread(ensure_bucket)
    version = await asyncio.to_thread(lambda: encoder().version)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO users(id,email,password_hash) VALUES (:id,'demo@example.com',:hash) ON CONFLICT(email) DO NOTHING"
            ),
            {
                "id": UUID("15000000-0000-0000-0000-000000000001"),
                "hash": hasher.hash("ImageTwinDemo123!"),
            },
        )
        user = await conn.scalar(text("SELECT id FROM users WHERE email='demo@example.com'"))
        await conn.execute(
            text(
                "INSERT INTO collections(id,user_id,name) VALUES (:id,:user,'Demo images') ON CONFLICT(user_id,name) DO NOTHING"
            ),
            {"id": UUID("15000000-0000-0000-0000-000000000010"), "user": user},
        )
    await engine.dispose()
    print("Seed ready; encoder:", version)


if __name__ == "__main__":
    asyncio.run(seed())
