import asyncio
import json
import logging
import signal
from contextlib import suppress
from uuid import UUID

import aio_pika

from .broker import open_channel
from .config import settings
from .db import engine
from .service import claim, complete, heartbeat
from .storage import get
from .vision import encoder


def infer(row):
    content = get(row["object_key"], row["source_sha256"])
    return encoder().extract(content)


async def handle(payload):
    image_id, generation = UUID(payload["image_id"]), UUID(payload["generation"])
    row = await claim(image_id, generation)
    if not row:
        return False
    if settings.inference_delay_seconds:
        await asyncio.sleep(settings.inference_delay_seconds)
    result = await asyncio.to_thread(infer, row)
    return await complete(image_id, generation, result)


async def main():
    logging.basicConfig(level=logging.INFO)
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set)
    connection = await aio_pika.connect_robust(settings.amqp_url)

    async def consume(message):
        async with message.process(requeue=False):
            try:
                await handle(json.loads(message.body))
            except Exception as error:
                # Содержимое изображения не пишем в лог. Повторная попытка начнётся после lease.
                logging.error("Image indexing failed: %s", type(error).__name__)

    try:
        channel, queue = await open_channel(connection)
        tag = await queue.consume(consume)
        while not stop.is_set():
            await heartbeat("indexer")
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), 2)
        await queue.cancel(tag)
    finally:
        await connection.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
