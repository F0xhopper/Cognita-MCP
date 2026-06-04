"""Celery worker for background book ingestion.

Run with:
    celery -A cognita.ingestion.worker worker --loglevel=info -Q ingestion
"""

import asyncio
import logging

import asyncpg
from celery import Celery

from cognita.core.config import settings
from cognita.core.logging import setup_logging
from cognita.ingestion.pipeline import ingest_book

setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = Celery(
    "cognita",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_routes={"cognita.ingestion.worker.ingest_book_task": {"queue": "ingestion"}},
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # one task at a time — ingestion is memory-intensive
)


@app.task(bind=True, name="cognita.ingestion.worker.ingest_book_task", max_retries=2)
def ingest_book_task(self, book_id: int) -> dict:
    logger.info("Worker starting ingestion: book_id=%d", book_id)
    try:
        pool = asyncio.get_event_loop().run_until_complete(_get_pool())
        asyncio.get_event_loop().run_until_complete(ingest_book(book_id, pool))
        return {"status": "ok", "book_id": book_id}
    except Exception as exc:
        logger.error("Task failed for book_id=%d: %s", book_id, exc)
        raise self.retry(exc=exc, countdown=60)


async def _get_pool() -> asyncpg.Pool:
    ssl = "require" if settings.DATABASE_SSL else None
    return await asyncpg.create_pool(settings.DATABASE_URL, min_size=1, max_size=3, ssl=ssl)


def run() -> None:
    app.worker_main(["worker", "--loglevel=info", "-Q", "ingestion"])
