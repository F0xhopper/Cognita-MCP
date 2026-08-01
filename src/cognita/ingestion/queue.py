"""A small background queue for ingestion jobs.

Adding a book returns immediately; the actual parsing and embedding happens
here. A queue rather than a bare ``asyncio.create_task`` for two reasons:

  * A task created and never referenced can be garbage-collected mid-run. The
    worker tasks here are held for the lifetime of the process.
  * Importing a folder of fifty books at once would otherwise fire fifty
    concurrent ingestions and hit every rate limit at the same time.
"""

import asyncio

import asyncpg

from cognita.core.config import settings
from cognita.core.logging import get_logger
from cognita.ingestion.pipeline import ingest_book

logger = get_logger(__name__)


class IngestionQueue:
    """FIFO queue draining into a fixed number of workers."""

    def __init__(self, pool: asyncpg.Pool, concurrency: int | None = None) -> None:
        self._pool = pool
        self._concurrency = max(1, concurrency or settings.INGEST_CONCURRENCY)
        self._queue: asyncio.Queue[int] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._in_flight: set[int] = set()

    def submit(self, book_id: int) -> None:
        """Schedule a book for ingestion. Returns as soon as it is queued."""
        self._ensure_workers()
        self._queue.put_nowait(book_id)

    @property
    def depth(self) -> int:
        """Books waiting or currently being ingested."""
        return self._queue.qsize() + len(self._in_flight)

    def _ensure_workers(self) -> None:
        # Started lazily so constructing the queue never requires a running loop.
        self._workers = [w for w in self._workers if not w.done()]
        while len(self._workers) < self._concurrency:
            self._workers.append(asyncio.create_task(self._worker(len(self._workers))))

    async def _worker(self, index: int) -> None:
        logger.debug("Ingestion worker %d started", index)
        while True:
            book_id = await self._queue.get()
            self._in_flight.add(book_id)
            try:
                await ingest_book(book_id, self._pool)
            except Exception as exc:  # noqa: BLE001
                # ingest_book already recorded the failure on the book row; the
                # worker must survive so the rest of the queue still drains.
                logger.warning("Ingestion of book_id=%d failed: %s", book_id, exc)
            finally:
                self._in_flight.discard(book_id)
                self._queue.task_done()

    async def drain(self) -> None:
        """Wait for everything queued so far to finish. Used by tests."""
        await self._queue.join()

    async def close(self) -> None:
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
