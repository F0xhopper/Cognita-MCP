import json
import logging
from datetime import datetime

import asyncpg

from cognita.books.domain import (
    Book,
    BookFormat,
    BookMetadata,
    BookStatus,
    BookSummary,
    TocEntry,
)

logger = logging.getLogger(__name__)

_CREATE_BOOKS_SQL = """
CREATE TABLE IF NOT EXISTS books (
    id              SERIAL PRIMARY KEY,
    user_id         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    format          TEXT NOT NULL,
    storage_path    TEXT NOT NULL,
    file_size_bytes BIGINT NOT NULL DEFAULT 0,
    title           TEXT NOT NULL,
    author          TEXT,
    year            INTEGER,
    publisher       TEXT,
    language        TEXT NOT NULL DEFAULT 'en',
    isbn            TEXT,
    description     TEXT,
    tags            JSONB NOT NULL DEFAULT '[]'::jsonb,
    toc             JSONB NOT NULL DEFAULT '[]'::jsonb,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_books_user_id ON books (user_id)",
    "CREATE INDEX IF NOT EXISTS idx_books_status ON books (status)",
    "CREATE INDEX IF NOT EXISTS idx_books_title_trgm ON books USING gin (title gin_trgm_ops)",
]


def _row_to_book(row: asyncpg.Record) -> Book:
    d = dict(row)
    toc_raw = d.pop("toc") or []
    toc = (
        [TocEntry(**e) for e in toc_raw]
        if isinstance(toc_raw, list)
        else [TocEntry(**e) for e in json.loads(toc_raw)]
    )
    meta = BookMetadata(
        title=d.pop("title"),
        author=d.pop("author"),
        year=d.pop("year"),
        publisher=d.pop("publisher"),
        language=d.pop("language"),
        isbn=d.pop("isbn"),
        description=d.pop("description"),
        tags=d.pop("tags") or [],
    )
    return Book(
        id=d["id"],
        user_id=d["user_id"],
        status=BookStatus(d["status"]),
        format=BookFormat(d["format"]),
        storage_path=d["storage_path"],
        file_size_bytes=d["file_size_bytes"],
        metadata=meta,
        toc=toc,
        chunk_count=d["chunk_count"],
        error_message=d["error_message"],
        created_at=d["created_at"],
        updated_at=d["updated_at"],
    )


class BookRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_BOOKS_SQL)
            for sql in _CREATE_INDEXES_SQL:
                try:
                    await conn.execute(sql)
                except Exception as exc:
                    logger.warning("Could not create index: %s — %s", sql[:60], exc)

    async def create(
        self,
        user_id: str,
        status: BookStatus,
        fmt: BookFormat,
        storage_path: str,
        file_size_bytes: int,
        meta: BookMetadata,
    ) -> Book:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO books
                    (user_id, status, format, storage_path, file_size_bytes,
                     title, author, year, publisher, language, isbn, description, tags)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                RETURNING *
                """,
                user_id, str(status), str(fmt), storage_path, file_size_bytes,
                meta.title, meta.author, meta.year, meta.publisher,
                meta.language, meta.isbn, meta.description,
                json.dumps(meta.tags),
            )
        return _row_to_book(row)

    async def get(self, book_id: int, user_id: str) -> Book | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM books WHERE id = $1 AND user_id = $2",
                book_id, user_id,
            )
        return _row_to_book(row) if row else None

    async def list_for_user(self, user_id: str) -> list[BookSummary]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, title, author, status, format, chunk_count, created_at
                FROM books WHERE user_id = $1 ORDER BY created_at DESC
                """,
                user_id,
            )
        return [
            BookSummary(
                id=r["id"],
                title=r["title"],
                author=r["author"],
                status=BookStatus(r["status"]),
                format=BookFormat(r["format"]),
                chunk_count=r["chunk_count"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    async def search_library(self, user_id: str, query: str, limit: int = 20) -> list[BookSummary]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, title, author, status, format, chunk_count, created_at
                FROM books
                WHERE user_id = $1
                  AND status = 'ready'
                  AND (
                    title ILIKE '%' || $2 || '%'
                    OR author ILIKE '%' || $2 || '%'
                    OR description ILIKE '%' || $2 || '%'
                  )
                ORDER BY similarity(title, $2) DESC
                LIMIT $3
                """,
                user_id, query, limit,
            )
        return [
            BookSummary(
                id=r["id"],
                title=r["title"],
                author=r["author"],
                status=BookStatus(r["status"]),
                format=BookFormat(r["format"]),
                chunk_count=r["chunk_count"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    async def update_status(
        self,
        book_id: int,
        status: BookStatus,
        error_message: str | None = None,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE books
                SET status = $1, error_message = $2, updated_at = $3
                WHERE id = $4
                """,
                str(status), error_message, datetime.utcnow(), book_id,
            )

    async def update_toc_and_count(
        self,
        book_id: int,
        toc: list[TocEntry],
        chunk_count: int,
    ) -> None:
        toc_data = json.dumps([
            {
                "title": e.title, "level": e.level, "sequence": e.sequence,
                "start_char": e.start_char, "page_start": e.page_start,
                "chunk_id": e.chunk_id,
            }
            for e in toc
        ])
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE books
                SET toc = $1, chunk_count = $2, updated_at = $3
                WHERE id = $4
                """,
                toc_data, chunk_count, datetime.utcnow(), book_id,
            )

    async def delete(self, book_id: int, user_id: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM books WHERE id = $1 AND user_id = $2",
                book_id, user_id,
            )
        return result == "DELETE 1"
