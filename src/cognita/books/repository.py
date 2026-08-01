import json
from datetime import datetime

import asyncpg

from cognita.books.domain import (
    Book,
    BookFormat,
    BookMetadata,
    BookStatus,
    BookSummary,
    LibraryStatus,
    TocEntry,
)
from cognita.core.logging import get_logger

logger = get_logger(__name__)

_CREATE_BOOKS_SQL = """
CREATE TABLE IF NOT EXISTS books (
    id              SERIAL PRIMARY KEY,
    status          TEXT NOT NULL DEFAULT 'pending',
    format          TEXT NOT NULL,
    file_data       BYTEA NOT NULL DEFAULT ''::bytea,
    file_size_bytes BIGINT NOT NULL DEFAULT 0,
    title           TEXT NOT NULL,
    author          TEXT,
    year            INTEGER,
    publisher       TEXT,
    language        TEXT NOT NULL DEFAULT 'en',
    isbn            TEXT,
    description     TEXT,
    source          TEXT,
    tags            JSONB NOT NULL DEFAULT '[]'::jsonb,
    toc             JSONB NOT NULL DEFAULT '[]'::jsonb,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# Brings a library created by an older, multi-user version up to date. Dropping
# user_id is required rather than cosmetic: the column was NOT NULL, so inserts
# would fail against a stale table.
_MIGRATE_BOOKS_SQL = [
    "ALTER TABLE books DROP COLUMN IF EXISTS user_id",
    "ALTER TABLE books DROP COLUMN IF EXISTS storage_path",
    "ALTER TABLE books ADD COLUMN IF NOT EXISTS source TEXT",
]

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_books_status ON books (status)",
    "CREATE INDEX IF NOT EXISTS idx_books_title_trgm ON books USING gin (title gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS idx_books_author_trgm ON books USING gin (author gin_trgm_ops)",
]

# Every column except the file bytes, which are large and rarely wanted.
_BOOK_COLUMNS = """
    id, status, format, file_size_bytes, title, author, year, publisher,
    language, isbn, description, source, tags, toc, chunk_count,
    error_message, created_at, updated_at
"""

_SUMMARY_COLUMNS = "id, title, author, status, format, chunk_count, created_at, error_message"


def _row_to_book(row: asyncpg.Record) -> Book:
    d = dict(row)
    toc_raw = d.get("toc") or []
    if isinstance(toc_raw, str):
        toc_raw = json.loads(toc_raw)
    tags_raw = d.get("tags") or []
    if isinstance(tags_raw, str):
        tags_raw = json.loads(tags_raw)

    meta = BookMetadata(
        title=d["title"],
        author=d["author"],
        year=d["year"],
        publisher=d["publisher"],
        language=d["language"],
        isbn=d["isbn"],
        description=d["description"],
        tags=list(tags_raw),
        source=d.get("source"),
    )
    return Book(
        id=d["id"],
        status=BookStatus(d["status"]),
        format=BookFormat(d["format"]),
        file_size_bytes=d["file_size_bytes"],
        metadata=meta,
        toc=[TocEntry(**e) for e in toc_raw],
        chunk_count=d["chunk_count"],
        error_message=d["error_message"],
        created_at=d["created_at"],
        updated_at=d["updated_at"],
    )


def _row_to_summary(row: asyncpg.Record) -> BookSummary:
    return BookSummary(
        id=row["id"],
        title=row["title"],
        author=row["author"],
        status=BookStatus(row["status"]),
        format=BookFormat(row["format"]),
        chunk_count=row["chunk_count"],
        created_at=row["created_at"],
        error_message=row["error_message"],
    )


class BookRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_BOOKS_SQL)
            for sql in _MIGRATE_BOOKS_SQL:
                try:
                    await conn.execute(sql)
                except Exception as exc:  # noqa: BLE001 — migrations are best-effort
                    logger.warning("Schema migration skipped (%s): %s", sql[:48], exc)
            for sql in _CREATE_INDEXES_SQL:
                try:
                    await conn.execute(sql)
                except Exception as exc:  # noqa: BLE001 — an index is an optimisation
                    logger.warning("Index creation skipped (%s): %s", sql[:48], exc)

    async def create(
        self,
        fmt: BookFormat,
        file_data: bytes,
        meta: BookMetadata,
    ) -> Book:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                INSERT INTO books
                    (status, format, file_data, file_size_bytes, title, author,
                     year, publisher, language, isbn, description, source, tags)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                RETURNING {_BOOK_COLUMNS}
                """,
                str(BookStatus.PENDING), str(fmt), file_data, len(file_data),
                meta.title, meta.author, meta.year, meta.publisher, meta.language,
                meta.isbn, meta.description, meta.source, json.dumps(meta.tags),
            )
        return _row_to_book(row)

    async def get(self, book_id: int) -> Book | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {_BOOK_COLUMNS} FROM books WHERE id = $1", book_id
            )
        return _row_to_book(row) if row else None

    async def get_file_data(self, book_id: int) -> bytes:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT file_data FROM books WHERE id = $1", book_id)
        if row is None:
            raise KeyError(f"Book {book_id} not found")
        return bytes(row["file_data"])

    async def list_all(self) -> list[BookSummary]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {_SUMMARY_COLUMNS} FROM books ORDER BY created_at DESC"
            )
        return [_row_to_summary(r) for r in rows]

    async def find(self, query: str, limit: int = 20) -> list[BookSummary]:
        """Fuzzy metadata search over title, author and description.

        Ranked by trigram similarity against title and author, so a near-miss
        spelling still finds the book.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT {_SUMMARY_COLUMNS}
                FROM books
                WHERE title ILIKE '%' || $1 || '%'
                   OR author ILIKE '%' || $1 || '%'
                   OR description ILIKE '%' || $1 || '%'
                   OR similarity(title, $1) > 0.3
                   OR similarity(coalesce(author, ''), $1) > 0.3
                ORDER BY GREATEST(
                    similarity(title, $1),
                    similarity(coalesce(author, ''), $1)
                ) DESC, created_at DESC
                LIMIT $2
                """,
                query, limit,
            )
        return [_row_to_summary(r) for r in rows]

    async def resolve_ids(
        self,
        book_ids: list[int] | None = None,
        authors: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> list[int] | None:
        """Narrow a search to a set of book ids.

        Returns None when no filter was requested, which callers read as
        "search everything". Returns an empty list when filters matched nothing,
        which correctly yields no results rather than silently searching all.
        """
        if not any((book_ids, authors, tags)):
            return None

        conditions: list[str] = []
        params: list = []
        if book_ids:
            params.append(book_ids)
            conditions.append(f"id = ANY(${len(params)}::int[])")
        if authors:
            # Any author term matching any book's author, case-insensitively.
            params.append(authors)
            conditions.append(
                f"EXISTS (SELECT 1 FROM unnest(${len(params)}::text[]) a "
                f"WHERE author ILIKE '%' || a || '%')"
            )
        if tags:
            params.append(tags)
            conditions.append(f"tags ?| ${len(params)}::text[]")

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id FROM books WHERE {' AND '.join(conditions)}", *params
            )
        return [r["id"] for r in rows]

    async def known_sources(self) -> set[str]:
        """Every recorded source path/URL — used to skip re-importing a folder."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT source FROM books WHERE source IS NOT NULL")
        return {r["source"] for r in rows}

    async def titles_for(self, book_ids: list[int]) -> dict[int, tuple[str, str | None]]:
        """Map book_id → (title, author) for a batch of books in one query."""
        if not book_ids:
            return {}
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, title, author FROM books WHERE id = ANY($1::int[])",
                list(set(book_ids)),
            )
        return {r["id"]: (r["title"], r["author"]) for r in rows}

    async def update_status(
        self,
        book_id: int,
        status: BookStatus,
        error_message: str | None = None,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE books SET status = $1, error_message = $2, updated_at = $3 WHERE id = $4",
                str(status), error_message, datetime.utcnow(), book_id,
            )

    async def update_metadata(self, book_id: int, meta: BookMetadata) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE books
                SET title = $1, author = $2, year = $3, publisher = $4,
                    language = $5, isbn = $6, description = $7, tags = $8,
                    updated_at = $9
                WHERE id = $10
                """,
                meta.title, meta.author, meta.year, meta.publisher, meta.language,
                meta.isbn, meta.description, json.dumps(meta.tags),
                datetime.utcnow(), book_id,
            )

    async def update_toc_and_count(
        self,
        book_id: int,
        toc: list[TocEntry],
        chunk_count: int,
    ) -> None:
        toc_data = json.dumps([
            {
                "title": e.title,
                "level": e.level,
                "sequence": e.sequence,
                "start_char": e.start_char,
                "page_start": e.page_start,
                "chunk_id": e.chunk_id,
            }
            for e in toc
        ])
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE books SET toc = $1, chunk_count = $2, updated_at = $3 WHERE id = $4",
                toc_data, chunk_count, datetime.utcnow(), book_id,
            )

    async def delete(self, book_id: int) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute("DELETE FROM books WHERE id = $1", book_id)
        return result == "DELETE 1"

    async def library_status(self) -> LibraryStatus:
        async with self._pool.acquire() as conn:
            counts = await conn.fetchrow(
                """
                SELECT
                    COUNT(*)                                          AS total,
                    COUNT(*) FILTER (WHERE status = 'ready')          AS ready,
                    COUNT(*) FILTER (WHERE status = 'processing')     AS processing,
                    COUNT(*) FILTER (WHERE status = 'pending')        AS pending,
                    COUNT(*) FILTER (WHERE status = 'failed')         AS failed,
                    COALESCE(SUM(chunk_count), 0)                     AS chunk_count
                FROM books
                """
            )
            failures = await conn.fetch(
                f"""
                SELECT {_SUMMARY_COLUMNS} FROM books
                WHERE status = 'failed' ORDER BY updated_at DESC LIMIT 10
                """
            )
        return LibraryStatus(
            total=counts["total"],
            ready=counts["ready"],
            processing=counts["processing"],
            pending=counts["pending"],
            failed=counts["failed"],
            chunk_count=counts["chunk_count"],
            queue_depth=0,  # filled in by the service from the live queue
            failures=[_row_to_summary(r) for r in failures],
        )
