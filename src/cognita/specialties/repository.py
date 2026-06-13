import json
import logging
from datetime import datetime

import asyncpg

from cognita.specialties.domain import SourceTier, SourceType, Specialty, SuggestedSource

logger = logging.getLogger(__name__)

_CREATE_SPECIALTIES_SQL = """
CREATE TABLE IF NOT EXISTS specialties (
    id              SERIAL PRIMARY KEY,
    user_id         TEXT NOT NULL,
    name            TEXT NOT NULL,
    description     TEXT,
    persona         TEXT,
    pending_corpus  JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, name)
);
"""

_ADD_PENDING_CORPUS_SQL = """
ALTER TABLE specialties
    ADD COLUMN IF NOT EXISTS pending_corpus JSONB NOT NULL DEFAULT '[]'::jsonb;
"""

_CREATE_SPECIALTY_BOOKS_SQL = """
CREATE TABLE IF NOT EXISTS specialty_books (
    specialty_id INTEGER NOT NULL REFERENCES specialties(id) ON DELETE CASCADE,
    book_id      INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    added_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (specialty_id, book_id)
);
"""

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_specialties_user_id ON specialties (user_id)",
    "CREATE INDEX IF NOT EXISTS idx_specialty_books_book_id ON specialty_books (book_id)",
]

_SELECT_WITH_BOOKS_SQL = """
SELECT s.*,
       COALESCE(
           ARRAY_AGG(sb.book_id ORDER BY sb.added_at) FILTER (WHERE sb.book_id IS NOT NULL),
           '{}'
       ) AS book_ids
FROM specialties s
LEFT JOIN specialty_books sb ON sb.specialty_id = s.id
"""


def _sources_to_json(items: list[SuggestedSource]) -> str:
    return json.dumps([
        {
            "title": s.title,
            "author": s.author,
            "tier": str(s.tier),
            "rationale": s.rationale,
            "source_url": s.source_url,
            "source_type": str(s.source_type),
            "approved": s.approved,
        }
        for s in items
    ])


def _json_to_sources(raw: list | str | None) -> list[SuggestedSource]:
    if not raw:
        return []
    items: list[dict] = raw if isinstance(raw, list) else json.loads(raw)
    return [
        SuggestedSource(
            title=d["title"],
            author=d["author"],
            tier=SourceTier(d["tier"]),
            rationale=d["rationale"],
            source_url=d.get("source_url"),
            source_type=SourceType(d["source_type"]),
            approved=d["approved"],
        )
        for d in items
    ]


def _row_to_specialty(row: asyncpg.Record) -> Specialty:
    d = dict(row)
    return Specialty(
        id=d["id"],
        user_id=d["user_id"],
        name=d["name"],
        description=d["description"],
        persona=d["persona"],
        book_ids=list(d.get("book_ids") or []),
        pending_corpus=_json_to_sources(d.get("pending_corpus")),
        created_at=d["created_at"],
        updated_at=d["updated_at"],
    )


class SpecialtyRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_SPECIALTIES_SQL)
            await conn.execute(_ADD_PENDING_CORPUS_SQL)
            await conn.execute(_CREATE_SPECIALTY_BOOKS_SQL)
            for sql in _CREATE_INDEXES_SQL:
                try:
                    await conn.execute(sql)
                except Exception as exc:
                    logger.warning("Could not create index: %s — %s", sql[:60], exc)

    async def create(
        self,
        user_id: str,
        name: str,
        description: str | None = None,
        persona: str | None = None,
    ) -> Specialty:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO specialties (user_id, name, description, persona)
                VALUES ($1, $2, $3, $4)
                RETURNING *
                """,
                user_id, name, description, persona,
            )
        d = dict(row)
        d["book_ids"] = []
        return _row_to_specialty(d)

    async def get(self, specialty_id: int, user_id: str) -> Specialty | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                _SELECT_WITH_BOOKS_SQL + " WHERE s.id = $1 AND s.user_id = $2 GROUP BY s.id",
                specialty_id, user_id,
            )
        return _row_to_specialty(row) if row else None

    async def get_by_name(self, user_id: str, name: str) -> Specialty | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                _SELECT_WITH_BOOKS_SQL + " WHERE s.user_id = $1 AND s.name = $2 GROUP BY s.id",
                user_id, name,
            )
        return _row_to_specialty(row) if row else None

    async def list_for_user(self, user_id: str) -> list[Specialty]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                _SELECT_WITH_BOOKS_SQL + " WHERE s.user_id = $1 GROUP BY s.id ORDER BY s.name",
                user_id,
            )
        return [_row_to_specialty(r) for r in rows]

    async def update(
        self,
        specialty_id: int,
        user_id: str,
        name: str,
        description: str | None,
        persona: str | None,
    ) -> Specialty | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE specialties
                SET name = $1, description = $2, persona = $3, updated_at = $4
                WHERE id = $5 AND user_id = $6
                RETURNING *
                """,
                name, description, persona, datetime.utcnow(), specialty_id, user_id,
            )
        if row is None:
            return None
        return await self.get(specialty_id, user_id)

    async def delete(self, specialty_id: int, user_id: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM specialties WHERE id = $1 AND user_id = $2",
                specialty_id, user_id,
            )
        return result == "DELETE 1"

    async def add_book(self, specialty_id: int, book_id: int) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO specialty_books (specialty_id, book_id)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
                """,
                specialty_id, book_id,
            )

    async def remove_book(self, specialty_id: int, book_id: int) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM specialty_books WHERE specialty_id = $1 AND book_id = $2",
                specialty_id, book_id,
            )
        return result == "DELETE 1"

    async def save_pending_corpus(self, specialty_id: int, items: list[SuggestedSource]) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE specialties SET pending_corpus = $1 WHERE id = $2",
                _sources_to_json(items), specialty_id,
            )

    async def clear_pending_corpus(self, specialty_id: int) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE specialties SET pending_corpus = '[]'::jsonb WHERE id = $1",
                specialty_id,
            )
