"""Chunk repository — stores text chunks and pgvector embeddings.

Hybrid search combines:
  - pgvector cosine similarity (semantic)
  - PostgreSQL tsvector (full-text / BM25-like)
  - RRF (Reciprocal Rank Fusion) to merge ranked lists
"""

import json
import logging

import asyncpg

from cognita.chunks.domain import Chunk, ChunkLevel, ChunkLocation

logger = logging.getLogger(__name__)

_CREATE_CHUNKS_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    id             SERIAL PRIMARY KEY,
    book_id        INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    user_id        TEXT NOT NULL,
    text           TEXT NOT NULL,
    level          TEXT NOT NULL DEFAULT 'paragraph',
    sequence       INTEGER NOT NULL,
    chapter_title  TEXT,
    chapter_n      INTEGER,
    section_title  TEXT,
    section_n      INTEGER,
    page_start     INTEGER,
    page_end       INTEGER,
    char_start     INTEGER,
    char_end       INTEGER,
    paragraph_n    INTEGER,
    token_count    INTEGER NOT NULL DEFAULT 0,
    embedding      vector(3072),
    fts            TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_CREATE_CHUNK_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_chunks_book_id ON chunks (book_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_user_id ON chunks (user_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_sequence ON chunks (book_id, sequence)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING gin (fts)",
]

_HYBRID_SEARCH_SQL = """
WITH semantic AS (
    SELECT id,
           1 - (embedding <=> $1::vector) AS sem_score,
           ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS sem_rank
    FROM chunks
    WHERE user_id = $2
      AND ($3::int[] IS NULL OR book_id = ANY($3::int[]))
      AND embedding IS NOT NULL
    ORDER BY embedding <=> $1::vector
    LIMIT $4
),
keyword AS (
    SELECT id,
           ts_rank_cd(fts, plainto_tsquery('english', $5)) AS kw_score,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank_cd(fts, plainto_tsquery('english', $5)) DESC
           ) AS kw_rank
    FROM chunks
    WHERE user_id = $2
      AND ($3::int[] IS NULL OR book_id = ANY($3::int[]))
      AND fts @@ plainto_tsquery('english', $5)
    ORDER BY kw_score DESC
    LIMIT $4
),
rrf AS (
    SELECT
        COALESCE(s.id, k.id) AS id,
        COALESCE(1.0 / (60 + s.sem_rank), 0) +
        COALESCE(1.0 / (60 + k.kw_rank), 0) AS rrf_score
    FROM semantic s
    FULL OUTER JOIN keyword k ON s.id = k.id
)
SELECT c.*, rrf.rrf_score
FROM rrf
JOIN chunks c ON rrf.id = c.id
ORDER BY rrf.rrf_score DESC
LIMIT $6
"""


def _row_to_chunk(row: asyncpg.Record) -> Chunk:
    d = dict(row)
    loc = ChunkLocation(
        chapter_title=d.get("chapter_title"),
        chapter_n=d.get("chapter_n"),
        section_title=d.get("section_title"),
        section_n=d.get("section_n"),
        page_start=d.get("page_start"),
        page_end=d.get("page_end"),
        char_start=d.get("char_start"),
        char_end=d.get("char_end"),
        paragraph_n=d.get("paragraph_n"),
    )
    raw_emb = d.get("embedding")
    embedding = list(raw_emb) if raw_emb is not None else []
    return Chunk(
        id=d["id"],
        book_id=d["book_id"],
        user_id=d["user_id"],
        text=d["text"],
        level=ChunkLevel(d["level"]),
        sequence=d["sequence"],
        location=loc,
        embedding=embedding,
        token_count=d.get("token_count", 0),
    )


class ChunkRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_CHUNKS_SQL)
            for sql in _CREATE_CHUNK_INDEXES_SQL:
                try:
                    await conn.execute(sql)
                except Exception as exc:
                    logger.warning("Index creation skipped: %s", exc)

    async def bulk_insert(self, chunks: list[Chunk]) -> list[int]:
        """Insert chunks in a single transaction. Returns assigned IDs."""
        if not chunks:
            return []
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                ids = []
                for c in chunks:
                    emb_str = f"[{','.join(str(v) for v in c.embedding)}]" if c.embedding else None
                    row = await conn.fetchrow(
                        """
                        INSERT INTO chunks
                            (book_id, user_id, text, level, sequence,
                             chapter_title, chapter_n, section_title, section_n,
                             page_start, page_end, char_start, char_end,
                             paragraph_n, token_count, embedding)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16::vector)
                        RETURNING id
                        """,
                        c.book_id, c.user_id, c.text, str(c.level), c.sequence,
                        c.location.chapter_title, c.location.chapter_n,
                        c.location.section_title, c.location.section_n,
                        c.location.page_start, c.location.page_end,
                        c.location.char_start, c.location.char_end,
                        c.location.paragraph_n, c.token_count, emb_str,
                    )
                    ids.append(row["id"])
        return ids

    async def hybrid_search(
        self,
        user_id: str,
        query_embedding: list[float],
        query_text: str,
        book_ids: list[int] | None = None,
        candidate_k: int = 40,
        top_k: int = 10,
    ) -> list[tuple[Chunk, float]]:
        emb_str = f"[{','.join(str(v) for v in query_embedding)}]"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                _HYBRID_SEARCH_SQL,
                emb_str,
                user_id,
                book_ids,
                candidate_k,
                query_text,
                top_k,
            )
        return [(_row_to_chunk(r), r["rrf_score"]) for r in rows]

    async def get_neighbours(
        self,
        chunk_id: int,
        book_id: int,
        window: int = 2,
    ) -> list[Chunk]:
        """Fetch the `window` chunks before and after `chunk_id` in sequence order."""
        async with self._pool.acquire() as conn:
            seq_row = await conn.fetchrow(
                "SELECT sequence FROM chunks WHERE id = $1 AND book_id = $2",
                chunk_id, book_id,
            )
            if not seq_row:
                return []
            seq = seq_row["sequence"]
            rows = await conn.fetch(
                """
                SELECT * FROM chunks
                WHERE book_id = $1
                  AND sequence BETWEEN $2 AND $3
                ORDER BY sequence
                """,
                book_id, seq - window, seq + window,
            )
        return [_row_to_chunk(r) for r in rows]

    async def get_by_location(
        self,
        book_id: int,
        user_id: str,
        chapter_n: int | None = None,
        section_n: int | None = None,
    ) -> list[Chunk]:
        conditions = ["book_id = $1", "user_id = $2"]
        params: list = [book_id, user_id]
        i = 3
        if chapter_n is not None:
            conditions.append(f"chapter_n = ${i}")
            params.append(chapter_n)
            i += 1
        if section_n is not None:
            conditions.append(f"section_n = ${i}")
            params.append(section_n)
        where = " AND ".join(conditions)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM chunks WHERE {where} ORDER BY sequence",
                *params,
            )
        return [_row_to_chunk(r) for r in rows]

    async def delete_for_book(self, book_id: int) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM chunks WHERE book_id = $1", book_id)
