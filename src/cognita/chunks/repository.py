"""Chunk storage and hybrid retrieval.

Search fuses two independent rankings with Reciprocal Rank Fusion:

  semantic — pgvector cosine distance over the chunk embedding
  keyword  — PostgreSQL full-text ranking over a tsvector of context + text

RRF is used rather than score blending because the two arms produce scores on
incomparable scales; ranks are comparable, scores are not. Each arm contributes
``weight / (k + rank)``, so a chunk that both arms rank highly beats a chunk
either arm loves alone.
"""

import asyncpg

from cognita.chunks.domain import Chunk, ChunkLocation
from cognita.core.config import settings
from cognita.core.logging import get_logger

logger = get_logger(__name__)

_CREATE_CHUNKS_SQL = f"""
CREATE TABLE IF NOT EXISTS chunks (
    id             SERIAL PRIMARY KEY,
    book_id        INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    text           TEXT NOT NULL,
    context        TEXT,
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
    embedding      vector({settings.EMBED_DIM}),
    fts            TSVECTOR GENERATED ALWAYS AS (
                       to_tsvector('english', coalesce(context, '') || ' ' || text)
                   ) STORED,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# Idempotent catch-up for libraries created by older versions.
_MIGRATE_CHUNKS_SQL = [
    "ALTER TABLE chunks DROP COLUMN IF EXISTS user_id",
    "ALTER TABLE chunks DROP COLUMN IF EXISTS level",
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS context TEXT",
]

# HNSW indexes a `vector` column of at most 2000 dimensions. The default
# embedding model produces 3072, so above that limit the column is indexed —
# and queried — through a halfvec cast, which HNSW supports up to 4000. halfvec
# is 16-bit: it halves index size for a recall cost that does not show up at
# this scale, and the stored vectors keep full precision either way.
_HALFVEC_LIMIT = 2000
_USE_HALFVEC = settings.EMBED_DIM > _HALFVEC_LIMIT

# The distance expression must match the indexed expression exactly, or the
# planner ignores the index and falls back to a sequential scan.
_DISTANCE = (
    f"embedding::halfvec({settings.EMBED_DIM}) <=> $1::halfvec({settings.EMBED_DIM})"
    if _USE_HALFVEC
    else "embedding <=> $1::vector"
)

_EMBEDDING_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_half ON chunks USING hnsw "
    f"((embedding::halfvec({settings.EMBED_DIM})) halfvec_cosine_ops) "
    "WITH (m = 16, ef_construction = 64)"
    if _USE_HALFVEC
    else "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw "
         "(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
)

_CREATE_CHUNK_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_chunks_book_id ON chunks (book_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_sequence ON chunks (book_id, sequence)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_location ON chunks (book_id, chapter_n, section_n)",
    _EMBEDDING_INDEX_SQL,
    "CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING gin (fts)",
]

# $1 embedding · $2 book_ids · $3 candidate_k · $4 query text
# $5 semantic weight · $6 keyword weight · $7 rrf k · $8 top_k
#
# websearch_to_tsquery (rather than plainto_tsquery) lets a caller use
# "quoted phrases", OR, and -exclusions in the query string.
_HYBRID_SEARCH_SQL = f"""
WITH q AS (
    SELECT websearch_to_tsquery('english', $4) AS tsq
),
semantic AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY {_DISTANCE}) AS rank
    FROM chunks
    WHERE ($2::int[] IS NULL OR book_id = ANY($2::int[]))
      AND embedding IS NOT NULL
    ORDER BY {_DISTANCE}
    LIMIT $3
),
keyword AS (
    SELECT c.id,
           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.fts, q.tsq) DESC) AS rank
    FROM chunks c, q
    WHERE ($2::int[] IS NULL OR c.book_id = ANY($2::int[]))
      AND q.tsq IS NOT NULL
      AND c.fts @@ q.tsq
    ORDER BY ts_rank_cd(c.fts, q.tsq) DESC
    LIMIT $3
),
fused AS (
    SELECT COALESCE(s.id, k.id) AS id,
           COALESCE($5::float / ($7 + s.rank), 0)
         + COALESCE($6::float / ($7 + k.rank), 0) AS score
    FROM semantic s
    FULL OUTER JOIN keyword k ON s.id = k.id
)
SELECT c.*, fused.score
FROM fused
JOIN chunks c ON c.id = fused.id
ORDER BY fused.score DESC
LIMIT $8
"""


# Each embedding is a long text literal, so batches stay modest to keep any one
# statement from carrying tens of megabytes.
_INSERT_BATCH_SIZE = 100

_INSERT_COLUMNS = (
    "book_id, text, context, sequence, chapter_title, chapter_n, section_title, "
    "section_n, page_start, page_end, char_start, char_end, paragraph_n, token_count"
)

_BULK_INSERT_SQL = f"""
INSERT INTO chunks ({_INSERT_COLUMNS}, embedding)
SELECT {_INSERT_COLUMNS}, embedding::vector
FROM unnest(
    $1::int[], $2::text[], $3::text[], $4::int[], $5::text[], $6::int[],
    $7::text[], $8::int[], $9::int[], $10::int[], $11::int[], $12::int[],
    $13::int[], $14::int[], $15::text[]
) AS t({_INSERT_COLUMNS}, embedding)
RETURNING id, sequence
"""


def _columns(chunks: list[Chunk]) -> tuple[list, ...]:
    """Transpose chunks into the parallel arrays the insert statement expects."""
    locations = [c.location for c in chunks]
    return (
        [c.book_id for c in chunks],
        [c.text for c in chunks],
        [c.context or None for c in chunks],
        [c.sequence for c in chunks],
        [loc.chapter_title for loc in locations],
        [loc.chapter_n for loc in locations],
        [loc.section_title for loc in locations],
        [loc.section_n for loc in locations],
        [loc.page_start for loc in locations],
        [loc.page_end for loc in locations],
        [loc.char_start for loc in locations],
        [loc.char_end for loc in locations],
        [loc.paragraph_n for loc in locations],
        [c.token_count for c in chunks],
        [f"[{','.join(map(str, c.embedding))}]" if c.embedding else None for c in chunks],
    )


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
    return Chunk(
        id=d["id"],
        book_id=d["book_id"],
        text=d["text"],
        sequence=d["sequence"],
        location=loc,
        token_count=d.get("token_count") or 0,
        context=d.get("context") or "",
    )


class ChunkRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_CHUNKS_SQL)
            for sql in _MIGRATE_CHUNKS_SQL:
                try:
                    await conn.execute(sql)
                except Exception as exc:  # noqa: BLE001 — migrations are best-effort
                    logger.warning("Schema migration skipped (%s): %s", sql[:48], exc)
            await self._ensure_contextual_fts(conn)
            for sql in _CREATE_CHUNK_INDEXES_SQL:
                try:
                    await conn.execute(sql)
                except Exception as exc:  # noqa: BLE001 — an index is an optimisation
                    logger.warning("Index creation skipped (%s): %s", sql[:48], exc)

    @staticmethod
    async def _ensure_contextual_fts(conn: asyncpg.Connection) -> None:
        """Make the full-text column index context + text.

        fts is a generated column, so changing its expression means dropping and
        re-adding it — cheap, since it is entirely derived. No-op once current.
        """
        expr = await conn.fetchval(
            """
            SELECT generation_expression FROM information_schema.columns
            WHERE table_name = 'chunks' AND column_name = 'fts'
            """
        )
        if expr is not None and "context" in expr:
            return
        try:
            await conn.execute("DROP INDEX IF EXISTS idx_chunks_fts")
            await conn.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS fts")
            await conn.execute(
                """
                ALTER TABLE chunks ADD COLUMN fts TSVECTOR GENERATED ALWAYS AS (
                    to_tsvector('english', coalesce(context, '') || ' ' || text)
                ) STORED
                """
            )
            logger.info("Rebuilt chunks.fts to index context alongside text")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Contextual fts migration skipped: %s", exc)

    async def bulk_insert(self, chunks: list[Chunk]) -> list[int]:
        """Insert chunks in one transaction. Returns their ids, in input order.

        Rows go in as parallel arrays unnested server-side, which is one round
        trip per batch instead of one per chunk — the difference between a few
        seconds and a few minutes on a full-length book.
        """
        if not chunks:
            return []

        by_sequence: dict[int, int] = {}
        async with self._pool.acquire() as conn, conn.transaction():
            for start in range(0, len(chunks), _INSERT_BATCH_SIZE):
                batch = chunks[start : start + _INSERT_BATCH_SIZE]
                rows = await conn.fetch(_BULK_INSERT_SQL, *_columns(batch))
                by_sequence.update({r["sequence"]: r["id"] for r in rows})

        # Mapped by sequence rather than trusting RETURNING to echo input order.
        return [by_sequence[c.sequence] for c in chunks]

    async def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        book_ids: list[int] | None = None,
        candidate_k: int = 200,
        top_k: int = 40,
    ) -> list[tuple[Chunk, float]]:
        emb = f"[{','.join(str(v) for v in query_embedding)}]"
        # pgvector applies the book filter after the index scan, so a scoped
        # search needs a wider sweep to return the same number of real hits.
        ef = settings.HNSW_EF_SEARCH * (2 if book_ids else 1)
        ef = max(ef, candidate_k)

        async with self._pool.acquire() as conn, conn.transaction():
            # SET LOCAL takes no bind parameters; ef is an int from settings.
            await conn.execute(f"SET LOCAL hnsw.ef_search = {int(ef)}")
            rows = await conn.fetch(
                _HYBRID_SEARCH_SQL,
                emb,
                book_ids,
                candidate_k,
                query_text,
                settings.RRF_SEMANTIC_WEIGHT,
                settings.RRF_KEYWORD_WEIGHT,
                settings.RRF_K,
                top_k,
            )
        return [(_row_to_chunk(r), float(r["score"])) for r in rows]

    async def get_neighbours(self, chunk_id: int, window: int = 2) -> list[Chunk]:
        """Fetch `window` chunks either side of `chunk_id`, in reading order."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH target AS (
                    SELECT book_id, sequence FROM chunks WHERE id = $1
                )
                SELECT c.* FROM chunks c, target t
                WHERE c.book_id = t.book_id
                  AND c.sequence BETWEEN t.sequence - $2 AND t.sequence + $2
                ORDER BY c.sequence
                """,
                chunk_id, window,
            )
        return [_row_to_chunk(r) for r in rows]

    async def get_by_location(
        self,
        book_id: int,
        chapter_n: int | None = None,
        section_n: int | None = None,
    ) -> list[Chunk]:
        conditions = ["book_id = $1"]
        params: list = [book_id]
        if chapter_n is not None:
            params.append(chapter_n)
            conditions.append(f"chapter_n = ${len(params)}")
        if section_n is not None:
            params.append(section_n)
            conditions.append(f"section_n = ${len(params)}")

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM chunks WHERE {' AND '.join(conditions)} ORDER BY sequence",
                *params,
            )
        return [_row_to_chunk(r) for r in rows]

    async def delete_for_book(self, book_id: int) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM chunks WHERE book_id = $1", book_id)
