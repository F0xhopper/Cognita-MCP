"""Connection pool — a lazily created, process-wide asyncpg pool."""

import asyncpg

from cognita.core.config import settings

_pool: asyncpg.Pool | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")


async def get_pool() -> asyncpg.Pool:
    """Return the shared pool, creating it on first use."""
    global _pool
    if _pool is None:
        if not settings.DATABASE_URL:
            raise RuntimeError("DATABASE_URL is not set — see .env.example")
        _pool = await asyncpg.create_pool(
            settings.DATABASE_URL,
            min_size=1,
            max_size=10,
            ssl="require" if settings.DATABASE_SSL else None,
            init=_init_connection,
            # Pooled Postgres (pgbouncer / Supabase) cannot share prepared
            # statements across connections.
            statement_cache_size=0,
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
