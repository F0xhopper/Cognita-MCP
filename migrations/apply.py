"""Run all .sql migration files in order against DATABASE_URL."""

import asyncio
from pathlib import Path

import asyncpg
from dotenv import load_dotenv
import os

load_dotenv()


async def main() -> None:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL not set")

    ssl = "require" if os.getenv("DATABASE_SSL", "false").lower() == "true" else None
    conn = await asyncpg.connect(url, ssl=ssl)

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS _migrations (
            filename TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    applied = {r["filename"] for r in await conn.fetch("SELECT filename FROM _migrations")}

    migrations_dir = Path(__file__).parent
    sql_files = sorted(migrations_dir.glob("*.sql"))

    for f in sql_files:
        if f.name in applied:
            print(f"  skip  {f.name}")
            continue
        print(f"  apply {f.name}…")
        sql = f.read_text()
        await conn.execute(sql)
        await conn.execute("INSERT INTO _migrations (filename) VALUES ($1)", f.name)
        print(f"  done  {f.name}")

    await conn.close()
    print("All migrations applied.")


if __name__ == "__main__":
    asyncio.run(main())
