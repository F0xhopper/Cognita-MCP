-- Migration 001: initial schema
-- Run with: python migrations/apply.py

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

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

CREATE INDEX IF NOT EXISTS idx_books_user_id ON books (user_id);
CREATE INDEX IF NOT EXISTS idx_books_status  ON books (status);
CREATE INDEX IF NOT EXISTS idx_books_title_trgm ON books USING gin (title gin_trgm_ops);

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

CREATE INDEX IF NOT EXISTS idx_chunks_book_id   ON chunks (book_id);
CREATE INDEX IF NOT EXISTS idx_chunks_user_id   ON chunks (user_id);
CREATE INDEX IF NOT EXISTS idx_chunks_sequence  ON chunks (book_id, sequence);
CREATE INDEX IF NOT EXISTS idx_chunks_fts       ON chunks USING gin (fts);

-- HNSW index for vector search (faster than ivfflat for < 1M vectors)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
