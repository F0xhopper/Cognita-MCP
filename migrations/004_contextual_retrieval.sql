-- Migration 004: Contextual Retrieval
-- Add a per-chunk context blurb (Anthropic's contextual retrieval technique) and
-- index it alongside the passage text. The blurb is embedded (in the ingestion
-- pipeline) and folded into the full-text index here. Existing rows get NULL
-- context and continue to work; re-ingest a book to populate its blurbs.
--
-- fts is a generated column, so changing its expression requires dropping and
-- re-adding it. This is safe — fts is derived from text/context.

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS context TEXT;

DROP INDEX IF EXISTS idx_chunks_fts;
ALTER TABLE chunks DROP COLUMN IF EXISTS fts;
ALTER TABLE chunks ADD COLUMN fts TSVECTOR GENERATED ALWAYS AS (
    to_tsvector('english', coalesce(context, '') || ' ' || text)
) STORED;
CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING gin (fts);
