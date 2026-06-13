-- Migration 003: store file content in DB instead of filesystem/S3
-- Drops storage_path, adds file_data BYTEA

ALTER TABLE books DROP COLUMN IF EXISTS storage_path;
ALTER TABLE books ADD COLUMN IF NOT EXISTS file_data BYTEA NOT NULL DEFAULT ''::bytea;
ALTER TABLE books ALTER COLUMN file_data DROP DEFAULT;
