-- Upgrading a library created by Cognita v1.
--
-- The server migrates the books and chunks tables itself on startup (dropping
-- user_id, adding the source and context columns), so the only thing left is
-- the specialties feature, which no longer exists. Run this once if you want
-- the tables gone; leaving them costs nothing but clutter.
--
--   psql "$DATABASE_URL" -f migrations/upgrade_from_v1.sql

DROP TABLE IF EXISTS specialty_books;
DROP TABLE IF EXISTS specialties;

-- v1 also tracked its own migration history, which is no longer used.
DROP TABLE IF EXISTS _migrations;
