-- Migration 002: specialties — named, persona-bearing slices of the library
-- A specialty groups books into a scoped "expert" that agents can query.

CREATE TABLE IF NOT EXISTS specialties (
    id          SERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    name        TEXT NOT NULL,
    description TEXT,
    persona     TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, name)
);

CREATE INDEX IF NOT EXISTS idx_specialties_user_id ON specialties (user_id);

CREATE TABLE IF NOT EXISTS specialty_books (
    specialty_id INTEGER NOT NULL REFERENCES specialties(id) ON DELETE CASCADE,
    book_id      INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    added_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (specialty_id, book_id)
);

CREATE INDEX IF NOT EXISTS idx_specialty_books_book_id ON specialty_books (book_id);
