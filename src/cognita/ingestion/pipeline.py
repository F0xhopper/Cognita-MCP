"""Ingestion pipeline — orchestrates parse → chunk → embed → store.

Called by the Celery worker. Also usable directly in tests.
"""

import asyncio
from pathlib import Path

import asyncpg

from cognita.books.domain import BookStatus, TocEntry
from cognita.books.repository import BookRepository
from cognita.chunks.repository import ChunkRepository
from cognita.core.exceptions import IngestionError
from cognita.core.logging import get_logger
from cognita.infrastructure.embeddings import embed_batch
from cognita.infrastructure.mistral import ocr_pdf
from cognita.infrastructure.storage import storage
from cognita.ingestion.chunker import build_chunks
from cognita.ingestion.parsers import ParsedDocument, parse_document

logger = get_logger(__name__)

_EMBED_BATCH_SIZE = 32  # stay within OpenAI rate limits


async def ingest_book(book_id: int, pool: asyncpg.Pool) -> None:
    book_repo = BookRepository(pool)
    chunk_repo = ChunkRepository(pool)

    book = await book_repo.get(book_id, user_id="__system__")  # worker bypasses user scoping
    if book is None:
        raise IngestionError(f"Book {book_id} not found")

    await book_repo.update_status(book_id, BookStatus.PROCESSING)
    logger.info("Ingesting book_id=%d title=%r", book_id, book.metadata.title)

    try:
        file_bytes = await storage().read(book.storage_path)
        tmp_path = Path(f"/tmp/cognita_{book_id}{Path(book.storage_path).suffix}")
        tmp_path.write_bytes(file_bytes)

        doc = _parse_with_fallback(tmp_path, book.metadata.title)
        chunks = build_chunks(doc, book_id, book.user_id)

        embeddings = await _embed_chunks([c.text for c in chunks])
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        ids = await chunk_repo.bulk_insert(chunks)

        toc = _build_toc(doc.native_toc, chunks, ids)
        await book_repo.update_toc_and_count(book_id, toc, len(chunks))
        await book_repo.update_status(book_id, BookStatus.READY)

        tmp_path.unlink(missing_ok=True)
        logger.info("Ingestion complete: book_id=%d, %d chunks", book_id, len(chunks))

    except Exception as exc:
        logger.exception("Ingestion failed for book_id=%d", book_id)
        await book_repo.update_status(book_id, BookStatus.FAILED, str(exc))
        raise IngestionError(str(exc)) from exc


def _parse_with_fallback(path: Path, title: str) -> ParsedDocument:
    doc = parse_document(path)
    if doc.is_scanned:
        logger.info("Scanned PDF detected — falling back to Mistral OCR")
        raw_text = asyncio.get_event_loop().run_until_complete(ocr_pdf(path))
        doc = ParsedDocument(
            raw_text=raw_text,
            pages=raw_text.split("\n\n---PAGE---\n\n"),
            native_toc=doc.native_toc,
            is_scanned=False,
        )
    return doc


async def _embed_chunks(texts: list[str]) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[i : i + _EMBED_BATCH_SIZE]
        batch_embs = await embed_batch(batch)
        all_embeddings.extend(batch_embs)
    return all_embeddings


def _build_toc(
    native_toc: list[dict],
    chunks,
    chunk_ids: list[int],
) -> list[TocEntry]:
    toc: list[TocEntry] = []
    seen_sections: set[tuple[int, int]] = set()

    for chunk, cid in zip(chunks, chunk_ids):
        key = (chunk.location.chapter_n or 0, chunk.location.section_n or 0)
        if key in seen_sections:
            continue
        seen_sections.add(key)

        level = 1 if not chunk.location.section_n or chunk.location.section_n <= 1 else 2
        title = chunk.location.section_title or chunk.location.chapter_title or "Section"
        toc.append(TocEntry(
            title=title,
            level=level,
            sequence=len(toc),
            start_char=chunk.location.char_start,
            chunk_id=cid,
        ))

    return sorted(toc, key=lambda e: e.sequence)
