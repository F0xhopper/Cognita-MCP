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
from cognita.ingestion.chunker import build_chunks
from cognita.ingestion.contextualizer import contextualize_chunks
from cognita.ingestion.parsers import ParsedDocument, parse_document

logger = get_logger(__name__)

_EMBED_BATCH_SIZE = 20  # ~10k tokens/batch; 1s inter-batch delay keeps well under 1M TPM


async def ingest_book(book_id: int, pool: asyncpg.Pool) -> None:
    book_repo = BookRepository(pool)
    chunk_repo = ChunkRepository(pool)

    book = await book_repo.get_by_id(book_id)
    if book is None:
        raise IngestionError(f"Book {book_id} not found")

    await book_repo.update_status(book_id, BookStatus.PROCESSING)
    logger.info("Ingesting book_id=%d title=%r", book_id, book.metadata.title)

    try:
        file_bytes = await book_repo.get_file_data(book_id)
        tmp_path = Path(f"/tmp/cognita_{book_id}.{book.format}")
        tmp_path.write_bytes(file_bytes)

        doc = await _parse_with_fallback(tmp_path, book.metadata.title)
        chunks = build_chunks(doc, book_id, book.user_id)

        # Contextual Retrieval: situate each chunk in its source, then embed the
        # contextualized text so retrieval matches the passage in context.
        contexts = await contextualize_chunks(
            chunks, book.metadata.title, book.metadata.author
        )
        for chunk, ctx in zip(chunks, contexts):
            chunk.context = ctx

        embeddings = await _embed_chunks([_embed_input(c) for c in chunks])
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


async def _parse_with_fallback(path: Path, title: str) -> ParsedDocument:
    doc = parse_document(path)
    if doc.is_scanned:
        logger.info("Scanned PDF detected — falling back to Mistral OCR")
        raw_text = await ocr_pdf(path)
        doc = ParsedDocument(
            raw_text=raw_text,
            pages=raw_text.split("\n\n---PAGE---\n\n"),
            native_toc=doc.native_toc,
            is_scanned=False,
        )
    return doc


def _embed_input(chunk) -> str:
    """Text sent to the embedding model: the contextual blurb prepended to the
    passage when present, otherwise the passage alone."""
    return f"{chunk.context}\n\n{chunk.text}" if chunk.context else chunk.text


async def _embed_chunks(texts: list[str]) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[i : i + _EMBED_BATCH_SIZE]
        batch_embs = await embed_batch(batch)
        all_embeddings.extend(batch_embs)
        if i + _EMBED_BATCH_SIZE < len(texts):
            await asyncio.sleep(1)
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
