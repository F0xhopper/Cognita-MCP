"""Ingestion: parse → chunk → contextualize → embed → store.

Runs in the background for one book at a time. Every failure mode ends with the
book marked FAILED and the reason recorded on the row, so a caller polling
``list_books`` always learns what went wrong.
"""

import asyncio
import tempfile
from pathlib import Path

import asyncpg

from cognita.books.domain import BookStatus, TocEntry
from cognita.books.repository import BookRepository
from cognita.chunks.domain import Chunk
from cognita.chunks.repository import ChunkRepository
from cognita.core.config import settings
from cognita.core.exceptions import IngestionError
from cognita.core.logging import get_logger
from cognita.infrastructure.embeddings import embed_batch
from cognita.ingestion.chunker import build_chunks
from cognita.ingestion.contextualizer import contextualize_chunks
from cognita.ingestion.parsers import ParsedDocument, detect_headings, parse_document

logger = get_logger(__name__)

# ~20 chunks of ~1.5k chars ≈ 10k tokens per request, well inside the limit.
_EMBED_BATCH_SIZE = 20
_EMBED_PAUSE_SECONDS = 0.5

# Mistral OCR returns pages joined by this marker.
_OCR_PAGE_SEPARATOR = "\n\n---PAGE---\n\n"


async def ingest_book(book_id: int, pool: asyncpg.Pool) -> None:
    book_repo = BookRepository(pool)
    chunk_repo = ChunkRepository(pool)

    book = await book_repo.get(book_id)
    if book is None:
        raise IngestionError(f"Book {book_id} not found")

    await book_repo.update_status(book_id, BookStatus.PROCESSING)
    logger.info("Ingesting book_id=%d %r", book_id, book.metadata.title)

    try:
        # Re-ingestion must not stack a second copy of every chunk.
        await chunk_repo.delete_for_book(book_id)

        file_bytes = await book_repo.get_file_data(book_id)
        with tempfile.TemporaryDirectory(prefix="cognita-") as tmpdir:
            path = Path(tmpdir) / f"book-{book_id}.{book.format}"
            path.write_bytes(file_bytes)
            doc = await _parse(path, book_id)

        if not doc.raw_text.strip():
            raise IngestionError("No extractable text — the file may be empty or image-only")

        chunks = build_chunks(doc, book_id)
        if not chunks:
            raise IngestionError("Document produced no chunks")

        contexts = await contextualize_chunks(chunks, book.metadata.title, book.metadata.author)
        for chunk, context in zip(chunks, contexts, strict=True):
            chunk.context = context

        embeddings = await _embed_all([_embed_input(c) for c in chunks])
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk.embedding = embedding

        chunk_ids = await chunk_repo.bulk_insert(chunks)

        await book_repo.update_toc_and_count(book_id, _build_toc(chunks, chunk_ids), len(chunks))
        await book_repo.update_status(book_id, BookStatus.READY)
        logger.info("Ingested book_id=%d: %d chunks", book_id, len(chunks))

    except Exception as exc:
        logger.exception("Ingestion failed for book_id=%d", book_id)
        await book_repo.update_status(book_id, BookStatus.FAILED, str(exc))
        raise IngestionError(str(exc)) from exc


async def _parse(path: Path, book_id: int) -> ParsedDocument:
    """Parse off the event loop; fall back to OCR when a PDF has no text layer."""
    doc = await asyncio.to_thread(parse_document, path)
    if not doc.is_scanned:
        return doc

    if not settings.ocr_enabled:
        logger.warning(
            "book_id=%d looks scanned but MISTRAL_API_KEY is not set — "
            "ingesting whatever text could be extracted",
            book_id,
        )
        return doc

    # Imported here so the OCR client is only constructed when actually needed.
    from cognita.infrastructure.mistral import ocr_pdf

    logger.info("book_id=%d has no text layer — running OCR", book_id)
    text = await ocr_pdf(path)
    pages = text.split(_OCR_PAGE_SEPARATOR)

    page_starts: list[int] = []
    cursor = 0
    for page in pages:
        page_starts.append(cursor)
        cursor += len(page) + 2

    raw_text = "\n\n".join(pages)
    return ParsedDocument(
        raw_text=raw_text,
        headings=detect_headings(raw_text),
        page_starts=page_starts,
        is_scanned=False,
    )


def _embed_input(chunk: Chunk) -> str:
    """What actually gets embedded: the situating blurb ahead of the passage."""
    return f"{chunk.context}\n\n{chunk.text}" if chunk.context else chunk.text


async def _embed_all(texts: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        embeddings.extend(await embed_batch(texts[i : i + _EMBED_BATCH_SIZE]))
        if i + _EMBED_BATCH_SIZE < len(texts):
            await asyncio.sleep(_EMBED_PAUSE_SECONDS)
    return embeddings


def _build_toc(chunks: list[Chunk], chunk_ids: list[int]) -> list[TocEntry]:
    """One entry per distinct section, anchored to the first chunk that starts it."""
    entries: list[TocEntry] = []
    seen: set[tuple[int, int]] = set()

    for chunk, chunk_id in zip(chunks, chunk_ids, strict=True):
        location = chunk.location
        key = (location.chapter_n or 0, location.section_n or 0)
        if key in seen:
            continue
        seen.add(key)

        is_chapter_start = (location.section_n or 1) <= 1
        entries.append(
            TocEntry(
                title=location.section_title or location.chapter_title or "Section",
                level=1 if is_chapter_start else 2,
                sequence=len(entries),
                start_char=location.char_start,
                page_start=location.page_start,
                chunk_id=chunk_id,
            )
        )
    return entries
