"""Everything that puts a book into the library, and everything that lists it.

The five ways in — a local file, a folder of files, a URL, a title to look up,
or pasted text — all converge on :meth:`BookService._store`, which writes the
bytes, derives whatever metadata it can, and hands the book to the ingestion
queue. Adding is therefore always fast; searchability arrives shortly after.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import asyncpg

from cognita.books.domain import (
    Book,
    BookFormat,
    BookMetadata,
    BookStatus,
    BookSummary,
    LibraryStatus,
)
from cognita.books.repository import BookRepository
from cognita.books.sources import ResolvedSource, resolve_source
from cognita.books.url_fetcher import fetch_book_from_url
from cognita.chunks.repository import ChunkRepository
from cognita.core.config import settings
from cognita.core.exceptions import NotFoundError, UnsupportedFormatError
from cognita.core.logging import get_logger
from cognita.ingestion.metadata import extract_metadata, merge_metadata
from cognita.ingestion.queue import IngestionQueue

logger = get_logger(__name__)

_EXTENSIONS: dict[str, BookFormat] = {
    ".pdf": BookFormat.PDF,
    ".epub": BookFormat.EPUB,
    ".txt": BookFormat.TXT,
    ".text": BookFormat.TXT,
    ".md": BookFormat.MD,
    ".markdown": BookFormat.MD,
    ".html": BookFormat.HTML,
    ".htm": BookFormat.HTML,
}

SUPPORTED_EXTENSIONS = tuple(sorted(_EXTENSIONS))


@dataclass
class FolderImport:
    """Outcome of importing a directory."""

    added: list[BookSummary]
    skipped: list[str]  # "path — reason", for files that were not imported


class BookService:
    def __init__(self, pool: asyncpg.Pool, queue: IngestionQueue | None = None) -> None:
        self._pool = pool
        self._repo = BookRepository(pool)
        self._chunk_repo = ChunkRepository(pool)
        self._queue = queue or IngestionQueue(pool)

    # ── Adding ────────────────────────────────────────────────────────────────

    async def add_from_path(self, path: str | Path, meta: BookMetadata | None = None) -> Book:
        """Add a single file from the local filesystem."""
        file_path = Path(path).expanduser()
        if not file_path.exists():
            raise NotFoundError("File", str(file_path))
        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        fmt = _format_for(file_path)
        data = file_path.read_bytes()
        self._check_size(data, file_path.name)

        extracted = extract_metadata(file_path)
        extracted.source = str(file_path)
        return await self._store(fmt, data, merge_metadata(extracted, meta))

    async def add_from_folder(
        self,
        path: str | Path,
        recursive: bool = True,
        tags: list[str] | None = None,
    ) -> FolderImport:
        """Add every supported file in a directory.

        Files already in the library (matched on their source path) are skipped,
        so re-running the import after adding a few books only ingests the new ones.
        """
        folder = Path(path).expanduser()
        if not folder.is_dir():
            raise NotFoundError("Folder", str(folder))

        known = await self._repo.known_sources()
        candidates = sorted(folder.rglob("*") if recursive else folder.glob("*"))

        added: list[BookSummary] = []
        skipped: list[str] = []

        for candidate in candidates:
            if not candidate.is_file() or candidate.name.startswith("."):
                continue
            if candidate.suffix.lower() not in _EXTENSIONS:
                continue
            if str(candidate) in known:
                skipped.append(f"{candidate.name} — already in the library")
                continue
            try:
                book = await self.add_from_path(candidate, BookMetadata(title="", tags=tags or []))
            except Exception as exc:  # noqa: BLE001 — one bad file must not stop the import
                logger.warning("Skipping %s: %s", candidate, exc)
                skipped.append(f"{candidate.name} — {exc}")
                continue
            added.append(_summarise(book))

        logger.info("Folder import from %s: %d added, %d skipped",
                    folder, len(added), len(skipped))
        return FolderImport(added=added, skipped=skipped)

    async def add_from_url(self, url: str, meta: BookMetadata | None = None) -> Book:
        """Download a book from a URL and add it."""
        data, fmt, filename = await fetch_book_from_url(url)
        self._check_size(data, filename)

        extracted = await self._metadata_from_bytes(data, filename, fmt)
        extracted.source = url
        return await self._store(fmt, data, merge_metadata(extracted, meta))

    async def add_by_title(
        self,
        title: str,
        author: str | None = None,
        meta: BookMetadata | None = None,
    ) -> tuple[Book, ResolvedSource]:
        """Look the book up in the public-domain libraries and add what is found."""
        resolved = await resolve_source(title, author)
        if resolved is None:
            raise NotFoundError("Public-domain source for", f"{title!r}")

        supplied = meta or BookMetadata(title=title, author=author)
        supplied.title = supplied.title or title
        supplied.author = supplied.author or author
        book = await self.add_from_url(resolved.url, supplied)
        return book, resolved

    async def add_text(
        self,
        title: str,
        text: str,
        author: str | None = None,
        tags: list[str] | None = None,
    ) -> Book:
        """Add pasted text — notes, an article, a transcript."""
        if not text.strip():
            raise ValueError("Text is empty")
        data = text.encode("utf-8")
        self._check_size(data, title)
        # Markdown so any headings in the text become real sections.
        meta = BookMetadata(title=title, author=author, tags=tags or [], source="text")
        return await self._store(BookFormat.MD, data, meta)

    # ── Reading ───────────────────────────────────────────────────────────────

    async def list_books(self) -> list[BookSummary]:
        return await self._repo.list_all()

    async def find_books(self, query: str, limit: int = 20) -> list[BookSummary]:
        return await self._repo.find(query, limit)

    async def get_book(self, book_id: int) -> Book:
        book = await self._repo.get(book_id)
        if book is None:
            raise NotFoundError("Book", book_id)
        return book

    async def library_status(self) -> LibraryStatus:
        status = await self._repo.library_status()
        status.queue_depth = self._queue.depth
        return status

    # ── Managing ──────────────────────────────────────────────────────────────

    async def delete_book(self, book_id: int) -> str:
        book = await self.get_book(book_id)
        # Chunks cascade on the foreign key, but deleting them first keeps the
        # window where a search could hit orphaned rows closed.
        await self._chunk_repo.delete_for_book(book_id)
        await self._repo.delete(book_id)
        logger.info("Deleted book_id=%d %r", book_id, book.metadata.title)
        return book.metadata.title

    async def reingest_book(self, book_id: int) -> Book:
        """Re-run ingestion — after a failure, or to pick up better settings."""
        book = await self.get_book(book_id)
        await self._repo.update_status(book_id, BookStatus.PENDING)
        self._queue.submit(book_id)
        return book

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _store(self, fmt: BookFormat, data: bytes, meta: BookMetadata) -> Book:
        if not meta.title:
            meta.title = "Untitled"
        book = await self._repo.create(fmt=fmt, file_data=data, meta=meta)
        self._queue.submit(book.id)
        logger.info("Queued book_id=%d %r (%s)", book.id, meta.title, fmt)
        return book

    async def _metadata_from_bytes(
        self, data: bytes, filename: str, fmt: BookFormat
    ) -> BookMetadata:
        """Extract metadata from downloaded bytes by staging them on disk."""
        with tempfile.TemporaryDirectory(prefix="cognita-meta-") as tmpdir:
            staged = Path(tmpdir) / (filename if Path(filename).suffix else f"{filename}.{fmt}")
            staged.write_bytes(data)
            return extract_metadata(staged)

    def _check_size(self, data: bytes, label: str) -> None:
        limit = settings.MAX_FILE_MB * 1024 * 1024
        if len(data) > limit:
            raise ValueError(
                f"{label} is {len(data) // (1024 * 1024)} MB, "
                f"over the {settings.MAX_FILE_MB} MB limit"
            )


def _format_for(path: Path) -> BookFormat:
    fmt = _EXTENSIONS.get(path.suffix.lower())
    if fmt is None:
        raise UnsupportedFormatError(path.suffix or path.name)
    return fmt


def _summarise(book: Book) -> BookSummary:
    return BookSummary(
        id=book.id,
        title=book.metadata.title,
        author=book.metadata.author,
        status=book.status,
        format=book.format,
        chunk_count=book.chunk_count,
        created_at=book.created_at,
        error_message=book.error_message,
    )
