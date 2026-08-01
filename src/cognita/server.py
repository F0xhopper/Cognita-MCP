"""Cognita — an MCP server over your personal library.

Two ways to run it:

    cognita                 # stdio, for Claude Desktop / Claude Code on this machine
    cognita --http          # HTTP, for a remote or shared deployment

In HTTP mode, set COGNITA_AUTH_TOKEN and callers must present it as
``Authorization: Bearer <token>``. The local-disk tools are refused over HTTP
unless explicitly enabled, since the server's filesystem is not the caller's.
"""

import argparse
import asyncio
import sys

import asyncpg
from mcp.server.fastmcp import FastMCP

from cognita.books.domain import Book, BookMetadata, BookSummary, TocEntry
from cognita.books.repository import BookRepository
from cognita.books.service import SUPPORTED_EXTENSIONS, BookService
from cognita.chunks.repository import ChunkRepository
from cognita.core.config import settings
from cognita.core.logging import get_logger, setup_logging
from cognita.infrastructure.database import get_pool
from cognita.ingestion.queue import IngestionQueue
from cognita.schemas import (
    AddedBook,
    BookItem,
    ExpandedPassage,
    FolderImportResult,
    LibraryStatusResult,
    PassageResult,
    SearchResult,
    TocItem,
)
from cognita.search.domain import Passage
from cognita.search.service import SearchService

logger = get_logger(__name__)

INSTRUCTIONS = """\
This is the user's personal library: the books, papers and notes they have \
collected, indexed passage by passage.

Use search_library whenever a question touches something the library would \
cover — their own reading, a book they mention, a subject they collect on — and \
prefer it over answering from memory, because it returns what their sources \
actually say.

Working with it:
  • search_library returns ranked passages with citations. Run it again with \
different phrasings if the first results are thin; each call is cheap.
  • Narrow with book_ids or authors when the user names a source.
  • expand_passage widens a hit that got cut off mid-argument.
  • read_chapter and read_section read straight through, for when the user \
wants an argument followed rather than the best-matching lines.
  • Quote the citation string with every claim taken from a passage.

If a search returns nothing, say so plainly rather than filling the gap from \
memory — an empty result usually just means that book is not in the library yet.\
"""

mcp = FastMCP("Cognita", instructions=INSTRUCTIONS)

_pool: asyncpg.Pool | None = None
_queue: IngestionQueue | None = None
_schema_ready = False
_allow_local_files = True


async def _services() -> tuple[BookService, SearchService]:
    """Resolve the services, ensuring the schema exists before the first query."""
    global _pool, _queue, _schema_ready

    if _pool is None:
        _pool = await get_pool()
    if not _schema_ready:
        await BookRepository(_pool).ensure_schema()
        await ChunkRepository(_pool).ensure_schema()
        _schema_ready = True
        logger.info("Schema ready")
    if _queue is None:
        _queue = IngestionQueue(_pool)

    return BookService(_pool, _queue), SearchService(_pool)


def _require_local_files() -> None:
    if not _allow_local_files:
        raise ValueError(
            "This server runs remotely and cannot read your local disk. "
            "Use add_book_from_url, add_book_by_title, or add_text instead."
        )


# ── Searching ─────────────────────────────────────────────────────────────────

@mcp.tool()
async def search_library(
    query: str,
    book_ids: list[int] | None = None,
    authors: list[str] | None = None,
    top_k: int = 10,
    min_score: float = 0.0,
) -> SearchResult:
    """Search the library and return the passages that best answer the query.

    Combines meaning-based and keyword search, so it finds a passage whether or
    not the user's wording matches the book's. Phrases in "double quotes" are
    matched literally and a leading - excludes a term.

    Args:
        query: What you want to find. A full question works better than keywords.
        book_ids: Restrict to these books. Get ids from find_books or list_books.
        authors: Restrict to books by these authors. Partial names are fine.
        top_k: How many passages to return. 5 for a quick check, 20 to read widely.
        min_score: Drop results below this relevance (0–1). Try 0.3 to cut noise.
    """
    _, search = await _services()
    response = await search.search(
        query=query,
        book_ids=book_ids,
        authors=authors,
        top_k=top_k,
        min_score=min_score,
    )
    return SearchResult(
        query=response.query,
        ranking=response.ranking,
        passages=[_passage(p) for p in response.passages],
    )


@mcp.tool()
async def expand_passage(chunk_id: int, window: int = 2) -> ExpandedPassage:
    """Read the text surrounding a passage, for when a hit is cut off.

    Args:
        chunk_id: The chunk_id from a search result.
        window: How many passages to include either side, 1–5.
    """
    _, search = await _services()
    context = await search.expand_passage(chunk_id, max(1, min(window, 5)))
    return ExpandedPassage(
        text=context.full_text(),
        citation=context.passage.citation.to_string(),
        chunk_ids=context.chunk_ids(),
    )


@mcp.tool()
async def read_chapter(book_id: int, chapter_n: int) -> list[PassageResult]:
    """Read a whole chapter in order, rather than by relevance.

    Use when the user wants an argument followed through rather than the
    best-matching lines. Chapter numbers come from get_table_of_contents.
    """
    _, search = await _services()
    passages = await search.read_location(book_id, chapter_n=chapter_n)
    return [_passage(p) for p in passages]


@mcp.tool()
async def read_section(book_id: int, chapter_n: int, section_n: int) -> list[PassageResult]:
    """Read one section of a chapter in order. Numbers come from get_table_of_contents."""
    _, search = await _services()
    passages = await search.read_location(book_id, chapter_n=chapter_n, section_n=section_n)
    return [_passage(p) for p in passages]


# ── Browsing ──────────────────────────────────────────────────────────────────

@mcp.tool()
async def list_books() -> list[BookItem]:
    """List every book in the library, newest first, with its ingestion status."""
    books, _ = await _services()
    return [_book(b) for b in await books.list_books()]


@mcp.tool()
async def find_books(query: str) -> list[BookItem]:
    """Find books by title, author or description — not by their contents.

    Use this to turn "the Graeber book" into a book_id for search_library.
    To search inside books, use search_library.
    """
    books, _ = await _services()
    return [_book(b) for b in await books.find_books(query)]


@mcp.tool()
async def get_table_of_contents(book_id: int) -> list[TocItem]:
    """Show a book's structure, with the numbers read_chapter and read_section take."""
    books, _ = await _services()
    book = await books.get_book(book_id)
    return _toc_items(book.toc)


@mcp.tool()
async def library_status() -> LibraryStatusResult:
    """Show how much of the library is indexed, and what failed.

    Call this after adding books to see when they become searchable.
    """
    books, _ = await _services()
    status = await books.library_status()

    active = [
        name
        for name, on in (
            ("contextual indexing", settings.context_enabled),
            ("reranking", settings.rerank_enabled),
            ("OCR for scanned PDFs", settings.ocr_enabled),
        )
        if on
    ]
    quality = "hybrid search (vector + keyword)"
    if active:
        quality += " with " + ", ".join(active)

    return LibraryStatusResult(
        total_books=status.total,
        ready=status.ready,
        processing=status.processing,
        pending=status.pending,
        failed=status.failed,
        total_passages=status.chunk_count,
        queue_depth=status.queue_depth,
        failures=[_book(b) for b in status.failures],
        search_quality=quality,
    )


# ── Adding books ──────────────────────────────────────────────────────────────

@mcp.tool()
async def add_book_from_path(
    path: str,
    title: str | None = None,
    author: str | None = None,
    tags: list[str] | None = None,
) -> AddedBook:
    """Add a book from a file on this machine.

    Title and author are read from the file itself, so pass them only to
    override what it declares. Supported: PDF, EPUB, TXT, Markdown, HTML.

    Args:
        path: Absolute path to the file.
        title: Overrides the title found in the file.
        author: Overrides the author found in the file.
        tags: Labels for later filtering, e.g. ["philosophy", "to-reread"].
    """
    _require_local_files()
    books, _ = await _services()
    return _added(await books.add_from_path(path, _meta(title, author, tags)))


@mcp.tool()
async def add_books_from_folder(
    path: str,
    recursive: bool = True,
    tags: list[str] | None = None,
) -> FolderImportResult:
    """Add every supported book in a folder.

    Files already in the library are skipped, so this is safe to re-run after
    adding a few more. Metadata is read from each file.

    Args:
        path: Absolute path to the folder.
        recursive: Include subfolders.
        tags: Labels applied to everything imported.
    """
    _require_local_files()
    books, _ = await _services()
    result = await books.add_from_folder(path, recursive=recursive, tags=tags)
    return FolderImportResult(
        added_count=len(result.added),
        skipped_count=len(result.skipped),
        added=[_added_summary(b) for b in result.added],
        skipped=result.skipped,
    )


@mcp.tool()
async def add_book_from_url(
    url: str,
    title: str | None = None,
    author: str | None = None,
    tags: list[str] | None = None,
) -> AddedBook:
    """Add a book by downloading it from a public URL.

    Metadata is read from the downloaded file, so title and author are optional.
    Supported: PDF, EPUB, TXT, Markdown, HTML.
    """
    books, _ = await _services()
    return _added(await books.add_from_url(url, _meta(title, author, tags)))


@mcp.tool()
async def add_book_by_title(
    title: str,
    author: str | None = None,
    tags: list[str] | None = None,
) -> AddedBook:
    """Find a public-domain edition of a book by name and add it.

    Searches Project Gutenberg, Standard Ebooks, Open Library, the Internet
    Archive and Wikisource, preferring clean transcriptions over scans. Only
    works for public-domain titles — for anything in copyright the user must
    supply the file.

    Args:
        title: The book's title.
        author: The author. Worth passing — it disambiguates common titles.
        tags: Labels for later filtering.
    """
    books, _ = await _services()
    book, resolved = await books.add_by_title(title, author, _meta(title, author, tags))
    added = _added(book)
    added.note = f"Found on {resolved.source_type.replace('_', ' ')}. {added.note}"
    return added


@mcp.tool()
async def add_text(
    title: str,
    text: str,
    author: str | None = None,
    tags: list[str] | None = None,
) -> AddedBook:
    """Add pasted text to the library — notes, an article, a transcript.

    Markdown headings in the text become chapter and section boundaries.
    """
    books, _ = await _services()
    return _added(await books.add_text(title=title, text=text, author=author, tags=tags))


# ── Managing ──────────────────────────────────────────────────────────────────

@mcp.tool()
async def delete_book(book_id: int) -> str:
    """Permanently remove a book and all of its passages. Confirm with the user first."""
    books, _ = await _services()
    title = await books.delete_book(book_id)
    return f"Deleted {title!r} and all of its passages."


@mcp.tool()
async def reingest_book(book_id: int) -> AddedBook:
    """Re-run ingestion for a book. Use to retry one that failed."""
    books, _ = await _services()
    added = _added(await books.reingest_book(book_id))
    added.note = "Re-queued for ingestion. Check library_status for progress."
    return added


# ── Conversions ───────────────────────────────────────────────────────────────

def _meta(title: str | None, author: str | None, tags: list[str] | None) -> BookMetadata:
    return BookMetadata(title=title or "", author=author, tags=tags or [])


def _book(summary: BookSummary) -> BookItem:
    return BookItem(
        id=summary.id,
        title=summary.title,
        author=summary.author,
        format=str(summary.format),
        status=str(summary.status),
        chunk_count=summary.chunk_count,
        error=summary.error_message,
    )


def _added(book: Book) -> AddedBook:
    return AddedBook(
        id=book.id,
        title=book.metadata.title,
        author=book.metadata.author,
        format=str(book.format),
        status=str(book.status),
        source=book.metadata.source,
    )


def _added_summary(summary: BookSummary) -> AddedBook:
    return AddedBook(
        id=summary.id,
        title=summary.title,
        author=summary.author,
        format=str(summary.format),
        status=str(summary.status),
    )


def _passage(passage: Passage) -> PassageResult:
    location = passage.location
    return PassageResult(
        text=passage.text,
        citation=passage.citation.to_string(),
        score=passage.score,
        book_id=passage.book_id,
        book_title=passage.citation.book_title,
        chunk_id=passage.chunk_id,
        chapter_title=location.chapter_title,
        section_title=location.section_title,
        page_start=location.page_start,
        chapter_n=location.chapter_n,
        section_n=location.section_n,
    )


def _toc_items(toc: list[TocEntry]) -> list[TocItem]:
    """Number the stored ToC the way read_chapter and read_section expect.

    The chunker numbers chapters and sections as it walks the book, so walking
    the ToC in the same order reproduces those numbers exactly.
    """
    items: list[TocItem] = []
    chapter_n = 0
    section_n = 0

    for entry in toc:
        if entry.level == 1:
            chapter_n += 1
            section_n = 1
        else:
            section_n += 1
        items.append(
            TocItem(
                title=entry.title,
                level=entry.level,
                chapter_n=max(chapter_n, 1),
                section_n=max(section_n, 1),
                page_start=entry.page_start,
            )
        )
    return items


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cognita", description="MCP server over your personal library."
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Serve over HTTP instead of stdio, for remote or shared use.",
    )
    parser.add_argument("--host", default=settings.MCP_HOST)
    parser.add_argument("--port", type=int, default=settings.MCP_PORT)
    parser.add_argument(
        "--allow-local-files",
        action="store_true",
        help="Permit the local-disk tools in HTTP mode. Off by default.",
    )
    return parser.parse_args(argv)


def _preflight() -> None:
    missing = settings.missing_required()
    if missing:
        raise SystemExit(
            f"Missing required environment variables: {', '.join(missing)}.\n"
            "Copy .env.example to .env and fill them in."
        )
    logger.info(
        "Formats: %s | contextual indexing: %s | reranking: %s | OCR: %s",
        ", ".join(e.lstrip(".") for e in SUPPORTED_EXTENSIONS),
        "on" if settings.context_enabled else "off",
        "on" if settings.rerank_enabled else "off",
        "on" if settings.ocr_enabled else "off",
    )


def run() -> None:
    global _allow_local_files

    args = _parse_args(sys.argv[1:])
    # In stdio mode the protocol owns stdout, so logs must go to stderr.
    setup_logging(settings.LOG_LEVEL, stream=sys.stderr)
    _preflight()

    if not args.http:
        logger.info("Cognita ready on stdio")
        mcp.run(transport="stdio")
        return

    from cognita.transport import serve_http

    _allow_local_files = args.allow_local_files or settings.ALLOW_LOCAL_FILES
    if _allow_local_files:
        logger.warning(
            "Local-file tools are enabled over HTTP — only do this on a server "
            "whose filesystem you intend callers to reach."
        )
    asyncio.run(serve_http(mcp, args.host, args.port))


if __name__ == "__main__":
    run()
