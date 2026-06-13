"""Cognita MCP Server — exposes library tools to AI agents via the MCP protocol.

Transport: SSE (HTTP-based, suitable for remote agents).
Can also be run in stdio mode for local Claude Desktop integration.

Usage:
    # SSE server (default):
    cognita-mcp

    # stdio (for Claude Desktop claude_desktop_config.json):
    cognita-mcp --stdio
"""

import asyncio
import sys

import asyncpg
from mcp.server.fastmcp import FastMCP

from cognita.books.domain import BookMetadata, BookStatus
from cognita.books.repository import BookRepository
from cognita.books.service import BookService
from cognita.core.exceptions import UnsupportedFormatError, UrlFetchError
from cognita.ingestion.worker import ingest_book_task
from cognita.core.config import settings
from cognita.core.logging import get_logger, setup_logging
from cognita.infrastructure.database import init_pool, get_pool
from cognita.infrastructure.storage import storage
from cognita.research.service import ResearchService
from cognita.search.service import SearchService
from cognita.specialties.service import SpecialtyService
from cognita.tools.schemas import (
    BookItem,
    ExpandedPassage,
    PassageResult,
    ResearchReportResult,
    SpecialtyItem,
    TocItem,
)

setup_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)

mcp = FastMCP(
    "Cognita MCP",
    instructions=(
        "You have access to the user's personal book library. "
        "For research questions, prefer deep_research — it plans queries, retrieves, "
        "and returns a synthesized answer with citations in one call. "
        "Use list_specialties to discover scoped experts (named slices of the library "
        "with their own persona) and pass specialty_id to deep_research or semantic_search. "
        "Use semantic_search for quick lookups and get_passage_context to expand a hit "
        "for richer context before quoting. "
        "Always include the citation string when referencing content."
    ),
)

# ── Dependency helpers ────────────────────────────────────────────────────────

_pool: asyncpg.Pool | None = None


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        ssl = "require" if settings.DATABASE_SSL else None
        _pool = await asyncpg.create_pool(settings.DATABASE_URL, min_size=2, max_size=5, ssl=ssl)
    return _pool


async def _book_service() -> BookService:
    pool = await _get_pool()
    return BookService(pool)


async def _search_service() -> SearchService:
    pool = await _get_pool()
    return SearchService(pool)


async def _specialty_service() -> SpecialtyService:
    pool = await _get_pool()
    return SpecialtyService(pool)


async def _research_service() -> ResearchService:
    pool = await _get_pool()
    return ResearchService(pool)


# ── Context helper — MCP tools must know which user they're serving ───────────
# The user_id is injected via MCP session metadata set during SSE handshake.
# For stdio / local use, fall back to COGNITA_USER_ID env var.

def _user_id_from_context(ctx) -> str:
    try:
        return ctx.session.metadata["user_id"]
    except (AttributeError, KeyError):
        import os
        uid = os.getenv("COGNITA_USER_ID")
        if not uid:
            raise ValueError(
                "user_id not found in session metadata. "
                "Set COGNITA_USER_ID env var for local stdio mode."
            )
        return uid


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def list_books(ctx) -> list[BookItem]:
    """List all books in the user's library."""
    user_id = _user_id_from_context(ctx)
    svc = await _book_service()
    books = await svc.list_books(user_id)
    return [
        BookItem(
            id=b.id,
            title=b.title,
            author=b.author,
            format=str(b.format),
            chunk_count=b.chunk_count,
            status=str(b.status),
        )
        for b in books
    ]


@mcp.tool()
async def search_library(query: str, ctx) -> list[BookItem]:
    """Search the library for books by title, author, or keyword."""
    user_id = _user_id_from_context(ctx)
    svc = await _book_service()
    books = await svc.search_library(user_id, query)
    return [
        BookItem(
            id=b.id,
            title=b.title,
            author=b.author,
            format=str(b.format),
            chunk_count=b.chunk_count,
            status=str(b.status),
        )
        for b in books
        if b.status == BookStatus.READY
    ]


@mcp.tool()
async def get_table_of_contents(book_id: int, ctx) -> list[TocItem]:
    """Get the table of contents for a specific book."""
    user_id = _user_id_from_context(ctx)
    svc = await _book_service()
    book = await svc.get_book(user_id, book_id)
    return [
        TocItem(
            title=e.title,
            level=e.level,
            sequence=e.sequence,
            page_start=e.page_start,
        )
        for e in book.toc
    ]


@mcp.tool()
async def get_chapter(book_id: int, chapter_n: int, ctx) -> list[PassageResult]:
    """Retrieve all passages from a specific chapter."""
    user_id = _user_id_from_context(ctx)
    svc = await _search_service()
    results = await svc.get_passage_by_location(user_id, book_id, chapter_n=chapter_n)
    return [_to_passage_result(r) for r in results]


@mcp.tool()
async def get_section(book_id: int, chapter_n: int, section_n: int, ctx) -> list[PassageResult]:
    """Retrieve passages from a specific section within a chapter."""
    user_id = _user_id_from_context(ctx)
    svc = await _search_service()
    results = await svc.get_passage_by_location(
        user_id, book_id, chapter_n=chapter_n, section_n=section_n
    )
    return [_to_passage_result(r) for r in results]


@mcp.tool()
async def semantic_search(
    query: str,
    ctx,
    book_ids: list[int] | None = None,
    specialty_id: int | None = None,
    top_k: int = 10,
) -> list[PassageResult]:
    """Search the library using semantic similarity.

    This is the primary quick-retrieval tool. Returns ranked passages with citations.
    Optionally restrict to specific book_ids, or pass specialty_id to search within
    a specialty's books (use list_specialties to discover IDs).
    """
    user_id = _user_id_from_context(ctx)
    if specialty_id is not None:
        spec_svc = await _specialty_service()
        _, book_ids = await spec_svc.resolve_scope(user_id, specialty_id, book_ids)
    svc = await _search_service()
    resp = await svc.search(user_id=user_id, query=query, book_ids=book_ids, top_k=top_k)
    return [_to_passage_result(r) for r in resp.results]


# ── Library management ────────────────────────────────────────────────────────

@mcp.tool()
async def add_book_from_url(
    url: str,
    title: str,
    ctx,
    author: str | None = None,
    year: int | None = None,
    language: str = "en",
    tags: list[str] | None = None,
) -> BookItem:
    """Add a book to the library by downloading it from a URL.

    The book is fetched immediately and then ingested in the background —
    it will appear as status='pending' and transition to 'ready' once ingestion
    completes (typically a few seconds to a few minutes depending on size).

    Supported formats: PDF, EPUB, plain-text (TXT).
    The URL must be publicly accessible; private/internal addresses are rejected.
    Use list_books to poll status after calling this.
    """
    user_id = _user_id_from_context(ctx)
    meta = BookMetadata(
        title=title,
        author=author,
        year=year,
        language=language,
        tags=tags or [],
    )
    svc = await _book_service()
    try:
        book = await svc.add_from_url(user_id=user_id, url=url, meta=meta)
    except UrlFetchError as exc:
        raise ValueError(str(exc)) from exc
    except UnsupportedFormatError as exc:
        raise ValueError(str(exc)) from exc

    ingest_book_task.delay(book.id)
    return BookItem(
        id=book.id,
        title=book.metadata.title,
        author=book.metadata.author,
        format=str(book.format),
        chunk_count=0,
        status=str(book.status),
    )


# ── Specialties — scoped experts over slices of the library ──────────────────

@mcp.tool()
async def list_specialties(ctx) -> list[SpecialtyItem]:
    """List the user's specialties — named expert scopes over subsets of the library.

    Each specialty groups books around a subject and may carry a persona.
    Pass its id as specialty_id to deep_research or semantic_search to consult
    that expert specifically.
    """
    user_id = _user_id_from_context(ctx)
    svc = await _specialty_service()
    return [_to_specialty_item(s) for s in await svc.list_specialties(user_id)]


@mcp.tool()
async def create_specialty(
    name: str,
    ctx,
    description: str | None = None,
    persona: str | None = None,
    book_ids: list[int] | None = None,
) -> SpecialtyItem:
    """Create a new specialty — a named expert scope over a subset of the library.

    Provide a persona to shape how the expert answers (e.g. "You are an expert on
    Stoic philosophy; prefer primary sources and cite precisely").
    """
    user_id = _user_id_from_context(ctx)
    svc = await _specialty_service()
    specialty = await svc.create(
        user_id=user_id,
        name=name,
        description=description,
        persona=persona,
        book_ids=book_ids,
    )
    return _to_specialty_item(specialty)


@mcp.tool()
async def add_books_to_specialty(
    specialty_id: int,
    book_ids: list[int],
    ctx,
) -> SpecialtyItem:
    """Add books to an existing specialty. Use list_books to find book IDs."""
    user_id = _user_id_from_context(ctx)
    svc = await _specialty_service()
    specialty = await svc.add_books(user_id, specialty_id, book_ids)
    return _to_specialty_item(specialty)


# ── Deep research — the agent-as-tool ─────────────────────────────────────────

@mcp.tool()
async def deep_research(
    question: str,
    ctx,
    specialty_id: int | None = None,
    book_ids: list[int] | None = None,
    depth: int = 2,
) -> ResearchReportResult:
    """Run multi-step research over the library and return a cited synthesis.

    Plans focused sub-queries, retrieves passages for each, checks for coverage
    gaps (depth >= 2), and synthesizes an answer in which every claim carries an
    inline marker like [1] referring to the returned citations list.

    Pass specialty_id to consult a specific expert scope (its books and persona).
    Prefer this over manual semantic_search loops for substantive questions.
    """
    user_id = _user_id_from_context(ctx)
    svc = await _research_service()
    report = await svc.deep_research(
        user_id=user_id,
        question=question,
        specialty_id=specialty_id,
        book_ids=book_ids,
        depth=depth,
    )
    return ResearchReportResult(
        question=report.question,
        answer=report.answer,
        citations=report.citations,
        sub_queries=report.sub_queries,
        specialty_name=report.specialty_name,
        passages=[_to_passage_result(f.result) for f in report.findings],
    )


@mcp.tool()
async def get_passage_context(
    chunk_id: int,
    book_id: int,
    ctx,
    window: int = 2,
) -> ExpandedPassage:
    """Expand a search result to include surrounding paragraphs for fuller context.

    Use this after semantic_search to get more context around a hit before quoting.
    """
    user_id = _user_id_from_context(ctx)
    svc = await _search_service()
    ctx_obj = await svc.get_passage_context(user_id, chunk_id, book_id, window)

    all_ids = (
        [c.id for c in ctx_obj.before]
        + [ctx_obj.hit.chunk.id]
        + [c.id for c in ctx_obj.after]
    )
    return ExpandedPassage(
        full_text=ctx_obj.full_text(),
        citation=ctx_obj.hit.citation.to_string(),
        chunk_ids=all_ids,
    )


@mcp.tool()
async def get_passage_by_location(
    book_id: int,
    ctx,
    chapter_n: int | None = None,
    section_n: int | None = None,
) -> list[PassageResult]:
    """Retrieve passages by exact structural location (chapter/section numbers).

    Use get_table_of_contents first to discover valid chapter_n and section_n values.
    """
    user_id = _user_id_from_context(ctx)
    svc = await _search_service()
    results = await svc.get_passage_by_location(user_id, book_id, chapter_n, section_n)
    return [_to_passage_result(r) for r in results]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_specialty_item(s) -> SpecialtyItem:
    return SpecialtyItem(
        id=s.id,
        name=s.name,
        description=s.description,
        persona=s.persona,
        book_ids=s.book_ids,
        book_count=s.book_count,
    )


def _to_passage_result(r) -> PassageResult:
    loc = r.chunk.location
    return PassageResult(
        chunk_id=r.chunk.id,
        book_id=r.chunk.book_id,
        text=r.chunk.text,
        score=round(r.score, 4),
        citation=r.citation.to_string(),
        chapter_title=loc.chapter_title,
        section_title=loc.section_title,
        page_start=loc.page_start,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> None:
    if "--stdio" in sys.argv:
        mcp.run(transport="stdio")
    else:
        import uvicorn
        sse_app = mcp.get_asgi_app()
        uvicorn.run(sse_app, host=settings.HOST, port=settings.MCP_PORT)
