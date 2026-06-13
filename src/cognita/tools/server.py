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
from cognita.books.service import BookService
from cognita.core.config import settings
from cognita.core.exceptions import UnsupportedFormatError, UrlFetchError
from cognita.core.logging import get_logger, setup_logging
from cognita.ingestion.pipeline import ingest_book
from cognita.research.service import ResearchService
from cognita.search.service import SearchService
from cognita.specialties.service import SpecialtyService
from cognita.tools.schemas import (
    BookItem,
    CorpusSuggestionItem,
    ExpandedPassage,
    PassageResult,
    ResearchReportResult,
    SpecialtyItem,
    SpecialtyWithSuggestionsItem,
    TocItem,
)

setup_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)

mcp = FastMCP(
    "Cognita MCP",
    instructions=(
        "You have access to the user's personal book library. "
        "Before answering any research question, call list_specialties first to check "
        "if a relevant expert scope exists, then pass its specialty_id to consult_specialist. "
        "Only skip this if the user specifies book_ids directly. "
        "For research questions, always prefer consult_specialist — it runs a specialist agent "
        "that autonomously searches, expands passages, and returns a synthesized cited answer. "
        "Use semantic_search only for quick, direct lookups when the user wants a specific passage. "
        "Use get_passage_context to expand a hit for richer context before quoting. "
        "Always include the citation string when referencing content."
    ),
)

# ── Dependency helpers ────────────────────────────────────────────────────────

_pool: asyncpg.Pool | None = None


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        ssl = "require" if settings.DATABASE_SSL else None
        _pool = await asyncpg.create_pool(
            settings.DATABASE_URL, min_size=2, max_size=5, ssl=ssl, statement_cache_size=0
        )
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

    asyncio.create_task(ingest_book(book.id, await _get_pool()))
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
    Pass its id as specialty_id to consult_specialist to engage that expert.
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
) -> SpecialtyWithSuggestionsItem:
    """Create a new specialty and receive a suggested corpus for user review.

    After calling this, present the suggestions list to the user. They approve
    or reject each item. Then call confirm_corpus with the approved indices to
    download and ingest the selected texts.

    Books with source_type='user_upload_required' have no auto-resolvable URL —
    the user must upload them manually via add_book_from_url and then add them
    via add_books_to_specialty.
    """
    user_id = _user_id_from_context(ctx)
    svc = await _specialty_service()
    specialty = await svc.create(
        user_id=user_id,
        name=name,
        description=description,
        persona=persona,
    )
    return SpecialtyWithSuggestionsItem(
        id=specialty.id,
        name=specialty.name,
        description=specialty.description,
        persona=specialty.persona,
        book_ids=specialty.book_ids,
        book_count=specialty.book_count,
        suggestions=[
            CorpusSuggestionItem(
                index=i,
                title=s.title,
                author=s.author,
                tier=str(s.tier),
                rationale=s.rationale,
                source_url=s.source_url,
                source_type=str(s.source_type),
                approved=s.approved,
            )
            for i, s in enumerate(specialty.pending_corpus)
        ],
    )


@mcp.tool()
async def confirm_corpus(
    specialty_id: int,
    approved_indices: list[int],
    ctx,
) -> SpecialtyItem:
    """Confirm which suggested sources to ingest for a specialty.

    Pass the index values from the suggestions returned by create_specialty.
    Approved sources are downloaded and ingested in the background.
    Use get_corpus_status to monitor progress.
    """
    user_id = _user_id_from_context(ctx)
    svc = await _specialty_service()
    try:
        specialty = await svc.confirm_corpus(
            user_id=user_id,
            specialty_id=specialty_id,
            approved_indices=approved_indices,
        )
    except Exception as exc:
        raise ValueError(str(exc)) from exc
    return _to_specialty_item(specialty)


@mcp.tool()
async def get_corpus_status(specialty_id: int, ctx) -> dict:
    """Check ingestion progress for a specialty's corpus.

    Returns counts of books by status: ready, processing, pending, failed.
    A specialty is queryable once at least one book reaches 'ready'.
    """
    user_id = _user_id_from_context(ctx)
    svc = await _specialty_service()
    try:
        result = await svc.corpus_status(user_id=user_id, specialty_id=specialty_id)
    except Exception as exc:
        raise ValueError(str(exc)) from exc
    return {
        "specialty_id": result.specialty_id,
        "total": result.total,
        "ready": result.ready,
        "processing": result.processing,
        "pending": result.pending,
        "failed": result.failed,
        "books": [
            {"book_id": b.book_id, "title": b.title, "author": b.author, "status": b.status}
            for b in result.books
        ],
    }


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


# ── Specialist sub-agent ──────────────────────────────────────────────────────

@mcp.tool()
async def consult_specialist(
    question: str,
    ctx,
    specialty_id: int | None = None,
    book_ids: list[int] | None = None,
) -> ResearchReportResult:
    """Delegate a research question to a specialist agent scoped to a specialty.

    The specialist autonomously searches the library, expands promising passages,
    and synthesizes a cited answer in which every claim carries an inline marker
    like [1] referring to the returned citations list.

    Pass specialty_id (from list_specialties) to consult a named expert — it will
    apply that specialty's persona and restrict retrieval to its books.
    Prefer this over manual semantic_search loops for any substantive question.
    """
    user_id = _user_id_from_context(ctx)
    svc = await _research_service()
    report = await svc.consult(
        user_id=user_id,
        question=question,
        specialty_id=specialty_id,
        book_ids=book_ids,
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
        uvicorn.run(mcp.sse_app(), host=settings.HOST, port=settings.MCP_PORT)
