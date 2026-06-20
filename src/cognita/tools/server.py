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
from cognita.search.service import SearchService
from cognita.specialties.service import SpecialtyService
from cognita.tools.coverage import assess_coverage as _do_assess_coverage
from cognita.tools.planner import plan_research as _do_plan_research
from cognita.tools.schemas import (
    BookItem,
    CoverageAssessment,
    CorpusSuggestionItem,
    ExpandedPassage,
    PassageHit,
    PassageResult,
    ResearchPlan,
    SpecialtyItem,
    SpecialtyWithSuggestionsItem,
    TocItem,
)

setup_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)

mcp = FastMCP(
    "Cognita MCP",
    instructions=(
        "You have access to the user's personal book library. Specialties are named expert "
        "scopes: each bundles a curated set of books with an optional persona — an instruction "
        "block describing how that expert should reason and write.\n\n"
        "To answer a research question, follow this four-step flow:\n\n"
        "1. PLAN — Call plan_research(question). It selects the best specialty, writes 2–5 "
        "retrieval subqueries, chooses a depth, and flags whether a chapter scan will be needed. "
        "If persona is set in the returned plan, adopt it as your guiding voice.\n\n"
        "2. RETRIEVE — Call execute_research_plan with the specialty_id, subqueries, and depth "
        "from the plan. All subqueries run in parallel server-side; you get back a single "
        "merged, reranked passage list. Use get_passage_context on hits that need more context, "
        "and get_chapter / get_section when needs_chapter_scan=true.\n\n"
        "3. ASSESS — Call assess_coverage(question, hits) with the passages from step 2. "
        "If satisfied=true, proceed to synthesis. If not, call execute_research_plan once more "
        "with the suggested_queries from the assessment. Do not loop more than twice.\n\n"
        "4. SYNTHESIZE — Write the answer from the retrieved passages only. Cite every claim "
        "using the citation string from each result. Do not speculate beyond what the passages "
        "support.\n\n"
        "Always include the citation string when referencing content. "
        "For simple factual questions you may skip plan_research and call semantic_search directly."
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


# ── Query planner + parallel retrieval + coverage assessment ──────────────────

_DEPTH_TOP_K: dict[str, int] = {"shallow": 5, "medium": 10, "deep": 15}


@mcp.tool()
async def plan_research(question: str, ctx) -> ResearchPlan:
    """Plan a research question into a structured retrieval plan.

    Always call this first when answering a non-trivial question. It selects the
    best specialty, produces 2–5 retrieval-phrased subqueries, chooses a depth,
    and flags whether a chapter scan will likely be needed.

    Pass specialty_id, subqueries, and depth directly to execute_research_plan.
    If persona is returned, adopt it as your guiding voice when synthesising.
    """
    user_id = _user_id_from_context(ctx)
    spec_svc = await _specialty_service()
    specialties = await spec_svc.list_specialties(user_id)
    spec_dicts = [
        {"id": s.id, "name": s.name, "description": s.description}
        for s in specialties
    ]

    raw = await _do_plan_research(question, spec_dicts)

    specialty_id: int | None = raw.get("specialty_id")
    spec_name: str | None = None
    persona: str | None = None
    if specialty_id is not None:
        match = next((s for s in specialties if s.id == specialty_id), None)
        if match:
            spec_name = match.name
            persona = match.persona

    return ResearchPlan(
        specialty_id=specialty_id,
        specialty_name=spec_name,
        persona=persona,
        intent=raw.get("intent", "overview"),
        subqueries=raw.get("subqueries", [question]),
        depth=raw.get("depth", "medium"),
        needs_chapter_scan=bool(raw.get("needs_chapter_scan", False)),
    )


@mcp.tool()
async def execute_research_plan(
    question: str,
    subqueries: list[str],
    ctx,
    specialty_id: int | None = None,
    depth: str = "medium",
) -> list[PassageResult]:
    """Run a research plan's subqueries in parallel and return merged results.

    Use the values returned by plan_research. All subqueries are embedded and
    searched concurrently; results are merged, deduplicated by chunk, and reranked
    against the original question before returning.

    depth controls the number of passages returned:
      shallow → 5  |  medium → 10  |  deep → 15
    """
    user_id = _user_id_from_context(ctx)
    book_ids: list[int] | None = None
    if specialty_id is not None:
        spec_svc = await _specialty_service()
        _, book_ids = await spec_svc.resolve_scope(user_id, specialty_id)

    top_k = _DEPTH_TOP_K.get(depth, 10)
    svc = await _search_service()
    resp = await svc.batch_search(
        user_id=user_id,
        question=question,
        queries=subqueries,
        book_ids=book_ids,
        top_k=top_k,
    )
    return [_to_passage_result(r) for r in resp.results]


@mcp.tool()
async def assess_coverage(
    question: str,
    hits: list[PassageHit],
    ctx,
) -> CoverageAssessment:
    """Decide whether the retrieved passages adequately answer the question.

    Pass the question and the passages from execute_research_plan. Returns:
      satisfied — true if the passages are sufficient to answer
      gaps      — aspects of the question not yet covered
      suggested_queries — follow-up queries to close the gaps

    If satisfied=false, call execute_research_plan again with suggested_queries.
    Limit to two coverage loops before synthesising with what you have.
    """
    passage_dicts = [{"text": h.text, "citation": h.citation} for h in hits]
    raw = await _do_assess_coverage(question, passage_dicts)
    return CoverageAssessment(
        satisfied=bool(raw.get("satisfied", False)),
        gaps=raw.get("gaps", []),
        suggested_queries=raw.get("suggested_queries", []),
    )


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

    Each specialty groups books around a subject and may carry a persona (an instruction
    block describing how that expert reasons and writes) plus the book_ids in its scope.
    To consult one: adopt its persona, then pass its id as specialty_id to semantic_search
    (and get_chapter / get_section / get_passage_context) to retrieve only from its books.
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


# ── Structural retrieval — read specific locations within a book ─────────────

@mcp.tool()
async def get_chapter(book_id: int, chapter_n: int, ctx) -> list[PassageResult]:
    """Retrieve all passages from a specific chapter by its sequential number.

    Use when you know the chapter number (e.g. from get_table_of_contents) and want
    to read it in full rather than relying on semantic_search hits.
    """
    user_id = _user_id_from_context(ctx)
    svc = await _search_service()
    results = await svc.get_passage_by_location(
        user_id=user_id, book_id=book_id, chapter_n=chapter_n
    )
    return [_to_passage_result(r) for r in results]


@mcp.tool()
async def get_section(
    book_id: int,
    chapter_n: int,
    section_n: int,
    ctx,
) -> list[PassageResult]:
    """Retrieve passages from a specific section within a chapter."""
    user_id = _user_id_from_context(ctx)
    svc = await _search_service()
    results = await svc.get_passage_by_location(
        user_id=user_id, book_id=book_id, chapter_n=chapter_n, section_n=section_n
    )
    return [_to_passage_result(r) for r in results]


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
