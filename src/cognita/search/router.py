from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from cognita.auth.dependencies import get_current_user_id
from cognita.chunks.schemas import ChunkLocationResponse, CitationResponse, SearchResultResponse
from cognita.core.exceptions import NotFoundError
from cognita.infrastructure.database import get_pool
from cognita.search.service import SearchService

router = APIRouter(prefix="/search", tags=["search"])


def _get_service(pool=Depends(get_pool)) -> SearchService:
    return SearchService(pool)


class SearchRequest(BaseModel):
    query: str
    book_ids: list[int] | None = None
    top_k: int = 10


@router.post("/", response_model=list[SearchResultResponse])
async def semantic_search(
    req: SearchRequest,
    user_id: str = Depends(get_current_user_id),
    svc: SearchService = Depends(_get_service),
):
    resp = await svc.search(
        user_id=user_id,
        query=req.query,
        book_ids=req.book_ids,
        top_k=req.top_k,
    )
    return [_to_response(r) for r in resp.results]


class PassageContextResponse(BaseModel):
    hit: SearchResultResponse
    before: list[str]
    after: list[str]
    full_text: str


@router.get("/context", response_model=PassageContextResponse)
async def get_passage_context(
    chunk_id: int = Query(...),
    book_id: int = Query(...),
    window: int = Query(2, ge=1, le=5),
    user_id: str = Depends(get_current_user_id),
    svc: SearchService = Depends(_get_service),
):
    try:
        ctx = await svc.get_passage_context(user_id, chunk_id, book_id, window)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return PassageContextResponse(
        hit=_to_response(ctx.hit),
        before=[c.text for c in ctx.before],
        after=[c.text for c in ctx.after],
        full_text=ctx.full_text(),
    )


@router.get("/location", response_model=list[SearchResultResponse])
async def get_passage_by_location(
    book_id: int = Query(...),
    chapter_n: int | None = Query(None),
    section_n: int | None = Query(None),
    user_id: str = Depends(get_current_user_id),
    svc: SearchService = Depends(_get_service),
):
    results = await svc.get_passage_by_location(user_id, book_id, chapter_n, section_n)
    return [_to_response(r) for r in results]


def _to_response(r) -> "SearchResultResponse":
    from cognita.chunks.schemas import SearchResultResponse as SR
    loc = r.chunk.location
    return SR(
        chunk_id=r.chunk.id,
        book_id=r.chunk.book_id,
        text=r.chunk.text,
        score=r.score,
        location=ChunkLocationResponse(
            chapter_title=loc.chapter_title,
            chapter_n=loc.chapter_n,
            section_title=loc.section_title,
            section_n=loc.section_n,
            page_start=loc.page_start,
            page_end=loc.page_end,
            paragraph_n=loc.paragraph_n,
        ),
        citation=CitationResponse(
            text=r.citation.to_string(),
            book_title=r.citation.book_title,
            author=r.citation.author,
            chapter=r.citation.chapter,
            section=r.citation.section,
            page_range=r.citation.page_range,
        ),
    )
