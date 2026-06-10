from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from cognita.auth.dependencies import get_current_user_id
from cognita.core.exceptions import ConflictError, NotFoundError
from cognita.infrastructure.database import get_pool
from cognita.research.service import ResearchService

router = APIRouter(prefix="/research", tags=["research"])


def _get_service(pool=Depends(get_pool)) -> ResearchService:
    return ResearchService(pool)


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=3)
    specialty_id: int | None = None
    book_ids: list[int] | None = None
    depth: int = Field(2, ge=1, le=3)


class ResearchFindingResponse(BaseModel):
    n: int
    chunk_id: int
    book_id: int
    text: str
    score: float
    citation: str


class ResearchReportResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]
    sub_queries: list[str]
    specialty_name: str | None
    findings: list[ResearchFindingResponse]


@router.post("/", response_model=ResearchReportResponse)
async def deep_research(
    req: ResearchRequest,
    user_id: str = Depends(get_current_user_id),
    svc: ResearchService = Depends(_get_service),
):
    try:
        report = await svc.deep_research(
            user_id=user_id,
            question=req.question,
            specialty_id=req.specialty_id,
            book_ids=req.book_ids,
            depth=req.depth,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ResearchReportResponse(
        question=report.question,
        answer=report.answer,
        citations=report.citations,
        sub_queries=report.sub_queries,
        specialty_name=report.specialty_name,
        findings=[
            ResearchFindingResponse(
                n=f.n,
                chunk_id=f.result.chunk.id,
                book_id=f.result.chunk.book_id,
                text=f.result.text,
                score=round(f.result.score, 4),
                citation=f.citation,
            )
            for f in report.findings
        ],
    )
