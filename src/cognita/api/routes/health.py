from fastapi import APIRouter, Depends
from pydantic import BaseModel

from cognita.infrastructure.database import get_pool

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    status: str
    db: str


@router.get("/health", response_model=HealthResponse)
async def health(pool=Depends(get_pool)):
    try:
        async with pool.acquire() as conn:
            await conn.execute("SELECT 1")
        db_status = "ok"
    except Exception:
        db_status = "error"
    return HealthResponse(status="ok", db=db_status)
