from fastapi import APIRouter, Depends, HTTPException, status

from cognita.auth.dependencies import get_current_user_id
from cognita.core.exceptions import ConflictError, NotFoundError
from cognita.infrastructure.database import get_pool
from cognita.specialties.domain import Specialty
from cognita.specialties.schemas import (
    SpecialtyBooksInput,
    SpecialtyCreateInput,
    SpecialtyResponse,
    SpecialtyUpdateInput,
)
from cognita.specialties.service import SpecialtyService

router = APIRouter(prefix="/specialties", tags=["specialties"])


def _get_service(pool=Depends(get_pool)) -> SpecialtyService:
    return SpecialtyService(pool)


def _to_response(s: Specialty) -> SpecialtyResponse:
    return SpecialtyResponse(
        id=s.id,
        name=s.name,
        description=s.description,
        persona=s.persona,
        book_ids=s.book_ids,
        book_count=s.book_count,
        created_at=s.created_at,
        updated_at=s.updated_at,
    )


@router.get("/", response_model=list[SpecialtyResponse])
async def list_specialties(
    user_id: str = Depends(get_current_user_id),
    svc: SpecialtyService = Depends(_get_service),
):
    return [_to_response(s) for s in await svc.list_specialties(user_id)]


@router.post("/", response_model=SpecialtyResponse, status_code=status.HTTP_201_CREATED)
async def create_specialty(
    req: SpecialtyCreateInput,
    user_id: str = Depends(get_current_user_id),
    svc: SpecialtyService = Depends(_get_service),
):
    try:
        specialty = await svc.create(
            user_id=user_id,
            name=req.name,
            description=req.description,
            persona=req.persona,
            book_ids=req.book_ids,
        )
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_response(specialty)


@router.get("/{specialty_id}", response_model=SpecialtyResponse)
async def get_specialty(
    specialty_id: int,
    user_id: str = Depends(get_current_user_id),
    svc: SpecialtyService = Depends(_get_service),
):
    try:
        specialty = await svc.get(user_id, specialty_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_response(specialty)


@router.put("/{specialty_id}", response_model=SpecialtyResponse)
async def update_specialty(
    specialty_id: int,
    req: SpecialtyUpdateInput,
    user_id: str = Depends(get_current_user_id),
    svc: SpecialtyService = Depends(_get_service),
):
    try:
        specialty = await svc.update(
            user_id, specialty_id, req.name, req.description, req.persona
        )
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_response(specialty)


@router.delete("/{specialty_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_specialty(
    specialty_id: int,
    user_id: str = Depends(get_current_user_id),
    svc: SpecialtyService = Depends(_get_service),
):
    try:
        await svc.delete(user_id, specialty_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{specialty_id}/books", response_model=SpecialtyResponse)
async def add_books(
    specialty_id: int,
    req: SpecialtyBooksInput,
    user_id: str = Depends(get_current_user_id),
    svc: SpecialtyService = Depends(_get_service),
):
    try:
        specialty = await svc.add_books(user_id, specialty_id, req.book_ids)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_response(specialty)


@router.delete("/{specialty_id}/books/{book_id}", response_model=SpecialtyResponse)
async def remove_book(
    specialty_id: int,
    book_id: int,
    user_id: str = Depends(get_current_user_id),
    svc: SpecialtyService = Depends(_get_service),
):
    try:
        specialty = await svc.remove_book(user_id, specialty_id, book_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_response(specialty)
