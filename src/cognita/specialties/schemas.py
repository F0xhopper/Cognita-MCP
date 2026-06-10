from datetime import datetime

from pydantic import BaseModel, Field


class SpecialtyCreateInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)
    persona: str | None = Field(
        None,
        max_length=4000,
        description="Instruction block shaping how the expert presents itself",
    )
    book_ids: list[int] = Field(default_factory=list)


class SpecialtyUpdateInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)
    persona: str | None = Field(None, max_length=4000)


class SpecialtyBooksInput(BaseModel):
    book_ids: list[int] = Field(..., min_length=1)


class SpecialtyResponse(BaseModel):
    id: int
    name: str
    description: str | None
    persona: str | None
    book_ids: list[int]
    book_count: int
    created_at: datetime
    updated_at: datetime
