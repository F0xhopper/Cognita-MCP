"""Pydantic result models returned by the MCP tools.

Input contracts are derived by FastMCP from each tool's function signature in
server.py; these models describe the structured outputs the tools return.
Keep them stable and well-documented — they are part of the agent-facing contract.
"""

from pydantic import BaseModel


class BookItem(BaseModel):
    id: int
    title: str
    author: str | None
    format: str
    chunk_count: int
    status: str


class TocItem(BaseModel):
    title: str
    level: int
    sequence: int
    page_start: int | None


class PassageResult(BaseModel):
    chunk_id: int
    book_id: int
    text: str
    score: float
    citation: str          # human-readable: "Author, Title › Chapter › Section › p. 42"
    chapter_title: str | None
    section_title: str | None
    page_start: int | None


class ExpandedPassage(BaseModel):
    full_text: str
    citation: str
    chunk_ids: list[int]   # ordered IDs of all chunks included in full_text


class CorpusSuggestionItem(BaseModel):
    index: int              # pass this in approved_indices to confirm_corpus
    title: str
    author: str
    tier: str               # primary | commentary | competing | synthesis
    rationale: str
    source_url: str | None
    source_type: str        # gutenberg | archive_org | user_upload_required
    approved: bool          # pre-selected default; user should review


class SpecialtyItem(BaseModel):
    id: int
    name: str
    description: str | None
    persona: str | None
    book_ids: list[int]
    book_count: int


class SpecialtyWithSuggestionsItem(SpecialtyItem):
    """Returned by create_specialty — includes suggested corpus for user review.

    Present the suggestions to the user; they approve or reject each one.
    Then call confirm_corpus with the approved indices to start ingestion.
    Items with source_type='user_upload_required' have no URL and must be
    uploaded manually via add_book_from_url after the user locates a copy.
    """
    suggestions: list[CorpusSuggestionItem]


# ── Query planner schemas ─────────────────────────────────────────────────────

class ResearchPlan(BaseModel):
    """Structured retrieval plan returned by plan_research.

    Pass specialty_id, subqueries, and depth directly to execute_research_plan.
    If persona is set, adopt it as your guiding voice when synthesising.
    """
    specialty_id: int | None
    specialty_name: str | None
    persona: str | None
    intent: str              # definition | comparison | causal | biographical | thematic | overview
    subqueries: list[str]   # 2–5 retrieval-phrased queries, ready for execute_research_plan
    depth: str              # shallow | medium | deep
    needs_chapter_scan: bool


class PassageHit(BaseModel):
    """A single retrieved passage, passed to assess_coverage."""
    text: str
    citation: str


class CoverageAssessment(BaseModel):
    """Returned by assess_coverage — signals whether retrieval is sufficient.

    If satisfied=false, pass suggested_queries back to execute_research_plan.
    Limit to two coverage loops before synthesising with what you have.
    """
    satisfied: bool
    gaps: list[str]             # aspects of the question not yet covered
    suggested_queries: list[str]  # retrieval-phrased follow-ups targeting the gaps
