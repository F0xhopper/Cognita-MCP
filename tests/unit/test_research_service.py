"""Unit tests for the deep-research agent loop."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cognita.books.domain import Book, BookFormat, BookMetadata, BookStatus, TocEntry
from cognita.chunks.domain import Chunk, ChunkLevel, ChunkLocation, Citation
from cognita.core.exceptions import ConflictError
from cognita.research.service import ResearchService, _book_outline, _merge_findings
from cognita.search.domain import SearchResponse, SearchResult
from cognita.specialties.domain import Specialty


def _make_result(chunk_id: int, score: float, text: str = "passage text") -> SearchResult:
    chunk = Chunk(
        id=chunk_id, book_id=1, user_id="u", text=text,
        level=ChunkLevel.PARAGRAPH, sequence=chunk_id,
        location=ChunkLocation(chapter_title="Ch 1", page_start=10, page_end=10),
        token_count=10,
    )
    citation = Citation(
        book_title="Meditations", author=None, chapter="Ch 1",
        section=None, page_range="p. 10", paragraph_n=None,
    )
    return SearchResult(chunk=chunk, score=score, citation=citation)


def _make_service(
    search_results: list[SearchResult],
    plan: dict | None = None,
    gaps: dict | None = None,
    answer: str = "Synthesized answer [1].",
) -> ResearchService:
    json_responses = [plan or {"queries": ["sub query one"]}, gaps or {"queries": []}]
    llm_json = AsyncMock(side_effect=json_responses)
    llm_text = AsyncMock(return_value=answer)

    svc = ResearchService(MagicMock(), llm_json=llm_json, llm_text=llm_text)
    svc._search = AsyncMock()
    svc._search.search.return_value = SearchResponse(
        query="q", results=search_results, total=len(search_results)
    )
    svc._specialties = AsyncMock()
    svc._specialties.resolve_scope.return_value = (None, None)
    return svc


def test_merge_findings_dedupes_keeping_best_score():
    a = _make_result(1, 0.5)
    b = _make_result(1, 0.9)
    c = _make_result(2, 0.7)
    merged = _merge_findings([a, c], [b])
    assert [r.chunk.id for r in merged] == [1, 2]
    assert merged[0].score == 0.9


async def test_deep_research_returns_cited_report():
    results = [_make_result(1, 0.9), _make_result(2, 0.8)]
    svc = _make_service(results)

    report = await svc.deep_research("u", "What does Marcus say about anger?", depth=2)

    assert report.answer == "Synthesized answer [1]."
    assert report.question == "What does Marcus say about anger?"
    # original question always runs as the first sub-query
    assert report.sub_queries[0] == "What does Marcus say about anger?"
    assert "sub query one" in report.sub_queries
    assert len(report.citations) == 2
    assert all("Meditations" in c for c in report.citations)
    assert [f.n for f in report.findings] == [1, 2]


async def test_deep_research_no_findings_skips_synthesis():
    svc = _make_service([])
    report = await svc.deep_research("u", "Unknown topic?", depth=1)
    assert "No relevant passages" in report.answer
    assert report.citations == []
    svc._llm_text.assert_not_called()


async def test_deep_research_gap_check_runs_follow_up_queries():
    results = [_make_result(1, 0.9)]
    svc = _make_service(results, gaps={"queries": ["follow up query"]})

    report = await svc.deep_research("u", "Question?", depth=2)

    assert "follow up query" in report.sub_queries
    searched = [call.kwargs["query"] for call in svc._search.search.call_args_list]
    assert "follow up query" in searched


async def test_deep_research_depth_one_skips_gap_check():
    results = [_make_result(1, 0.9)]
    svc = _make_service(results)

    await svc.deep_research("u", "Question?", depth=1)

    # only the planning call — no gap-check round
    assert svc._llm_json.call_count == 1


async def test_plan_failure_falls_back_to_raw_question():
    results = [_make_result(1, 0.9)]
    svc = _make_service(results)
    svc._llm_json = AsyncMock(side_effect=RuntimeError("LLM down"))

    report = await svc.deep_research("u", "Question?", depth=1)

    assert report.sub_queries == ["Question?"]
    assert report.answer == "Synthesized answer [1]."


def _make_book(
    book_id: int, title: str, author: str | None = None, chapters: list[str] | None = None
) -> Book:
    return Book(
        id=book_id, user_id="u", status=BookStatus.READY, format=BookFormat.EPUB,
        storage_path="x", file_size_bytes=1,
        metadata=BookMetadata(title=title, author=author),
        toc=[
            TocEntry(title=t, level=1, sequence=i)
            for i, t in enumerate(chapters or [])
        ],
    )


def _make_gap_service(llm_response: dict) -> ResearchService:
    svc = ResearchService(MagicMock(), llm_json=AsyncMock(return_value=llm_response))
    svc._specialties = AsyncMock()
    svc._specialties.get.return_value = Specialty(
        id=7, user_id="u", name="Stoic Philosophy",
        description="Stoicism corpus", persona=None, book_ids=[1, 2],
    )
    svc._books = AsyncMock()
    svc._books.get.side_effect = [
        _make_book(1, "Meditations", "Marcus Aurelius", ["On Anger", "On Death"]),
        _make_book(2, "Letters", "Seneca"),
    ]
    return svc


def test_book_outline_includes_author_and_chapters():
    book = _make_book(1, "Meditations", "Marcus Aurelius", ["On Anger", "On Death"])
    outline = _book_outline(book)
    assert outline == "- Meditations — Marcus Aurelius (contents: On Anger; On Death)"


async def test_analyze_gaps_builds_report():
    svc = _make_gap_service({
        "summary": "Coverage is strong on ethics but weak on physics.",
        "gaps": [
            {"topic": "Stoic physics", "reason": "No coverage", "suggested_reading": ["Hahm"]},
        ],
    })

    analysis = await svc.analyze_gaps("u", 7)

    assert analysis.specialty_name == "Stoic Philosophy"
    assert analysis.books_analyzed == 2
    assert analysis.summary.startswith("Coverage is strong")
    assert [g.topic for g in analysis.gaps] == ["Stoic physics"]
    assert analysis.gaps[0].suggested_reading == ["Hahm"]
    # the corpus outline is included in the LLM prompt
    prompt = svc._llm_json.call_args.args[0]
    assert "Meditations — Marcus Aurelius" in prompt
    assert "Stoic Philosophy" in prompt


async def test_analyze_gaps_empty_specialty_raises():
    svc = _make_gap_service({"summary": "", "gaps": []})
    svc._specialties.get.return_value = Specialty(
        id=7, user_id="u", name="Empty", book_ids=[],
    )
    with pytest.raises(ConflictError):
        await svc.analyze_gaps("u", 7)


async def test_analyze_gaps_skips_malformed_entries():
    svc = _make_gap_service({
        "summary": "ok",
        "gaps": [
            {"topic": "Valid gap", "reason": "r", "suggested_reading": ["Book", 42]},
            {"reason": "missing topic"},
            "not a dict",
            {"topic": "   "},
        ],
    })

    analysis = await svc.analyze_gaps("u", 7)

    assert [g.topic for g in analysis.gaps] == ["Valid gap"]
    assert analysis.gaps[0].suggested_reading == ["Book"]


async def test_deep_research_uses_specialty_scope_and_name():
    results = [_make_result(1, 0.9)]
    svc = _make_service(results)
    svc._specialties.resolve_scope.return_value = ("You are an expert.", [1, 2])
    svc._specialties.get.return_value = MagicMock(name_attr="x")
    svc._specialties.get.return_value.name = "Stoicism"

    report = await svc.deep_research("u", "Question?", specialty_id=7, depth=1)

    assert report.specialty_name == "Stoicism"
    svc._specialties.resolve_scope.assert_awaited_once_with("u", 7, None)
    for call in svc._search.search.call_args_list:
        assert call.kwargs["book_ids"] == [1, 2]
    # persona is threaded into the synthesis prompt's system message
    assert svc._llm_text.call_args.kwargs["system"] == "You are an expert."
