"""The shaping stage between retrieval and the caller.

Covers score normalisation, the per-book cap, adjacent-chunk merging, overlap
removal, and citation formatting — everything the search service does to raw
hits that does not require a database.
"""

from cognita.chunks.domain import Chunk, ChunkLocation
from cognita.search.domain import join_without_overlap
from cognita.search.service import (
    _Candidate,
    _cap_per_book,
    _citation,
    _merge_adjacent,
    _normalise,
)


def _chunk(chunk_id: int, book_id: int = 1, sequence: int = 0, text: str = "text", **loc) -> Chunk:
    return Chunk(
        id=chunk_id,
        book_id=book_id,
        text=text,
        sequence=sequence,
        location=ChunkLocation(**loc),
    )


# ── Score normalisation ───────────────────────────────────────────────────────

def test_normalise_puts_the_best_hit_at_one():
    candidates = _normalise([
        _Candidate(_chunk(1), 0.032),
        _Candidate(_chunk(2), 0.016),
        _Candidate(_chunk(3), 0.008),
    ])

    assert candidates[0].score == 1.0
    assert candidates[1].score == 0.5
    assert candidates[2].score == 0.25


def test_normalise_preserves_relative_ordering():
    scores = [0.4, 0.9, 0.1]
    normalised = _normalise([_Candidate(_chunk(i), s) for i, s in enumerate(scores)])

    assert [c.score for c in normalised] == [0.4 / 0.9, 1.0, 0.1 / 0.9]


def test_normalise_handles_degenerate_input():
    assert _normalise([]) == []
    zeroed = _normalise([_Candidate(_chunk(1), 0.0)])
    assert zeroed[0].score == 0.0


# ── Per-book cap ──────────────────────────────────────────────────────────────

def test_cap_limits_each_book_but_keeps_rank_order():
    candidates = [
        _Candidate(_chunk(1, book_id=1), 1.0),
        _Candidate(_chunk(2, book_id=1), 0.9),
        _Candidate(_chunk(3, book_id=1), 0.8),
        _Candidate(_chunk(4, book_id=2), 0.7),
        _Candidate(_chunk(5, book_id=3), 0.6),
    ]
    kept = _cap_per_book(candidates, cap=2)

    assert [c.chunk.id for c in kept] == [1, 2, 4, 5]


def test_cap_of_one_gives_a_book_apiece():
    candidates = [_Candidate(_chunk(i, book_id=i % 2), 1.0 - i / 10) for i in range(6)]
    kept = _cap_per_book(candidates, cap=1)

    assert len({c.chunk.book_id for c in kept}) == len(kept)


# ── Overlap removal ───────────────────────────────────────────────────────────

def test_join_drops_the_paragraphs_two_chunks_share():
    first = "Alpha para.\n\nBeta para.\n\nGamma para."
    second = "Beta para.\n\nGamma para.\n\nDelta para."

    joined = join_without_overlap([first, second])

    assert joined == "Alpha para.\n\nBeta para.\n\nGamma para.\n\nDelta para."
    assert joined.count("Gamma para.") == 1


def test_join_leaves_unrelated_chunks_alone():
    joined = join_without_overlap(["One.", "Two.", "Three."])
    assert joined == "One.\n\nTwo.\n\nThree."


def test_join_handles_full_containment():
    assert join_without_overlap(["A.\n\nB.", "A.\n\nB."]) == "A.\n\nB."


def test_join_of_nothing_is_empty():
    assert join_without_overlap([]) == ""


# ── Adjacent merging ──────────────────────────────────────────────────────────

def test_consecutive_chunks_merge_into_one_passage():
    merged = _merge_adjacent([
        _Candidate(_chunk(1, sequence=4, text="First half."), 0.9),
        _Candidate(_chunk(2, sequence=5, text="Second half."), 0.7),
    ])

    assert len(merged) == 1
    assert merged[0].ids == [1, 2]
    assert merged[0].score == 0.9, "the merged passage keeps the better score"
    assert "First half." in merged[0].chunk.text
    assert "Second half." in merged[0].chunk.text


def test_distant_chunks_stay_separate():
    merged = _merge_adjacent([
        _Candidate(_chunk(1, sequence=2), 0.9),
        _Candidate(_chunk(2, sequence=40), 0.8),
    ])
    assert len(merged) == 2


def test_chunks_from_different_books_never_merge():
    merged = _merge_adjacent([
        _Candidate(_chunk(1, book_id=1, sequence=5), 0.9),
        _Candidate(_chunk(2, book_id=2, sequence=6), 0.8),
    ])
    assert len(merged) == 2


def test_a_long_run_collapses_to_a_single_passage():
    merged = _merge_adjacent([
        _Candidate(_chunk(i, sequence=i, text=f"Part {i}."), 1.0 - i / 10) for i in range(5)
    ])

    assert len(merged) == 1
    assert merged[0].ids == [0, 1, 2, 3, 4]
    assert all(f"Part {i}." in merged[0].chunk.text for i in range(5))


def test_merging_removes_the_chunker_overlap():
    """Adjacent chunks share trailing paragraphs; the merge must not repeat them."""
    merged = _merge_adjacent([
        _Candidate(_chunk(1, sequence=0, text="One.\n\nTwo.\n\nThree."), 0.9),
        _Candidate(_chunk(2, sequence=1, text="Three.\n\nFour."), 0.8),
    ])

    assert merged[0].chunk.text.count("Three.") == 1


def test_merged_passage_spans_the_full_page_range():
    merged = _merge_adjacent([
        _Candidate(_chunk(1, sequence=0, page_start=10, page_end=10), 0.9),
        _Candidate(_chunk(2, sequence=1, page_start=11, page_end=12), 0.8),
    ])

    assert merged[0].chunk.location.page_start == 10
    assert merged[0].chunk.location.page_end == 12


def test_merging_a_single_candidate_is_a_no_op():
    only = _Candidate(_chunk(1), 1.0)
    assert _merge_adjacent([only]) == [only]


# ── Citations ─────────────────────────────────────────────────────────────────

def test_citation_includes_author_chapter_and_page():
    citation = _citation(
        _chunk(1, chapter_title="On Duty", section_title="Part II", page_start=42, page_end=43),
        "Meditations",
        "Marcus Aurelius",
    )
    rendered = citation.to_string()

    assert rendered == "Marcus Aurelius, Meditations › On Duty › Part II › pp. 42–43"


def test_citation_without_an_author_leads_with_the_title():
    assert _citation(_chunk(1), "Some Book", None).to_string() == "Some Book"


def test_citation_single_page_is_not_a_range():
    citation = _citation(_chunk(1, page_start=7, page_end=7), "Book", None)
    assert "p. 7" in citation.to_string()
    assert "pp." not in citation.to_string()


def test_citation_infers_the_end_page_when_missing():
    citation = _citation(_chunk(1, page_start=7), "Book", None)
    assert "p. 7" in citation.to_string()


def test_citation_does_not_repeat_a_section_matching_its_chapter():
    citation = _citation(
        _chunk(1, chapter_title="Full Text", section_title="Full Text"), "Book", None
    )
    assert citation.to_string() == "Book › Full Text"


def test_citation_does_not_repeat_the_book_title():
    """A short document's only heading is often its own title."""
    citation = _citation(
        _chunk(1, chapter_title="Reading notes", section_title="Reading notes"),
        "Reading notes",
        None,
    )
    assert citation.to_string() == "Reading notes"


def test_citation_keeps_a_section_that_differs_from_the_title():
    citation = _citation(
        _chunk(1, chapter_title="Notes", section_title="Monday"), "Notes", None
    )
    assert citation.to_string() == "Notes › Monday"
