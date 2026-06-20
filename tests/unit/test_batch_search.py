"""Unit tests for _merge_hits — the pure deduplication helper in SearchService.

_merge_hits is the core logic of batch_search: it takes the per-subquery hit
lists, deduplicates by chunk_id (keeping the best RRF score), and sorts
descending. Because it is a pure function with no I/O, it is straightforward
to unit-test exhaustively.
"""

from cognita.chunks.domain import Chunk, ChunkLevel, ChunkLocation
from cognita.search.service import _merge_hits


# ── helpers ───────────────────────────────────────────────────────────────────

def _chunk(chunk_id: int, book_id: int = 1) -> Chunk:
    return Chunk(
        id=chunk_id,
        book_id=book_id,
        user_id="u",
        text=f"text for chunk {chunk_id}",
        level=ChunkLevel.PARAGRAPH,
        sequence=chunk_id,
        location=ChunkLocation(),
        token_count=10,
    )


# ── basic behaviour ───────────────────────────────────────────────────────────

def test_empty_input_returns_empty():
    assert _merge_hits([]) == []


def test_single_subquery_passthrough():
    hits = [(_chunk(1), 0.9), (_chunk(2), 0.7), (_chunk(3), 0.5)]
    result = _merge_hits([hits])

    assert [(c.id, s) for c, s in result] == [(1, 0.9), (2, 0.7), (3, 0.5)]


def test_no_overlap_concatenates_and_sorts():
    hits_a = [(_chunk(1), 0.8), (_chunk(2), 0.6)]
    hits_b = [(_chunk(3), 0.9), (_chunk(4), 0.3)]

    result = _merge_hits([hits_a, hits_b])

    ids = [c.id for c, _ in result]
    scores = [s for _, s in result]

    assert ids == [3, 1, 2, 4]
    assert scores == sorted(scores, reverse=True)


# ── deduplication ─────────────────────────────────────────────────────────────

def test_duplicate_chunk_keeps_highest_score():
    hits_a = [(_chunk(1), 0.5)]
    hits_b = [(_chunk(1), 0.9)]   # same chunk, higher score in second list

    result = _merge_hits([hits_a, hits_b])

    assert len(result) == 1
    chunk, score = result[0]
    assert chunk.id == 1
    assert score == 0.9


def test_duplicate_chunk_keeps_highest_score_regardless_of_order():
    hits_a = [(_chunk(1), 0.9)]   # higher score first this time
    hits_b = [(_chunk(1), 0.5)]

    result = _merge_hits([hits_a, hits_b])

    assert len(result) == 1
    _, score = result[0]
    assert score == 0.9


def test_three_subqueries_same_chunk_takes_max():
    chunk = _chunk(42)
    hits_a = [(chunk, 0.3)]
    hits_b = [(chunk, 0.8)]
    hits_c = [(chunk, 0.5)]

    result = _merge_hits([hits_a, hits_b, hits_c])

    assert len(result) == 1
    _, score = result[0]
    assert score == 0.8


def test_partial_overlap():
    chunk_shared = _chunk(10)
    hits_a = [(chunk_shared, 0.4), (_chunk(11), 0.7)]
    hits_b = [(chunk_shared, 0.9), (_chunk(12), 0.2)]

    result = _merge_hits([hits_a, hits_b])

    assert len(result) == 3
    ids = [c.id for c, _ in result]
    assert 10 in ids
    assert 11 in ids
    assert 12 in ids

    shared_score = next(s for c, s in result if c.id == 10)
    assert shared_score == 0.9


# ── ordering guarantees ───────────────────────────────────────────────────────

def test_output_is_sorted_descending_by_score():
    hits_a = [(_chunk(1), 0.3), (_chunk(2), 0.9)]
    hits_b = [(_chunk(3), 0.6), (_chunk(4), 0.1)]

    result = _merge_hits([hits_a, hits_b])

    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


def test_many_subqueries_all_unique():
    all_hits = [
        [(_chunk(i * 10 + j), float(j) / 10) for j in range(5)]
        for i in range(4)
    ]

    result = _merge_hits(all_hits)

    assert len(result) == 20
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


# ── chunk identity ────────────────────────────────────────────────────────────

def test_chunk_object_from_highest_scoring_list_is_preserved():
    """When the same chunk_id appears in multiple lists, the Chunk object
    from whichever list had the highest score should be in the output."""
    chunk_low = _chunk(99)
    chunk_low.text = "low score version"

    chunk_high = _chunk(99)
    chunk_high.text = "high score version"

    result = _merge_hits([[(chunk_low, 0.2)], [(chunk_high, 0.8)]])

    assert len(result) == 1
    obj, score = result[0]
    assert score == 0.8
    assert obj.text == "high score version"
