"""Search — turn a question into ranked, citable passages.

The pipeline, in order:

  1. **Filter** — author/tag/book filters resolve to a set of book ids up front,
     so the expensive search only ever touches candidate books.
  2. **Retrieve** — one hybrid (vector + full-text, RRF-fused) query over the
     chunk table, over-fetching when a reranker is available to give it a wider
     pool to choose from.
  3. **Rerank** — optional. A model scores each candidate against the query and
     reorders. Without it the fusion ordering stands.
  4. **Shape** — scores normalised to 0–1, weak hits dropped, one book barred
     from crowding out the rest, and adjacent chunks merged into single passages.
  5. **Cite** — one batched metadata lookup builds a citation per passage.
"""

from dataclasses import dataclass, field

import asyncpg

from cognita.books.repository import BookRepository
from cognita.chunks.domain import Chunk, ChunkLocation, Citation
from cognita.chunks.repository import ChunkRepository
from cognita.core.config import settings
from cognita.core.exceptions import NotFoundError
from cognita.core.logging import get_logger
from cognita.infrastructure.embeddings import embed_text
from cognita.infrastructure.reranker import rerank
from cognita.search.domain import Passage, PassageContext, SearchResponse, join_without_overlap

logger = get_logger(__name__)

# Chunks adjacent in reading order merge into one passage. A gap of one means
# "directly next to"; the chunker's overlap makes such a pair continuous text.
_ADJACENT_GAP = 1


@dataclass
class _Candidate:
    """A chunk in flight through the ranking pipeline."""

    chunk: Chunk
    score: float
    ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.ids:
            self.ids = [self.chunk.id]


class SearchService:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._chunks = ChunkRepository(pool)
        self._books = BookRepository(pool)

    async def search(
        self,
        query: str,
        book_ids: list[int] | None = None,
        authors: list[str] | None = None,
        tags: list[str] | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
        max_per_book: int | None = None,
        merge_adjacent: bool | None = None,
    ) -> SearchResponse:
        query = query.strip()
        if not query:
            raise ValueError("Query is empty")

        scope = await self._books.resolve_ids(book_ids, authors, tags)
        if scope is not None and not scope:
            logger.info("Search filters matched no books")
            return SearchResponse(query=query, passages=[])

        should_merge = settings.MERGE_ADJACENT if merge_adjacent is None else merge_adjacent
        rerank_on = settings.rerank_enabled

        # With a reranker, retrieve a wider pool and let it choose. Merging
        # consumes results in runs, so ask for headroom there too.
        fetch_k = max(settings.RERANK_CANDIDATES, top_k) if rerank_on else top_k
        if should_merge:
            fetch_k = max(fetch_k, top_k * 2)

        embedding = await embed_text(query)
        hits = await self._chunks.hybrid_search(
            query_embedding=embedding,
            query_text=query,
            book_ids=scope,
            candidate_k=max(fetch_k * 4, 200),
            top_k=fetch_k,
        )
        if not hits:
            return SearchResponse(query=query, passages=[])

        candidates = _normalise([_Candidate(chunk, score) for chunk, score in hits])
        ranking = "fusion"

        if rerank_on and len(candidates) > 1:
            reranked = await rerank(
                query,
                [(c.chunk.text, c.chunk.context) for c in candidates],
                top_n=top_k * 2 if should_merge else top_k,
            )
            if reranked is not None:
                candidates = [
                    _Candidate(candidates[index].chunk, score) for index, score in reranked
                ]
                ranking = "reranked"

        if min_score > 0:
            candidates = [c for c in candidates if c.score >= min_score]

        cap = settings.MAX_PER_BOOK if max_per_book is None else max_per_book
        if cap > 0:
            candidates = _cap_per_book(candidates, cap)

        if should_merge:
            candidates = _merge_adjacent(candidates)
            candidates.sort(key=lambda c: c.score, reverse=True)

        passages = await self._to_passages(candidates[:top_k])
        return SearchResponse(query=query, passages=passages, ranking=ranking)

    async def expand_passage(self, chunk_id: int, window: int = 2) -> PassageContext:
        """Widen a hit with the chunks either side of it."""
        neighbours = await self._chunks.get_neighbours(chunk_id, window)
        target = next((c for c in neighbours if c.id == chunk_id), None)
        if target is None:
            raise NotFoundError("Passage", chunk_id)

        passages = await self._to_passages([_Candidate(target, 1.0)])
        return PassageContext(
            passage=passages[0],
            before=[c for c in neighbours if c.sequence < target.sequence],
            after=[c for c in neighbours if c.sequence > target.sequence],
        )

    async def read_location(
        self,
        book_id: int,
        chapter_n: int | None = None,
        section_n: int | None = None,
    ) -> list[Passage]:
        """Read a chapter or section straight through, in reading order."""
        chunks = await self._chunks.get_by_location(book_id, chapter_n, section_n)
        if not chunks:
            target = f"book {book_id} chapter {chapter_n}"
            if section_n is not None:
                target += f" section {section_n}"
            raise NotFoundError("Chapter" if section_n is None else "Section", target)

        candidates = _merge_adjacent([_Candidate(c, 1.0) for c in chunks])
        candidates.sort(key=lambda c: c.chunk.sequence)
        return await self._to_passages(candidates)

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _to_passages(self, candidates: list[_Candidate]) -> list[Passage]:
        if not candidates:
            return []
        titles = await self._books.titles_for([c.chunk.book_id for c in candidates])
        return [
            Passage(
                chunk_id=c.chunk.id,
                book_id=c.chunk.book_id,
                text=c.chunk.text,
                score=round(c.score, 4),
                citation=_citation(c.chunk, *titles.get(c.chunk.book_id, ("Unknown", None))),
                location=c.chunk.location,
                chunk_ids=c.ids,
            )
            for c in candidates
        ]


def _normalise(candidates: list[_Candidate]) -> list[_Candidate]:
    """Rescale fusion scores to 0–1.

    Raw RRF scores are tiny and unit-free (~0.03 for a strong hit), which tells
    a reader nothing. Dividing by the best score in the set makes the top hit 1.0
    and every other result legible as a fraction of it.
    """
    if not candidates:
        return []
    best = max(c.score for c in candidates)
    if best <= 0:
        return candidates
    for candidate in candidates:
        candidate.score /= best
    return candidates


def _cap_per_book(candidates: list[_Candidate], cap: int) -> list[_Candidate]:
    """Keep at most `cap` results per book, preserving rank order."""
    counts: dict[int, int] = {}
    kept: list[_Candidate] = []
    for candidate in candidates:
        book_id = candidate.chunk.book_id
        if counts.get(book_id, 0) >= cap:
            continue
        counts[book_id] = counts.get(book_id, 0) + 1
        kept.append(candidate)
    return kept


def _merge_adjacent(candidates: list[_Candidate]) -> list[_Candidate]:
    """Fold neighbouring chunks of the same book into single, longer passages.

    Two hits from consecutive chunks are two halves of one thought; returned
    separately they read as near-duplicates, because the chunker's overlap means
    they literally share paragraphs. Merged, they become one clean passage
    carrying the better score and the widest span.
    """
    if len(candidates) < 2:
        return candidates

    by_book: dict[int, list[_Candidate]] = {}
    for candidate in candidates:
        by_book.setdefault(candidate.chunk.book_id, []).append(candidate)

    merged: list[_Candidate] = []
    for group in by_book.values():
        group.sort(key=lambda c: c.chunk.sequence)
        run = [group[0]]
        for candidate in group[1:]:
            if candidate.chunk.sequence - run[-1].chunk.sequence <= _ADJACENT_GAP:
                run.append(candidate)
            else:
                merged.append(_fuse(run))
                run = [candidate]
        merged.append(_fuse(run))

    return merged


def _fuse(run: list[_Candidate]) -> _Candidate:
    """Collapse a run of adjacent chunks into one candidate."""
    if len(run) == 1:
        return run[0]

    chunks = [c.chunk for c in run]
    lead = max(run, key=lambda c: c.score).chunk
    starts = [c.location.page_start for c in chunks]
    ends = [c.location.page_end for c in chunks]

    fused = Chunk(
        id=lead.id,
        book_id=lead.book_id,
        text=join_without_overlap([c.text for c in chunks]),
        sequence=chunks[0].sequence,
        location=ChunkLocation(
            chapter_title=lead.location.chapter_title,
            chapter_n=lead.location.chapter_n,
            section_title=lead.location.section_title,
            section_n=lead.location.section_n,
            page_start=_min(starts),
            page_end=_max(ends or starts),
            char_start=_min([c.location.char_start for c in chunks]),
            char_end=_max([c.location.char_end for c in chunks]),
            paragraph_n=chunks[0].location.paragraph_n,
        ),
        token_count=sum(c.token_count for c in chunks),
        context=lead.context,
    )
    return _Candidate(fused, max(c.score for c in run), [c.id for c in chunks])


def _min(values: list[int | None]) -> int | None:
    present = [v for v in values if v is not None]
    return min(present) if present else None


def _max(values: list[int | None]) -> int | None:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def _citation(chunk: Chunk, book_title: str, author: str | None) -> Citation:
    location = chunk.location
    page_range = None
    if location.page_start is not None:
        end = location.page_end or location.page_start
        page_range = (
            f"p. {location.page_start}"
            if end == location.page_start
            else f"pp. {location.page_start}–{end}"
        )
    return Citation(
        book_title=book_title,
        author=author,
        chapter=location.chapter_title,
        section=location.section_title,
        page_range=page_range,
        paragraph_n=location.paragraph_n,
    )
