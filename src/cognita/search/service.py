import asyncpg

from cognita.books.repository import BookRepository
from cognita.chunks.domain import Citation
from cognita.chunks.repository import ChunkRepository
from cognita.core.config import settings
from cognita.core.exceptions import NotFoundError
from cognita.infrastructure.embeddings import embed_text
from cognita.infrastructure.reranker import rerank
from cognita.search.domain import PassageContext, SearchResponse, SearchResult


class SearchService:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self._chunk_repo = ChunkRepository(pool)
        self._book_repo = BookRepository(pool)

    async def search(
        self,
        user_id: str,
        query: str,
        book_ids: list[int] | None = None,
        top_k: int = 10,
    ) -> SearchResponse:
        query_embedding = await embed_text(query)

        # When reranking is on, over-fetch a wider candidate pool from hybrid
        # search, then let the reranker pick and order the final top_k.
        rerank_on = settings.RERANK_ENABLED and bool(settings.ANTHROPIC_API_KEY)
        fetch_k = max(settings.RERANK_CANDIDATES, top_k) if rerank_on else top_k

        hits = await self._chunk_repo.hybrid_search(
            user_id=user_id,
            query_embedding=query_embedding,
            query_text=query,
            book_ids=book_ids,
            candidate_k=max(fetch_k * 4, 100),
            top_k=fetch_k,
        )
        book_cache: dict[int, str] = {}
        results: list[SearchResult] = []
        for chunk, score in hits:
            if chunk.book_id not in book_cache:
                book = await self._book_repo.get(chunk.book_id, user_id)
                book_cache[chunk.book_id] = book.metadata.title if book else "Unknown"
            citation = _build_citation(chunk, book_cache[chunk.book_id])
            results.append(SearchResult(chunk=chunk, score=score, citation=citation))

        if rerank_on and len(results) > 1:
            ranking = await rerank(query, [r.text for r in results], top_n=top_k)
            reranked: list[SearchResult] = []
            for idx, score in ranking:
                hit = results[idx]
                hit.score = round(float(score), 4)
                reranked.append(hit)
            results = reranked
        else:
            results = results[:top_k]

        return SearchResponse(query=query, results=results, total=len(results))

    async def get_passage_context(
        self,
        user_id: str,
        chunk_id: int,
        book_id: int,
        window: int = 2,
    ) -> PassageContext:
        neighbours = await self._chunk_repo.get_neighbours(chunk_id, book_id, window)
        if not neighbours:
            raise NotFoundError("Chunk", chunk_id)

        target = next((c for c in neighbours if c.id == chunk_id), None)
        if not target:
            raise NotFoundError("Chunk", chunk_id)

        book = await self._book_repo.get(book_id, user_id)
        title = book.metadata.title if book else "Unknown"
        citation = _build_citation(target, title)
        hit = SearchResult(chunk=target, score=1.0, citation=citation)

        before = [c for c in neighbours if c.sequence < target.sequence]
        after = [c for c in neighbours if c.sequence > target.sequence]
        return PassageContext(hit=hit, before=before, after=after)

    async def get_passage_by_location(
        self,
        user_id: str,
        book_id: int,
        chapter_n: int | None = None,
        section_n: int | None = None,
    ) -> list[SearchResult]:
        chunks = await self._chunk_repo.get_by_location(
            book_id=book_id,
            user_id=user_id,
            chapter_n=chapter_n,
            section_n=section_n,
        )
        book = await self._book_repo.get(book_id, user_id)
        title = book.metadata.title if book else "Unknown"
        return [
            SearchResult(chunk=c, score=1.0, citation=_build_citation(c, title))
            for c in chunks
        ]


def _build_citation(chunk, book_title: str) -> Citation:
    loc = chunk.location
    page_range = None
    if loc.page_start is not None:
        page_range = f"p. {loc.page_start}" if loc.page_start == loc.page_end else \
            f"pp. {loc.page_start}–{loc.page_end}"
    return Citation(
        book_title=book_title,
        author=None,
        chapter=loc.chapter_title,
        section=loc.section_title,
        page_range=page_range,
        paragraph_n=loc.paragraph_n,
    )
