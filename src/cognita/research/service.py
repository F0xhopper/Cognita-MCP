"""Deep research agent — plans queries, retrieves, gap-checks, and synthesizes.

This is the "agent-as-tool" core: a single deep_research() call runs a small
research loop over the user's library (optionally scoped to a specialty) and
returns a synthesized answer in which every claim is grounded in a numbered,
citable passage.
"""

import asyncio
from collections.abc import Awaitable, Callable

import asyncpg

from cognita.core.config import settings
from cognita.core.logging import get_logger
from cognita.infrastructure.mistral import chat_json, chat_text
from cognita.research.domain import ResearchFinding, ResearchReport
from cognita.search.domain import SearchResult
from cognita.search.service import SearchService
from cognita.specialties.service import SpecialtyService

logger = get_logger(__name__)

_DEFAULT_PERSONA = (
    "You are a meticulous research assistant answering strictly from the "
    "user's personal library."
)

_PLAN_PROMPT = """\
You are planning library research for the question below.
Break it into at most {max_queries} focused search queries that together cover the question.
Queries run against semantic search over book passages, so phrase them as the text
of a relevant passage would read, not as questions.

Question: {question}

Respond with JSON only: {{"queries": ["...", "..."]}}"""

_GAP_PROMPT = """\
You are mid-way through library research on the question below.
Review what has been found so far and identify what is still missing.

Question: {question}

Passages found so far (citations only):
{found}

If important aspects of the question are not yet covered, propose at most {max_queries} \
follow-up search queries targeting the gaps. If coverage is sufficient, return an empty list.

Respond with JSON only: {{"queries": ["..."]}}"""

_SYNTHESIS_PROMPT = """\
Answer the research question using ONLY the numbered passages below.
Cite every claim with inline markers like [1] or [2][5] referring to passage numbers.
If the passages do not fully answer the question, say so explicitly — never invent
content that is not in the passages.

Question: {question}

Passages:
{passages}"""


class ResearchService:
    def __init__(
        self,
        pool: asyncpg.Pool,
        llm_json: Callable[..., Awaitable[dict]] = chat_json,
        llm_text: Callable[..., Awaitable[str]] = chat_text,
    ) -> None:
        self._search = SearchService(pool)
        self._specialties = SpecialtyService(pool)
        self._llm_json = llm_json
        self._llm_text = llm_text

    async def deep_research(
        self,
        user_id: str,
        question: str,
        specialty_id: int | None = None,
        book_ids: list[int] | None = None,
        depth: int = 2,
    ) -> ResearchReport:
        """Run the research loop: plan → retrieve → gap-check → synthesize.

        depth 1 = single retrieval pass; depth 2-3 add a gap-check round and
        widen the query plan.
        """
        depth = max(1, min(depth, 3))
        persona, scoped_ids = await self._specialties.resolve_scope(
            user_id, specialty_id, book_ids
        )
        specialty_name = None
        if specialty_id is not None:
            specialty_name = (await self._specialties.get(user_id, specialty_id)).name
        persona = persona or _DEFAULT_PERSONA

        queries = await self._plan_queries(question, persona, depth)
        findings = await self._retrieve(user_id, queries, scoped_ids)

        if depth >= 2 and findings:
            follow_ups = await self._gap_check(question, persona, findings)
            if follow_ups:
                queries.extend(follow_ups)
                more = await self._retrieve(user_id, follow_ups, scoped_ids)
                findings = _merge_findings(findings, more)

        numbered = [
            ResearchFinding(n=i + 1, result=r)
            for i, r in enumerate(findings[: settings.RESEARCH_MAX_PASSAGES])
        ]
        answer = await self._synthesize(question, persona, numbered)

        return ResearchReport(
            question=question,
            answer=answer,
            citations=[f.citation for f in numbered],
            sub_queries=queries,
            specialty_name=specialty_name,
            findings=numbered,
        )

    async def _plan_queries(self, question: str, persona: str, depth: int) -> list[str]:
        max_queries = 2 * depth
        prompt = _PLAN_PROMPT.format(question=question, max_queries=max_queries)
        try:
            data = await self._llm_json(prompt, system=persona)
            planned = _clean_queries(data.get("queries", []))
        except Exception as exc:
            logger.warning("Query planning failed, falling back to raw question: %s", exc)
            planned = []
        queries = [question] + [q for q in planned if q != question]
        return queries[: max_queries + 1]

    async def _gap_check(
        self, question: str, persona: str, findings: list[SearchResult]
    ) -> list[str]:
        found = "\n".join(f"- {r.citation.to_string()}" for r in findings)
        prompt = _GAP_PROMPT.format(question=question, found=found, max_queries=3)
        try:
            data = await self._llm_json(prompt, system=persona)
            return _clean_queries(data.get("queries", []))[:3]
        except Exception as exc:
            logger.warning("Gap check failed, skipping follow-up round: %s", exc)
            return []

    async def _retrieve(
        self,
        user_id: str,
        queries: list[str],
        book_ids: list[int] | None,
    ) -> list[SearchResult]:
        responses = await asyncio.gather(
            *(
                self._search.search(
                    user_id=user_id,
                    query=q,
                    book_ids=book_ids,
                    top_k=settings.RESEARCH_TOP_K_PER_QUERY,
                )
                for q in queries
            ),
            return_exceptions=True,
        )
        results: list[SearchResult] = []
        for q, resp in zip(queries, responses, strict=True):
            if isinstance(resp, BaseException):
                logger.warning("Search failed for sub-query %r: %s", q, resp)
                continue
            results.extend(resp.results)
        return _merge_findings([], results)

    async def _synthesize(
        self, question: str, persona: str, findings: list[ResearchFinding]
    ) -> str:
        if not findings:
            return (
                "No relevant passages were found in the library for this question. "
                "Try rephrasing it, widening the scope, or adding books on the topic."
            )
        passages = "\n\n".join(
            f"[{f.n}] ({f.citation})\n{f.result.text}" for f in findings
        )
        prompt = _SYNTHESIS_PROMPT.format(question=question, passages=passages)
        return await self._llm_text(prompt, system=persona)


def _clean_queries(raw: list) -> list[str]:
    return [q.strip() for q in raw if isinstance(q, str) and q.strip()]


def _merge_findings(
    existing: list[SearchResult], new: list[SearchResult]
) -> list[SearchResult]:
    """Dedupe by chunk_id (keeping the best score) and sort by score descending."""
    by_chunk: dict[int, SearchResult] = {}
    for r in existing + new:
        current = by_chunk.get(r.chunk.id)
        if current is None or r.score > current.score:
            by_chunk[r.chunk.id] = r
    return sorted(by_chunk.values(), key=lambda r: r.score, reverse=True)
