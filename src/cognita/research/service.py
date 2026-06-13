"""Research service — resolves specialty scope and delegates to the specialist agent."""

import asyncpg

from cognita.research.agent import _DEFAULT_PERSONA, run_specialist_agent
from cognita.research.domain import ResearchReport
from cognita.search.service import SearchService
from cognita.specialties.service import SpecialtyService


class ResearchService:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self._specialties = SpecialtyService(pool)

    async def consult(
        self,
        user_id: str,
        question: str,
        specialty_id: int | None = None,
        book_ids: list[int] | None = None,
    ) -> ResearchReport:
        """Resolve specialty scope and run the specialist agent."""
        persona, scoped_ids = await self._specialties.resolve_scope(
            user_id, specialty_id, book_ids
        )
        specialty_name: str | None = None
        if specialty_id is not None:
            specialty_name = (await self._specialties.get(user_id, specialty_id)).name

        return await run_specialist_agent(
            question=question,
            persona=persona or _DEFAULT_PERSONA,
            book_ids=scoped_ids,
            user_id=user_id,
            specialty_name=specialty_name,
            search_svc=SearchService(self._pool),
        )
