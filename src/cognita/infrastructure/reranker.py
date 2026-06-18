"""Reranking — reorder hybrid-search candidates by relevance to the query.

Hybrid search (semantic + keyword fused with RRF) is good at *recall* but weak at
ordering: the most relevant passage is often not ranked first. A reranker scores
each candidate against the query and reorders them, so the top-k handed to the
caller are the ones that actually answer the question.

This implementation uses Claude as a listwise reranker (no extra API key beyond the
one already required). It is best-effort: when disabled, unconfigured, or on any
error it returns the candidates in their original RRF order, so search never breaks.
"""

from cognita.core.config import settings
from cognita.core.logging import get_logger
from cognita.infrastructure.anthropic_client import get_anthropic_client

logger = get_logger(__name__)

_MAX_DOC_CHARS = 1500  # cap each passage shown to the reranker to control token cost

_TOOL = {
    "name": "rank_passages",
    "description": "Score how well each passage answers the query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "rankings": {
                "type": "array",
                "description": "One entry per passage, by its index.",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "The passage's index."},
                        "relevance": {
                            "type": "number",
                            "description": "0.0 (irrelevant) to 1.0 (directly answers the query).",
                        },
                    },
                    "required": ["index", "relevance"],
                },
            }
        },
        "required": ["rankings"],
    },
}


async def rerank(
    query: str,
    documents: list[str],
    top_n: int,
) -> list[tuple[int, float]]:
    """Rank `documents` against `query`. Returns (original_index, score) pairs,
    best first, length ≤ top_n. Falls back to original order on any problem."""
    n = len(documents)
    identity = [(i, 0.0) for i in range(min(n, top_n))]
    if not settings.RERANK_ENABLED or not settings.ANTHROPIC_API_KEY or n <= 1:
        return identity

    try:
        client = get_anthropic_client()
        passages = "\n\n".join(f"[{i}]\n{documents[i][:_MAX_DOC_CHARS]}" for i in range(n))
        resp = await client.messages.create(
            model=settings.RERANK_MODEL,
            max_tokens=2048,
            system=(
                "You are a search reranker. Score how directly each passage answers "
                "the query. Score every passage exactly once, by its index."
            ),
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "rank_passages"},
            messages=[{
                "role": "user",
                "content": f"Query: {query}\n\nPassages:\n\n{passages}",
            }],
        )
        block = next(b for b in resp.content if getattr(b, "type", None) == "tool_use")
        rankings = block.input.get("rankings", [])

        seen: set[int] = set()
        scored: list[tuple[int, float]] = []
        for r in rankings:
            idx, score = r.get("index"), r.get("relevance")
            if isinstance(idx, int) and 0 <= idx < n and idx not in seen \
                    and isinstance(score, (int, float)):
                seen.add(idx)
                scored.append((idx, float(score)))

        if not scored:
            return identity

        scored.sort(key=lambda x: x[1], reverse=True)
        # Append any passages the model skipped, preserving their RRF order.
        scored.extend((i, 0.0) for i in range(n) if i not in seen)
        return scored[:top_n]

    except Exception as exc:
        logger.warning("Rerank failed, falling back to RRF order: %s", exc)
        return identity
