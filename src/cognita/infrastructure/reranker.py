"""Reranking — reorder retrieved candidates by how well they answer the query.

Hybrid search is strong on recall and weak on ordering: the passage that
actually answers the question is often ranked fourth. A reranker reads each
candidate against the query and scores it directly, which is a much better
judgement than either arm's ranking signal.

Claude does the scoring here rather than a dedicated cross-encoder — no extra
service to run, no second model to host. Candidates are scored in batches so
the prompt stays small and the batches run concurrently.

Everything is best-effort. Disabled, unconfigured, or failing, this returns
None and the caller keeps its existing order — search never breaks because
reranking is unavailable.
"""

import asyncio

from cognita.core.config import settings
from cognita.core.logging import get_logger
from cognita.infrastructure.anthropic_client import get_anthropic_client

logger = get_logger(__name__)

# Enough to judge relevance; short enough to keep a batch affordable.
_MAX_DOC_CHARS = 1200

_SYSTEM = (
    "You are a search reranker for a personal library. Score how directly each "
    "passage answers the query: 1.0 means it contains the answer, 0.5 means it is "
    "on-topic but does not answer, 0.0 means it is irrelevant. Judge the passage "
    "text, not the surrounding context line. Score every passage exactly once."
)

_TOOL = {
    "name": "rank_passages",
    "description": "Score how well each passage answers the query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "rankings": {
                "type": "array",
                "description": "One entry per passage, identified by its index.",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "The passage's index."},
                        "relevance": {
                            "type": "number",
                            "description": "0.0 (irrelevant) to 1.0 (directly answers).",
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
    documents: list[tuple[str, str]],
    top_n: int,
) -> list[tuple[int, float]] | None:
    """Score `documents` — (text, context) pairs — against `query`.

    Returns (original_index, score) pairs, best first, at most `top_n` long.
    Returns None when reranking is unavailable, meaning "keep your own order".
    """
    if not settings.rerank_enabled or len(documents) <= 1:
        return None

    size = max(1, settings.RERANK_BATCH_SIZE)
    batches = [
        list(range(start, min(start + size, len(documents))))
        for start in range(0, len(documents), size)
    ]

    results = await asyncio.gather(
        *(_score_batch(query, documents, batch) for batch in batches),
        return_exceptions=True,
    )

    scored: list[tuple[int, float]] = []
    failed = 0
    for batch, result in zip(batches, results, strict=True):
        if isinstance(result, dict):
            scored.extend(result.items())
        else:
            failed += 1
            logger.warning("Rerank batch failed: %s", result)
            # Keep the batch's own retrieval order by scoring it below anything
            # the model actually judged, rather than dropping it outright.
            scored.extend((index, 0.0) for index in batch)

    if failed == len(batches):
        return None

    # Ties keep their retrieval order, so an unscored passage never leapfrogs.
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:top_n]


async def _score_batch(
    query: str,
    documents: list[tuple[str, str]],
    indices: list[int],
) -> dict[int, float]:
    client = get_anthropic_client()

    blocks = []
    for index in indices:
        text, context = documents[index]
        header = f"[{index}]"
        if context:
            header += f"\n(context: {context.strip()[:300]})"
        blocks.append(f"{header}\n{text[:_MAX_DOC_CHARS]}")

    response = await client.messages.create(
        model=settings.RERANK_MODEL,
        max_tokens=1024,
        system=_SYSTEM,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "rank_passages"},
        messages=[
            {
                "role": "user",
                "content": f"Query: {query}\n\nPassages:\n\n" + "\n\n".join(blocks),
            }
        ],
    )

    block = next(b for b in response.content if getattr(b, "type", None) == "tool_use")
    allowed = set(indices)
    scores: dict[int, float] = {}

    for entry in block.input.get("rankings", []):
        index, relevance = entry.get("index"), entry.get("relevance")
        if index in allowed and index not in scores and isinstance(relevance, int | float):
            scores[index] = max(0.0, min(1.0, float(relevance)))

    # A passage the model declined to score still belongs in the output.
    for index in indices:
        scores.setdefault(index, 0.0)
    return scores
