"""Coverage assessment — decides when retrieval is sufficient to answer a question.

Calls Claude with the question and retrieved passages. Claude scores coverage and
returns gaps + suggested follow-up queries, giving the agent a principled stopping
condition rather than an open-ended loop.

Best-effort: on any error, returns ``satisfied=False`` with an empty suggestions
list so search never silently succeeds on a broken signal.
"""

from cognita.core.config import settings
from cognita.core.logging import get_logger
from cognita.infrastructure.anthropic_client import get_anthropic_client

logger = get_logger(__name__)

# Cap to keep the prompt within a reasonable token budget.
_MAX_PASSAGES = 15
_MAX_PASSAGE_CHARS = 800

_TOOL: dict = {
    "name": "coverage_assessment",
    "description": "Assess whether the retrieved passages adequately answer the question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "satisfied": {
                "type": "boolean",
                "description": (
                    "True if the passages collectively provide enough evidence to answer "
                    "the question with adequate depth and accuracy."
                ),
            },
            "gaps": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Distinct aspects of the question not covered or only partially "
                    "addressed by the retrieved passages."
                ),
            },
            "suggested_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Retrieval-phrased follow-up queries targeting the gaps — phrased as "
                    "text a relevant passage would contain, not as questions. "
                    "Empty list when satisfied=true."
                ),
            },
        },
        "required": ["satisfied", "gaps", "suggested_queries"],
    },
}

_SYSTEM = (
    "You are a retrieval evaluator for a research assistant. Given a research question "
    "and passages retrieved from a book library, decide whether the passages are sufficient "
    "to answer the question fully.\n\n"
    "Rules:\n"
    "- Mark satisfied=true only when the passages provide direct, substantive evidence that "
    "answers the question — not merely tangentially related content.\n"
    "- List every distinct aspect of the question that is missing or only partially covered.\n"
    "- Write suggested_queries as text a relevant passage would contain — declarative "
    "phrases that target the specific gaps.\n"
    "- If satisfied=true, gaps and suggested_queries must be empty lists."
)


async def assess_coverage(
    question: str,
    passages: list[dict],
) -> dict:
    """Assess whether *passages* adequately answer *question*.

    *passages* is a list of ``{"text": str, "citation": str}``.
    Returns the raw tool-call input dict; the caller maps it to a ``CoverageAssessment``.
    Falls back to ``satisfied=False`` with empty suggestions on any error.
    """
    if not settings.ANTHROPIC_API_KEY:
        return _fallback()

    client = get_anthropic_client()

    capped = passages[:_MAX_PASSAGES]
    passage_block = "\n\n".join(
        f"[{i}] {p['citation']}\n{p['text'][:_MAX_PASSAGE_CHARS]}"
        for i, p in enumerate(capped)
    )

    try:
        resp = await client.messages.create(
            model=settings.COVERAGE_MODEL,
            max_tokens=512,
            system=_SYSTEM,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "coverage_assessment"},
            messages=[{
                "role": "user",
                "content": f"Question: {question}\n\nRetrieved passages:\n\n{passage_block}",
            }],
        )
        block = next(b for b in resp.content if getattr(b, "type", None) == "tool_use")
        return dict(block.input)
    except Exception as exc:
        logger.warning("assess_coverage failed, returning unsatisfied: %s", exc)
        return _fallback()


def _fallback() -> dict:
    return {
        "satisfied": False,
        "gaps": ["Coverage assessment unavailable"],
        "suggested_queries": [],
    }
