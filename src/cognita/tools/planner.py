"""Query planner — converts a research question into a structured retrieval plan.

Calls Claude with the available specialties and forces a `create_research_plan`
tool call, returning a plan the caller can hand directly to execute_research_plan.

Making planning explicit:
- prevents redundant semantic_search calls by committing to subqueries upfront
- enables server-side parallelisation of all subqueries in one shot
- decouples "thinking about what to search" from "actually searching"

Falls back gracefully on any error: returns a minimal plan with the original
question as the only subquery and no specialty restriction.
"""

from cognita.core.config import settings
from cognita.core.logging import get_logger
from cognita.infrastructure.anthropic_client import get_anthropic_client

logger = get_logger(__name__)

_TOOL: dict = {
    "name": "create_research_plan",
    "description": "Produce a structured retrieval plan for a research question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "specialty_id": {
                "type": ["integer", "null"],
                "description": (
                    "ID of the best-matching specialty, or null to search the whole library."
                ),
            },
            "intent": {
                "type": "string",
                "enum": ["definition", "comparison", "causal", "biographical", "thematic", "overview"],
                "description": "The primary intent of the question.",
            },
            "subqueries": {
                "type": "array",
                "description": (
                    "2–5 retrieval-phrased queries. Each should be phrased as text a relevant "
                    "passage would contain — declarative phrases, not questions."
                ),
                "minItems": 2,
                "maxItems": 5,
                "items": {"type": "string"},
            },
            "depth": {
                "type": "string",
                "enum": ["shallow", "medium", "deep"],
                "description": (
                    "shallow: top 5 hits. medium: top 10 with context expansion. "
                    "deep: top 15 with chapter scans."
                ),
            },
            "needs_chapter_scan": {
                "type": "boolean",
                "description": (
                    "True when a structural read (get_chapter / get_section) would likely be "
                    "needed — e.g. 'summarise chapter 3' or 'what does book II argue'."
                ),
            },
        },
        "required": ["specialty_id", "intent", "subqueries", "depth", "needs_chapter_scan"],
    },
}

_SYSTEM = (
    "You are a research planner for a personal book library. Given a question and a list of "
    "available specialties (named expert scopes over subsets of the library), produce a "
    "structured retrieval plan.\n\n"
    "Rules:\n"
    "- Pick the specialty whose books are most likely to contain the answer. Use null if "
    "no specialty fits or the question spans multiple domains.\n"
    "- Write subqueries as text a relevant passage would contain — declarative phrases, "
    "not questions. Vary them to cover different angles: definitions, causes, examples, "
    "contrasts.\n"
    "- Use 'shallow' for simple factual lookups (one clear answer expected), "
    "'medium' for explanatory or interpretive questions, "
    "'deep' for comparative, comprehensive, or cross-book analyses.\n"
    "- Set needs_chapter_scan=true only when the question requires reading a full structural "
    "unit, not just scattered passages."
)


async def plan_research(
    question: str,
    specialties: list[dict],
) -> dict:
    """Produce a structured research plan for *question*.

    *specialties* is a list of ``{"id": int, "name": str, "description": str | None}``.
    Returns the raw tool-call input dict; the caller maps it to a ``ResearchPlan``.
    Falls back to a minimal single-query plan on any error.
    """
    if not settings.ANTHROPIC_API_KEY:
        return _fallback(question)

    client = get_anthropic_client()

    if specialties:
        spec_lines = "\n".join(
            f"  id={s['id']}: {s['name']}"
            + (f" — {s['description']}" if s.get("description") else "")
            for s in specialties
        )
        spec_block = f"Available specialties:\n{spec_lines}"
    else:
        spec_block = "Available specialties: (none — will search the whole library)"

    user_content = f"Question: {question}\n\n{spec_block}"

    try:
        resp = await client.messages.create(
            model=settings.PLANNER_MODEL,
            max_tokens=512,
            system=_SYSTEM,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "create_research_plan"},
            messages=[{"role": "user", "content": user_content}],
        )
        block = next(b for b in resp.content if getattr(b, "type", None) == "tool_use")
        return dict(block.input)
    except Exception as exc:
        logger.warning("plan_research failed, using fallback plan: %s", exc)
        return _fallback(question)


def _fallback(question: str) -> dict:
    return {
        "specialty_id": None,
        "intent": "overview",
        "subqueries": [question],
        "depth": "medium",
        "needs_chapter_scan": False,
    }
