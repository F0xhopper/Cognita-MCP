"""LLM-powered corpus suggestion via Claude tool use."""

from cognita.core.config import settings
from cognita.infrastructure.anthropic_client import get_anthropic_client
from cognita.specialties.domain import SourceTier, SourceType, SuggestedSource

_TOOL: dict = {
    "name": "propose_corpus",
    "description": "Propose a tiered reading corpus for a scholarly specialty.",
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "description": "Suggested texts, 4–16 total across all tiers.",
                "minItems": 4,
                "maxItems": 16,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "author": {"type": "string"},
                        "tier": {
                            "type": "string",
                            "enum": ["primary", "commentary", "competing", "synthesis"],
                        },
                        "rationale": {
                            "type": "string",
                            "description": "1–2 sentences on why this text belongs in the corpus.",
                        },
                    },
                    "required": ["title", "author", "tier", "rationale"],
                },
            }
        },
        "required": ["items"],
    },
}

_SYSTEM = """You are a scholarly librarian with deep expertise across philosophy, theology, science, history, and literature.

When asked to build a corpus for a specialty, propose canonical texts in four tiers:
- primary: foundational texts by the tradition's central figures
- commentary: major interpretive works that shaped how primary texts are understood
- competing: significant rival schools or critics — essential for intellectual honesty
- synthesis: modern works that integrate or re-assess the tradition

Rules:
- Prefer texts in the public domain (pre-1928) where possible — they can be sourced automatically.
- Be exact: give the canonical title and full author name.
- Limit to 3–4 texts per tier, 12–16 total.
- Do not pad with minor works; every suggestion must be defensible."""


async def suggest_corpus(name: str, description: str | None) -> list[SuggestedSource]:
    client = get_anthropic_client()

    prompt = f"Build a scholarly corpus for the specialty: **{name}**"
    if description:
        prompt += f"\n\nContext: {description}"

    response = await client.messages.create(
        model=settings.CORPUS_MODEL,
        max_tokens=2048,
        system=_SYSTEM,
        tools=[_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": prompt}],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    items: list[dict] = tool_block.input["items"]

    return [
        SuggestedSource(
            title=item["title"],
            author=item["author"],
            tier=SourceTier(item["tier"]),
            rationale=item["rationale"],
            source_url=None,
            source_type=SourceType.USER_UPLOAD_REQUIRED,
            approved=item["tier"] == "primary",
        )
        for item in items
    ]
