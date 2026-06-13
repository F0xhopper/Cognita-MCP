"""Specialist sub-agent loop — runs a Claude agent with scoped retrieval tools.

Each call to run_specialist_agent() spins up a Claude agent that autonomously
decides how many searches to run, when to expand passages, and when it has
gathered sufficient evidence. It returns a ResearchReport with cited prose.
"""

from cognita.core.config import settings
from cognita.core.logging import get_logger
from cognita.infrastructure.anthropic_client import get_anthropic_client
from cognita.research.domain import ResearchFinding, ResearchReport
from cognita.search.domain import SearchResult
from cognita.search.service import SearchService

logger = get_logger(__name__)

_DEFAULT_PERSONA = (
    "You are a meticulous research assistant answering strictly from the "
    "user's personal library."
)

_RESEARCH_INSTRUCTIONS = """
You have access to the user's personal book library through retrieval tools.

To answer the question:
1. Call semantic_search with focused queries phrased as text a relevant passage would contain, not as questions. Run multiple searches to cover different angles of the question.
2. Call get_passage_context to expand promising hits and get surrounding paragraphs for fuller context.
3. Use get_chapter or get_section to read specific structural locations when you know where to look (e.g. after consulting a table of contents).
4. When you have sufficient evidence, write your final answer using ONLY retrieved passages.
5. Cite every claim with inline markers like [1] or [2][5] — the number refers to the passage number shown in tool results.
6. If the library does not contain enough information to fully answer the question, say so explicitly rather than speculating.
"""

_TOOL_DEFS: list[dict] = [
    {
        "name": "semantic_search",
        "description": (
            "Search the library using semantic + keyword hybrid search. Returns numbered passages "
            "with citations. Phrase queries as text a relevant passage would contain, not as questions. "
            "Run multiple queries to cover different angles."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 8)",
                    "default": 8,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_passage_context",
        "description": (
            "Expand a passage to include surrounding paragraphs for fuller context. "
            "Use after semantic_search when a hit looks relevant but needs more surrounding text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "integer", "description": "ID of the passage to expand"},
                "book_id": {"type": "integer", "description": "ID of the book containing the passage"},
                "window": {
                    "type": "integer",
                    "description": "Number of surrounding chunks on each side (default 2)",
                    "default": 2,
                },
            },
            "required": ["chunk_id", "book_id"],
        },
    },
    {
        "name": "get_chapter",
        "description": (
            "Retrieve all passages from a specific chapter by its sequential number. "
            "Use when you know the chapter number (e.g. from a table of contents)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "book_id": {"type": "integer"},
                "chapter_n": {"type": "integer", "description": "Chapter number (1-based)"},
            },
            "required": ["book_id", "chapter_n"],
        },
    },
    {
        "name": "get_section",
        "description": "Retrieve passages from a specific section within a chapter.",
        "input_schema": {
            "type": "object",
            "properties": {
                "book_id": {"type": "integer"},
                "chapter_n": {"type": "integer"},
                "section_n": {"type": "integer"},
            },
            "required": ["book_id", "chapter_n", "section_n"],
        },
    },
]


def _build_system(persona: str) -> str:
    return persona.rstrip() + "\n\n" + _RESEARCH_INSTRUCTIONS.strip()


def _format_results(
    results: list[SearchResult],
    existing: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Number new results sequentially and format them for the agent.

    Passages already in existing are noted by their prior number.
    Returns (text_for_agent, list_of_new_unique_results).
    """
    existing_by_id: dict[int, int] = {r.chunk.id: (i + 1) for i, r in enumerate(existing)}
    next_n = len(existing) + 1
    lines: list[str] = []
    new_results: list[SearchResult] = []

    for r in results:
        if r.chunk.id in existing_by_id:
            lines.append(f"(already retrieved as [{existing_by_id[r.chunk.id]}]) {r.citation.to_string()}")
        else:
            n = next_n
            next_n += 1
            existing_by_id[r.chunk.id] = n
            new_results.append(r)
            lines.append(f"[{n}] {r.citation.to_string()}\n{r.chunk.text}")

    if not lines:
        return "No passages found for this query.", []

    return "\n\n".join(lines), new_results


async def _dispatch(
    block,
    user_id: str,
    book_ids: list[int] | None,
    search_svc: SearchService,
    findings: list[SearchResult],
) -> tuple[str, list[SearchResult]]:
    """Execute a tool call. Returns (text_for_agent, new_unique_results)."""
    name = block.name
    inp = block.input

    try:
        if name == "semantic_search":
            resp = await search_svc.search(
                user_id=user_id,
                query=inp["query"],
                book_ids=book_ids,
                top_k=int(inp.get("top_k", settings.RESEARCH_TOP_K_PER_QUERY)),
            )
            return _format_results(resp.results, findings)

        if name == "get_passage_context":
            ctx = await search_svc.get_passage_context(
                user_id=user_id,
                chunk_id=int(inp["chunk_id"]),
                book_id=int(inp["book_id"]),
                window=int(inp.get("window", 2)),
            )
            hit_text, new = _format_results([ctx.hit], findings)
            full = ctx.full_text()
            return f"{hit_text}\n\nExpanded context (including surrounding passages):\n{full}", new

        if name == "get_chapter":
            results = await search_svc.get_passage_by_location(
                user_id=user_id,
                book_id=int(inp["book_id"]),
                chapter_n=int(inp["chapter_n"]),
            )
            return _format_results(results, findings)

        if name == "get_section":
            results = await search_svc.get_passage_by_location(
                user_id=user_id,
                book_id=int(inp["book_id"]),
                chapter_n=int(inp["chapter_n"]),
                section_n=int(inp["section_n"]),
            )
            return _format_results(results, findings)

    except Exception as exc:
        logger.warning("Specialist tool %r failed: %s", name, exc)
        return f"Tool {name!r} failed: {exc}", []

    return f"Unknown tool: {name!r}", []


async def run_specialist_agent(
    question: str,
    persona: str,
    book_ids: list[int] | None,
    user_id: str,
    specialty_name: str | None,
    search_svc: SearchService,
) -> ResearchReport:
    """Run a Claude specialist agent that autonomously retrieves and synthesizes an answer.

    The agent receives the specialty persona as its system prompt and has access to
    scoped retrieval tools. It decides how many searches to run and when it has
    enough evidence, then returns a cited prose answer.
    """
    client = get_anthropic_client()
    messages: list[dict] = [{"role": "user", "content": question}]
    findings: list[SearchResult] = []
    sub_queries: list[str] = []

    logger.info("Specialist agent starting for question: %r (specialty=%r)", question, specialty_name)

    while True:
        resp = await client.messages.create(
            model=settings.SPECIALIST_MODEL,
            max_tokens=settings.SPECIALIST_MAX_TOKENS,
            system=_build_system(persona),
            tools=_TOOL_DEFS,  # type: ignore[arg-type]
            messages=messages,
        )

        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            answer_parts = [b.text for b in resp.content if hasattr(b, "text")]
            answer = "\n\n".join(answer_parts) if answer_parts else (
                "The specialist agent did not produce a text response."
            )
            break

        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue

            logger.debug("Specialist calling tool %r with %s", block.name, block.input)
            result_text, new_results = await _dispatch(
                block, user_id, book_ids, search_svc, findings
            )
            findings = findings + new_results

            if block.name == "semantic_search":
                sub_queries.append(str(block.input.get("query", "")))

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text,
            })

        messages.append({"role": "user", "content": tool_results})

    numbered = [
        ResearchFinding(n=i + 1, result=r)
        for i, r in enumerate(findings[: settings.RESEARCH_MAX_PASSAGES])
    ]

    logger.info(
        "Specialist agent done: %d passages retrieved, %d sub-queries",
        len(numbered),
        len(sub_queries),
    )

    return ResearchReport(
        question=question,
        answer=answer,
        citations=[f.citation for f in numbered],
        sub_queries=sub_queries,
        specialty_name=specialty_name,
        findings=numbered,
    )
