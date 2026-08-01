# Cognita

**An MCP server that lets your AI agent search your personal library.**

Point it at your books — PDFs, EPUBs, notes, articles — and Claude can search
them the way it searches the web: by meaning, with citations, quoting what your
sources actually say instead of what it half-remembers about them.

```
You    → What did Graeber actually argue about the origin of money?

Claude → [searches your library]

         Graeber rejects the barter-then-money story outright. He argues credit
         came first, and that the barter economy is "a myth" with no
         ethnographic support:

         > "The problem is that there's no evidence that it ever happened, and
         > an enormous amount of evidence suggesting that it did not."
         — David Graeber, Debt: The First 5000 Years › Chapter 2 › pp. 28–29
```

No new interface to learn, no separate app. Your library becomes something your
agent can consult mid-conversation.

---

## Contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Connecting it to Claude](#connecting-it-to-claude)
- [Filling your library](#filling-your-library)
- [The tools](#the-tools)
- [How search works](#how-search-works)
- [Hosting it](#hosting-it)
- [Configuration](#configuration)
- [Development](#development)
- [Upgrading from v1](#upgrading-from-v1)

---

## What it does

**Ingests** a book, splits it into passages along its real chapter and section
boundaries, and stores each one with an embedding in Postgres.

**Searches** those passages two ways at once — by meaning and by exact wording —
fuses the rankings, and optionally has a model rerank the result by how well
each passage actually answers the question.

**Cites** every passage with its author, chapter, section, and page, so a claim
can be traced back to the page it came from.

That's the whole thing. It's a good search tool over your own books, exposed
over MCP. It does not try to do your agent's thinking for it.

**Formats:** PDF, EPUB, TXT, Markdown, HTML. Scanned PDFs go through OCR if you
supply a Mistral key.

**Requirements:** Postgres with `pgvector`, and an OpenAI key for embeddings.
An Anthropic key is optional and meaningfully improves search — see
[How search works](#how-search-works).

---

## Quick start

```bash
git clone https://github.com/F0xhopper/cognita-mcp
cd cognita-mcp
pip install -e .

cp .env.example .env        # add your OPENAI_API_KEY
docker compose up db -d     # Postgres + pgvector on :5432

cognita                     # stdio — this is what Claude connects to
```

The schema is created on first run. There is no migration step.

Already have a Postgres? Point `DATABASE_URL` at it. It needs the `vector` and
`pg_trgm` extensions, which the server creates itself if it has permission.
Supabase, Neon, and Fly Postgres all work.

---

## Connecting it to Claude

### Claude Code

```bash
claude mcp add cognita --env DATABASE_URL=postgresql://postgres:postgres@localhost:5432/cognita \
                       --env OPENAI_API_KEY=sk-... \
                       -- cognita
```

### Claude Desktop

Edit `claude_desktop_config.json`:

**macOS** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "cognita": {
      "command": "cognita",
      "env": {
        "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/cognita",
        "OPENAI_API_KEY": "sk-...",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

Restart Claude Desktop. Ask it "what's in my library?" to confirm the
connection.

> If `cognita` is not on your PATH, use the absolute path from `which cognita`.

---

## Filling your library

Five ways in, all of them things you can just ask for in conversation.

**From your disk** — the fastest way to start:

> "Add everything in ~/Books to my library"

Reads the whole folder, pulls title and author out of each file's own metadata,
and skips anything already imported — so it's safe to re-run after you buy a few
more books.

**From a URL:**

> "Add https://example.com/paper.pdf to my library"

**By name** — for anything in the public domain:

> "Add Marcus Aurelius' Meditations to my library"

Searches Project Gutenberg, Standard Ebooks, Open Library, the Internet Archive
and Wikisource, preferring clean transcriptions over scans, and ingests what it
finds. For books still in copyright, supply the file yourself.

**Pasted text** — notes, an article, a transcript:

> "Save these meeting notes to my library under 'Q3 planning'"

**Individual files:**

> "Add ~/Downloads/debt-the-first-5000-years.epub"

Ingestion runs in the background — adding returns immediately, and a book
becomes searchable a minute or two later. Ask for `library_status` to see
progress and anything that failed.

---

## The tools

Fifteen tools, each doing one thing.

### Searching

| Tool | What it's for |
|---|---|
| `search_library` | The main one. Returns ranked passages with citations. Supports `"quoted phrases"` and `-exclusions`, and filters by book or author. |
| `expand_passage` | Widen a hit that got cut off mid-argument. |
| `read_chapter` | Read a chapter straight through, in order, rather than by relevance. |
| `read_section` | The same for one section. |

### Browsing

| Tool | What it's for |
|---|---|
| `list_books` | Everything in the library, with ingestion status. |
| `find_books` | Find a book by title, author or description — turns "the Graeber book" into an id. |
| `get_table_of_contents` | A book's structure, with the chapter numbers `read_chapter` takes. |
| `library_status` | How much is indexed, what's still processing, what failed and why. |

### Adding

| Tool | What it's for |
|---|---|
| `add_book_from_path` | A file on this machine. |
| `add_books_from_folder` | A whole directory, recursively. |
| `add_book_from_url` | Download and add. |
| `add_book_by_title` | Find a public-domain edition by name. |
| `add_text` | Pasted text. Markdown headings become sections. |

### Managing

| Tool | What it's for |
|---|---|
| `delete_book` | Remove a book and its passages. |
| `reingest_book` | Retry one that failed. |

---

## How search works

A query goes through four stages.

### 1. Two searches, not one

Every query runs as a **vector search** (cosine similarity over embeddings,
finds passages that mean the same thing in different words) and a **full-text
search** (Postgres `tsvector`, finds exact terminology, names, and quotes) at
the same time.

Neither alone is enough. Vector search misses a proper noun it has never seen;
keyword search misses the passage that makes your point without using your
words.

### 2. Fusing the rankings

The two result lists are merged with **Reciprocal Rank Fusion** — each arm
contributes `weight / (k + rank)` to a passage's score. Ranks are fused rather
than scores because cosine distance and `ts_rank_cd` are not on comparable
scales. A passage both arms like beats a passage either arm loves alone.

Tune with `RRF_SEMANTIC_WEIGHT` and `RRF_KEYWORD_WEIGHT`.

### 3. Reranking *(needs `ANTHROPIC_API_KEY`)*

Hybrid search has good recall and mediocre ordering — the passage that answers
the question is often ranked fourth. So the top candidates are handed to a fast
model that scores each one on how directly it answers *this* query, and the list
is reordered. Candidates are scored in concurrent batches to keep it quick.

This is the single biggest quality lever available. Without a key, the fusion
ordering stands and everything still works.

### 4. Shaping

- **Normalised scores** — raw fusion scores are tiny and unit-free (~0.03).
  They're rescaled so the best hit is `1.0` and everything else reads as a
  fraction of it.
- **Adjacent merging** — two hits from consecutive passages are two halves of
  one thought, and they literally share paragraphs because chunks overlap.
  They're merged into one clean passage with the duplication removed.
- **Per-book cap** *(optional)* — set `MAX_PER_BOOK` to stop one book crowding
  out the rest of the library.

### Contextual indexing *(needs `ANTHROPIC_API_KEY`)*

At ingestion time, each passage gets a one-sentence blurb saying where it sits
in the book, which is embedded and indexed alongside it. This is
[Anthropic's contextual retrieval technique](https://www.anthropic.com/news/contextual-retrieval),
and it exists because a passage lifted out of a book loses what it refers to:

> "He argued that this was the central error of the entire school."

That is unfindable on its own. Indexed together with *"This passage from Chapter
4 of Debt discusses Graeber's critique of Adam Smith's account of barter"*, it
is findable by anyone asking about Graeber and Smith. The blurb is only ever
used for retrieval — it is never quoted back to you.

It costs one cheap model call per passage at ingestion, once. Set
`CONTEXT_ENABLED=false` to skip it.

### Without an Anthropic key

Everything works. You get hybrid search with fused ranking, real citations, and
adjacent merging — you lose reranking and contextual indexing. The server tells
you which features are live in `library_status`.

---

## Hosting it

Running it locally over stdio is the simplest thing and needs no auth — the
process boundary *is* the security model. Host it when you want the same library
available from Claude on any device.

Hosted mode adds two requirements: a **bearer token**, without which anyone who
finds the URL owns your library, and an **allowed hostname**, because MCP's
DNS-rebinding protection rejects Host headers it doesn't recognise.

```bash
cognita --http
```

Serves the MCP endpoint at `/mcp` (Streamable HTTP) and a health check at
`/health`. Everything but `/health` requires `Authorization: Bearer <token>`.

### Deploying to Fly.io

```bash
fly launch --no-deploy --copy-config

# Postgres — Fly's own, or point DATABASE_URL at Supabase/Neon instead
fly postgres create --name cognita-db
fly postgres attach cognita-db

# pgvector needs enabling once
fly postgres connect -a cognita-db
  CREATE EXTENSION vector;
  CREATE EXTENSION pg_trgm;
  \q

fly secrets set \
  OPENAI_API_KEY=sk-... \
  ANTHROPIC_API_KEY=sk-ant-... \
  COGNITA_AUTH_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

fly deploy
```

Then set `COGNITA_ALLOWED_HOSTS` in `fly.toml` to your app's hostname and
redeploy. Check it:

```bash
curl https://your-app.fly.dev/health
# {"status":"ok","service":"cognita"}
```

Connect Claude to `https://your-app.fly.dev/mcp` with your token as the bearer
credential.

Two things worth knowing about the shipped `fly.toml`:

- `auto_stop_machines = false`. Ingestion runs in-process, so a machine that
  stops mid-import drops the rest of the queue.
- `COGNITA_ALLOW_LOCAL_FILES = "false"`. On a hosted server, "the disk" is the
  server's disk, not yours — so `add_book_from_path` and
  `add_books_from_folder` refuse, and tell the agent to use the URL, by-title,
  or paste-text tools instead. Flip it only if you genuinely want callers
  reaching that filesystem.

### Deploying on every push

`.github/workflows/ci.yml` lints, tests, and then deploys `main` to Fly. One
secret is needed:

```bash
fly tokens create deploy -a cognita-mcp
```

Add the output as `FLY_API_TOKEN` under **Settings → Secrets and variables →
Actions**. Pull requests run lint, tests and a Docker build; only `main`
deploys, and only if lint and tests pass. `workflow_dispatch` gives you a
manual redeploy button.

The workflow assumes the app already exists — run `fly launch --no-deploy
--copy-config` once by hand first. If you renamed the app, change `FLY_APP`
and the `environment.url` at the top of the workflow.

> **A deploy interrupts ingestion.** Machines are replaced on release, and the
> ingestion queue lives in the process — so a merge to `main` while a book is
> importing loses whatever was still queued. Those books stay `pending` until
> you re-ingest them. Check `library_status` before merging, or add a required
> reviewer to the `production` environment to gate releases.

### Anywhere else

The `Dockerfile` is a plain Python image with no Fly-specific anything:

```bash
docker build -t cognita .
docker run -p 8080:8080 --env-file .env cognita
```

---

## Configuration

Everything lives in `.env`. Only two variables are required.

### Required

| Variable | |
|---|---|
| `DATABASE_URL` | Postgres with `pgvector`. |
| `OPENAI_API_KEY` | Embeddings. |

### Optional — better search

| Variable | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Enables reranking and contextual indexing. |
| `RERANK_ENABLED` | `true` | Reorder results by relevance. |
| `RERANK_CANDIDATES` | `40` | How many candidates to consider. |
| `CONTEXT_ENABLED` | `true` | Situate each passage at ingestion. |
| `MISTRAL_API_KEY` | — | OCR for scanned PDFs. |

### Optional — hosting

| Variable | Default | |
|---|---|---|
| `COGNITA_AUTH_TOKEN` | — | **Required for `--http`.** Bearer token. |
| `COGNITA_ALLOWED_HOSTS` | localhost | Comma-separated hostnames to accept. |
| `COGNITA_ALLOW_LOCAL_FILES` | `false` | Permit local-disk tools over HTTP. |
| `MCP_HOST` / `MCP_PORT` | `127.0.0.1` / `8001` | Bind address. |

### Tuning

| Variable | Default | |
|---|---|---|
| `EMBED_MODEL` | `text-embedding-3-large` | |
| `EMBED_DIM` | `3072` | Passed to the API — lowering it really does shorten vectors. |
| `CHUNK_SIZE_CHARS` | `1500` | Bigger = more context per hit, fuzzier matching. |
| `CHUNK_OVERLAP_CHARS` | `200` | Keeps an idea findable across a boundary. |
| `RRF_SEMANTIC_WEIGHT` | `1.0` | Raise to favour meaning over wording. |
| `RRF_KEYWORD_WEIGHT` | `1.0` | Raise to favour exact terms. |
| `HNSW_EF_SEARCH` | `100` | Higher = better recall, slower. |
| `MERGE_ADJACENT` | `true` | Fold neighbouring hits into one passage. |
| `MAX_PER_BOOK` | `0` (off) | Cap results from any one book. |
| `INGEST_CONCURRENCY` | `2` | Books ingested at once. |
| `MAX_FILE_MB` | `100` | Size ceiling per file. |

> **On `EMBED_DIM`:** `pgvector`'s HNSW index only handles 2000 dimensions
> directly, and the default model produces 3072. Above the limit, the column is
> indexed and queried through a `halfvec` cast, which HNSW supports to 4000 —
> so vector search stays indexed rather than degrading to a sequential scan.
> Handled automatically; you only need to know if you're reading the SQL.

---

## Development

```bash
pip install -e ".[dev]"
pytest                       # 125 tests, no database or network needed
ruff check src/ tests/
```

Layout:

```
src/cognita/
  server.py         MCP tool definitions — the agent-facing surface
  transport.py      HTTP transport, bearer auth, health check
  schemas.py        Structured tool outputs
  books/            Library: repository, service, URL fetching, source lookup
  chunks/           Passage storage and hybrid search SQL
  ingestion/        Parsing → chunking → contextualizing → embedding → storing
  search/           Ranking, merging, citations
  infrastructure/   Postgres, OpenAI, Anthropic, Mistral clients
  core/             Config, logging, exceptions
```

Each domain follows the same shape: `domain.py` for dataclasses, `repository.py`
for SQL, `service.py` for orchestration.

---

## Upgrading from v1

v2 removes the research-planning loop (`plan_research`, `execute_research_plan`,
`assess_coverage`) and the specialties system, along with the REST API and its
multi-user auth. What's left is the search tool that was underneath them.

Your books and passages carry over. The server migrates its own tables on
startup — dropping the now-unused `user_id` column, adding `source` and
`context`. The only manual step is optional:

```bash
psql "$DATABASE_URL" -f migrations/upgrade_from_v1.sql   # drops the specialties tables
```

**Worth re-ingesting.** Two bugs in v1 meant every book was stored as a single
undifferentiated "Full Text" section with no page numbers — chapter titles,
`get_chapter`, and page citations never worked. Books ingested by v2 get real
structure. To fix the ones you already have:

> "Re-ingest every book in my library"

---

## Licence

MIT
