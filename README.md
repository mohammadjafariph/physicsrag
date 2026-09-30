# PhysicsRAG

Self-hosted retrieval-augmented research system for physics papers, with a
built-in web UI and an autonomous research loop. Works with **any LLM API
provider** — Groq, OpenAI, Anthropic, OpenRouter, DeepSeek, Mistral, xAI,
Together, or fully local via Ollama / LM Studio / vLLM.

```
ROOT TOPIC ──► N research cycles
                 planner ─► arXiv search ─► download ─► chunk ─► embed
                                     │
                                     ▼
                 analyze findings/gaps ◄── hybrid retrieval + rerank
                                     │
                                     ▼
                 dedup ─► next topic ─► (loop)

On demand:  ask the library a question ─► cited answer in the web UI
```

## Quickstart

```bash
cd RAG
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt     # Windows
copy .env.example .env                            # then edit .env
.venv\Scripts\uvicorn src.api.server:app --reload
# open http://127.0.0.1:8000
```

Or with Docker:

```bash
cp .env.example .env   # set LLM_PROVIDER / LLM_MODEL / key
docker compose up --build
```

## Choosing a provider

Everything is driven by `.env` (or changed live in the web UI's Settings
page, which persists back to `.env`):

```ini
LLM_PROVIDER=groq            # any name from the table below
LLM_MODEL=openai/gpt-oss-120b
LLM_API_KEY=...              # not needed for local servers
LLM_BASE_URL=...             # optional; overrides the preset
```

| Provider    | Base URL (preset)              | Example model            |
|-------------|--------------------------------|--------------------------|
| groq        | https://api.groq.com/openai/v1 | openai/gpt-oss-120b      |
| openai      | https://api.openai.com/v1      | gpt-4o-mini              |
| anthropic   | https://api.anthropic.com      | claude-sonnet-4-5        |
| openrouter  | https://openrouter.ai/api/v1   | any OpenRouter model     |
| deepseek    | https://api.deepseek.com/v1    | deepseek-chat            |
| mistral     | https://api.mistral.ai/v1      | mistral-large-latest     |
| together    | https://api.together.xyz/v1    | meta-llama/...           |
| xai         | https://api.x.ai/v1            | grok-3                   |
| ollama      | http://localhost:11434/v1      | llama3.1:8b (local)      |
| lmstudio    | http://localhost:1234/v1       | any loaded model (local) |
| vllm        | http://localhost:8001/v1       | any served model (local) |
| custom      | (set LLM_BASE_URL)             | any OpenAI-compatible    |

Legacy compatibility: `GROQ_KEY` still works when `LLM_PROVIDER=groq`.

Per-stage overrides (`PLANNER_MODEL`, `ANALYZER_MODEL`, `DEDUP_MODEL`) let
you mix models, e.g. a cheap local model for planning and a strong model
for analysis. Embeddings and reranking always run locally
(sentence-transformers), so your paper library never leaves your machine.

## The web UI (http://127.0.0.1:8000)

- **Ask the library** — streaming RAG chat. Answers cite retrieved chunks
  as `[n]`; source cards link to the exact arXiv paper, section, and page.
  Math renders with KaTeX (vendored, works offline) for `$...$`,
  `$$...$$`, `\(...\)` and `\[...\]` notation.
- **Papers** — every downloaded paper with authors, categories, and chunk counts.
- **Research runs** — start an autonomous multi-cycle exploration of any
  topic; watch the live pipeline log and the growing topic tree.
- **Settings** — switch provider/model at runtime; the API key stays
  server-side.

## HTTP API

| Method | Path                     | Purpose                                   |
|--------|--------------------------|-------------------------------------------|
| GET    | `/api/health`            | status + library counts + provider        |
| GET    | `/api/papers`            | paper library                             |
| GET    | `/api/papers/{id}`       | one paper's metadata                      |
| GET    | `/api/topics`            | topics + cycle history                    |
| GET    | `/api/topics/tree`       | research trajectory (text + nested JSON)  |
| POST   | `/api/chat`              | RAG answer with citations (JSON)          |
| POST   | `/api/chat/stream`       | same, as SSE (`citations`→`token`→`done`) |
| POST   | `/api/research/runs`     | start a background research run           |
| GET    | `/api/research/runs[/{id}]` | run status, live log, tree             |
| GET/POST | `/api/settings`        | inspect / switch provider at runtime      |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How does entanglement entropy scale at the critical point?"}'
```

## CLI (the original pipeline)

```bash
python main.py                     # interactive
python main.py 3 "Open quantum systems"
```

## Configuration reference

| Variable         | Default                          | Meaning                       |
|------------------|----------------------------------|-------------------------------|
| LLM_PROVIDER     | groq                             | provider preset name          |
| LLM_MODEL        | openai/gpt-oss-120b              | default model for all stages  |
| LLM_API_KEY      | –                                | provider API key              |
| LLM_BASE_URL     | preset                           | override endpoint             |
| PLANNER/ANALYZER/DEDUP_MODEL | (LLM_MODEL)          | per-stage overrides           |
| EMBEDDING_MODEL  | all-MiniLM-L6-v2                 | local sentence-transformer    |
| RERANKER_MODEL   | cross-encoder/ms-marco-MiniLM-L-6-v2 | local cross-encoder       |
| RETRIEVAL_K      | 8                                | evidence chunks per question  |
| DATA_DIR         | ./data                           | papers, vectors, SQLite       |

## Reliability design

- **Provenance end-to-end**: chunks keep paper/section/page; findings cite
  chunk ids that are validated against the evidence actually retrieved;
  paper ids are derived from chunk-id prefixes, never trusted from the LLM.
- **Anti-loop memory**: candidate topics are rejected by exact/acronym/
  substring prefilter, embedding similarity, and an LLM judge.
- **Citation safety in chat**: the UI only maps `[n]` citations to chunks
  that were really retrieved; the answer is instructed to refuse when
  evidence is insufficient.
- **arXiv discipline**: TLS-fingerprint-compliant HTTP client, rate-limit
  delays, progressive query relaxation, PDF magic-byte validation.

## Tests

```bash
.venv\Scripts\python -m pytest
```

## Layout

```
config.py            settings (pydantic-settings) + provider presets
main.py              CLI entry point
src/
  llm/               provider-agnostic LLM layer (factory + adapters)
  topic/             planner, deduplicator
  search/            arXiv search, paper ranking
  documents/         downloader, PDF loader, section-aware chunker
  embeddings/        local embedder
  database/          Chroma vector store
  retrieval/         hybrid retriever (BM25 + vectors + RRF + rerank)
  analysis/          evidence analyzer (findings/gaps/candidates)
  controller/        research-cycle controller + topic tree
  memory/            SQLite research memory
  service/           chat service, background run manager
  api/               FastAPI app + routers
web/                 self-hosted single-page UI (no build step)
tests/               pytest suite
```
