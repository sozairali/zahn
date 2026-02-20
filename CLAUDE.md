# Zahn — Dental Lab Sentiment Analysis Worker

## Project Overview

Python worker that reads pending jobs from PostgreSQL, classifies customer messages as
`frustration | satisfaction | neutral` using a local Ollama LLM, and writes results back.

## Architecture

- **Integration**: Shared PostgreSQL polling (no message broker)
- **LLM**: Ollama with `llama3.2:3b` (local, on-premise)
- **Languages**: EN, FR, ES

## Running

```bash
# Install
pip install -e ".[dev]"

# Copy and fill env
cp .env.example .env

# Run worker
zahn-worker

# Tests
pytest tests/unit/          # no external deps
pytest tests/integration/   # requires Postgres + Ollama
```

## Key Files

- `src/zahn/config.py` — Pydantic BaseSettings; fails fast on missing vars
- `src/zahn/models.py` — Pydantic contracts (SentimentJob, LLMResponse, SentimentResult)
- `src/zahn/prompt.py` — Domain context builder and prompt engineering (build_prompt)
- `src/zahn/llm.py` — Ollama HTTP client (call_ollama, parse_llm_response)
- `src/zahn/db.py` — PostgreSQL claim/write/release (FOR UPDATE SKIP LOCKED)
- `src/zahn/analysis.py` — Orchestrates prompt→llm→validate per job
- `src/zahn/worker.py` — Polling loop + CLI entry point

## Sentiment Classification

The LLM holistically assesses each message and returns a label, a verbatim excerpt
that drives the classification, and a short reasoning in English.

## Error Handling

One `try/except` per job in `worker.py`. All other failures propagate and trigger `release_job`.
Max attempts before marking a job `failed`: configurable via `MAX_ATTEMPTS` env var (default 3).

## Database

Run the migration SQL against your Postgres instance.
Rails wraps it with `execute File.read(...)` in an ActiveRecord migration.
Note: `data/` and `migrations/` are excluded from version control.
