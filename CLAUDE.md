# Zahn — Dental Lab Sentiment Analysis Worker

## Project Overview

Python worker that reads pending jobs from PostgreSQL, classifies customer messages
using two independent binary classifiers (frustration yes/no, satisfaction yes/no)
via a local Ollama LLM, and writes results back.

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
- `src/zahn/models.py` — Pydantic contracts (SentimentJob, BinaryLLMResponse, SentimentResult)
- `src/zahn/prompt.py` — Prompt engineering; separate frustration/satisfaction prompt builders (build_frustration_prompt, build_satisfaction_prompt)
- `src/zahn/llm.py` — Ollama HTTP client (call_ollama, parse_binary_response, validate_excerpt)
- `src/zahn/db.py` — PostgreSQL claim/write/release (FOR UPDATE SKIP LOCKED)
- `src/zahn/analysis.py` — Orchestrates two binary classifier calls per job (run_classifier)
- `src/zahn/worker.py` — Polling loop + CLI entry point

## Sentiment Classification

Two independent binary classifiers run per message:

- **Frustration** (`build_frustration_prompt`): label `yes | no`
- **Satisfaction** (`build_satisfaction_prompt`): label `yes | no`

A message can be both frustrated and satisfied simultaneously.

Each classifier returns:
- **label**: `yes | no`
- **detected_language**: `en | fr | es` (required, validated)
- **excerpt**: verbatim substring from the message that drives the label (required when `yes`, may be empty when `no`)
- **reasoning**: 1-2 sentences in English explaining the label

Dental lab domain knowledge (case types, quality signals, multilingual terms) is
hardcoded in the prompt templates in `prompt.py`.

## Error Handling

Error handling uses two layers, each with a clear purpose:

- **`worker.py`** — single `try/except` per job. On failure, calls `release_job` which
  re-queues the job or marks it `failed` after `MAX_ATTEMPTS` (env var, default 3).
- **`run_classifier`** — retries LLM parse/validation failures (up to 3 attempts per
  classifier call). Only catches expected retryable errors: `JSONDecodeError`,
  `ValidationError`, `HTTPStatusError`, `ExcerptValidationError`. Programming bugs
  propagate immediately to `worker.py`.

Do not add `try/except` blocks elsewhere. Keep error handling graceful and meaningful —
catch only what you can act on (retry or record), and let everything else propagate.

## Database

The schema is documented in the README. Run it against your Postgres instance before
starting the worker. Rails wraps it with `execute File.read(...)` in an ActiveRecord migration.
