# Zahn — Dental Lab Sentiment Analysis Worker

LLM-powered Python worker that classifies dental lab customer messages as
`frustration`, `satisfaction`, or `neutral`, returning a verbatim excerpt and reasoning.

## Requirements

- Python 3.11+
- PostgreSQL 14+
- [Ollama](https://ollama.ai) running locally with `llama3.2:3b` pulled

## Quick Start

```bash
# Pull the model
ollama pull llama3.2:3b

# Create the database schema (run against your Postgres instance)
psql your_database < schema.sql

# Install Python dependencies
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env: set DATABASE_URL at minimum

# Run the worker
zahn-worker
```

## Database Schema

```sql
CREATE TABLE sentiment_jobs (
    id                  BIGSERIAL PRIMARY KEY,
    message_text        TEXT         NOT NULL,
    source_record_id    BIGINT,
    source_record_type  VARCHAR(100),
    language_hint       VARCHAR(10)  DEFAULT NULL,

    status              VARCHAR(20)  NOT NULL DEFAULT 'pending',
    claimed_at          TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    claimed_by          VARCHAR(100) DEFAULT NULL,
    attempts            SMALLINT     NOT NULL DEFAULT 0,
    last_error          TEXT         DEFAULT NULL,

    sentiment_label     VARCHAR(20)  DEFAULT NULL,
    excerpt             TEXT         DEFAULT NULL,
    reasoning           TEXT         DEFAULT NULL,
    raw_llm_response    TEXT         DEFAULT NULL,
    keyword_hits        JSONB        DEFAULT NULL,

    created_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sentiment_jobs_pending
    ON sentiment_jobs (created_at) WHERE status = 'pending';

CREATE INDEX idx_sentiment_jobs_source
    ON sentiment_jobs (source_record_type, source_record_id);
```

## Configuration

All settings are read from environment variables (or a `.env` file):

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | *(required)* | PostgreSQL connection string |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | Model name |
| `OLLAMA_TIMEOUT` | `60` | HTTP timeout in seconds |
| `POLL_INTERVAL` | `5` | Seconds between polling cycles |
| `MAX_ATTEMPTS` | `3` | Max retries before marking job `failed` |

## Tests

```bash
pytest tests/unit/           # fast, no external dependencies
pytest tests/integration/    # requires running Postgres + Ollama
```
