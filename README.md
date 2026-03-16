# Zahn — Dental Lab Sentiment Analysis Worker

LLM-powered Python worker that classifies dental lab customer messages using two
independent binary classifiers (`is_frustrated`, `is_satisfied`), detecting the
message language and returning a verbatim excerpt and reasoning for each dimension.

## Requirements

- Python 3.11+
- PostgreSQL 14+
- [Ollama](https://ollama.ai) running locally with `llama3.2:3b` pulled

## Quick Start

```bash
# Pull the model
ollama pull llama3.2:3b

# Create the database, then apply the schema
createdb zahn_dev
psql zahn_dev < schema.sql

# Install Python dependencies
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env: set DATABASE_URL at minimum

# Run the worker
zahn-worker
```

## Demo

To insert 10 sample conversations (5 EN, 3 ES, 2 FR) and process them end-to-end:

```bash
python demo.py
```

## Database Schema

The schema is in `schema.sql`. Key columns written on completion:

| Column | Type | Description |
|---|---|---|
| `frustration_label` | `yes\|no` | Whether the customer is frustrated |
| `satisfaction_label` | `yes\|no` | Whether the customer is satisfied |
| `detected_language` | `en\|fr\|es` | Language detected by the frustration call |
| `frustration_excerpt` | text | Verbatim substring driving the frustration label |
| `frustration_reasoning` | text | 1-2 sentence explanation (frustration) |
| `satisfaction_excerpt` | text | Verbatim substring driving the satisfaction label |
| `satisfaction_reasoning` | text | 1-2 sentence explanation (satisfaction) |
| `raw_frustration_response` | text | Full frustration JSON from Ollama (auditing) |
| `raw_satisfaction_response` | text | Full satisfaction JSON from Ollama (auditing) |

The two dimensions are independent — a job can have `frustration_label = 'yes'` and
`satisfaction_label = 'yes'` simultaneously (e.g. a customer frustrated about one thing
but genuinely complimentary about another).

### Migrating an existing database

If you have an existing instance using the old single-label schema, apply:

```bash
psql zahn_dev < migrations/002_binary_classifiers.sql
```

This drops `sentiment_label`, `excerpt`, `reasoning`, `raw_llm_response`; adds the nine
new columns; and resets completed rows to `pending` so they are re-processed.

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
