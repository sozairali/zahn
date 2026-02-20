# Zahn — Dental Lab Sentiment Analysis Worker

LLM-powered Python worker that classifies dental lab customer messages as
`frustration`, `satisfaction`, or `neutral`, returning an excerpt and reasoning.

## Requirements

- Python 3.11+
- PostgreSQL 14+
- [Ollama](https://ollama.ai) running locally with `llama3.2:3b` pulled

## Quick Start

```bash
# Pull the model
ollama pull llama3.2:3b

# Create and migrate the database (run the migration SQL against your Postgres instance)

# Install Python dependencies
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env: set DATABASE_URL at minimum

# Run the worker
zahn-worker
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
