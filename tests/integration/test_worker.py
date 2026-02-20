"""
Integration tests for worker.py.
Requires running PostgreSQL + Ollama.

Run with: pytest tests/integration/test_worker.py
"""
import pytest
from unittest.mock import patch

from zahn.config import Settings
from zahn.db import get_connection
from zahn.worker import run_one_iteration
from zahn.prompt import load_domain_context

pytestmark = pytest.mark.integration

VALID_RAW = """{
  "label": "frustration",
  "excerpt": "extremely late",
  "reasoning": "The case was delivered late causing frustration.",
  "detected_language": "en"
}"""


@pytest.fixture(scope="module")
def config():
    return Settings()


@pytest.fixture(scope="module")
def domain_context(config):
    return load_domain_context(config.keywords_csv_path)


@pytest.fixture
def db_conn(config):
    with get_connection(config) as conn:
        yield conn


@pytest.fixture(autouse=True)
def clean_jobs(db_conn):
    db_conn.execute("DELETE FROM sentiment_jobs WHERE message_text LIKE '[itest]%'")
    db_conn.commit()
    yield
    db_conn.execute("DELETE FROM sentiment_jobs WHERE message_text LIKE '[itest]%'")
    db_conn.commit()


def insert_job(conn, message: str, attempts: int = 0) -> int:
    cur = conn.execute(
        "INSERT INTO sentiment_jobs (message_text, attempts) VALUES (%s, %s) RETURNING id",
        (message, attempts),
    )
    conn.commit()
    return cur.fetchone()["id"]


class TestRunOneIteration:
    def test_returns_false_when_no_jobs(self, config, domain_context):
        with patch("zahn.analysis.call_ollama", return_value=VALID_RAW):
            result = run_one_iteration(config, domain_context)
        assert result is False

    def test_processes_pending_job(self, db_conn, config, domain_context):
        job_id = insert_job(db_conn, "[itest] this case is extremely late")

        with patch("zahn.analysis.call_ollama", return_value=VALID_RAW):
            processed = run_one_iteration(config, domain_context)

        assert processed is True
        row = db_conn.execute(
            "SELECT status, sentiment_label FROM sentiment_jobs WHERE id=%s", (job_id,)
        ).fetchone()
        assert row["status"] == "completed"
        assert row["sentiment_label"] == "frustration"

    def test_releases_job_on_error(self, db_conn, config, domain_context):
        job_id = insert_job(db_conn, "[itest] error test job")

        import httpx
        with patch("zahn.analysis.call_ollama", side_effect=httpx.TimeoutException("timeout")):
            processed = run_one_iteration(config, domain_context)

        assert processed is True
        row = db_conn.execute(
            "SELECT status, last_error, attempts FROM sentiment_jobs WHERE id=%s", (job_id,)
        ).fetchone()
        assert row["status"] == "pending"
        assert "TimeoutException" in row["last_error"]

    def test_marks_failed_after_max_attempts(self, db_conn, config, domain_context):
        job_id = insert_job(db_conn, "[itest] max attempts job", attempts=config.max_attempts - 1)

        import httpx
        with patch("zahn.analysis.call_ollama", side_effect=httpx.TimeoutException("timeout")):
            run_one_iteration(config, domain_context)

        row = db_conn.execute(
            "SELECT status FROM sentiment_jobs WHERE id=%s", (job_id,)
        ).fetchone()
        assert row["status"] == "failed"
