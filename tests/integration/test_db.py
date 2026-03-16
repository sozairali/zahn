"""
Integration tests for db.py.
Requires a running PostgreSQL with the sentiment_jobs table.
Set DATABASE_URL env var or .env file.

Run with: pytest tests/integration/test_db.py
"""
import os
import pytest
import psycopg
from psycopg.rows import dict_row

from zahn.config import Settings
from zahn.db import get_connection, claim_job, write_result, release_job, reset_stale_claims
from zahn.models import SentimentResult

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def config():
    return Settings()


@pytest.fixture
def db_conn(config):
    with get_connection(config) as conn:
        yield conn


@pytest.fixture(autouse=True)
def clean_jobs(db_conn):
    db_conn.execute("DELETE FROM sentiment_jobs WHERE message_text LIKE '[test]%'")
    db_conn.commit()
    yield
    db_conn.execute("DELETE FROM sentiment_jobs WHERE message_text LIKE '[test]%'")
    db_conn.commit()


def insert_job(conn, message: str, status: str = "pending", attempts: int = 0) -> int:
    cur = conn.execute(
        "INSERT INTO sentiment_jobs (message_text, status, attempts) VALUES (%s, %s, %s) RETURNING id",
        (message, status, attempts),
    )
    conn.commit()
    return cur.fetchone()["id"]


class TestClaimJob:
    def test_claims_pending_job(self, db_conn, config):
        job_id = insert_job(db_conn, "[test] claim basic")
        job = claim_job(db_conn, "test-worker", max_attempts=3)
        assert job is not None
        assert job.id == job_id
        assert job.message_text == "[test] claim basic"

    def test_returns_none_when_no_pending(self, db_conn, config):
        result = claim_job(db_conn, "test-worker", max_attempts=3)
        assert result is None

    def test_increments_attempts(self, db_conn, config):
        job_id = insert_job(db_conn, "[test] attempts inc")
        claim_job(db_conn, "test-worker")
        row = db_conn.execute(
            "SELECT attempts, status FROM sentiment_jobs WHERE id = %s", (job_id,)
        ).fetchone()
        assert row["attempts"] == 1
        assert row["status"] == "claimed"

    def test_skips_max_attempts_exceeded(self, db_conn, config):
        insert_job(db_conn, "[test] max attempts", attempts=3)
        result = claim_job(db_conn, "test-worker", max_attempts=3)
        assert result is None


class TestWriteResult:
    def test_writes_completed(self, db_conn, config):
        job_id = insert_job(db_conn, "[test] write result")
        # Manually claim it
        db_conn.execute(
            "UPDATE sentiment_jobs SET status='claimed' WHERE id=%s", (job_id,)
        )
        db_conn.commit()

        result = SentimentResult(
            job_id=job_id,
            frustration_label="yes",
            satisfaction_label="no",
            detected_language="en",
            frustration_excerpt="write result",
            frustration_reasoning="Test frustration reasoning.",
            satisfaction_excerpt="no positive signals",
            satisfaction_reasoning="Test satisfaction reasoning.",
            raw_frustration_response='{"label":"yes"}',
            raw_satisfaction_response='{"label":"no"}',
        )
        write_result(db_conn, result)

        row = db_conn.execute(
            """SELECT status, frustration_label, satisfaction_label, detected_language,
                      frustration_excerpt, frustration_reasoning
               FROM sentiment_jobs WHERE id=%s""",
            (job_id,),
        ).fetchone()
        assert row["status"] == "completed"
        assert row["frustration_label"] == "yes"
        assert row["satisfaction_label"] == "no"
        assert row["detected_language"] == "en"
        assert row["frustration_excerpt"] == "write result"

    def test_raises_if_not_claimed(self, db_conn, config):
        job_id = insert_job(db_conn, "[test] guard check")
        result = SentimentResult(
            job_id=job_id,
            frustration_label="no",
            satisfaction_label="no",
            detected_language="en",
            frustration_excerpt="guard check",
            frustration_reasoning="Test.",
            satisfaction_excerpt="guard check",
            satisfaction_reasoning="Test.",
            raw_frustration_response="{}",
            raw_satisfaction_response="{}",
        )
        with pytest.raises(RuntimeError, match="guard failed"):
            write_result(db_conn, result)


class TestReleaseJob:
    def test_release_back_to_pending(self, db_conn, config):
        job_id = insert_job(db_conn, "[test] release pending")
        db_conn.execute(
            "UPDATE sentiment_jobs SET status='claimed', attempts=1 WHERE id=%s", (job_id,)
        )
        db_conn.commit()
        release_job(db_conn, job_id, "test error", max_attempts=3)
        row = db_conn.execute(
            "SELECT status, last_error FROM sentiment_jobs WHERE id=%s", (job_id,)
        ).fetchone()
        assert row["status"] == "pending"
        assert "test error" in row["last_error"]

    def test_release_to_failed_at_max(self, db_conn, config):
        job_id = insert_job(db_conn, "[test] release failed", attempts=3)
        db_conn.execute(
            "UPDATE sentiment_jobs SET status='claimed' WHERE id=%s", (job_id,)
        )
        db_conn.commit()
        release_job(db_conn, job_id, "permanent error", max_attempts=3)
        row = db_conn.execute(
            "SELECT status FROM sentiment_jobs WHERE id=%s", (job_id,)
        ).fetchone()
        assert row["status"] == "failed"
