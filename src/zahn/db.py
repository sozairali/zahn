from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

import psycopg
from psycopg.rows import dict_row

from zahn.config import Settings
from zahn.models import SentimentJob, SentimentResult

logger = logging.getLogger(__name__)

_STALE_MINUTES = 10


@contextmanager
def get_connection(config: Settings) -> Generator[psycopg.Connection, None, None]:
    with psycopg.connect(config.database_url, row_factory=dict_row) as conn:
        yield conn


def reset_stale_claims(conn: psycopg.Connection) -> int:
    cur = conn.execute(
        f"""
        UPDATE sentiment_jobs
        SET status = 'pending', claimed_at = NULL, claimed_by = NULL, updated_at = NOW()
        WHERE status = 'claimed'
          AND claimed_at < NOW() - INTERVAL '{_STALE_MINUTES} minutes'
        """
    )
    conn.commit()
    count = cur.rowcount
    if count:
        logger.info("Reset %d stale claimed jobs to pending", count)
    return count


def claim_job(
    conn: psycopg.Connection, worker_id: str, max_attempts: int = 3
) -> SentimentJob | None:
    cur = conn.execute(
        """
        UPDATE sentiment_jobs
        SET status = 'claimed',
            claimed_at = NOW(),
            claimed_by = %(worker_id)s,
            attempts = attempts + 1,
            updated_at = NOW()
        WHERE id = (
            SELECT id FROM sentiment_jobs
            WHERE status = 'pending'
              AND attempts < %(max_attempts)s
            ORDER BY created_at ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        RETURNING id, message_text, source_record_id, source_record_type,
                  language_hint, attempts
        """,
        {"worker_id": worker_id, "max_attempts": max_attempts},
    )
    conn.commit()
    row = cur.fetchone()
    if row is None:
        return None
    return SentimentJob.model_validate(dict(row))


def write_result(conn: psycopg.Connection, result: SentimentResult) -> None:
    cur = conn.execute(
        """
        UPDATE sentiment_jobs
        SET status = 'completed',
            sentiment_label = %(label)s,
            excerpt = %(excerpt)s,
            reasoning = %(reasoning)s,
            raw_llm_response = %(raw)s,
            claimed_by = NULL,
            claimed_at = NULL,
            updated_at = NOW()
        WHERE id = %(job_id)s
          AND status = 'claimed'
        """,
        {
            "label": result.label,
            "excerpt": result.excerpt,
            "reasoning": result.reasoning,
            "raw": result.raw_llm_response,
            "job_id": result.job_id,
        },
    )
    if cur.rowcount != 1:
        conn.rollback()
        raise RuntimeError(
            f"write_result guard failed for job {result.job_id}: "
            f"expected 1 updated row, got {cur.rowcount} (job may have been reclaimed)"
        )
    conn.commit()


def release_job(
    conn: psycopg.Connection, job_id: int, error: str, max_attempts: int = 3
) -> None:
    cur = conn.execute(
        """
        UPDATE sentiment_jobs
        SET status = CASE
                WHEN attempts >= %(max_attempts)s THEN 'failed'
                ELSE 'pending'
            END,
            claimed_at = NULL,
            claimed_by = NULL,
            last_error = %(error)s,
            updated_at = NOW()
        WHERE id = %(job_id)s
        """,
        {"max_attempts": max_attempts, "error": error[:2000], "job_id": job_id},
    )
    conn.commit()
    logger.warning(
        "Released job %d after error (rowcount=%d): %s",
        job_id,
        cur.rowcount,
        error[:200],
    )
