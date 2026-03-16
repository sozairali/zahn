from __future__ import annotations

import logging
import time

from zahn.analysis import analyze_message
from zahn.config import Settings, load_settings
from zahn.db import claim_job, get_connection, release_job, reset_stale_claims, write_result

logger = logging.getLogger(__name__)


def run_one_iteration(config: Settings) -> bool:
    """Claim and process one job. Returns True if a job was processed, False if none pending."""
    with get_connection(config) as conn:
        job = claim_job(conn, config.worker_id, config.max_attempts)

    if job is None:
        return False

    logger.info("Claimed job %d (attempt %d)", job.id, job.attempts)

    try:
        result = analyze_message(job, config)
        with get_connection(config) as conn:
            write_result(conn, result)
        logger.info(
            "Completed job %d: frust=%s sat=%s excerpt=%r",
            job.id,
            result.frustration_label,
            result.satisfaction_label,
            result.frustration_excerpt[:80],
        )
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("Job %d failed: %s", job.id, error_msg)
        with get_connection(config) as conn:
            release_job(conn, job.id, error_msg, config.max_attempts)

    return True


def run_worker(config: Settings) -> None:
    logger.info("Worker starting: id=%s model=%s", config.worker_id, config.ollama_model)

    with get_connection(config) as conn:
        reset_stale_claims(conn)

    while True:
        processed = run_one_iteration(config)
        if not processed:
            time.sleep(config.poll_interval)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = load_settings()
    run_worker(config)


if __name__ == "__main__":
    main()
