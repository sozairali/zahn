from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable

import httpx
from pydantic import ValidationError

from zahn.config import Settings
from zahn.llm import ExcerptValidationError, call_ollama, parse_binary_response, validate_excerpt
from zahn.models import SentimentJob, SentimentResult
from zahn.prompt import build_frustration_prompt, build_satisfaction_prompt

logger = logging.getLogger(__name__)


@dataclass
class ClassifierResult:
    """Outcome of a single binary classifier call (frustration or satisfaction).

    On success all label/excerpt/reasoning/detected_language fields are populated
    and ``exception`` is None.  On total failure (all retries exhausted)
    label/excerpt/reasoning are empty and ``exception`` holds the original error.

    ``parse_error`` is a computed property derived from ``exception`` — the two
    are always in sync by construction, with no separate stored string to drift.

    ``excerpt_retries`` counts how many attempts were rejected specifically by
    the excerpt validator, regardless of whether the call ultimately succeeded.
    """

    label: str = ""
    excerpt: str = ""
    reasoning: str = ""
    detected_language: str = ""
    raw: str = ""
    excerpt_retries: int = 0
    exception: Exception | None = field(default=None, repr=False, compare=False)

    @property
    def parse_error(self) -> str:
        return str(self.exception) if self.exception is not None else ""

    def raise_if_failed(self) -> None:
        """Re-raise the stored exception if the classifier failed, preserving its original type."""
        if self.exception is not None:
            raise self.exception


def run_classifier(
    message: str,
    build_prompt_fn: Callable[[str], str],
    config: Settings,
    max_attempts: int = 3,
) -> ClassifierResult:
    """Call the LLM for a single binary classifier dimension with retry.

    Retries on expected failures: JSON parse errors, Pydantic validation
    errors, HTTP errors, and ``ExcerptValidationError``.  Programming bugs
    (e.g. ``AttributeError``, ``TypeError``) propagate immediately to the
    single ``try/except`` boundary in ``worker.py``.

    Always returns a ``ClassifierResult`` on expected failures — never raises
    for retryable errors.
    """
    last_exc: Exception | None = None
    last_raw: str = ""
    excerpt_retries = 0

    for attempt in range(1, max_attempts + 1):
        raw = ""
        try:
            raw = call_ollama(build_prompt_fn(message), config)
            last_raw = raw
            resp = parse_binary_response(raw)
            validate_excerpt(message, resp)
            return ClassifierResult(
                label=resp.label,
                excerpt=resp.excerpt,
                reasoning=resp.reasoning,
                detected_language=resp.detected_language,
                raw=raw,
                excerpt_retries=excerpt_retries,
            )
        except ExcerptValidationError as exc:
            excerpt_retries += 1
            last_exc = exc
            logger.debug("Excerpt validation failed (attempt %d/%d): %s", attempt, max_attempts, exc)
        except (json.JSONDecodeError, ValidationError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            logger.debug("Classifier call failed (attempt %d/%d): %s", attempt, max_attempts, exc)

    return ClassifierResult(
        raw=last_raw,
        excerpt_retries=excerpt_retries,
        exception=last_exc,
    )


def analyze_message(job: SentimentJob, config: Settings) -> SentimentResult:
    frust = run_classifier(job.message_text, build_frustration_prompt, config)
    frust.raise_if_failed()

    sat = run_classifier(job.message_text, build_satisfaction_prompt, config)
    sat.raise_if_failed()

    if frust.detected_language != sat.detected_language:
        logger.warning(
            "Language mismatch for job %d: frustration=%s, satisfaction=%s (using frustration)",
            job.id, frust.detected_language, sat.detected_language,
        )

    return SentimentResult(
        job_id=job.id,
        detected_language=frust.detected_language,
        frustration_label=frust.label,
        frustration_excerpt=frust.excerpt,
        frustration_reasoning=frust.reasoning,
        raw_frustration_response=frust.raw,
        satisfaction_label=sat.label,
        satisfaction_excerpt=sat.excerpt,
        satisfaction_reasoning=sat.reasoning,
        raw_satisfaction_response=sat.raw,
    )
