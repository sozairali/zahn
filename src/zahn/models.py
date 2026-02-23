from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, field_validator


VALID_LABELS = frozenset({"frustration", "satisfaction", "neutral"})
VALID_LANGUAGES = frozenset({"en", "fr", "es"})


class SentimentJob(BaseModel):
    id: int
    message_text: str
    source_record_id: Optional[int] = None
    source_record_type: Optional[str] = None
    attempts: int = 0


class LLMResponse(BaseModel):
    label: str
    excerpt: str
    reasoning: str
    detected_language: str

    @field_validator("label")
    @classmethod
    def label_must_be_valid(cls, v: str) -> str:
        if v not in VALID_LABELS:
            raise ValueError(f"label must be one of {sorted(VALID_LABELS)}, got {v!r}")
        return v

    @field_validator("detected_language")
    @classmethod
    def language_must_be_valid(cls, v: str) -> str:
        if v not in VALID_LANGUAGES:
            raise ValueError(f"detected_language must be one of {sorted(VALID_LANGUAGES)}, got {v!r}")
        return v

    @field_validator("excerpt")
    @classmethod
    def excerpt_must_not_be_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("excerpt must not be empty or whitespace-only")
        return stripped

    @field_validator("reasoning")
    @classmethod
    def reasoning_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reasoning must not be empty")
        return v


class SentimentResult(BaseModel):
    job_id: int
    label: str
    excerpt: str
    reasoning: str
    raw_llm_response: str
    detected_language: Optional[str] = None

    @field_validator("label")
    @classmethod
    def label_must_be_valid(cls, v: str) -> str:
        if v not in VALID_LABELS:
            raise ValueError(f"label must be one of {sorted(VALID_LABELS)}, got {v!r}")
        return v
