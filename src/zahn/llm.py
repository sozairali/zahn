from __future__ import annotations

import json
import re
from typing import Type, TypeVar

import httpx
from pydantic import BaseModel

from zahn.config import Settings
from zahn.models import BinaryLLMResponse

# Matches a JSON object even if surrounded by stray text or partial fences
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_M = TypeVar("_M", bound=BaseModel)


def call_ollama(prompt: str, config: Settings) -> str:
    url = f"{config.ollama_base_url.rstrip('/')}/api/generate"
    payload = {
        "model": config.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    with httpx.Client(timeout=config.ollama_timeout) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
    data = response.json()
    return data["response"]


def _parse_response(raw: str, model_cls: Type[_M]) -> _M:
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        raise json.JSONDecodeError("No JSON object found in LLM response", raw, 0)
    return model_cls.model_validate_json(match.group())


def parse_binary_response(raw: str) -> BinaryLLMResponse:
    """Parse a binary LLM response, recovering from common truncation patterns.

    Models occasionally stop generating just before the final ``}``.  Two forms
    are handled:
    - ends with ``"``   — model stopped after the last field value
    - ends with ``,``   — model emitted a trailing comma expecting more fields

    In both cases we append the missing brace and retry the parse before
    raising, so callers never need separate recovery logic.
    """
    try:
        return _parse_response(raw, BinaryLLMResponse)
    except Exception:
        stripped = raw.rstrip()
        candidate = stripped.rstrip(",") if stripped.endswith(",") else stripped
        if candidate.endswith('"'):
            try:
                return _parse_response(candidate + "\n}", BinaryLLMResponse)
            except Exception:
                pass
        raise


class ExcerptValidationError(ValueError):
    """Raised when the LLM's excerpt is not a verbatim substring of the message."""


def validate_excerpt(message: str, response: BinaryLLMResponse) -> None:
    if response.label != "yes":
        return
    # response.excerpt is guaranteed non-empty by BinaryLLMResponse's Pydantic validator
    if response.excerpt.lower() not in message.lower():
        raise ExcerptValidationError(
            f"excerpt is not a verbatim substring of the message: {response.excerpt!r}"
        )
