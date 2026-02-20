from __future__ import annotations

import json
import re

import httpx

from zahn.config import Settings
from zahn.models import LLMResponse

# Matches a JSON object even if surrounded by stray text or partial fences
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


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


def parse_llm_response(raw: str) -> LLMResponse:
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        raise json.JSONDecodeError("No JSON object found in LLM response", raw, 0)
    return LLMResponse.model_validate_json(match.group())
