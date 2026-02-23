from __future__ import annotations

from zahn.config import Settings
from zahn.llm import call_ollama, parse_llm_response
from zahn.models import SentimentJob, SentimentResult
from zahn.prompt import build_prompt


def analyze_message(job: SentimentJob, config: Settings) -> SentimentResult:
    prompt = build_prompt(job.message_text)
    raw = call_ollama(prompt, config)
    llm_result = parse_llm_response(raw)

    return SentimentResult(
        job_id=job.id,
        label=llm_result.label,
        excerpt=llm_result.excerpt,
        reasoning=llm_result.reasoning,
        raw_llm_response=raw,
    )
