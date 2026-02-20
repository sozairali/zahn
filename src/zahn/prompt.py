from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass

import chardet


# ---------------------------------------------------------------------------
# Internal keyword type (dental lab terminology loaded from CSV)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _KeywordEntry:
    language: str
    keyword: str
    keyword_type: str
    score: int


# ---------------------------------------------------------------------------
# CSV loading + domain context builder
# ---------------------------------------------------------------------------

def load_keywords(csv_path: str) -> list[_KeywordEntry]:
    with open(csv_path, "rb") as fb:
        encoding = chardet.detect(fb.read())["encoding"] or "utf-8"

    entries: list[_KeywordEntry] = []
    with open(csv_path, encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lang = row["Language"].strip().lower()
                keyword = row["keyword"].strip()
                ktype = row["keyword type"].strip()
                score = int(row["score"].strip())
            except (KeyError, ValueError):
                continue
            if not keyword or not lang:
                continue
            entries.append(
                _KeywordEntry(language=lang, keyword=keyword, keyword_type=ktype, score=score)
            )
    return entries


def build_domain_context(keywords: list[_KeywordEntry]) -> str:
    by_type: dict[str, list[str]] = defaultdict(list)
    for kw in keywords:
        lang_tag = f"/{kw.language}" if kw.language != "en" else ""
        by_type[kw.keyword_type].append(f"{kw.keyword}{lang_tag}({kw.score})")

    lines: list[str] = []
    for ktype, entries in sorted(by_type.items()):
        lines.append(f"  [{ktype}]  " + ", ".join(entries))
    return "\n".join(lines)


def load_domain_context(csv_path: str) -> str:
    """Load keywords from CSV and return a formatted domain context string."""
    return build_domain_context(load_keywords(csv_path))


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a dental lab customer-sentiment analyst. Your only output is a single valid JSON object.
Do not include markdown code fences, explanations, or any text outside the JSON object.\
"""

_TEMPLATE = """\
DOMAIN SIGNALS — dental lab terminology and sentiment keywords (keyword/lang(score)):
{domain_context}

TASK — read the message below and holistically assess the customer's overall sentiment.
Use the domain signals above as context to understand dental lab terminology, not as a checklist.

ALWAYS label as "frustration" when the message clearly conveys:
- Missed call or no response from the lab
- Case delivered late or significantly delayed
- Quality issues requiring a remake (especially a 2nd or 3rd remake)
- Customer expresses intent to leave / switch labs

SUPPORTED LANGUAGES: en / fr / es — analyze the message as-is, do not translate.

OUTPUT FORMAT (JSON only, no markdown):
{
  "label": "frustration|satisfaction|neutral",
  "excerpt": "<verbatim substring from the message that most drives the label>",
  "reasoning": "<1-2 sentences in English explaining the label>",
  "detected_language": "en|fr|es"
}

MESSAGE:
{message_text}\
"""


def build_prompt(message: str, domain_context: str) -> str:
    # Use str.replace instead of .format() so that curly braces in the
    # customer message do not raise KeyError.
    user_content = (
        _TEMPLATE
        .replace("{domain_context}", domain_context)
        .replace("{message_text}", message)
    )
    return f"{_SYSTEM}\n\n{user_content}"
