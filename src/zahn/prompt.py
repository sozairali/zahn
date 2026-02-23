from __future__ import annotations

_SYSTEM = """\
You are a dental lab customer-sentiment analyst. Your only output is a single valid JSON object.
Do not include markdown code fences, explanations, or any text outside the JSON object.\
"""

_TEMPLATE = """\
DENTAL LAB CONTEXT:
Common case types: crown, bridge, veneer, prosthesis, implant, denture.
Key quality signals: shade/colour match, margins, occlusion, fit.
A "remake" means a case must be redone due to quality issues — multiple remakes indicate serious problems.
Relevant terms in other languages: retake/refaire (FR), rehacer/rehecho (ES), furieux (FR), furioso (ES).

TASK — read the message below and holistically assess the customer's overall sentiment.

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


def build_prompt(message: str) -> str:
    # Use str.replace instead of .format() so that curly braces in the
    # customer message do not raise KeyError.
    return f"{_SYSTEM}\n\n{_TEMPLATE.replace('{message_text}', message)}"
