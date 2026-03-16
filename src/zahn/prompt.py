from __future__ import annotations

_SYSTEM = """\
You are a dental lab customer-sentiment analyst. Your only output is a single valid JSON object.
Do not include markdown code fences, explanations, or any text outside the JSON object.\
"""

_SHARED_CONTEXT = """\
DENTAL LAB CONTEXT:
Common case types: crown, bridge, veneer, prosthesis, implant, denture.
Key quality signals: shade/colour match, margins, occlusion, fit.
A "remake" means a case must be redone due to quality issues — multiple remakes indicate serious problems.
Relevant terms in other languages: retake/refaire (FR), rehacer/rehecho (ES), furieux (FR), furioso (ES).

The main reasons customers leave dental labs:
- Communication failures: not informed, poorly informed, or no response about case status
- Late cases / delivery delays
- Quality issues (fit, shade, margins) that may or may not result in remakes
- Price concerns or references to a competing lab

ABOUT THE MESSAGE:
- These notes are entered into a CRM by a dental lab sales representative.
- Sales reps are trained to frame interactions positively — they apologise, offer solutions, leave gifts.
  DO NOT let the rep's optimistic framing override the customer's underlying experience.
- Classify the CUSTOMER'S sentiment, not the lab rep's response to it.
- Names and identifiers have been redacted to preserve PII.\
"""

# Appended to every task section — defines language scope, output schema, and message slot.
# Use str.replace("{message_text}", ...) to inject the message; never use .format().
_TASK_FOOTER = """\
SUPPORTED LANGUAGES: en / fr / es — analyze the message as-is, do not translate.

OUTPUT FORMAT (JSON only, no markdown):
{
  "label": "yes|no",
  "detected_language": "en|fr|es",
  "excerpt": "<verbatim substring from the message that most drives the label>",
  "reasoning": "<1-2 sentences in English explaining the label>"
}

MESSAGE:
{message_text}\
"""

_FRUSTRATION_TASK = """\
TASK — assess whether the customer is frustrated.

LABEL AS "yes" when ANY of the following is present — even if the rep resolved it:
- The note uses words like: upset, frustrated, unhappy, angry, annoyed, disappointed
- A case was late, delayed, rescheduled, or the patient had to wait
- Any quality problem: bad fit, wrong shade, margins off, remake or adjustment required
- The customer switched labs, is trialling another lab, or has expressed intent to leave
- Missed call, no response, or any communication failure by the lab
- The customer did not receive what they ordered or expected
- Cumulative small issues ("little things that added up")
- The note is a complaint follow-up call — the fact of the complaint is itself the signal

HARD TRIGGERS — label "yes" unconditionally if ANY of these appear:
- The note references a complaint made by or about a doctor or customer, regardless of
  how the conversation went or whether things are "getting better". The existence of a
  complaint is the signal.
- A doctor or customer is described as "unhappy" — even if the rep is "checking in" or
  "offering help", the underlying unhappiness is the signal.
- A case was lost, returned to the lab multiple times for adjustments, or is at risk.
- The office reports persistent or recurring issues with cases.

THE FOLLOWING DO NOT CANCEL FRUSTRATION:
- The rep apologised or offered a solution — if there is something to fix, there was a problem
- The customer was polite or friendly during the call
- The office thanked the rep for calling or checking in — politeness is not satisfaction
- The customer said "maybe we'll come back" or "hopeful for the future" — they still left
- The issue was resolved — assess what prompted the note, not the outcome
- The rep's tone is positive or solution-focused — classify the customer's experience, not the rep's framing

LAB-AUTHORED OUTBOUND MESSAGES — label these "no":
Some notes are written BY the lab to the doctor, not by the customer. These contain no
customer sentiment and must be labeled "no":
- Due-date notification emails ("we are unable to meet your requested Due Date of…")
- Apology boilerplate ("We apologize for any inconvenience. Please contact me…")
- UPS/carrier tracking updates ("Your shipment 1ZJ… Delivered On Thursday…")
- Return-date confirmations ("Just wanted to provide you with the return date of…")
The presence of "We apologize" in a lab-authored email is NOT the customer expressing
frustration — it is routine communication.

LABEL AS "no" only when none of the above conditions are present:
- The note contains no complaint, no delay, no quality issue, no communication failure
- The relationship is in good standing and no problem is mentioned anywhere
- The note is a clinical or lab instruction (shade specs, occlusal adjustments, implant
  specs, remake parameters) with no emotional language from the customer. Describing a
  clinical problem to fix is not the same as a customer expressing frustration.
  Examples: "move midlines to left by 1mm", "MAILING PREVIOUS CROWN FOR SHADE MATCH",
  "D2 gingival and middle B1 occlusal" — label these "no".
- The note is too short or cryptic to determine customer sentiment ("READ RX",
  "set up 2025-87097", "for tooth #7", shade codes). When no emotional signal is
  detectable, default to "no". Do not infer frustration from brevity alone.
- The note is an internal staff action with no customer-expressed sentiment
  ("DELETED CASE, IT WAS A DUPE", "problem list for same reason as last note",
  "cancel case they sent it by mistake"). If no customer voice is present, label "no".\
"""

_SATISFACTION_TASK = """\
TASK — assess whether the customer is satisfied.

LABEL AS "yes" ONLY when the message is genuinely positive with no underlying complaint:
- Customer explicitly praises quality, communication, or service with no issue raised
- Relationship is in good standing and no problem is mentioned anywhere in the note
- The customer expresses appreciation, loyalty, or enthusiasm unprompted

THE FOLLOWING DO NOT QUALIFY AS SATISFACTION:
- A polite tone or lack of overt anger — neutral does not equal satisfied
- The rep resolved an issue — satisfaction requires genuine positivity, not just resolution
- The customer said "maybe we'll come back" or expressed conditional optimism
- Any complaint, delay, quality issue, or communication failure is present, even if minor

LABEL AS "no" when:
- The note is routine/administrative with no positive sentiment expressed
- The customer is frustrated (frustration and satisfaction are independent dimensions)
- The note is ambiguous or contains mixed signals\
"""


def _build_prompt(task: str, message: str) -> str:
    footer = _TASK_FOOTER.replace("{message_text}", message)
    return f"{_SYSTEM}\n\n{_SHARED_CONTEXT}\n\n{task}\n\n{footer}"


def build_frustration_prompt(message: str) -> str:
    return _build_prompt(_FRUSTRATION_TASK, message)


def build_satisfaction_prompt(message: str) -> str:
    return _build_prompt(_SATISFACTION_TASK, message)
