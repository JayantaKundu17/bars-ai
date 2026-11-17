"""BARS AI structured case-building module.

Converts user-provided facts into an editable, structured case record.
It does not determine legal liability or invent missing facts.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from google import genai
from google.genai import types

from backend.config import GEMINI_API_KEY, GEMINI_MODEL

_client = genai.Client(api_key=GEMINI_API_KEY)


def _clean(value: Any, limit: int = 10000) -> str:
    return str(value or "").strip()[:limit]


def _as_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [_clean(item, 1000) for item in value if _clean(item, 1000)]


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("The AI response did not contain valid structured data.")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("The AI response was not a JSON object.")
    return data


def build_case_record(
    description: str,
    case_type: str = "Other",
    jurisdiction: str = "India",
    user_role: str = "Not specified",
    desired_outcome: str = "",
) -> Dict[str, Any]:
    """Generate a structured case record from facts supplied by the user."""
    description = _clean(description)
    if len(description) < 20:
        raise ValueError("Please provide at least 20 characters describing the situation.")

    prompt = f"""
Convert the user's description into a structured case file. Return ONLY valid JSON.
Do not give legal advice or decide who is legally right. Do not invent facts, dates,
people, evidence, communications, amounts, legal provisions, deadlines, or events.
If information is absent, use an empty string or empty list. Separate confirmed
facts from allegations, assumptions, and unresolved questions. If dates are vague,
retain the user's wording and do not manufacture a precise date.

Use exactly this JSON structure:
{{
  "case_title": "short neutral title",
  "case_summary": "concise neutral summary",
  "parties": [{{"name_or_role": "", "position": "", "known_details": ""}}],
  "timeline": [{{"date_or_period": "", "event": "", "status": "user_stated"}}],
  "key_facts": ["..."],
  "allegations_or_claims": ["..."],
  "disputed_or_unclear_facts": ["..."],
  "evidence_available": [{{"item": "", "what_it_may_support": "", "provided_or_mentioned": "mentioned"}}],
  "missing_evidence": ["..."],
  "questions_to_clarify": ["..."],
  "desired_outcome": "",
  "issues_to_review": ["neutral issue descriptions; not legal conclusions"]
}}

Case type: {case_type}
Jurisdiction supplied: {jurisdiction}
User's role: {user_role}
Desired outcome supplied: {desired_outcome or 'Not specified'}
User's description:
{description}
"""
    response = _client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json"),
    )
    data = _extract_json(getattr(response, "text", ""))

    record = {
        "case_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_type": case_type,
        "jurisdiction": jurisdiction,
        "user_role": user_role,
        "original_description": description,
        "case_title": _clean(data.get("case_title"), 200) or "Untitled case",
        "case_summary": _clean(data.get("case_summary"), 3000),
        "parties": data.get("parties", []) if isinstance(data.get("parties"), list) else [],
        "timeline": data.get("timeline", []) if isinstance(data.get("timeline"), list) else [],
        "key_facts": _as_list(data.get("key_facts")),
        "allegations_or_claims": _as_list(data.get("allegations_or_claims")),
        "disputed_or_unclear_facts": _as_list(data.get("disputed_or_unclear_facts")),
        "evidence_available": data.get("evidence_available", []) if isinstance(data.get("evidence_available"), list) else [],
        "missing_evidence": _as_list(data.get("missing_evidence")),
        "questions_to_clarify": _as_list(data.get("questions_to_clarify")),
        "desired_outcome": _clean(data.get("desired_outcome"), 1500) or _clean(desired_outcome, 1500),
        "issues_to_review": _as_list(data.get("issues_to_review")),
    }
    return record
