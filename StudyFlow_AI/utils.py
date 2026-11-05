"""Small local helpers for StudyFlow AI."""
from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
STUDENT_ID_RE = re.compile(
    r"\b(?:student\s*(?:id|number)|university\s*(?:id|number)|id\s*(?:no\.?|number))\s*[:#-]?\s*[A-Za-z0-9-]{5,}\b",
    re.IGNORECASE,
)
LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{8,}(?!\d)")


def redact_sensitive_text(text: str) -> tuple[str, dict[str, int]]:
    """Mask common identifiers before text is sent to the model.

    This is a basic prototype safeguard, not a guarantee that every sensitive
    detail is removed. Users should avoid uploading confidential materials.
    """
    counts = {"emails": 0, "phones": 0, "student_ids": 0, "long_numbers": 0}

    def replace_email(match: re.Match[str]) -> str:
        counts["emails"] += 1
        return "[EMAIL REDACTED]"

    def replace_phone(match: re.Match[str]) -> str:
        counts["phones"] += 1
        return "[PHONE REDACTED]"

    def replace_student_id(match: re.Match[str]) -> str:
        counts["student_ids"] += 1
        return "[STUDENT ID REDACTED]"

    def replace_long_number(match: re.Match[str]) -> str:
        counts["long_numbers"] += 1
        return "[NUMBER REDACTED]"

    redacted = EMAIL_RE.sub(replace_email, text)
    redacted = STUDENT_ID_RE.sub(replace_student_id, redacted)
    redacted = PHONE_RE.sub(replace_phone, redacted)
    redacted = LONG_NUMBER_RE.sub(replace_long_number, redacted)
    return redacted, counts


def safe_json_loads(content: str) -> dict[str, Any]:
    """Load a JSON object even when a model wraps it in a Markdown fence."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("The model did not return a valid JSON object.")
        parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("The model response was JSON, but not a JSON object.")
    return parsed


def make_export_record(result: dict[str, Any]) -> dict[str, Any]:
    """Create a downloadable record without raw PDF text or uploaded bytes."""
    safe = deepcopy(result)
    safe.pop("source_text", None)
    safe.pop("redacted_text", None)

    # The metadata object is sometimes passed directly from PDF processing.
    # Remove any in-memory text there as well before offering a download.
    metadata = safe.get("document_metadata")
    if isinstance(metadata, dict):
        metadata.pop("redacted_text", None)
        metadata.pop("source_text", None)
        metadata.pop("file_bytes", None)

    safe["exported_at_utc"] = datetime.now(timezone.utc).isoformat()
    safe["privacy_note"] = (
        "This export excludes the original PDF bytes and raw extracted PDF text. "
        "It contains generated study material and file-safe metadata only."
    )
    return safe
