from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from aico.contracts.errors import FailureCategory, TypedFailure
from aico.contracts.models import CitedAnswer, ResponseEnvelope

LOGGER = logging.getLogger("aico.contracts.validator")

# One bounded markdown unwrap: strip a single leading/trailing fence if the
# entire payload is fenced. Arbitrary prose around JSON is not accepted.
_FENCE = "```"


def unwrap_markdown_json(raw: str) -> str:
    """Strip at most one markdown code fence. Documented bounded unwrap."""
    text = raw.strip()
    if not text.startswith(_FENCE):
        return text
    lines = text.splitlines()
    if not lines:
        return text
    body = lines[1:]
    if body and body[-1].strip() == _FENCE:
        body = body[:-1]
    return "\n".join(body).strip()


def parse_json(raw: str) -> dict[str, Any] | TypedFailure:
    """Turn untrusted text into a JSON object, or a typed parse failure."""
    if not isinstance(raw, str) or not raw.strip():
        return _parse_failure("response is empty")
    candidate = unwrap_markdown_json(raw)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return _parse_failure("response is not valid JSON")
    if not isinstance(payload, dict):
        return _parse_failure("response JSON must be an object")
    return payload


def validate_cited_answer(payload: dict[str, Any]) -> CitedAnswer | TypedFailure:
    try:
        return CitedAnswer.model_validate(payload)
    except ValidationError as exc:
        return _contract_failure(exc)


def validate_envelope(payload: dict[str, Any]) -> ResponseEnvelope | TypedFailure:
    try:
        return ResponseEnvelope.model_validate(payload)
    except ValidationError as exc:
        return _contract_failure(exc)


def parse_and_validate_cited_answer(raw: str) -> CitedAnswer | TypedFailure:
    parsed = parse_json(raw)
    if isinstance(parsed, TypedFailure):
        return parsed
    return validate_cited_answer(parsed)


def _parse_failure(message: str) -> TypedFailure:
    LOGGER.info("stage=parse outcome=failure")
    return TypedFailure(
        category=FailureCategory.PARSE,
        message=message,
        repairable=False,
    )


def _contract_failure(exc: ValidationError) -> TypedFailure:
    error = exc.errors()[0] if exc.errors() else {}
    field_path = _field_path(error.get("loc", ()))
    message = _safe_contract_message(error)
    LOGGER.info(
        "stage=contract outcome=failure field_path=%s error_type=%s",
        field_path or "-",
        error.get("type", "unknown"),
    )
    return TypedFailure(
        category=FailureCategory.CONTRACT,
        message=message,
        field_path=field_path,
        repairable=True,
    )


def _field_path(loc: tuple[Any, ...]) -> str | None:
    if not loc:
        return None
    return ".".join(str(part) for part in loc)


def _safe_contract_message(error: dict[str, Any]) -> str:
    """Map a Pydantic error to a message that does not echo raw model output."""
    err_type = str(error.get("type", ""))
    if err_type == "missing":
        return "required field is missing"
    if err_type == "extra_forbidden":
        return "unknown field is not allowed"
    if err_type in {"enum", "literal_error"}:
        return "value is not an allowed enum or literal"
    if err_type.endswith("_type") or err_type in {"string_type", "list_type", "int_type"}:
        return "field has the wrong type"
    if err_type in {
        "string_too_short",
        "string_too_long",
        "too_short",
        "too_long",
        "greater_than_equal",
        "less_than_equal",
        "greater_than",
        "less_than",
    }:
        return "value is out of range"
    return "contract validation failed"
