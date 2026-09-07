"""Optional-field evolution stays backward compatible with the existing v1 caller."""

from __future__ import annotations

import json
from pathlib import Path

from aico.contracts.errors import TypedFailure
from aico.contracts.models import SCHEMA_VERSION_V1, CitedAnswer, ResponseEnvelope
from aico.contracts.service import wrap_envelope
from aico.contracts.validator import parse_and_validate_cited_answer, validate_envelope

CALLER_PATH = Path("tests/fixtures/day04/existing_caller_v1.json")
CASES_PATH = Path("tests/fixtures/day04/structured_output_cases.json")


def _caller() -> dict:
    return json.loads(CALLER_PATH.read_text(encoding="utf-8"))


def test_existing_caller_without_warning_still_validates() -> None:
    payload = _caller()
    sample = payload["sample_response"]
    assert "warning" not in sample
    result = validate_envelope(sample)
    assert isinstance(result, ResponseEnvelope)
    assert result.warning is None
    assert result.schema_version == SCHEMA_VERSION_V1
    assert result.request_id == "REQ-001"
    for field in payload["expected_fields"]:
        assert getattr(result, field) is not None


def test_optional_warning_is_accepted_when_present() -> None:
    sample = dict(_caller()["sample_response"])
    sample["warning"] = "citation coverage is thin"
    result = validate_envelope(sample)
    assert isinstance(result, ResponseEnvelope)
    assert result.warning == "citation coverage is thin"


def test_schema_version_is_present_on_envelope_metadata() -> None:
    sample = _caller()["sample_response"]
    result = validate_envelope(sample)
    assert isinstance(result, ResponseEnvelope)
    dumped = result.model_dump(exclude_none=True)
    assert dumped["schema_version"] == "1.0"
    assert dumped["result"]["schema_version"] == "1.0"


def test_wrap_envelope_keeps_existing_caller_fields() -> None:
    cited = parse_and_validate_cited_answer(_valid_cited_raw())
    assert isinstance(cited, CitedAnswer)
    envelope = wrap_envelope(
        cited,
        request_id="REQ-001",
        model_alias="chat-primary",
    )
    assert envelope.warning is None
    round_trip = validate_envelope(envelope.model_dump(exclude_none=True))
    assert round_trip == envelope


def _valid_cited_raw() -> str:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    return next(item["raw"] for item in cases if item["id"] == "D04-01")


def test_empty_optional_warning_is_rejected() -> None:
    sample = dict(_caller()["sample_response"])
    sample["warning"] = ""
    result = validate_envelope(sample)
    assert isinstance(result, TypedFailure)


def test_documented_breaking_changes_would_fail_existing_caller() -> None:
    """These are examples of changes that require a schema-version decision."""
    sample = _caller()["sample_response"]

    missing_required = dict(sample)
    missing_required.pop("request_id")
    assert isinstance(validate_envelope(missing_required), TypedFailure)

    type_change = dict(sample)
    type_change["request_id"] = 123
    assert isinstance(validate_envelope(type_change), TypedFailure)

    enum_change = json.loads(json.dumps(sample))
    enum_change["result"]["status"] = "partial"
    assert isinstance(validate_envelope(enum_change), TypedFailure)
