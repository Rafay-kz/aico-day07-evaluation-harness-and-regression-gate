"""Day 4 bounded repair uses the Model Gateway once and never loops."""

from __future__ import annotations

import json
from pathlib import Path

from aico.contracts.errors import FailureCategory, TypedFailure
from aico.contracts.models import CitedAnswer
from aico.contracts.repair import MAX_REPAIR_ATTEMPTS, RepairBudget, is_repairable
from aico.contracts.service import ingest_cited_answer
from aico.contracts.validator import parse_and_validate_cited_answer
from aico.platform.config import testing_config as make_testing_config
from aico.platform.fake_transport import FakeTransport
from aico.platform.model_gateway import ModelGateway

FIXTURE_PATH = Path("tests/fixtures/day04/structured_output_cases.json")


def _cases() -> dict[str, dict]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return {item["id"]: item for item in payload["cases"]}


def _gateway(chat_texts: list[str]) -> tuple[ModelGateway, FakeTransport]:
    transport = FakeTransport(chat_texts=chat_texts)
    gateway = ModelGateway(transport, make_testing_config(), sleep=lambda _seconds: None)
    return gateway, transport


def test_invalid_then_repaired_valid_is_success() -> None:
    case = _cases()["D04-11"]
    gateway, transport = _gateway([case["fake_repair_response"]])
    result = ingest_cited_answer(case["raw"], gateway=gateway)
    assert isinstance(result, CitedAnswer)
    assert result.answer == "A corrected typed answer."
    assert transport.chat_calls == 1


def test_invalid_then_repaired_invalid_is_typed_failure() -> None:
    case = _cases()["D04-12"]
    gateway, transport = _gateway([case["fake_repair_response"]])
    result = ingest_cited_answer(case["raw"], gateway=gateway)
    assert isinstance(result, TypedFailure)
    assert result.category == FailureCategory.REPAIR_EXHAUSTED
    assert result.repair_attempted is True
    assert result.repairable is False
    assert transport.chat_calls == 1


def test_repair_cap_makes_a_second_attempt_impossible() -> None:
    assert MAX_REPAIR_ATTEMPTS == 1
    budget = RepairBudget()
    assert budget.consume() is True
    assert budget.consume() is False
    assert budget.consume() is False
    assert budget.attempts == 1

    valid = _cases()["D04-01"]["raw"]
    still_invalid = _cases()["D04-12"]["fake_repair_response"]
    gateway, transport = _gateway([still_invalid, valid])
    result = ingest_cited_answer(_cases()["D04-12"]["raw"], gateway=gateway)
    assert isinstance(result, TypedFailure)
    assert result.repair_attempted is True
    assert transport.chat_calls == 1


def test_parse_failure_is_non_repairable_and_does_not_call_gateway() -> None:
    gateway, transport = _gateway([_cases()["D04-01"]["raw"]])
    result = ingest_cited_answer(_cases()["D04-02"]["raw"], gateway=gateway)
    assert isinstance(result, TypedFailure)
    assert result.category == FailureCategory.PARSE
    assert is_repairable(result) is False
    assert transport.chat_calls == 0


def test_repair_request_includes_validation_error() -> None:
    case = _cases()["D04-11"]
    first = parse_and_validate_cited_answer(case["raw"])
    assert isinstance(first, TypedFailure)
    gateway, transport = _gateway([case["fake_repair_response"]])
    ingest_cited_answer(case["raw"], gateway=gateway)
    assert transport.last_chat_messages is not None
    user_text = transport.last_chat_messages[-1].content
    assert first.category.value in user_text
    assert (first.field_path or "answer") in user_text
    assert first.message in user_text


def test_allow_repair_false_does_not_call_gateway() -> None:
    case = _cases()["D04-11"]
    gateway, transport = _gateway([case["fake_repair_response"]])
    result = ingest_cited_answer(case["raw"], gateway=gateway, allow_repair=False)
    assert isinstance(result, TypedFailure)
    assert result.category == FailureCategory.CONTRACT
    assert transport.chat_calls == 0
