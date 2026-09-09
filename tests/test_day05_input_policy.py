"""Deterministic input policy over the supplied attack corpus."""

from __future__ import annotations

from aico.rag.answer_service import Blocked, Clarify, GroundedAnswer
from aico.security.input_policy import evaluate_policy, run_attack_fixtures, write_attack_results
from aico.security.normalization import normalize
from tests.day05_helpers import (
    attack_fixtures,
    cited_answer_json,
    chunks_from_case,
    make_service,
    answer_cases,
)


def test_all_supplied_attack_fixtures_match_expected_outcomes() -> None:
    rows = run_attack_fixtures()
    assert len(rows) >= 8
    failed = [row for row in rows if not row["pass"]]
    assert failed == [], failed
    write_attack_results(rows)


def test_instruction_override_is_blocked() -> None:
    fixture = attack_fixtures()["ATK-001"]
    decision = evaluate_policy(fixture["input"])
    assert decision.outcome == "block"
    service, transport, retriever = make_service(["unused"], [])
    result = service.answer(fixture["input"])
    assert isinstance(result, Blocked)
    assert result.policy.outcome == "block"
    assert transport.chat_calls == 0
    assert retriever.calls == []


def test_role_escalation_is_blocked() -> None:
    fixture = attack_fixtures()["ATK-002"]
    assert evaluate_policy(fixture["input"]).outcome == "block"
    service, transport, retriever = make_service(["unused"], [])
    result = service.answer(fixture["input"])
    assert isinstance(result, Blocked)
    assert transport.chat_calls == 0
    assert retriever.calls == []


def test_tool_coercion_is_blocked() -> None:
    fixture = attack_fixtures()["ATK-004"]
    assert evaluate_policy(fixture["input"]).outcome == "block"


def test_system_prompt_extraction_is_blocked() -> None:
    fixture = attack_fixtures()["ATK-003"]
    assert evaluate_policy(fixture["input"]).outcome == "block"


def test_citation_forgery_request_is_blocked() -> None:
    fixture = attack_fixtures()["ATK-005"]
    assert evaluate_policy(fixture["input"]).outcome == "block"


def test_obfuscated_attack_is_normalized_before_policy() -> None:
    fixture = attack_fixtures()["ATK-006"]
    raw = fixture["input"]
    assert "ignore" not in raw.lower()
    normalized = normalize(raw)
    assert "IGNORE" in normalized.upper()
    assert "previous" in normalized.lower()
    assert evaluate_policy(raw).outcome == "block"
    service, transport, retriever = make_service(["unused"], [])
    result = service.answer(raw)
    assert isinstance(result, Blocked)
    assert transport.chat_calls == 0
    assert retriever.calls == []


def test_benign_question_is_not_needlessly_rewritten() -> None:
    fixture = attack_fixtures()["ATK-007"]
    assert normalize(fixture["input"]) == fixture["input"].strip()
    assert evaluate_policy(fixture["input"]).outcome == "allow"


def test_clarification_case() -> None:
    fixture = attack_fixtures()["ATK-008"]
    assert evaluate_policy(fixture["input"]).outcome == "clarify"
    service, transport, retriever = make_service(["unused"], [])
    result = service.answer(fixture["input"])
    assert isinstance(result, Clarify)
    assert result.policy.outcome == "clarify"
    assert transport.chat_calls == 0
    assert retriever.calls == []


def test_quoted_poisoned_text_as_data_is_allowed() -> None:
    fixture = attack_fixtures()["ATK-009"]
    assert evaluate_policy(fixture["input"]).outcome == "allow"


def test_policy_is_deterministic() -> None:
    for fixture in attack_fixtures().values():
        first = evaluate_policy(fixture["input"])
        second = evaluate_policy(fixture["input"])
        assert first == second
        assert first.outcome == fixture["expected"]


def test_policy_does_not_call_the_model() -> None:
    case = answer_cases()["ANS-001"]
    raw = cited_answer_json(
        status="answered",
        answer=case["retrieved"][0]["text"],
        citations=[{"chunk_id": "CHUNK-101", "source_file": "synthetic"}],
    )
    service, transport, retriever = make_service([raw], chunks_from_case(case))
    blocked = service.answer(attack_fixtures()["ATK-001"]["input"])
    assert isinstance(blocked, Blocked)
    assert transport.chat_calls == 0
    assert retriever.calls == []
    allowed = service.answer(case["question"])
    assert isinstance(allowed, GroundedAnswer)
    assert transport.chat_calls == 1
