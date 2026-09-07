"""Day 4 semantic rules are separate from contract/schema validation."""

from __future__ import annotations

import json
from pathlib import Path

from aico.contracts.errors import FailureCategory, TypedFailure
from aico.contracts.models import CitedAnswer
from aico.contracts.semantic import validate_semantics
from aico.contracts.validator import parse_and_validate_cited_answer

FIXTURE_PATH = Path("tests/fixtures/day04/structured_output_cases.json")


def _cases() -> dict[str, dict]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return {item["id"]: item for item in payload["cases"]}


def _schema_then_semantic(raw: str) -> tuple[CitedAnswer | TypedFailure, CitedAnswer | TypedFailure]:
    contract = parse_and_validate_cited_answer(raw)
    if isinstance(contract, TypedFailure):
        return contract, contract
    return contract, validate_semantics(contract)


def test_schema_valid_answered_without_citation_fails_semantics() -> None:
    contract, semantic = _schema_then_semantic(_cases()["D04-09"]["raw"])
    assert isinstance(contract, CitedAnswer)
    assert isinstance(semantic, TypedFailure)
    assert semantic.category == FailureCategory.SEMANTIC
    assert semantic.field_path == "citations"
    assert contract.citations == []


def test_schema_valid_insufficient_high_confidence_fails_semantics() -> None:
    contract, semantic = _schema_then_semantic(_cases()["D04-10"]["raw"])
    assert isinstance(contract, CitedAnswer)
    assert isinstance(semantic, TypedFailure)
    assert semantic.category == FailureCategory.SEMANTIC
    assert semantic.field_path == "confidence_label"


def test_schema_and_semantic_failures_are_distinguishable() -> None:
    schema_fail = parse_and_validate_cited_answer(_cases()["D04-04"]["raw"])
    _, semantic_fail = _schema_then_semantic(_cases()["D04-09"]["raw"])
    assert isinstance(schema_fail, TypedFailure)
    assert isinstance(semantic_fail, TypedFailure)
    assert schema_fail.category == FailureCategory.CONTRACT
    assert semantic_fail.category == FailureCategory.SEMANTIC
    assert schema_fail.is_schema_failure()
    assert semantic_fail.is_semantic_failure()
    assert not semantic_fail.is_schema_failure()


def test_duplicate_chunk_ids_fail_s3() -> None:
    raw = json.dumps(
        {
            "schema_version": "1.0",
            "status": "answered",
            "answer": "Policy applies.",
            "citations": [
                {"chunk_id": "CHK-001", "source_file": "DOC-001.md"},
                {"chunk_id": "CHK-001", "source_file": "DOC-002.md"},
            ],
            "confidence_label": "low",
        }
    )
    contract, semantic = _schema_then_semantic(raw)
    assert isinstance(contract, CitedAnswer)
    assert isinstance(semantic, TypedFailure)
    assert "S3" in semantic.message


def test_insufficient_evidence_with_citations_fails_s4() -> None:
    raw = json.dumps(
        {
            "schema_version": "1.0",
            "status": "insufficient_evidence",
            "answer": "INSUFFICIENT_EVIDENCE: not in corpus.",
            "citations": [{"chunk_id": "CHK-001", "source_file": "DOC-001.md"}],
            "confidence_label": "low",
        }
    )
    contract, semantic = _schema_then_semantic(raw)
    assert isinstance(contract, CitedAnswer)
    assert isinstance(semantic, TypedFailure)
    assert "S4" in semantic.message


def test_answered_prefix_mismatch_fails_s5() -> None:
    raw = json.dumps(
        {
            "schema_version": "1.0",
            "status": "answered",
            "answer": "INSUFFICIENT_EVIDENCE: this should not be answered.",
            "citations": [{"chunk_id": "CHK-001", "source_file": "DOC-001.md"}],
            "confidence_label": "medium",
        }
    )
    contract, semantic = _schema_then_semantic(raw)
    assert isinstance(contract, CitedAnswer)
    assert isinstance(semantic, TypedFailure)
    assert "S5" in semantic.message


def test_semantic_validation_does_not_mutate_the_model() -> None:
    contract = parse_and_validate_cited_answer(_cases()["D04-09"]["raw"])
    assert isinstance(contract, CitedAnswer)
    before = contract.model_dump()
    validate_semantics(contract)
    assert contract.model_dump() == before
    assert contract.citations == []
