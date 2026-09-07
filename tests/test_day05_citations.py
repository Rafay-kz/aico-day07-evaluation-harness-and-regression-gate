"""Citation membership: valid pass, forged/mixed fail closed."""

from __future__ import annotations

from aico.contracts.errors import FailureCategory, TypedFailure
from aico.rag.answer_service import GroundedAnswer
from aico.rag.citation_validator import validate_citations
from aico.rag.prompt_builder import RetrievedChunk
from tests.day05_helpers import citation_cases, cited_answer_json, make_service


def test_supplied_citation_cases() -> None:
    for case in citation_cases().values():
        result = validate_citations(case["model_citations"], case["retrieved_context_ids"])
        if case["expected"] == "pass":
            assert result.passed is True
            assert result.forged_ids == ()
            assert result.reason == "pass"
        else:
            assert case["expected"] == "fail_closed"
            assert result.passed is False
            assert result.reason == "fail_closed"
            assert result.forged_ids


def test_valid_citation_passes() -> None:
    case = citation_cases()["CIT-001"]
    result = validate_citations(case["model_citations"], case["retrieved_context_ids"])
    assert result.passed is True
    assert result.cited_ids == ("CHUNK-001",)


def test_forged_citation_fails_closed() -> None:
    case = citation_cases()["CIT-003"]
    result = validate_citations(case["model_citations"], case["retrieved_context_ids"])
    assert result.passed is False
    assert result.forged_ids == ("CHUNK-999",)
    failure = result.to_failure()
    assert failure.category == FailureCategory.CITATION
    assert failure.repairable is False
    assert failure.is_citation_failure() is True


def test_multiple_citations_every_id_is_validated() -> None:
    case = citation_cases()["CIT-002"]
    result = validate_citations(case["model_citations"], case["retrieved_context_ids"])
    assert result.passed is True
    assert set(result.cited_ids) == {"CHUNK-001", "CHUNK-004"}

    mixed = citation_cases()["CIT-004"]
    mixed_result = validate_citations(mixed["model_citations"], mixed["retrieved_context_ids"])
    assert mixed_result.passed is False
    assert mixed_result.forged_ids == ("CHUNK-999",)
    assert "CHUNK-001" in mixed_result.cited_ids


def test_service_does_not_drop_forged_citation_and_keep_answer() -> None:
    chunks = [
        RetrievedChunk(chunk_id="CHUNK-001", text="Policy A.", source_file="synthetic"),
        RetrievedChunk(chunk_id="CHUNK-004", text="Policy B.", source_file="synthetic"),
    ]
    raw = cited_answer_json(
        status="answered",
        answer="Policy A and a forged source.",
        citations=[
            {"chunk_id": "CHUNK-001", "source_file": "synthetic"},
            {"chunk_id": "CHUNK-999", "source_file": "forged"},
        ],
    )
    service, _transport, _retriever = make_service([raw], chunks)
    result = service.answer("What does the policy say?")
    assert isinstance(result, TypedFailure)
    assert result.category == FailureCategory.CITATION
    assert not isinstance(result, GroundedAnswer)
    assert "CHUNK-999" in result.message


def test_format_lookalike_is_not_enough() -> None:
    result = validate_citations(["CHUNK-002"], ["CHUNK-001", "CHUNK-004"])
    assert result.passed is False
    assert result.forged_ids == ("CHUNK-002",)
