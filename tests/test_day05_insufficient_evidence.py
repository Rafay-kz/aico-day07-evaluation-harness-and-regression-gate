from __future__ import annotations

from pathlib import Path

from aico.contracts.errors import FailureCategory, TypedFailure
from aico.contracts.models import Status
from aico.rag.answer_service import GroundedAnswer, InsufficientEvidence
from tests.day05_helpers import answer_cases, cited_answer_json, chunks_from_case, make_service

ARTIFACT = Path("artifacts/day05/insufficient_evidence.md")
INVENTED_DOB = "The synthetic supplier CEO was born on 1 January 1970."


def test_unsupported_question_returns_insufficient_evidence() -> None:
    case = answer_cases()["ANS-002"]
    chunks = chunks_from_case(case)
    raw = cited_answer_json(
        status="insufficient_evidence",
        answer=(
            "INSUFFICIENT_EVIDENCE: available evidence does not support the "
            "synthetic supplier CEO date of birth."
        ),
        citations=[],
        confidence_label="low",
    )
    service, transport, retriever = make_service([raw], chunks)
    result = service.answer(case["question"])
    assert isinstance(result, InsufficientEvidence)
    assert result.cited_answer.status == Status.insufficient_evidence
    assert result.cited_answer.citations == []
    assert result.invented_fact is False
    assert result.invented_citation is False
    assert INVENTED_DOB not in result.explanation
    assert "1970" not in result.explanation
    assert result.retrieved_chunk_ids == ("CHUNK-102",)
    assert retriever.calls == [case["question"]]
    assert transport.chat_calls == 1
    _write_artifact(case["question"], result)


def test_unanswerable_case_invents_no_chunk_id() -> None:
    case = answer_cases()["ANS-002"]
    chunks = chunks_from_case(case)
    raw = cited_answer_json(
        status="answered",
        answer=INVENTED_DOB,
        citations=[{"chunk_id": "CHUNK-999", "source_file": "invented"}],
        confidence_label="high",
    )
    service, _transport, _retriever = make_service([raw], chunks)
    result = service.answer(case["question"])
    assert not isinstance(result, GroundedAnswer)
    assert not isinstance(result, InsufficientEvidence) or result.invented_citation is False
    assert isinstance(result, TypedFailure)
    assert result.category == FailureCategory.CITATION
    assert "CHUNK-999" in result.message


def test_nearest_neighbor_is_not_treated_as_support() -> None:
    case = answer_cases()["ANS-002"]
    chunks = chunks_from_case(case)
    assert "date of birth" not in chunks[0].text.lower()
    assert "ceo" not in chunks[0].text.lower()
    raw = cited_answer_json(
        status="insufficient_evidence",
        answer=(
            "INSUFFICIENT_EVIDENCE: retrieved CHUNK-102 states payment terms, "
            "not a date of birth."
        ),
        citations=[],
        confidence_label="low",
    )
    service, _transport, _retriever = make_service([raw], chunks)
    result = service.answer(case["question"])
    assert isinstance(result, InsufficientEvidence)
    assert result.cited_answer.citations == []


def _write_artifact(question: str, result: InsufficientEvidence) -> None:
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        [
            "# Insufficient-evidence result",
            "",
            "Synthetic lab artifact. No production or personal data.",
            "",
            f"- User question: {question}",
            f"- Retrieved chunk IDs: {', '.join(result.retrieved_chunk_ids)}",
            f"- Insufficient-evidence result: {result.explanation}",
            f"- Status: {result.cited_answer.status.value}",
            f"- Citations produced: {list(result.cited_answer.citations)}",
            f"- Invented fact: {result.invented_fact}",
            f"- Invented citation: {result.invented_citation}",
            "",
            "Confirmation: no unsupported fact and no citation ID were produced.",
            "",
        ]
    )
    ARTIFACT.write_text(body, encoding="utf-8")
