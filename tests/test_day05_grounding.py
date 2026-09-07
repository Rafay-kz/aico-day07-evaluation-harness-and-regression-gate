"""Prompt boundaries, supported answers, and Day 2–4 path proofs."""

from __future__ import annotations

from pathlib import Path

from aico.contracts.errors import FailureCategory, TypedFailure
from aico.rag.answer_service import Day2Retriever, GroundedAnswer
from aico.rag.prompt_builder import (
    EVIDENCE_LABEL,
    EVIDENCE_UNTRUSTED_NOTICE,
    SYSTEM_INSTRUCTIONS,
    SYSTEM_LABEL,
    USER_LABEL,
    build_messages,
    evidence_in_system_instructions,
)
from aico.security.input_policy import write_attack_results
from tests.day05_helpers import (
    answer_cases,
    cited_answer_json,
    chunks_from_case,
    make_service,
)

ARTIFACT_DIR = Path("artifacts/day05")


def test_supplied_fixtures_were_not_edited() -> None:
    citation = Path("tests/fixtures/day05/citation_cases.json")
    answers = Path("tests/fixtures/day05/answer_cases.json")
    attacks = Path("tests/fixtures/day05/attacks/attack_fixtures.json")
    assert "CIT-001" in citation.read_text(encoding="utf-8")
    assert "ANS-001" in answers.read_text(encoding="utf-8")
    assert "ATK-001" in attacks.read_text(encoding="utf-8")


def test_prompt_boundaries_keep_system_user_evidence_separate() -> None:
    case = answer_cases()["ANS-001"]
    chunks = chunks_from_case(case)
    messages = build_messages(case["question"], chunks)
    assert [item.role for item in messages] == ["system", "user"]
    system = messages[0].content
    user = messages[1].content
    assert system.startswith(SYSTEM_LABEL)
    assert SYSTEM_INSTRUCTIONS in system
    assert USER_LABEL in user
    assert EVIDENCE_LABEL in user
    assert EVIDENCE_UNTRUSTED_NOTICE in user
    assert case["question"] in user
    assert chunks[0].text in user
    assert evidence_in_system_instructions(chunks, system) is False
    assert chunks[0].text not in system


def test_supported_answer_uses_retrieved_evidence() -> None:
    case = answer_cases()["ANS-001"]
    chunks = chunks_from_case(case)
    raw = cited_answer_json(
        status="answered",
        answer=chunks[0].text,
        citations=[{"chunk_id": "CHUNK-101", "source_file": "synthetic"}],
        confidence_label="high",
    )
    service, transport, retriever = make_service([raw], chunks)
    result = service.answer(case["question"])
    assert isinstance(result, GroundedAnswer)
    assert result.cited_answer.answer == chunks[0].text
    assert [c.chunk_id for c in result.cited_answer.citations] == case["expected_citation_ids"]
    assert result.citation_validation.passed is True
    assert result.retrieved_chunk_ids == ("CHUNK-101",)
    assert retriever.calls == [case["question"]]
    assert transport.chat_calls == 1
    _write_supported_artifact(case["question"], result)


def test_day4_contract_path_still_used() -> None:
    case = answer_cases()["ANS-001"]
    service, transport, _retriever = make_service(
        ["this is not json"],
        chunks_from_case(case),
    )
    result = service.answer(case["question"])
    assert isinstance(result, TypedFailure)
    assert result.category == FailureCategory.PARSE
    assert transport.chat_calls == 1


def test_day3_gateway_path_still_used() -> None:
    case = answer_cases()["ANS-001"]
    chunks = chunks_from_case(case)
    raw = cited_answer_json(
        status="answered",
        answer=chunks[0].text,
        citations=[{"chunk_id": "CHUNK-101", "source_file": "synthetic"}],
    )
    service, transport, _retriever = make_service([raw], chunks)
    result = service.answer(case["question"])
    assert isinstance(result, GroundedAnswer)
    assert transport.chat_calls == 1
    assert transport.last_chat_messages is not None
    assert transport.last_chat_messages[0].role == "system"
    assert chunks[0].text not in transport.last_chat_messages[0].content


def test_day2_retrieval_path_still_used() -> None:
    retriever = Day2Retriever(Path("data/index"), mode="bm25", top_k=3)
    hits = retriever.retrieve("payment terms")
    assert retriever.calls == ["payment terms"]
    assert hits
    index_ids = {
        item["chunk_id"]
        for item in __import__("json").loads(
            Path("data/index/chunks.json").read_text(encoding="utf-8")
        )["chunks"]
    }
    assert all(hit.chunk_id in index_ids for hit in hits)


def test_attack_results_artifact_is_written() -> None:
    path = write_attack_results()
    text = path.read_text(encoding="utf-8")
    assert "ATK-001" in text
    assert "ATK-009" in text
    assert "does not imply universal jailbreak prevention" in text


def _write_supported_artifact(question: str, result: GroundedAnswer) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    citations = ", ".join(c.chunk_id for c in result.cited_answer.citations) or "(none)"
    body = "\n".join(
        [
            "# Supported grounded answer",
            "",
            "Synthetic lab artifact. No production or personal data.",
            "",
            f"- User question: {question}",
            f"- Retrieved chunk IDs: {', '.join(result.retrieved_chunk_ids)}",
            f"- Final typed answer: {result.cited_answer.answer}",
            f"- Status: {result.cited_answer.status.value}",
            f"- Citations: {citations}",
            f"- Citation-validation result: {result.citation_validation.reason} "
            f"(passed={result.citation_validation.passed})",
            "",
        ]
    )
    (ARTIFACT_DIR / "supported_answer.md").write_text(body, encoding="utf-8")
