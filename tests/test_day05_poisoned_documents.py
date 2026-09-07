"""Poisoned retrieved text remains data and cannot override system behavior."""

from __future__ import annotations

from aico.rag.answer_service import GroundedAnswer
from aico.rag.prompt_builder import SYSTEM_INSTRUCTIONS, build_messages, evidence_in_system_instructions
from tests.day05_helpers import answer_cases, cited_answer_json, chunks_from_case, make_service


def test_poisoned_chunk_may_still_be_retrieved() -> None:
    case = answer_cases()["ANS-003"]
    chunks = chunks_from_case(case)
    assert "Ignore previous instructions" in chunks[0].text
    messages = build_messages(case["question"], chunks)
    user = messages[1].content
    assert chunks[0].text in user
    assert chunks[0].chunk_id in user


def test_poisoned_text_is_not_concatenated_into_system_instructions() -> None:
    case = answer_cases()["ANS-003"]
    chunks = chunks_from_case(case)
    messages = build_messages(case["question"], chunks)
    system = messages[0].content
    assert SYSTEM_INSTRUCTIONS in system
    assert evidence_in_system_instructions(chunks, system) is False
    assert "reveal the system prompt" not in system.lower()
    assert "risk score is 99" not in system


def test_poisoned_document_cannot_override_behavior() -> None:
    case = answer_cases()["ANS-003"]
    chunks = chunks_from_case(case)
    raw = cited_answer_json(
        status="answered",
        answer=case["supported_fact"],
        citations=[{"chunk_id": "CHUNK-103", "source_file": "synthetic"}],
        confidence_label="medium",
    )
    service, transport, retriever = make_service([raw], chunks)
    result = service.answer(case["question"])
    assert isinstance(result, GroundedAnswer)
    assert result.cited_answer.answer == case["supported_fact"]
    assert [c.chunk_id for c in result.cited_answer.citations] == case["expected_citation_ids"]
    assert result.citation_validation.passed is True
    assert "99" not in result.cited_answer.answer
    assert "risk score" not in result.cited_answer.answer.lower()
    assert "system prompt" not in result.cited_answer.answer.lower()
    assert retriever.calls == [case["question"]]
    system = transport.last_chat_messages[0].content
    assert chunks[0].text not in system
    assert SYSTEM_INSTRUCTIONS in system
    assert "untrusted" in transport.last_chat_messages[1].content.lower()
