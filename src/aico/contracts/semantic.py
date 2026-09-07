"""Application meaning checks. Run only after contract/schema validation succeeds."""

from __future__ import annotations

import logging

from aico.contracts.errors import FailureCategory, TypedFailure
from aico.contracts.models import CitedAnswer, ConfidenceLabel, Status

LOGGER = logging.getLogger("aico.contracts.semantic")

INSUFFICIENT_PREFIX = "INSUFFICIENT_EVIDENCE"


def validate_semantics(answer: CitedAnswer) -> CitedAnswer | TypedFailure:
    """Return the same object, or a typed semantic failure. Never mutates the answer."""
    checks = (
        _s1_answered_requires_citation,
        _s2_insufficient_not_high_confidence,
        _s3_unique_chunk_ids,
        _s4_insufficient_has_no_citations,
        _s5_answer_agrees_with_status,
    )
    for check in checks:
        failure = check(answer)
        if failure is not None:
            LOGGER.info(
                "stage=semantic outcome=failure rule=%s field_path=%s",
                failure.message.split(":")[0],
                failure.field_path or "-",
            )
            return failure
    return answer


def _failure(message: str, field_path: str) -> TypedFailure:
    return TypedFailure(
        category=FailureCategory.SEMANTIC,
        message=message,
        field_path=field_path,
        repairable=True,
    )


def _s1_answered_requires_citation(answer: CitedAnswer) -> TypedFailure | None:
    if answer.status == Status.answered and len(answer.citations) < 1:
        return _failure("S1: answered responses require at least one citation", "citations")
    return None


def _s2_insufficient_not_high_confidence(answer: CitedAnswer) -> TypedFailure | None:
    if (
        answer.status == Status.insufficient_evidence
        and answer.confidence_label == ConfidenceLabel.high
    ):
        return _failure(
            "S2: insufficient_evidence must not claim high confidence",
            "confidence_label",
        )
    return None


def _s3_unique_chunk_ids(answer: CitedAnswer) -> TypedFailure | None:
    seen: set[str] = set()
    for citation in answer.citations:
        if citation.chunk_id in seen:
            return _failure("S3: citation chunk_id values must be unique", "citations.chunk_id")
        seen.add(citation.chunk_id)
    return None


def _s4_insufficient_has_no_citations(answer: CitedAnswer) -> TypedFailure | None:
    if answer.status == Status.insufficient_evidence and answer.citations:
        return _failure(
            "S4: insufficient_evidence responses must not contain citations",
            "citations",
        )
    return None


def _s5_answer_agrees_with_status(answer: CitedAnswer) -> TypedFailure | None:
    starts_with_flag = answer.answer.startswith(INSUFFICIENT_PREFIX)
    if answer.status == Status.answered and starts_with_flag:
        return _failure(
            "S5: answered text must not begin with INSUFFICIENT_EVIDENCE",
            "answer",
        )
    if answer.status == Status.insufficient_evidence and not starts_with_flag:
        return _failure(
            "S5: insufficient_evidence text must begin with INSUFFICIENT_EVIDENCE",
            "answer",
        )
    return None
