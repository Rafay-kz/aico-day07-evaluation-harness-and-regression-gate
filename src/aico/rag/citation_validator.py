from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from aico.contracts.errors import FailureCategory, TypedFailure
from aico.contracts.models import CitedAnswer


@dataclass(frozen=True)
class CitationValidationResult:
    passed: bool
    cited_ids: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    forged_ids: tuple[str, ...]
    reason: str

    def to_failure(self) -> TypedFailure:
        forged = ", ".join(self.forged_ids) or "(none)"
        return TypedFailure(
            category=FailureCategory.CITATION,
            message=(
                "citation validation fail-closed: cited IDs are not a subset of "
                f"retrieved context IDs; forged={forged}"
            ),
            field_path="citations.chunk_id",
            repairable=False,
        )


def cited_ids_of(answer: CitedAnswer) -> tuple[str, ...]:
    return tuple(citation.chunk_id for citation in answer.citations)


def validate_citations(
    cited_ids: Sequence[str],
    retrieved_ids: Sequence[str],
) -> CitationValidationResult:
    retrieved = tuple(retrieved_ids)
    retrieved_set = set(retrieved)
    cited = tuple(cited_ids)
    forged = tuple(chunk_id for chunk_id in cited if chunk_id not in retrieved_set)
    if forged:
        return CitationValidationResult(
            passed=False,
            cited_ids=cited,
            retrieved_ids=retrieved,
            forged_ids=forged,
            reason="fail_closed",
        )
    return CitationValidationResult(
        passed=True,
        cited_ids=cited,
        retrieved_ids=retrieved,
        forged_ids=(),
        reason="pass",
    )


def validate_answer_citations(
    answer: CitedAnswer,
    retrieved_ids: Sequence[str],
) -> CitationValidationResult:
    return validate_citations(cited_ids_of(answer), retrieved_ids)
