from __future__ import annotations

import logging

from aico.contracts.errors import FailureCategory, TypedFailure
from aico.platform.model_gateway import (
    CancellationToken,
    ChatMessage,
    ChatRequest,
    ModelGateway,
)

LOGGER = logging.getLogger("aico.contracts.repair")

MAX_REPAIR_ATTEMPTS = 1

_REPAIRABLE = {FailureCategory.CONTRACT, FailureCategory.SEMANTIC}


class RepairBudget:
    """Hard cap: one repair attempt. Callers cannot raise the limit."""

    def __init__(self) -> None:
        self._attempts = 0

    @property
    def attempts(self) -> int:
        return self._attempts

    def consume(self) -> bool:
        if self._attempts >= MAX_REPAIR_ATTEMPTS:
            return False
        self._attempts += 1
        return True

_REPAIR_SYSTEM = (
    "You repair invalid structured output. Return only a JSON object that matches "
    "the cited-answer contract. Required fields: schema_version (exactly \"1.0\"), "
    "status (answered or insufficient_evidence), answer (non-empty string), "
    "citations (list of {chunk_id, source_file} with non-empty strings), "
    "confidence_label (low, medium, or high). Do not add extra fields. "
    "Do not wrap the JSON in markdown. Do not include explanations."
)


def is_repairable(failure: TypedFailure) -> bool:
    if failure.repair_attempted:
        return False
    return failure.category in _REPAIRABLE and failure.repairable


def build_repair_messages(original_raw: str, failure: TypedFailure) -> list[ChatMessage]:
    details = [
        f"failure_category={failure.category.value}",
        f"field_path={failure.field_path or '-'}",
        f"validation_error={failure.message}",
        "Return corrected JSON only.",
        "Invalid output follows:",
        original_raw,
    ]
    return [
        ChatMessage(role="system", content=_REPAIR_SYSTEM),
        ChatMessage(role="user", content="\n".join(details)),
    ]


def request_repair(
    gateway: ModelGateway,
    original_raw: str,
    failure: TypedFailure,
    cancellation: CancellationToken | None = None,
) -> str:
    """Exactly one Model Gateway chat call. Caller must revalidate the result."""
    LOGGER.info(
        "stage=repair attempt=1 category=%s field_path=%s",
        failure.category.value,
        failure.field_path or "-",
    )
    result = gateway.chat(
        ChatRequest(
            messages=build_repair_messages(original_raw, failure),
            cancellation=cancellation,
        )
    )
    return result.text
