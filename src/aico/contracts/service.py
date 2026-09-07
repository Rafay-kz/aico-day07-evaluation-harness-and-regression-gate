from __future__ import annotations

import logging

from aico.contracts.errors import FailureCategory, TypedFailure
from aico.contracts.models import SCHEMA_VERSION_V1, CitedAnswer, ResponseEnvelope
from aico.contracts.repair import RepairBudget, is_repairable, request_repair
from aico.contracts.semantic import validate_semantics
from aico.contracts.validator import parse_and_validate_cited_answer
from aico.platform.model_gateway import CancellationToken, ModelGateway

LOGGER = logging.getLogger("aico.contracts.service")


def ingest_cited_answer(
    raw: str,
    *,
    gateway: ModelGateway | None = None,
    allow_repair: bool = True,
    cancellation: CancellationToken | None = None,
) -> CitedAnswer | TypedFailure:
    first = _validate_pipeline(raw)
    if isinstance(first, CitedAnswer):
        return first

    budget = RepairBudget()
    if (
        not allow_repair
        or gateway is None
        or not is_repairable(first)
        or not budget.consume()
    ):
        return first

    repaired_raw = request_repair(gateway, raw, first, cancellation=cancellation)
    second = _validate_pipeline(repaired_raw)
    if isinstance(second, CitedAnswer):
        LOGGER.info("stage=repair outcome=success")
        return second

    LOGGER.info(
        "stage=repair outcome=failure category=%s field_path=%s",
        second.category.value,
        second.field_path or "-",
    )
    # Cap is already consumed. A second consume() cannot succeed.
    if budget.consume():
        raise RuntimeError("repair cap was violated")
    return TypedFailure(
        category=FailureCategory.REPAIR_EXHAUSTED,
        message="repair did not produce a valid contract",
        field_path=second.field_path,
        repairable=False,
        repair_attempted=True,
    )


def wrap_envelope(
    result: CitedAnswer,
    *,
    request_id: str,
    model_alias: str,
    trace_id: str | None = None,
    warning: str | None = None,
) -> ResponseEnvelope:
    return ResponseEnvelope(
        schema_version=SCHEMA_VERSION_V1,
        request_id=request_id,
        result=result,
        model_alias=model_alias,
        trace_id=trace_id,
        warning=warning,
    )


def _validate_pipeline(raw: str) -> CitedAnswer | TypedFailure:
    contract = parse_and_validate_cited_answer(raw)
    if isinstance(contract, TypedFailure):
        return contract
    return validate_semantics(contract)
