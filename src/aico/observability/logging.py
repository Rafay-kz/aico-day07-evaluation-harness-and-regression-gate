from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from typing import Any

LOGGER = logging.getLogger("aico.observability")

request_id_var: ContextVar[str] = ContextVar("aico_request_id", default="")
correlation_id_var: ContextVar[str] = ContextVar("aico_correlation_id", default="")

_ALLOWED_FIELDS = frozenset(
    {
        "request_id",
        "correlation_id",
        "stage",
        "outcome",
        "latency_ms",
        "error_category",
        "operation",
    }
)

_FORBIDDEN_SUBSTRINGS = (
    "authorization",
    "bearer ",
    "api_key",
    "secret",
)

_events: list[dict[str, Any]] = []


def bind_request_context(*, request_id: str, correlation_id: str) -> None:
    request_id_var.set(request_id)
    correlation_id_var.set(correlation_id)


def reset_log_events() -> None:
    _events.clear()


def get_log_events() -> list[dict[str, Any]]:
    return list(_events)


def log_operation(
    *,
    stage: str,
    outcome: str,
    latency_ms: float | None = None,
    error_category: str | None = None,
    operation: str | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "request_id": request_id_var.get(),
        "correlation_id": correlation_id_var.get(),
        "stage": stage,
        "outcome": outcome,
    }
    if latency_ms is not None:
        event["latency_ms"] = round(latency_ms, 3)
    if error_category is not None:
        event["error_category"] = error_category
    if operation is not None:
        event["operation"] = operation
    _assert_safe(event)
    _events.append(event)
    LOGGER.info("%s", json.dumps(event, sort_keys=True))
    return event


def _assert_safe(event: dict[str, Any]) -> None:
    extra = set(event) - _ALLOWED_FIELDS
    if extra:
        raise ValueError(f"log fields are not in the operational allow-list: {sorted(extra)}")
    blob = json.dumps(event).lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        if token in blob:
            raise ValueError("log event would contain a forbidden sensitive token")
