"""Day 6 telemetry: structured logs, metrics, and OpenTelemetry spans."""

from aico.observability.logging import (
    bind_request_context,
    get_log_events,
    log_operation,
    reset_log_events,
)
from aico.observability.metrics import MetricsRegistry, get_metrics, reset_metrics
from aico.observability.telemetry import (
    PipelineStats,
    get_finished_spans,
    get_tracer,
    reset_telemetry,
    setup_telemetry,
    span_attributes,
)

__all__ = [
    "MetricsRegistry",
    "PipelineStats",
    "bind_request_context",
    "get_finished_spans",
    "get_log_events",
    "get_metrics",
    "get_tracer",
    "log_operation",
    "reset_log_events",
    "reset_metrics",
    "reset_telemetry",
    "setup_telemetry",
    "span_attributes",
]
