"""OpenTelemetry setup. A local in-memory exporter is enough for tests."""

from __future__ import annotations

from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

_exporter: InMemorySpanExporter | None = None
_provider: TracerProvider | None = None


@dataclass
class PipelineStats:
    retrieval_latency_ms: float = 0.0
    gateway_latency_ms: float = 0.0
    token_input: int | None = None
    token_output: int | None = None
    token_total: int | None = None
    retry_count: int = 0
    cache_hit: bool | None = None
    model_alias: str | None = None


def setup_telemetry(*, service_name: str = "aico-api") -> InMemorySpanExporter:
    global _exporter, _provider
    if _provider is not None and _exporter is not None:
        return _exporter
    _exporter = InMemorySpanExporter()
    _provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    _provider.add_span_processor(SimpleSpanProcessor(_exporter))
    trace.set_tracer_provider(_provider)
    return _exporter


def reset_telemetry() -> InMemorySpanExporter:
    exporter = setup_telemetry()
    exporter.clear()
    return exporter


def get_tracer() -> Tracer:
    setup_telemetry()
    return trace.get_tracer("aico")


def get_finished_spans():
    exporter = setup_telemetry()
    return list(exporter.get_finished_spans())


def span_attributes(
    *,
    request_id: str,
    correlation_id: str,
    result_category: str | None = None,
) -> dict[str, str]:
    attrs = {
        "aico.request_id": request_id,
        "aico.correlation_id": correlation_id,
    }
    if result_category is not None:
        attrs["aico.result_category"] = result_category
    return attrs
