"""Logs, metrics, traces, redaction, correlation, and dependency injection."""

from __future__ import annotations

from aico.api.app import create_app
from aico.api.dependencies import AppContainer, ApiSettings, get_trusted_claims
from aico.api.health import DependencySnapshot, HealthService
from aico.observability.logging import get_log_events
from aico.observability.metrics import get_metrics
from aico.observability.telemetry import get_finished_spans, reset_telemetry
from aico.platform.config import testing_config as gateway_testing_config
from aico.platform.fake_transport import FakeTransport
from aico.platform.model_gateway import ModelGateway
from aico.rag.answer_service import AnswerService
from aico.rag.prompt_builder import RetrievedChunk
from aico.security.input_policy import PolicyDecision
from tests.day06_helpers import (
    SUPPORTED_QUESTION,
    VALID_CLAIMS,
    ask_headers,
    make_client,
    supported_chat_json,
)

FORBIDDEN = (
    SUPPORTED_QUESTION,
    "Synthetic supplier payment terms are net 30.",
    "TENANT-SYN-001",
    "USER-SYN-001",
    "CHUNK-101",
)


def test_structured_logs_have_operational_fields_without_sensitive_content() -> None:
    client, _, _ = make_client()
    response = client.post(
        "/ask",
        json={"question": SUPPORTED_QUESTION},
        headers=ask_headers(**{"X-Correlation-ID": "corr-log-1", "X-Request-ID": "req-log-1"}),
    )
    assert response.status_code == 200
    events = get_log_events()
    assert events
    stages = {event["stage"] for event in events}
    assert "api" in stages
    assert "policy" in stages
    assert "retrieval" in stages
    assert "model_gateway" in stages
    assert "validation" in stages
    for event in events:
        assert event["request_id"] == "req-log-1"
        assert event["correlation_id"] == "corr-log-1"
        assert "outcome" in event
        blob = str(event).lower()
        for item in FORBIDDEN:
            assert item.lower() not in blob


def test_metrics_record_latency_tokens_retries_and_cache() -> None:
    client, _, _ = make_client()
    response = client.post("/ask", json={"question": SUPPORTED_QUESTION}, headers=ask_headers())
    assert response.status_code == 200
    metrics = get_metrics()
    assert metrics.request_latency_ms
    assert metrics.retrieval_latency_ms
    assert metrics.gateway_latency_ms
    assert metrics.outcomes["answered"] >= 1
    assert metrics.retry_count_total >= 0
    assert metrics.token_usage_total >= 8
    assert metrics.cache_hits + metrics.cache_misses >= 1


def test_trace_hierarchy_shares_correlation_and_is_redacted() -> None:
    client, _, _ = make_client()
    response = client.post(
        "/ask",
        json={"question": SUPPORTED_QUESTION},
        headers=ask_headers(**{"X-Correlation-ID": "corr-trace-9"}),
    )
    assert response.status_code == 200
    spans = get_finished_spans()
    names = [span.name for span in spans]
    for required in (
        "api",
        "policy",
        "retrieval",
        "model_gateway",
        "validation",
        "response_composition",
    ):
        assert required in names, names
    for span in spans:
        attrs = dict(span.attributes or {})
        assert attrs.get("aico.correlation_id") == "corr-trace-9"
        dumped = str(attrs)
        for item in FORBIDDEN:
            assert item not in dumped
            assert item not in span.name


def test_dependency_injection_replaces_answer_service_gateway_retriever_policy_and_health() -> None:
    reset_telemetry()
    calls: dict[str, int] = {"answer": 0, "policy": 0, "retrieve": 0, "health": 0}

    class FakeRetriever:
        last_cache_hit = False

        def retrieve(self, question: str):
            calls["retrieve"] += 1
            return [
                RetrievedChunk(
                    chunk_id="CHUNK-101",
                    text="Synthetic supplier payment terms are net 30.",
                    source_file="synthetic",
                )
            ]

    def fake_policy(question: str) -> PolicyDecision:
        calls["policy"] += 1
        return PolicyDecision(outcome="allow", category="allow", reason="ok")

    def _count_health(name: str) -> DependencySnapshot:
        calls["health"] += 1
        return DependencySnapshot(name=name, status="healthy", detail="injected")

    transport = FakeTransport(chat_texts=[supported_chat_json()])
    gateway = ModelGateway(transport, gateway_testing_config(), sleep=lambda _seconds: None)
    inner = AnswerService(gateway, FakeRetriever(), policy=fake_policy)

    class InjectedService:
        def answer(self, question: str, cancellation=None, **kwargs):
            calls["answer"] += 1
            return inner.answer(question, cancellation, **kwargs)

    health = HealthService(
        {
            "retrieval": lambda: _count_health("retrieval"),
            "model_gateway": lambda: _count_health("model_gateway"),
            "configuration": lambda: _count_health("configuration"),
        }
    )

    container = AppContainer(
        answer_service=InjectedService(),
        health=health,
        settings=ApiSettings(),
        gateway=gateway,
        retriever=FakeRetriever(),
        policy=fake_policy,
    )
    application = create_app(container)
    application.dependency_overrides[get_trusted_claims] = lambda: VALID_CLAIMS
    from fastapi.testclient import TestClient

    client = TestClient(application)
    asked = client.post("/ask", json={"question": SUPPORTED_QUESTION}, headers=ask_headers())
    ready = client.get("/health/ready")
    assert asked.status_code == 200
    assert ready.status_code == 200
    assert calls["answer"] == 1
    assert calls["policy"] == 1
    assert calls["retrieve"] == 1
    assert calls["health"] >= 1
    assert asked.json()["outcome"] == "answered"
