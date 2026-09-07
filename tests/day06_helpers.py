"""Helpers for Day 6 deterministic API tests. No cloud calls."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from aico.api.app import create_app
from aico.api.dependencies import ApiSettings, AppContainer, TrustedClaims, get_trusted_claims
from aico.api.health import (
    DEGRADED_MODE_POLICY,
    DependencySnapshot,
    HealthService,
)
from aico.observability.logging import reset_log_events
from aico.observability.metrics import reset_metrics
from aico.observability.telemetry import reset_telemetry
from aico.platform.config import testing_config
from aico.platform.fake_transport import FakeTransport
from aico.platform.model_gateway import ModelGateway, TokenUsage, TransportChatResponse
from aico.rag.answer_service import AnswerService, FixedRetriever
from aico.rag.prompt_builder import RetrievedChunk
from aico.security.input_policy import evaluate_policy
from tests.day05_helpers import cited_answer_json

API_CASES = Path("tests/fixtures/day06/api_cases.json")
IDENTITY_CASES = Path("tests/fixtures/day06/identity_claim_cases.json")
HEALTH_CASES = Path("tests/fixtures/day06/dependency_health_cases.json")

VALID_CLAIMS = TrustedClaims(tenant_id="TENANT-SYN-001", user_id="USER-SYN-001")
SUPPORTED_QUESTION = "What payment terms are stated in the synthetic supplier policy?"
SUPPORTED_ANSWER = "Synthetic supplier payment terms are net 30."


class CountingAnswerService:
    def __init__(self, inner: AnswerService) -> None:
        self.inner = inner
        self.calls: list[str] = []

    def answer(self, question: str, cancellation=None, **kwargs):
        self.calls.append(question)
        return self.inner.answer(question, cancellation, **kwargs)


def load_cases(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["id"]: item for item in payload["cases"]}


def supported_chat_json() -> str:
    return cited_answer_json(
        status="answered",
        answer=SUPPORTED_ANSWER,
        citations=[{"chunk_id": "CHUNK-101", "source_file": "synthetic"}],
        confidence_label="high",
    )


class MeteredFakeTransport(FakeTransport):
    def chat(self, messages, *, model_alias, timeout_seconds, cancellation):
        response = super().chat(
            messages,
            model_alias=model_alias,
            timeout_seconds=timeout_seconds,
            cancellation=cancellation,
        )
        return TransportChatResponse(
            text=response.text,
            token_usage=TokenUsage(input_tokens=3, output_tokens=5, total_tokens=8),
        )


def make_container(
    *,
    transport: FakeTransport | None = None,
    max_request_bytes: int = 8192,
    health: HealthService | None = None,
    chat_texts: list[str] | None = None,
) -> tuple[AppContainer, CountingAnswerService, FakeTransport]:
    resolved_transport = transport or MeteredFakeTransport(
        chat_texts=chat_texts or [supported_chat_json()]
    )
    gateway = ModelGateway(
        resolved_transport, testing_config(), sleep=lambda _seconds: None
    )
    retriever = FixedRetriever(
        [
            RetrievedChunk(
                chunk_id="CHUNK-101",
                text=SUPPORTED_ANSWER,
                source_file="synthetic",
            )
        ]
    )
    inner = AnswerService(gateway, retriever, policy=evaluate_policy)
    service = CountingAnswerService(inner)
    probes = health or HealthService(
        {
            "retrieval": lambda: DependencySnapshot("retrieval", "healthy", "ok"),
            "model_gateway": lambda: DependencySnapshot("model_gateway", "healthy", "ok"),
            "configuration": lambda: DependencySnapshot("configuration", "healthy", "ok"),
        }
    )
    container = AppContainer(
        answer_service=service,
        health=probes,
        settings=ApiSettings(max_request_bytes=max_request_bytes),
        gateway=gateway,
        retriever=retriever,
        policy=evaluate_policy,
    )
    return container, service, resolved_transport


def make_client(
    *,
    claims: TrustedClaims | None = VALID_CLAIMS,
    max_request_bytes: int = 8192,
    transport: FakeTransport | None = None,
    health: HealthService | None = None,
    chat_texts: list[str] | None = None,
) -> tuple[TestClient, CountingAnswerService, FakeTransport]:
    reset_telemetry()
    reset_log_events()
    reset_metrics()
    container, service, resolved_transport = make_container(
        transport=transport,
        max_request_bytes=max_request_bytes,
        health=health,
        chat_texts=chat_texts,
    )
    application = create_app(container)
    if claims is not None:
        application.dependency_overrides[get_trusted_claims] = lambda: claims
    client = TestClient(application)
    return client, service, resolved_transport


def ask_headers(**extra: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    headers.update(extra)
    return headers


__all__ = [
    "API_CASES",
    "DEGRADED_MODE_POLICY",
    "HEALTH_CASES",
    "IDENTITY_CASES",
    "SUPPORTED_QUESTION",
    "VALID_CLAIMS",
    "ask_headers",
    "load_cases",
    "make_client",
    "make_container",
]
