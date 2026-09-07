"""Task 3 — timeout, cancellation and bounded retry. Fake transport only."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from random import Random

import pytest

from aico.platform.config import testing_config as make_testing_config
from aico.platform.errors import (
    AuthenticationError,
    BadRequestError,
    GatewayTimeout,
    OperationCancelled,
    RateLimitError,
    ServerError,
)
from aico.platform.fake_transport import FakeTransport
from aico.platform.model_gateway import (
    CancellationToken,
    ChatMessage,
    ChatRequest,
    EmbedRequest,
    ModelGateway,
)

CASES = json.loads(Path("fixtures/gateway_cases.json").read_text(encoding="utf-8"))["cases"]
ERROR_TYPES = {
    "timeout": GatewayTimeout,
    "rate_limit": RateLimitError,
    "authentication": AuthenticationError,
    "bad_request": BadRequestError,
    "server_error": ServerError,
}


def _invoke(gateway: ModelGateway, operation: str):
    if operation == "embed":
        return gateway.embed(EmbedRequest(texts=["fixture-text"]))
    return gateway.chat(ChatRequest(messages=[ChatMessage(role="user", content="fixture-text")]))


def _gateway(transport: FakeTransport, **config_kwargs) -> ModelGateway:
    return ModelGateway(
        transport,
        make_testing_config(**config_kwargs),
        sleep=lambda _seconds: None,
    )


def test_timeout_is_normalized() -> None:
    transport = FakeTransport(outcomes=["timeout"])
    gateway = _gateway(transport, max_attempts=1)
    with pytest.raises(GatewayTimeout) as exc:
        gateway.embed(EmbedRequest(texts=["fixture-text"]))
    assert exc.value.category == "timeout"
    assert exc.value.retryable is True


def test_rate_limit_is_normalized() -> None:
    transport = FakeTransport(outcomes=["rate_limit"])
    gateway = _gateway(transport, max_attempts=1)
    with pytest.raises(RateLimitError) as exc:
        gateway.chat(ChatRequest(messages=[ChatMessage(role="user", content="fixture-text")]))
    assert exc.value.category == "rate_limit"


def test_non_retryable_authentication_fails_immediately() -> None:
    transport = FakeTransport(outcomes=["authentication"])
    gateway = _gateway(transport, max_attempts=3)
    with pytest.raises(AuthenticationError):
        gateway.chat(ChatRequest(messages=[ChatMessage(role="user", content="fixture-text")]))
    assert transport.chat_calls == 1


def test_non_retryable_bad_request_fails_immediately() -> None:
    transport = FakeTransport(outcomes=["bad_request"])
    gateway = _gateway(transport, max_attempts=3)
    with pytest.raises(BadRequestError):
        gateway.chat(ChatRequest(messages=[ChatMessage(role="user", content="fixture-text")]))
    assert transport.chat_calls == 1


def test_retryable_failure_stops_at_configured_ceiling() -> None:
    transport = FakeTransport(outcomes=["server_error"])
    gateway = _gateway(transport, max_attempts=3)
    with pytest.raises(ServerError):
        gateway.chat(ChatRequest(messages=[ChatMessage(role="user", content="fixture-text")]))
    assert transport.chat_calls == 3


def test_retry_then_success_reports_retry_count() -> None:
    transport = FakeTransport(outcomes=["rate_limit", "success"])
    gateway = _gateway(transport, max_attempts=3)
    result = gateway.embed(EmbedRequest(texts=["fixture-text"]))
    assert result.metadata.retry_count == 1
    assert transport.embed_calls == 2


def test_exponential_backoff_without_jitter_is_bounded() -> None:
    delays: list[float] = []
    transport = FakeTransport(outcomes=["rate_limit", "rate_limit", "success"])
    gateway = ModelGateway(
        transport,
        make_testing_config(max_attempts=3, base_delay_ms=250, max_delay_ms=2000, jitter=False),
        sleep=delays.append,
    )
    result = gateway.embed(EmbedRequest(texts=["fixture-text"]))
    assert result.metadata.retry_count == 2
    assert delays == [0.25, 0.5]


def test_jitter_keeps_delay_within_ceiling() -> None:
    delays: list[float] = []
    transport = FakeTransport(outcomes=["server_error", "success"])
    gateway = ModelGateway(
        transport,
        make_testing_config(max_attempts=3, base_delay_ms=250, max_delay_ms=2000, jitter=True),
        sleep=delays.append,
        rng=Random(0),
    )
    gateway.embed(EmbedRequest(texts=["fixture-text"]))
    assert len(delays) == 1
    assert 0 <= delays[0] <= 0.25


def test_cancellation_stops_in_flight_operation() -> None:
    transport = FakeTransport(wait_for_cancel=True)
    gateway = _gateway(transport, max_attempts=1)
    token = CancellationToken()
    caught: dict[str, BaseException] = {}

    def run() -> None:
        try:
            gateway.embed(EmbedRequest(texts=["fixture-text"], cancellation=token))
        except BaseException as exc:  # noqa: BLE001 — test captures the gateway error
            caught["exc"] = exc

    worker = threading.Thread(target=run)
    worker.start()
    time.sleep(0.05)
    token.cancel()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert isinstance(caught.get("exc"), OperationCancelled)


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if case["id"] in {"G03", "G04", "G05", "G06", "G07", "G08", "G09"}],
    ids=lambda case: case["id"],
)
def test_fixture_retry_cases(case: dict) -> None:
    sequence = case.get("transport_sequence") or [case["transport_outcome"]]
    transport = FakeTransport(outcomes=list(sequence))
    gateway = _gateway(transport, max_attempts=3)
    if case.get("expected") == "success":
        result = _invoke(gateway, case["operation"])
        assert result.metadata.retry_count == case["expected_retry_count"]
        return
    expected = ERROR_TYPES[case["expected_error"]]
    with pytest.raises(expected) as exc:
        _invoke(gateway, case["operation"])
    assert exc.value.retryable is case.get("retryable", expected().retryable)
    if case.get("expected") == "retry_ceiling_reached":
        calls = transport.embed_calls + transport.chat_calls
        assert calls == 3
