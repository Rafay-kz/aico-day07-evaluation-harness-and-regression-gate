"""Task 4 — routing policy and safe fallback. Fake transport only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aico.platform.config import testing_config as make_testing_config
from aico.platform.errors import FallbackBlockedError, ServerError
from aico.platform.fake_transport import FAKE_CHAT_TEXT, FakeTransport
from aico.platform.model_gateway import ChatMessage, ChatRequest, ModelGateway

CASES = json.loads(Path("fixtures/gateway_cases.json").read_text(encoding="utf-8"))["cases"]
FALLBACK_CASES = [case for case in CASES if case["id"].startswith("G1")]


def _chat(gateway: ModelGateway):
    return gateway.chat(ChatRequest(messages=[ChatMessage(role="user", content="fixture-text")]))


def _config_from_policy(policy: dict, **kwargs):
    return make_testing_config(
        fallback_enabled=True,
        max_attempts=1,
        fallback_provider=(
            "microsoft-foundry" if policy["provider_compatible"] else "other-provider"
        ),
        fallback_region="test-region" if policy["region_compatible"] else "other-region",
        fallback_data_boundary=(
            "test-boundary" if policy["data_boundary_compatible"] else "other-boundary"
        ),
        fallback_risk_class="standard" if policy["risk_compatible"] else "elevated",
        budget_available=policy["budget_compatible"],
        **kwargs,
    )


def test_allowed_route_proceeds() -> None:
    transport = FakeTransport()
    gateway = ModelGateway(
        transport,
        make_testing_config(fallback_enabled=False),
        sleep=lambda _seconds: None,
    )
    result = _chat(gateway)
    assert result.text == FAKE_CHAT_TEXT
    assert transport.chat_calls == 1


def test_compatible_fallback_is_used_after_primary_failure() -> None:
    primary = FakeTransport(outcomes=["server_error"])
    fallback = FakeTransport()
    gateway = ModelGateway(
        primary,
        make_testing_config(fallback_enabled=True, max_attempts=1),
        fallback_transport=fallback,
        sleep=lambda _seconds: None,
    )
    result = _chat(gateway)
    assert result.text == FAKE_CHAT_TEXT
    assert primary.chat_calls == 1
    assert fallback.chat_calls == 1


def test_fallback_disabled_surfaces_primary_error() -> None:
    primary = FakeTransport(outcomes=["server_error"])
    fallback = FakeTransport()
    gateway = ModelGateway(
        primary,
        make_testing_config(fallback_enabled=False, max_attempts=1),
        fallback_transport=fallback,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(ServerError):
        _chat(gateway)
    assert fallback.chat_calls == 0


def test_provider_mismatch_blocks_fallback() -> None:
    primary = FakeTransport(outcomes=["server_error"])
    fallback = FakeTransport()
    gateway = ModelGateway(
        primary,
        make_testing_config(
            fallback_enabled=True,
            max_attempts=1,
            fallback_provider="other-provider",
        ),
        fallback_transport=fallback,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(FallbackBlockedError) as exc:
        _chat(gateway)
    assert "provider" in exc.value.failed_checks
    assert fallback.chat_calls == 0


@pytest.mark.parametrize("case", FALLBACK_CASES, ids=lambda case: case["id"])
def test_fixture_fallback_cases(case: dict) -> None:
    policy = case["fallback_policy"]
    primary = FakeTransport(outcomes=[case["primary_outcome"]])
    fallback = FakeTransport()
    gateway = ModelGateway(
        primary,
        _config_from_policy(policy),
        fallback_transport=fallback,
        sleep=lambda _seconds: None,
    )
    if case["expected"] == "fallback_allowed":
        result = _chat(gateway)
        assert result.text == FAKE_CHAT_TEXT
        assert fallback.chat_calls == 1
        return
    with pytest.raises(FallbackBlockedError) as exc:
        _chat(gateway)
    assert fallback.chat_calls == 0
    expected_check = {
        "G11": "region",
        "G12": "data_boundary",
        "G13": "risk",
        "G14": "budget",
    }[case["id"]]
    assert expected_check in exc.value.failed_checks
