"""Liveness, readiness, and dependency health stay distinct."""

from __future__ import annotations

from aico.api.health import DEGRADED_MODE_POLICY, DependencySnapshot, HealthService
from tests.day06_helpers import HEALTH_CASES, load_cases, make_client


def _health(retrieval: str, model_gateway: str) -> HealthService:
    def status(name: str, value: str) -> DependencySnapshot:
        mapped = "healthy" if value == "healthy" else "unhealthy"
        return DependencySnapshot(name=name, status=mapped, detail=value)

    return HealthService(
        {
            "retrieval": lambda: status("retrieval", retrieval),
            "model_gateway": lambda: status("model_gateway", model_gateway),
            "configuration": lambda: DependencySnapshot("configuration", "healthy", "ok"),
        }
    )


def test_all_dependencies_healthy() -> None:
    case = load_cases(HEALTH_CASES)["HLT-001"]
    deps = case["dependencies"]
    client, _, _ = make_client(health=_health(deps["retrieval"], deps["model_gateway"]))
    live = client.get("/health/live")
    ready = client.get("/health/ready")
    detail = client.get("/health/dependencies")
    assert live.status_code == 200
    assert live.json()["status"] == "healthy"
    assert live.json()["signal"] == "liveness"
    assert ready.status_code == 200
    assert ready.json()["ready"] is True
    assert detail.json()["status"] == "healthy"
    assert DEGRADED_MODE_POLICY in live.json()["policy"]
    assert DEGRADED_MODE_POLICY in ready.json()["policy"]


def test_model_outage_keeps_process_live_and_matches_fail_closed_readiness() -> None:
    case = load_cases(HEALTH_CASES)["HLT-002"]
    deps = case["dependencies"]
    client, _, _ = make_client(health=_health(deps["retrieval"], deps["model_gateway"]))
    live = client.get("/health/live")
    ready = client.get("/health/ready")
    detail = client.get("/health/dependencies")
    assert live.status_code == 200
    assert live.json()["status"] == "healthy"
    assert detail.json()["status"] in {"degraded", "unhealthy"}
    gateway = next(
        item for item in detail.json()["dependencies"] if item["name"] == "model_gateway"
    )
    assert gateway["status"] != "healthy"
    assert ready.status_code == 503
    assert ready.json()["ready"] is False
    assert ready.json()["signal"] == "readiness"
    assert "secret" not in detail.text.lower()
    assert "prompt" not in detail.text.lower()


def test_retrieval_outage_reported_separately() -> None:
    case = load_cases(HEALTH_CASES)["HLT-003"]
    deps = case["dependencies"]
    client, _, _ = make_client(health=_health(deps["retrieval"], deps["model_gateway"]))
    live = client.get("/health/live")
    ready = client.get("/health/ready")
    detail = client.get("/health/dependencies")
    assert live.status_code == 200
    retrieval = next(
        item for item in detail.json()["dependencies"] if item["name"] == "retrieval"
    )
    assert retrieval["status"] != "healthy"
    assert live.json()["signal"] != ready.json()["signal"]
    assert detail.json()["signal"] == "dependency_health"
    assert ready.json()["ready"] is False
