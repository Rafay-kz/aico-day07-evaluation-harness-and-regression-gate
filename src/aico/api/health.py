
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

HealthState = Literal["healthy", "unhealthy", "degraded"]

DEGRADED_MODE_POLICY = (
    "Fail-closed readiness: liveness stays healthy when a remote dependency is "
    "unavailable; readiness is false until retrieval and model_gateway are both "
    "healthy; dependency health is reported separately without secrets or content."
)


@dataclass(frozen=True)
class DependencySnapshot:
    name: str
    status: HealthState
    detail: str


Probe = Callable[[], DependencySnapshot]


class HealthService:
    def __init__(self, probes: dict[str, Probe]) -> None:
        self._probes = dict(probes)

    def liveness(self) -> dict:
        return {
            "status": "healthy",
            "signal": "liveness",
            "policy": DEGRADED_MODE_POLICY,
        }

    def snapshots(self) -> list[DependencySnapshot]:
        return [probe() for probe in self._probes.values()]

    def dependencies(self) -> dict:
        items = self.snapshots()
        overall: HealthState = "healthy"
        if any(item.status == "unhealthy" for item in items):
            overall = "unhealthy"
        elif any(item.status == "degraded" for item in items):
            overall = "degraded"
        return {
            "status": overall,
            "signal": "dependency_health",
            "dependencies": [
                {"name": item.name, "status": item.status, "detail": item.detail}
                for item in items
            ],
        }

    def ready(self) -> tuple[bool, dict]:
        items = self.snapshots()
        is_ready = all(item.status == "healthy" for item in items)
        body = {
            "status": "ready" if is_ready else "not_ready",
            "signal": "readiness",
            "policy": DEGRADED_MODE_POLICY,
            "ready": is_ready,
        }
        return is_ready, body


def retrieval_index_probe(index_path: Path | None = None) -> Probe:
    path = index_path or Path("data/index/chunks.json")

    def _probe() -> DependencySnapshot:
        if path.is_file():
            return DependencySnapshot(
                name="retrieval",
                status="healthy",
                detail="index artifact present",
            )
        return DependencySnapshot(
            name="retrieval",
            status="unhealthy",
            detail="index artifact missing",
        )

    return _probe


def gateway_configured_probe(gateway: object | None) -> Probe:
    def _probe() -> DependencySnapshot:
        if gateway is None:
            return DependencySnapshot(
                name="model_gateway",
                status="unhealthy",
                detail="gateway not configured",
            )
        return DependencySnapshot(
            name="model_gateway",
            status="healthy",
            detail="gateway configured",
        )

    return _probe


def configuration_probe(config_ok: bool = True) -> Probe:
    def _probe() -> DependencySnapshot:
        if config_ok:
            return DependencySnapshot(
                name="configuration",
                status="healthy",
                detail="required configuration present",
            )
        return DependencySnapshot(
            name="configuration",
            status="unhealthy",
            detail="required configuration missing",
        )

    return _probe
