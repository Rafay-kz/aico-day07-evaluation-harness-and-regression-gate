"""Fallback is a policy decision. Silent cross-boundary switching is prohibited."""

from __future__ import annotations

from aico.platform.config import GatewayConfig

CHECK_ORDER = ("provider", "region", "data_boundary", "risk", "budget")


def failed_fallback_checks(config: GatewayConfig, *, budget_ok: bool) -> list[str]:
    """Return failed compatibility check names. Empty means fallback may proceed."""
    if not config.fallback_enabled:
        return ["disabled"]
    failed: list[str] = []
    required = config.require_compatibility
    primary = config.primary
    fallback = config.fallback
    if required.provider and primary.provider != fallback.provider:
        failed.append("provider")
    if required.region and primary.region != fallback.region:
        failed.append("region")
    if required.data_boundary and primary.data_boundary != fallback.data_boundary:
        failed.append("data_boundary")
    if required.risk and primary.risk_class != fallback.risk_class:
        failed.append("risk")
    if required.budget and not budget_ok:
        failed.append("budget")
    return failed
