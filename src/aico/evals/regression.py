"""Threshold comparison, reviewed baseline, and gate verdict."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_THRESHOLDS = Path("evals/thresholds_v1.json")
DEFAULT_BASELINE = Path("evals/baseline_v1.json")
BASELINE_METRICS = (
    "hit_at_k",
    "mrr",
    "citation_validity",
    "refusal_accuracy",
    "groundedness",
    "attack_pass_rate",
)


@dataclass(frozen=True)
class Thresholds:
    version: str
    dataset_version: str
    k: int
    metrics: dict[str, float]
    zero_tolerance: bool


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    exit_code: int
    reasons: tuple[str, ...]
    safety_failed: bool
    threshold_failures: tuple[str, ...]
    baseline_failures: tuple[str, ...]


def load_thresholds(path: Path | str | None = None) -> Thresholds:
    target = Path(path) if path is not None else DEFAULT_THRESHOLDS
    payload = json.loads(target.read_text(encoding="utf-8"))
    metrics = {key: float(value) for key, value in payload["metrics"].items()}
    safety = payload.get("safety") or {}
    return Thresholds(
        version=str(payload.get("version") or "thresholds_v1"),
        dataset_version=str(payload.get("dataset_version") or "golden_v1"),
        k=int(payload.get("k") or 5),
        metrics=metrics,
        zero_tolerance=bool(safety.get("zero_tolerance", True)),
    )


def load_baseline(path: Path | str | None = None) -> dict[str, Any] | None:
    target = Path(path) if path is not None else DEFAULT_BASELINE
    if not target.is_file():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def apply_gate(
    candidate_metrics: dict[str, float],
    thresholds: Thresholds,
    *,
    baseline: dict[str, Any] | None,
    safety_failures: int,
) -> GateDecision:
    reasons: list[str] = []
    threshold_failures: list[str] = []
    baseline_failures: list[str] = []
    safety_failed = thresholds.zero_tolerance and safety_failures > 0
    if safety_failed:
        reasons.append(
            f"safety zero-tolerance: {safety_failures} adversarial case(s) failed"
        )

    for name, minimum in thresholds.metrics.items():
        actual = float(candidate_metrics.get(name, 0.0))
        if actual + 1e-12 < minimum:
            detail = f"{name} {actual:.4f} < threshold {minimum:.4f}"
            threshold_failures.append(detail)
            reasons.append(detail)

    baseline_metrics = (baseline or {}).get("metrics") or {}
    for name in BASELINE_METRICS:
        if name not in baseline_metrics:
            continue
        actual = float(candidate_metrics.get(name, 0.0))
        expected = float(baseline_metrics[name])
        if actual + 1e-12 < expected:
            detail = f"{name} {actual:.4f} < baseline {expected:.4f}"
            baseline_failures.append(detail)
            reasons.append(detail)

    passed = not reasons
    return GateDecision(
        passed=passed,
        exit_code=0 if passed else 1,
        reasons=tuple(reasons),
        safety_failed=safety_failed,
        threshold_failures=tuple(threshold_failures),
        baseline_failures=tuple(baseline_failures),
    )


def build_baseline_document(
    *,
    dataset_version: str,
    evaluator_version: str,
    model_alias: str,
    retrieval: dict[str, Any],
    metrics: dict[str, float],
    review_version: str = "baseline_v1",
) -> dict[str, Any]:
    return {
        "version": review_version,
        "dataset_version": dataset_version,
        "evaluator_version": evaluator_version,
        "model_alias": model_alias,
        "retrieval": retrieval,
        "metrics": {name: float(metrics[name]) for name in BASELINE_METRICS if name in metrics},
        "reviewed": True,
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "notes": "Reviewed baseline. Normal evaluation must not overwrite this file.",
    }


def write_baseline(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def baseline_fingerprint(path: Path) -> tuple[int, str]:
    payload = path.read_text(encoding="utf-8") if path.is_file() else ""
    return len(payload), payload
