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
RETRIEVAL_METRICS = frozenset({"hit_at_k", "mrr"})


@dataclass(frozen=True)
class Thresholds:
    version: str
    dataset_version: str
    k: int
    metrics: dict[str, float]
    by_category: dict[str, dict[str, float]]
    zero_tolerance: bool
    evaluator_zero_tolerance: bool


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    exit_code: int
    reasons: tuple[str, ...]
    safety_failed: bool
    evaluator_failed: bool
    threshold_failures: tuple[str, ...]
    baseline_failures: tuple[str, ...]
    category_failures: tuple[str, ...]


def load_thresholds(path: Path | str | None = None) -> Thresholds:
    target = Path(path) if path is not None else DEFAULT_THRESHOLDS
    payload = json.loads(target.read_text(encoding="utf-8"))
    metrics = {key: float(value) for key, value in payload["metrics"].items()}
    raw_category = payload.get("by_category") or {}
    by_category = {
        str(name): {str(metric): float(value) for metric, value in minima.items()}
        for name, minima in raw_category.items()
    }
    safety = payload.get("safety") or {}
    return Thresholds(
        version=str(payload.get("version") or "thresholds_v1"),
        dataset_version=str(payload.get("dataset_version") or "golden_v1"),
        k=int(payload.get("k") or 5),
        metrics=metrics,
        by_category=by_category,
        zero_tolerance=bool(safety.get("zero_tolerance", True)),
        evaluator_zero_tolerance=bool(safety.get("evaluator_zero_tolerance", True)),
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
    evaluator_failures: int = 0,
    by_category: dict[str, dict[str, float]] | None = None,
) -> GateDecision:
    reasons: list[str] = []
    threshold_failures: list[str] = []
    baseline_failures: list[str] = []
    category_failures: list[str] = []
    safety_failed = thresholds.zero_tolerance and safety_failures > 0
    if safety_failed:
        reasons.append(
            f"safety zero-tolerance: {safety_failures} adversarial case(s) failed"
        )
    evaluator_failed = thresholds.evaluator_zero_tolerance and evaluator_failures > 0
    if evaluator_failed:
        reasons.append(
            f"evaluator zero-tolerance: {evaluator_failures} evaluator failure(s)"
        )

    for name, minimum in thresholds.metrics.items():
        actual = float(candidate_metrics.get(name, 0.0))
        if actual + 1e-12 < minimum:
            detail = f"{name} {actual:.4f} < threshold {minimum:.4f}"
            threshold_failures.append(detail)
            reasons.append(detail)

    for category, minima in thresholds.by_category.items():
        actuals = (by_category or {}).get(category) or {}
        if float(actuals.get("n") or 0.0) <= 0:
            continue
        for metric, minimum in minima.items():
            if metric in RETRIEVAL_METRICS and float(actuals.get("n_retrieval") or 0.0) <= 0:
                continue
            if metric == "groundedness" and float(actuals.get("n_groundedness") or 0.0) <= 0:
                continue
            actual = float(actuals.get(metric, 0.0))
            if actual + 1e-12 < minimum:
                detail = f"{category}.{metric} {actual:.4f} < threshold {minimum:.4f}"
                category_failures.append(detail)
                reasons.append(detail)

    baseline_metrics = (baseline or {}).get("metrics") or {}
    if baseline is not None and baseline.get("reviewed") is False:
        reasons.append("baseline is not reviewed; candidate cannot auto-approve itself")
    for name in BASELINE_METRICS:
        if name not in baseline_metrics:
            continue
        if baseline is not None and baseline.get("reviewed") is False:
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
        evaluator_failed=evaluator_failed,
        threshold_failures=tuple(threshold_failures),
        baseline_failures=tuple(baseline_failures),
        category_failures=tuple(category_failures),
    )


def build_baseline_document(
    *,
    dataset_version: str,
    evaluator_version: str,
    model_alias: str,
    retrieval: dict[str, Any],
    metrics: dict[str, float],
    review_version: str = "baseline_v1",
    reviewed_by: str = "",
    splits_used: tuple[str, ...] = ("train", "development"),
    evaluator_transport: str = "lab",
) -> dict[str, Any]:
    reviewed = bool(reviewed_by.strip())
    return {
        "version": review_version,
        "dataset_version": dataset_version,
        "evaluator_version": evaluator_version,
        "model_alias": model_alias,
        "retrieval": retrieval,
        "metrics": {name: float(metrics[name]) for name in BASELINE_METRICS if name in metrics},
        "reviewed": reviewed,
        "reviewed_by": reviewed_by.strip(),
        "holdout_excluded_from_metrics": True,
        "splits_used": list(splits_used),
        "evaluator_transport": evaluator_transport,
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "notes": (
            "Reviewed baseline. Metrics are train+development only. "
            "Normal evaluation must not overwrite this file. "
            "reviewed=true requires --reviewed-by."
        ),
    }


def write_baseline(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def baseline_fingerprint(path: Path) -> tuple[int, str]:
    payload = path.read_text(encoding="utf-8") if path.is_file() else ""
    return len(payload), payload
