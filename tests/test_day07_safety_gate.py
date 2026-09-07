"""Safety zero-tolerance: one adversarial failure fails the gate."""

from __future__ import annotations

from aico.evals.regression import apply_gate, load_thresholds


def test_one_safety_failure_fails_gate_regardless_of_aggregate_score() -> None:
    thresholds = load_thresholds()
    perfect = {name: 1.0 for name in thresholds.metrics}
    decision = apply_gate(perfect, thresholds, baseline=None, safety_failures=1)
    assert decision.passed is False
    assert decision.exit_code != 0
    assert decision.safety_failed is True
    assert any("zero-tolerance" in reason for reason in decision.reasons)


def test_perfect_safety_with_meeting_thresholds_passes() -> None:
    thresholds = load_thresholds()
    candidate = {name: max(value, 1.0) for name, value in thresholds.metrics.items()}
    decision = apply_gate(candidate, thresholds, baseline=None, safety_failures=0)
    assert decision.passed is True
    assert decision.exit_code == 0
