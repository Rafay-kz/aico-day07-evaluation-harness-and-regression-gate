"""Normal evaluation cannot rewrite the reviewed baseline; update is explicit."""

from __future__ import annotations

from pathlib import Path

from aico.evals.day07 import build_parser, run_evaluation
from aico.evals.regression import baseline_fingerprint


def test_normal_evaluation_cannot_rewrite_baseline(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline_v1.json"
    baseline.write_text(Path("evals/baseline_v1.json").read_text(encoding="utf-8"), encoding="utf-8")
    before = baseline_fingerprint(baseline)
    run_evaluation(
        artifacts_dir=tmp_path / "out",
        baseline_path=baseline,
        update_baseline=False,
        write_reports=False,
    )
    assert baseline_fingerprint(baseline) == before


def test_explicit_baseline_update_is_a_separate_path(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline_v1.json"
    run_evaluation(
        artifacts_dir=tmp_path / "out",
        baseline_path=baseline,
        update_baseline=True,
        reviewed_by="Abdul Rafay",
        write_reports=False,
    )
    assert baseline.is_file()
    payload = baseline.read_text(encoding="utf-8")
    assert "dataset_version" in payload
    assert "evaluator_version" in payload
    assert "model_alias" in payload
    assert "retrieval" in payload
    assert '"reviewed": true' in payload
    assert "Abdul Rafay" in payload
    assert "holdout_excluded_from_metrics" in payload
    parser = build_parser()
    default = parser.parse_args([])
    assert default.update_baseline is False
    flagged = parser.parse_args(["--update-baseline", "--reviewed-by", "lead"])
    assert flagged.update_baseline is True
    assert flagged.reviewed_by == "lead"
    ci = Path(".github/workflows/day07-quality-gate.yml").read_text(encoding="utf-8")
    assert "--update-baseline" not in ci


def test_update_without_reviewer_cannot_auto_approve(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline_v1.json"
    run_evaluation(
        artifacts_dir=tmp_path / "out",
        baseline_path=baseline,
        update_baseline=True,
        reviewed_by="",
        write_reports=False,
    )
    import json

    payload = json.loads(baseline.read_text(encoding="utf-8"))
    assert payload["reviewed"] is False
    assert payload["holdout_excluded_from_metrics"] is True
