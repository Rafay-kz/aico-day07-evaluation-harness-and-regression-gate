"""Holdout is measured, never used to tune labels, thresholds, or retrieval."""

from __future__ import annotations

import json
from pathlib import Path

from aico.evals.dataset import HOLOUT_SPLIT, TUNING_SPLITS, iter_holdout_cases, iter_tuning_cases, load_dataset
from aico.evals.day07 import run_evaluation


def test_holdout_cases_are_excluded_from_tuning_helpers() -> None:
    dataset = load_dataset()
    tuning_ids = {case.case_id for case in iter_tuning_cases(dataset)}
    holdout_ids = {case.case_id for case in iter_holdout_cases(dataset)}
    assert holdout_ids
    assert tuning_ids.isdisjoint(holdout_ids)
    for case in dataset.cases:
        if case.split == HOLOUT_SPLIT:
            assert case.split not in TUNING_SPLITS


def test_evaluation_does_not_rewrite_golden_or_thresholds(tmp_path: Path) -> None:
    golden = Path("evals/golden_v1.json").read_text(encoding="utf-8")
    thresholds = Path("evals/thresholds_v1.json").read_text(encoding="utf-8")
    run_evaluation(artifacts_dir=tmp_path / "artifacts", write_reports=True)
    assert Path("evals/golden_v1.json").read_text(encoding="utf-8") == golden
    assert Path("evals/thresholds_v1.json").read_text(encoding="utf-8") == thresholds


def test_holdout_is_reported_separately(tmp_path: Path) -> None:
    run = run_evaluation(artifacts_dir=tmp_path, write_reports=True)
    report = json.loads((tmp_path / "evaluation_report.json").read_text(encoding="utf-8"))
    assert "holdout" in report["by_split"]
    assert report["holdout_separation"]["holdout_excluded_from_tuning"] is True
    assert report["holdout_separation"]["tuning_splits"] == ["train", "development"]
    assert run.by_split["holdout"]["n"] == float(
        sum(1 for case in run.dataset.cases if case.split == "holdout")
    )


def test_update_baseline_does_not_copy_holdout_into_tuning_labels(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    original = json.loads(Path("evals/golden_v1.json").read_text(encoding="utf-8"))
    run_evaluation(
        artifacts_dir=tmp_path / "out",
        baseline_path=baseline,
        update_baseline=True,
        write_reports=False,
    )
    after = json.loads(Path("evals/golden_v1.json").read_text(encoding="utf-8"))
    assert after == original
    holdout_ids = {item["case_id"] for item in original["cases"] if item["split"] == "holdout"}
    written = json.loads(baseline.read_text(encoding="utf-8"))
    assert "metrics" in written
    assert holdout_ids
    assert "labels" not in written
