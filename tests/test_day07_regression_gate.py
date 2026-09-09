"""Regression gate exit codes, reports, controlled weakening, and uv workflow."""

from __future__ import annotations

import json
from pathlib import Path

from aico.evals.day07 import WEAK_TOP_K, main, prove_controlled_regression, run_evaluation
from aico.evals.regression import apply_gate, load_thresholds


def test_failed_threshold_returns_nonzero() -> None:
    thresholds = load_thresholds()
    decision = apply_gate(
        {
            "hit_at_k": 0.0,
            "mrr": 0.0,
            "citation_validity": 0.0,
            "refusal_accuracy": 0.0,
            "groundedness": 0.0,
            "attack_pass_rate": 0.0,
        },
        thresholds,
        baseline=None,
        safety_failures=0,
    )
    assert decision.passed is False
    assert decision.exit_code != 0
    assert decision.threshold_failures


def test_reports_are_generated_from_a_run(tmp_path: Path) -> None:
    run = run_evaluation(artifacts_dir=tmp_path, write_reports=True)
    assert (tmp_path / "evaluation_report.json").is_file()
    assert (tmp_path / "evaluation_report.md").is_file()
    assert (tmp_path / "failure_classification.md").is_file()
    assert (tmp_path / "stability_report.md").is_file()
    payload = json.loads((tmp_path / "evaluation_report.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "evaluation_report.md").read_text(encoding="utf-8")
    assert payload["deterministic"]["hit_at_k"] == run.metrics["hit_at_k"]
    assert payload["model_based"]["groundedness"] == run.metrics["groundedness"]
    assert payload["stability"]["cases"]
    assert payload["stability"]["cases"][0]["groundedness"]["values"]
    assert payload["holdout_separation"]["holdout_excluded_from_release_metrics"] is True
    assert "AI score" not in markdown
    assert "Deterministic checks" in markdown
    assert "Model-based groundedness" in markdown
    assert "holdout" in payload["by_split"]


def test_stability_reporting_records_repeated_results(tmp_path: Path) -> None:
    run = run_evaluation(artifacts_dir=tmp_path, write_reports=True)
    assert run.stability["repeats"] > 1
    assert run.stability["case_ids"]
    for item in run.stability["cases"]:
        assert item["repeats"] > 1
        assert len(item["outcomes"]) == item["repeats"]
        assert "values" in item["groundedness"]


def test_controlled_regression_weakened_retrieval_is_detected(tmp_path: Path) -> None:
    proof = prove_controlled_regression(artifacts_dir=tmp_path)
    assert proof["normal"]["exit_code"] == 0
    assert proof["regressed"]["exit_code"] != 0
    assert proof["restored"]["exit_code"] == 0
    assert proof["passed"] is True
    assert (tmp_path / "controlled_regression.md").is_file()
    assert proof["regressed"]["top_k"] == WEAK_TOP_K


def test_cli_nonzero_on_weakened_retrieval() -> None:
    code = main(["--weaken-retrieval", "--artifacts", "artifacts/day07/_cli_weak"])
    assert code != 0


def test_deterministic_and_model_based_remain_separate(tmp_path: Path) -> None:
    run = run_evaluation(artifacts_dir=tmp_path, write_reports=True)
    assert "hit_at_k" in run.report["deterministic"]
    assert "groundedness" in run.report["model_based"]
    assert "ai_score" not in run.report
    assert "deterministic" in run.report
    assert "model_based" in run.report


def test_uv_project_files_are_the_source_of_truth() -> None:
    assert Path("pyproject.toml").is_file()
    assert Path("uv.lock").is_file()
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "uv sync --frozen" in readme
    assert "uv run ruff check ." in readme
    assert "uv run pytest -q" in readme
    assert "uv run python -m aico.evals.day07" in readme
    workflow = Path(".github/workflows/day07-quality-gate.yml").read_text(encoding="utf-8")
    assert "uv sync --frozen" in workflow
    assert "--update-baseline" not in workflow
    assert "allow_failure" not in workflow
    assert "continue-on-error" not in workflow
