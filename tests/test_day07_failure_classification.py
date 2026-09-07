"""Every failed case receives exactly one allowed primary type."""

from __future__ import annotations

from aico.evals.failure_classifier import FAILURE_TYPES, classify_failure


def test_adversarial_miss_is_refusal() -> None:
    assert (
        classify_failure(
            {
                "category": "adversarial",
                "expected_outcome": "block",
                "observed_outcome": "answer",
                "retrieval_hit": True,
            }
        )
        == "refusal"
    )


def test_forged_citation_is_citation() -> None:
    assert (
        classify_failure(
            {
                "category": "answerable",
                "expected_outcome": "answer",
                "observed_outcome": "answer",
                "retrieval_hit": True,
                "citation_valid": False,
            }
        )
        == "citation"
    )


def test_missed_sources_are_retrieval() -> None:
    assert (
        classify_failure(
            {
                "category": "answerable",
                "expected_outcome": "answer",
                "observed_outcome": "refuse",
                "retrieval_hit": False,
                "expected_sources": ["DOC-001"],
            }
        )
        == "retrieval"
    )


def test_fact_split_out_of_window_is_chunking() -> None:
    assert (
        classify_failure(
            {
                "category": "answerable",
                "expected_outcome": "answer",
                "observed_outcome": "refuse",
                "retrieval_hit": True,
                "critical_facts_in_retrieved": False,
            }
        )
        == "chunking"
    )


def test_evaluator_error_is_evaluator() -> None:
    assert classify_failure({"evaluator_error": "parse failed"}) == "evaluator"


def test_every_failed_case_from_a_run_has_an_allowed_type() -> None:
    from pathlib import Path

    from aico.evals.day07 import run_evaluation

    run = run_evaluation(artifacts_dir=Path("artifacts/day07/_class"), write_reports=False)
    for row in run.rows:
        if row["passed"]:
            assert row["failure_type"] is None
            continue
        assert row["failure_type"] in FAILURE_TYPES
        assert row["failure_reason"]
