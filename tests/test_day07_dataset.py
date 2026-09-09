"""Golden dataset schema, coverage, and uniqueness."""

from __future__ import annotations

from aico.evals.dataset import (
    MIN_CASES,
    REQUIRED_CATEGORIES,
    REQUIRED_SPLITS,
    category_counts,
    load_dataset,
    split_counts,
    validate_payload,
)


def test_dataset_has_required_fields_and_valid_categories_splits() -> None:
    dataset = load_dataset()
    for case in dataset.cases:
        assert case.case_id
        assert case.category in REQUIRED_CATEGORIES
        assert case.split in REQUIRED_SPLITS
        assert case.question
        assert case.answerability in {"answerable", "unanswerable", "ambiguous"}
        assert isinstance(case.expected_sources, tuple)
        assert isinstance(case.critical_facts, tuple)
        assert isinstance(case.prohibited_claims, tuple)


def test_minimum_case_count() -> None:
    dataset = load_dataset()
    assert len(dataset.cases) >= MIN_CASES


def test_all_required_categories_are_represented() -> None:
    counts = category_counts(load_dataset())
    for name in REQUIRED_CATEGORIES:
        assert counts[name] >= 1, name


def test_unique_case_ids() -> None:
    ids = [case.case_id for case in load_dataset().cases]
    assert len(ids) == len(set(ids))


def test_ambiguous_questions_are_not_near_duplicates() -> None:
    questions = [
        case.question.lower()
        for case in load_dataset().cases
        if case.category == "ambiguous"
    ]
    assert len(questions) == len(set(questions))
    reused = [q for q in questions if "supplier is good" in q]
    assert len(reused) <= 1


def test_exactly_one_valid_split_per_case() -> None:
    counts = split_counts(load_dataset())
    assert set(counts) == set(REQUIRED_SPLITS)
    assert all(value >= 1 for value in counts.values())
    for case in load_dataset().cases:
        assert case.split in REQUIRED_SPLITS


def test_validate_payload_rejects_duplicate_ids() -> None:
    payload = {
        "cases": [
            {
                "case_id": "X1",
                "category": "answerable",
                "split": "train",
                "question": "alpha",
                "expected_sources": ["DOC-001"],
                "answerability": "answerable",
                "critical_facts": [],
                "prohibited_claims": [],
            },
            {
                "case_id": "X1",
                "category": "unanswerable",
                "split": "holdout",
                "question": "beta",
                "expected_sources": [],
                "answerability": "unanswerable",
                "critical_facts": [],
                "prohibited_claims": [],
            },
        ]
    }
    errors = validate_payload(payload)
    assert any("duplicate case IDs" in error for error in errors)
