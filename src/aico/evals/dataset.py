"""Day 7 golden dataset load and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_CATEGORIES = (
    "answerable",
    "ambiguous",
    "multi_chunk",
    "synonym_heavy",
    "unanswerable",
    "adversarial",
)
REQUIRED_SPLITS = ("train", "development", "holdout")
TUNING_SPLITS = frozenset({"train", "development"})
HOLOUT_SPLIT = "holdout"
MIN_CASES = 25
REQUIRED_CASE_FIELDS = (
    "case_id",
    "category",
    "split",
    "question",
    "expected_sources",
    "answerability",
    "critical_facts",
    "prohibited_claims",
)
ANSWERABILITY_VALUES = ("answerable", "unanswerable", "ambiguous")
EXPECTED_OUTCOMES = ("answer", "refuse", "clarify", "block")

DEFAULT_DATASET = Path("evals/golden_v1.json")


class DatasetError(ValueError):
    """Golden dataset is not valid for evaluation."""


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    category: str
    split: str
    question: str
    expected_sources: tuple[str, ...]
    answerability: str
    critical_facts: tuple[str, ...]
    prohibited_claims: tuple[str, ...]
    expected_outcome: str

    @property
    def retrieval_scored(self) -> bool:
        return bool(self.expected_sources)

    @property
    def is_adversarial(self) -> bool:
        return self.category == "adversarial"

    @property
    def is_holdout(self) -> bool:
        return self.split == HOLOUT_SPLIT


@dataclass(frozen=True)
class GoldenDataset:
    version: str
    k: int
    cases: tuple[GoldenCase, ...]
    stability_case_ids: tuple[str, ...]
    stability_repeats: int
    stability_justification: str
    payload: dict[str, Any]

    def case_by_id(self, case_id: str) -> GoldenCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(case_id)


def load_dataset(path: Path | str | None = None) -> GoldenDataset:
    target = Path(path) if path is not None else DEFAULT_DATASET
    payload = json.loads(target.read_text(encoding="utf-8"))
    return parse_dataset(payload)


def parse_dataset(payload: dict[str, Any]) -> GoldenDataset:
    errors = validate_payload(payload)
    if errors:
        raise DatasetError("; ".join(errors))
    cases = tuple(_case_from_mapping(item) for item in payload["cases"])
    stability = payload.get("stability") or {}
    return GoldenDataset(
        version=str(payload.get("version") or "golden_v1"),
        k=int(payload.get("k") or 5),
        cases=cases,
        stability_case_ids=tuple(stability.get("case_ids") or ()),
        stability_repeats=int(stability.get("repeats") or 2),
        stability_justification=str(stability.get("justification") or ""),
        payload=payload,
    )


def validate_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        return ["dataset must contain a non-empty cases list"]
    if len(cases) < MIN_CASES:
        errors.append(f"dataset must contain at least {MIN_CASES} cases, got {len(cases)}")

    ids: list[str] = []
    questions: list[str] = []
    categories: set[str] = set()
    splits: set[str] = set()
    for index, item in enumerate(cases):
        if not isinstance(item, dict):
            errors.append(f"case {index} is not an object")
            continue
        missing = [field for field in REQUIRED_CASE_FIELDS if field not in item]
        if missing:
            errors.append(f"case {index} missing fields: {', '.join(missing)}")
            continue
        case_id = str(item["case_id"]).strip()
        if not case_id:
            errors.append(f"case {index} has an empty case_id")
        ids.append(case_id)
        category = str(item["category"])
        split = str(item["split"])
        answerability = str(item["answerability"])
        categories.add(category)
        splits.add(split)
        if category not in REQUIRED_CATEGORIES:
            errors.append(f"{case_id} has invalid category {category!r}")
        if split not in REQUIRED_SPLITS:
            errors.append(f"{case_id} has invalid split {split!r}")
        if answerability not in ANSWERABILITY_VALUES:
            errors.append(f"{case_id} has invalid answerability {answerability!r}")
        outcome = str(item.get("expected_outcome") or "")
        if outcome and outcome not in EXPECTED_OUTCOMES:
            errors.append(f"{case_id} has invalid expected_outcome {outcome!r}")
        if not isinstance(item["expected_sources"], list):
            errors.append(f"{case_id} expected_sources must be a list")
        if not isinstance(item["critical_facts"], list):
            errors.append(f"{case_id} critical_facts must be a list")
        if not isinstance(item["prohibited_claims"], list):
            errors.append(f"{case_id} prohibited_claims must be a list")
        question = str(item["question"]).strip()
        if not question:
            errors.append(f"{case_id} has an empty question")
        questions.append(_fold_question(question))

    if len(ids) != len(set(ids)):
        dupes = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        errors.append(f"duplicate case IDs: {', '.join(dupes)}")
    if len(questions) != len(set(questions)):
        errors.append("duplicate questions detected; cases must not leak across splits")
    missing_categories = [name for name in REQUIRED_CATEGORIES if name not in categories]
    if missing_categories:
        errors.append(f"missing required categories: {', '.join(missing_categories)}")
    missing_splits = [name for name in REQUIRED_SPLITS if name not in splits]
    if missing_splits:
        errors.append(f"missing required splits: {', '.join(missing_splits)}")
    return errors


def iter_tuning_cases(dataset: GoldenDataset) -> tuple[GoldenCase, ...]:
    """Cases that may inform local development. Holdout is excluded."""
    return tuple(case for case in dataset.cases if case.split in TUNING_SPLITS)


def iter_holdout_cases(dataset: GoldenDataset) -> tuple[GoldenCase, ...]:
    return tuple(case for case in dataset.cases if case.split == HOLOUT_SPLIT)


def split_counts(dataset: GoldenDataset) -> dict[str, int]:
    counts = {name: 0 for name in REQUIRED_SPLITS}
    for case in dataset.cases:
        counts[case.split] += 1
    return counts


def category_counts(dataset: GoldenDataset) -> dict[str, int]:
    counts = {name: 0 for name in REQUIRED_CATEGORIES}
    for case in dataset.cases:
        counts[case.category] += 1
    return counts


def _case_from_mapping(item: dict[str, Any]) -> GoldenCase:
    outcome = str(item.get("expected_outcome") or _default_outcome(item))
    return GoldenCase(
        case_id=str(item["case_id"]),
        category=str(item["category"]),
        split=str(item["split"]),
        question=str(item["question"]),
        expected_sources=tuple(str(value) for value in item["expected_sources"]),
        answerability=str(item["answerability"]),
        critical_facts=tuple(str(value) for value in item["critical_facts"]),
        prohibited_claims=tuple(str(value) for value in item["prohibited_claims"]),
        expected_outcome=outcome,
    )


def _default_outcome(item: dict[str, Any]) -> str:
    category = str(item.get("category") or "")
    answerability = str(item.get("answerability") or "")
    if category == "adversarial":
        return "block"
    if answerability == "unanswerable":
        return "refuse"
    if answerability == "ambiguous" or category == "ambiguous":
        return "clarify"
    return "answer"


def _fold_question(question: str) -> str:
    return " ".join(question.lower().split())
