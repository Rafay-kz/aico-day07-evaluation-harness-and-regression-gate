"""Primary failure type for each failed Day 7 case."""

from __future__ import annotations

from typing import Any

FAILURE_TYPES = (
    "chunking",
    "retrieval",
    "prompt",
    "citation",
    "refusal",
    "evaluator",
)


def classify_failure(observation: dict[str, Any]) -> str:
    """Return exactly one allowed primary type."""
    if observation.get("evaluator_error"):
        return "evaluator"
    category = observation.get("category")
    outcome = observation.get("observed_outcome")
    expected = observation.get("expected_outcome")
    retrieval_hit = bool(observation.get("retrieval_hit"))
    citation_valid = observation.get("citation_valid")
    fact_in_retrieved = observation.get("critical_facts_in_retrieved")

    if category == "adversarial" and expected == "block" and outcome != "block":
        return "refusal"
    if citation_valid is False:
        return "citation"
    if expected in {"refuse", "clarify", "block"} and outcome == "answer":
        return "refusal"
    if expected == "answer" and outcome in {"refuse", "clarify", "block"}:
        if not retrieval_hit:
            return "retrieval"
        if fact_in_retrieved is False:
            return "chunking"
        return "prompt"
    if expected == "answer" and outcome == "answer":
        if not retrieval_hit:
            return "retrieval"
        if observation.get("prohibited_hits"):
            return "prompt"
        if observation.get("missing_critical_facts"):
            if fact_in_retrieved is False:
                return "chunking"
            return "prompt"
        if observation.get("groundedness_score", 1.0) < 0.5:
            return "prompt"
    if not retrieval_hit and observation.get("expected_sources"):
        return "retrieval"
    return "prompt"


def allowed_type(value: str) -> bool:
    return value in FAILURE_TYPES
