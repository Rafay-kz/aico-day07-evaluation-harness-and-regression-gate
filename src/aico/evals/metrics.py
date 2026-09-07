"""Deterministic Day 7 metrics. Hit@K and MRR keep Day 1 semantics."""

from __future__ import annotations

from collections.abc import Sequence

from aico.evals.day01 import _mean
from aico.rag.citation_validator import validate_citations

HIT_K = 5


def source_id(source_file: str) -> str:
    """DOC-004-onboarding-screening.md → DOC-004."""
    name = source_file.rsplit("/", 1)[-1]
    if name.upper().startswith("DOC-") and len(name) >= 7:
        return name[:7].upper()
    return name


def hit_matches_expected(source_file: str, expected_sources: Sequence[str]) -> bool:
    """Match a retrieved hit against labelled sources. Never consults the raw corpus."""
    retrieved_id = source_id(source_file)
    for expected in expected_sources:
        token = expected.strip().upper()
        if not token:
            continue
        if retrieved_id == token or token in source_file.upper():
            return True
    return False


def first_match_rank(hits: Sequence[dict], expected_sources: Sequence[str]) -> int | None:
    for hit in hits:
        if hit_matches_expected(str(hit.get("source_file") or ""), expected_sources):
            return int(hit["rank"])
    return None


def hit_at_k(
    hits: Sequence[dict],
    expected_sources: Sequence[str],
    k: int = HIT_K,
) -> bool:
    """Day 1 Hit@K: a labelled source appears at rank <= K in the retrieved list."""
    rank = first_match_rank(hits, expected_sources)
    return rank is not None and rank <= k


def reciprocal_rank(
    hits: Sequence[dict],
    expected_sources: Sequence[str],
    k: int = HIT_K,
) -> float:
    """Day 1 MRR contribution: 1/rank of the first match, else 0 if missing from top K."""
    rank = first_match_rank(hits, expected_sources)
    if rank is None or rank > k:
        return 0.0
    return 1.0 / rank


def mean_hit_at_k(values: Sequence[bool]) -> float:
    return round(_mean([1.0 if value else 0.0 for value in values]), 4)


def mean_mrr(values: Sequence[float]) -> float:
    return round(_mean(list(values)), 4)


def citation_is_valid(cited_ids: Sequence[str], retrieved_ids: Sequence[str]) -> bool:
    return validate_citations(cited_ids, retrieved_ids).passed


def expected_outcome_for_answerability(answerability: str, *, category: str = "") -> str:
    """Map labels to the outcome refusal-accuracy scores against."""
    if category == "adversarial":
        return "block"
    if answerability == "unanswerable":
        return "refuse"
    if answerability == "ambiguous" or category == "ambiguous":
        return "clarify"
    return "answer"


def refusal_is_correct(expected_outcome: str, observed_outcome: str) -> bool:
    """True when the system outcome matches the labelled answer/refuse/clarify/block."""
    return observed_outcome == expected_outcome


def refusal_accuracy(pairs: Sequence[tuple[str, str]]) -> float:
    """Mean of refusal_is_correct over (expected, observed) pairs. Same averaging as Hit@K."""
    return mean_hit_at_k([refusal_is_correct(expected, observed) for expected, observed in pairs])


def contains_fact(text: str, fact: str) -> bool:
    return _fold(fact) in _fold(text)


def prohibited_claim_hit(text: str, claims: Sequence[str]) -> tuple[str, ...]:
    return tuple(claim for claim in claims if claim and contains_fact(text, claim))


def _fold(text: str) -> str:
    chars: list[str] = []
    for char in text.lower():
        chars.append(char if char.isalnum() or char.isspace() else " ")
    return " ".join("".join(chars).split())
