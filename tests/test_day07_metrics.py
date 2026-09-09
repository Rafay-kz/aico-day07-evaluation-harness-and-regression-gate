"""Hit@K, MRR, citation, and refusal scoring on known fixtures."""

from __future__ import annotations

from aico.evals.metrics import (
    citation_is_valid,
    contains_fact,
    expected_outcome_for_answerability,
    hit_at_k,
    prohibited_claim_hit,
    reciprocal_rank,
    refusal_accuracy,
    refusal_is_correct,
)


def _hit(rank: int, source_file: str) -> dict:
    return {
        "rank": rank,
        "chunk_id": f"c{rank}",
        "source_file": source_file,
        "text": "placeholder",
        "score": 1.0,
    }


def test_hit_at_k_known_result() -> None:
    hits = [
        _hit(1, "DOC-003-pricing-payment.md"),
        _hit(2, "DOC-001-sourcing-policy.md"),
    ]
    assert hit_at_k(hits, ["DOC-001"], k=5) is True
    assert hit_at_k(hits, ["DOC-001"], k=1) is False
    assert hit_at_k(hits, ["DOC-004"], k=5) is False


def test_mrr_known_reciprocal_rank() -> None:
    hits = [
        _hit(1, "DOC-003-pricing-payment.md"),
        _hit(2, "DOC-004-onboarding-screening.md"),
    ]
    assert reciprocal_rank(hits, ["DOC-004"], k=5) == 0.5
    assert reciprocal_rank(hits, ["DOC-001"], k=5) == 0.0


def test_hit_at_k_does_not_use_raw_corpus_membership() -> None:
    hits = [_hit(1, "DOC-001-sourcing-policy.md")]
    assert hit_at_k(hits, ["DOC-004"], k=5) is False


def test_hit_at_k_requires_relevant_text_not_just_document_id() -> None:
    hits = [
        {
            "rank": 1,
            "chunk_id": "c1",
            "source_file": "DOC-004-onboarding-screening.md",
            "text": "warehouse opening hours and visitor parking only",
            "score": 1.0,
        }
    ]
    assert hit_at_k(hits, ["DOC-004"], k=5) is True
    assert hit_at_k(hits, ["DOC-004"], k=5, anchors=["five million pounds"]) is False


def test_citation_validity_accepts_retrieved_ids_and_rejects_forged() -> None:
    assert citation_is_valid(["a"], ["a", "b"]) is True
    assert citation_is_valid(["forged"], ["a", "b"]) is False


def test_refusal_and_prohibited_claim_helpers() -> None:
    assert contains_fact("cover of five million pounds is required", "five million pounds")
    assert prohibited_claim_hit("net sixty days is standard", ["net sixty days"]) == ("net sixty days",)
    assert prohibited_claim_hit("net thirty days", ["net sixty days"]) == ()


def test_refusal_accuracy_answerable_and_unanswerable_scoring() -> None:
    """Known fixture: two answerable + two unanswerable, one miss of each kind → 0.5."""
    answerable = expected_outcome_for_answerability("answerable")
    unanswerable = expected_outcome_for_answerability("unanswerable")
    assert answerable == "answer"
    assert unanswerable == "refuse"
    assert refusal_is_correct(answerable, "answer") is True
    assert refusal_is_correct(unanswerable, "refuse") is True
    assert refusal_is_correct(unanswerable, "answer") is False
    assert refusal_is_correct(answerable, "refuse") is False
    pairs = [
        (answerable, "answer"),
        (answerable, "refuse"),
        (unanswerable, "refuse"),
        (unanswerable, "answer"),
    ]
    assert refusal_accuracy(pairs) == 0.5
