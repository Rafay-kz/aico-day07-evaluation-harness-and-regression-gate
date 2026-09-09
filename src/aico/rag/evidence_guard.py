"""Reject unsupported or instruction-following claims even when citation IDs match."""

from __future__ import annotations

import re

from aico.rag.prompt_builder import RetrievedChunk
from aico.retrieval.bm25 import tokenize_for_search

INSTRUCTION_MARKERS = (
    "ignore previous",
    "ignore all previous",
    "reveal the system",
    "reveal the hidden",
    "system prompt",
    "then state ",
    "you are now",
    "override policy",
    "risk score is",
)


def is_instruction_like(text: str) -> bool:
    folded = text.lower()
    return any(marker in folded for marker in INSTRUCTION_MARKERS)


def factual_evidence_text(text: str) -> str:
    """Drop instruction-like sentences so poisoned retrieval cannot become evidence."""
    kept = [sentence.strip() for sentence in _sentences(text) if sentence.strip() and not is_instruction_like(sentence)]
    return " ".join(kept)


def unsupported_poisoned_claims(answer: str, chunks: list[RetrievedChunk]) -> str | None:
    """Return a reason when the answer is not supported by factual retrieved text."""
    if not answer.strip() or answer.upper().startswith("INSUFFICIENT_EVIDENCE"):
        return None
    if any(is_instruction_like(sentence) for sentence in _sentences(answer)):
        return "answer follows embedded retrieval instructions rather than policy facts"
    evidence = " ".join(factual_evidence_text(chunk.text) for chunk in chunks)
    if not evidence.strip():
        return "no factual evidence remained after stripping embedded instructions"
    for sentence in _sentences(answer):
        if not _claim_supported(sentence, evidence):
            return "answer contains claims not supported by factual retrieved evidence"
    return None


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _claim_supported(claim: str, evidence: str) -> bool:
    folded_claim = " ".join(claim.lower().split())
    folded_evidence = " ".join(evidence.lower().split())
    if folded_claim and folded_claim in folded_evidence:
        return True
    terms = set(tokenize_for_search(claim))
    if len(terms) < 2:
        return folded_claim in folded_evidence
    evidence_terms = set(tokenize_for_search(evidence))
    return (len(terms & evidence_terms) / len(terms)) >= 0.6
