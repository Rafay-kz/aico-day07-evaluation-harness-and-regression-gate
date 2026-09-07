"""BM25 ranking tests. Query and document must share one tokenizer."""

from __future__ import annotations

import inspect
import re

from aico.retrieval.bm25 import B, K1, BM25Index, tokenize_for_search


def _chunk(chunk_id: str, text: str, source_file: str = "t.md", start: int = 0) -> dict:
    return {
        "chunk_id": chunk_id,
        "source_file": source_file,
        "char_start": start,
        "char_end": start + len(text),
        "text": text,
    }


def test_k1_and_b_are_named_constants() -> None:
    source = inspect.getsource(BM25Index)
    assert "K1" in source and "B" in source
    assert re.search(r"\b1\.5\b", source) is None
    assert re.search(r"\b0\.75\b", source) is None


def test_query_and_document_tokenise_identically() -> None:
    query = "Net Thirty Days."
    document = "net thirty days."
    assert tokenize_for_search(query) == tokenize_for_search(document)


def test_case_folding_does_not_change_ranking() -> None:
    chunks = [
        _chunk("a:0:1", "Public liability cover of five million pounds."),
        _chunk("b:0:1", "Packaging must be appropriate to the season."),
    ]
    index = BM25Index(chunks)
    upper = index.search("PUBLIC LIABILITY", top_k=1)
    lower = index.search("public liability", top_k=1)
    assert upper[0]["chunk_id"] == lower[0]["chunk_id"] == "a:0:1"
    assert upper[0]["score"] == lower[0]["score"]


def test_punctuation_does_not_block_a_match() -> None:
    chunks = [_chunk("a:0:1", "Standard terms are net thirty days.")]
    hits = BM25Index(chunks).search("thirty days", top_k=1)
    assert hits[0]["chunk_id"] == "a:0:1"
    assert hits[0]["score"] > 0


def test_unambiguous_query_returns_expected_chunk_first() -> None:
    chunks = [
        _chunk("noise:0:1", "Delivery windows and packaging rules for the warehouse."),
        _chunk("hit:0:1", "Every supplier must hold public liability cover of five million pounds."),
        _chunk("other:0:1", "Invoices are paid by bank transfer after validation."),
    ]
    hits = BM25Index(chunks).search("public liability five million pounds", top_k=3)
    assert hits[0]["chunk_id"] == "hit:0:1"
    assert hits[0]["score"] > hits[1]["score"]


def test_ranking_is_deterministic() -> None:
    chunks = [
        _chunk("c:0:1", "Notice period of sixty days written notice."),
        _chunk("a:0:1", "Inspection window of ten working days."),
        _chunk("b:0:1", "Ninety days written notice of withdrawal."),
    ]
    index = BM25Index(chunks)
    first = index.search("written notice", top_k=3)
    second = index.search("written notice", top_k=3)
    assert [(h["chunk_id"], h["score"]) for h in first] == [
        (h["chunk_id"], h["score"]) for h in second
    ]


def test_equal_scores_break_ties_by_chunk_id() -> None:
    text = "alpha unique-term-xyz omega"
    chunks = [
        _chunk("b.md:0:10", text, source_file="b.md"),
        _chunk("a.md:0:10", text, source_file="a.md"),
    ]
    hits = BM25Index(chunks).search("unique-term-xyz", top_k=2)
    assert hits[0]["score"] == hits[1]["score"]
    assert hits[0]["chunk_id"] == "a.md:0:10"
    assert hits[1]["chunk_id"] == "b.md:0:10"


def test_empty_query_or_empty_index() -> None:
    chunks = [_chunk("a:0:1", "hello world")]
    assert BM25Index(chunks).search("   ", top_k=5) == []
    assert BM25Index([]).search("hello", top_k=5) == []
