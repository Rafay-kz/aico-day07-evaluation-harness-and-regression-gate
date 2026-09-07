"""Eval matching, metrics, and no_match separation."""

from __future__ import annotations

import json
from pathlib import Path

from aico.evals.day01 import (
    SCORE_FLOOR,
    chunk_matches_anchor,
    evaluate_chunks,
    normalise,
    run_comparison,
    run_eval,
)


def test_normalise_follows_assignment_rule() -> None:
    assert normalise("Net Thirty Days.") == "net thirty days"
    assert normalise("  five   million\tpounds ") == "five million pounds"
    assert normalise("assign-or-novate") == "assign or novate"


def test_anchor_match_is_normalised_substring() -> None:
    chunk = "Every supplier must hold public liability cover of at least five million pounds."
    assert chunk_matches_anchor(chunk, "public liability cover of at least five million pounds")
    assert not chunk_matches_anchor(chunk, "may not assign or novate the agreement")


def test_hit_at_1_hit_at_5_and_mrr_on_fixture() -> None:
    chunks = [
        {
            "chunk_id": "hit:0:1",
            "source_file": "a.md",
            "char_start": 0,
            "char_end": 10,
            "text": "public liability cover of at least five million pounds is required.",
        },
        {
            "chunk_id": "noise:0:1",
            "source_file": "b.md",
            "char_start": 0,
            "char_end": 10,
            "text": "packaging must be appropriate to the season.",
        },
    ]
    queries = [
        {
            "query_id": "Q01",
            "text": "public liability five million pounds",
            "category": "exact_term",
            "relevant": [
                {"doc_id": "DOC-004", "anchor": "public liability cover of at least five million pounds"}
            ],
        },
        {
            "query_id": "Q05",
            "text": "hand the agreement over",
            "category": "synonym_poor",
            "relevant": [{"doc_id": "DOC-002", "anchor": "may not assign or novate the agreement"}],
        },
        {
            "query_id": "Q09",
            "text": "what interest rate is charged on a late payment",
            "category": "no_match",
            "relevant": [],
        },
    ]
    result = evaluate_chunks(queries, chunks, score_floor=100.0)
    assert result["scored_queries"]["n"] == 2
    assert result["scored_queries"]["hit_at_1"] == 0.5
    assert result["scored_queries"]["hit_at_5"] == 0.5
    assert result["scored_queries"]["mrr"] == 0.5
    assert result["no_match"]["n"] == 1
    assert result["no_match"]["correct_count"] == 1


def test_multi_chunk_full_hit_needs_every_anchor() -> None:
    chunks = [
        {
            "chunk_id": "a:0:1",
            "source_file": "DOC-001.md",
            "char_start": 0,
            "char_end": 20,
            "text": "the supplier is entitled to ninety days written notice of withdrawal from the list.",
        },
        {
            "chunk_id": "b:0:1",
            "source_file": "DOC-002.md",
            "char_start": 0,
            "char_end": 20,
            "text": "this notice period supersedes any notice period stated in the sourcing policy.",
        },
    ]
    query = {
        "query_id": "Q07",
        "text": "ninety days written notice supersedes sourcing policy",
        "category": "multi_chunk",
        "relevant": [
            {"doc_id": "DOC-001", "anchor": "ninety days written notice of withdrawal"},
            {"doc_id": "DOC-002", "anchor": "supersedes any notice period stated in the sourcing policy"},
        ],
    }
    full = evaluate_chunks([query], chunks)
    assert full["per_query"][0]["full_hit"] is True
    half = evaluate_chunks([query], chunks[:1])
    assert half["per_query"][0]["hit_at_5"] is True
    assert half["per_query"][0]["full_hit"] is False


def test_no_match_not_folded_into_scored_average() -> None:
    chunks = [
        {
            "chunk_id": "a:0:1",
            "source_file": "a.md",
            "char_start": 0,
            "char_end": 10,
            "text": "late payment interest under statutory provisions.",
        }
    ]
    queries = [
        {
            "query_id": "Q01",
            "text": "late payment interest",
            "category": "exact_term",
            "relevant": [{"doc_id": "DOC-003", "anchor": "late payment interest under statutory provisions"}],
        },
        {
            "query_id": "Q09",
            "text": "late payment interest rate",
            "category": "no_match",
            "relevant": [],
        },
    ]
    result = evaluate_chunks(queries, chunks, score_floor=0.0)
    assert result["scored_queries"]["hit_at_1"] == 1.0
    assert result["no_match"]["correct_count"] == 0
    assert result["scored_queries"]["n"] == 1


def test_run_eval_on_corpus_index(tmp_path: Path) -> None:
    from aico.retrieval.ingest import ingest

    ingest(Path("data/documents"), tmp_path, tokens=300, overlap=50)
    result = run_eval(Path("data/evals/day01_queries.json"), tmp_path)
    assert result["scored_queries"]["n"] == 8
    assert result["no_match"]["n"] == 2
    assert result["score_floor"] == SCORE_FLOOR
    ids = [row["query_id"] for row in result["per_query"]]
    assert ids == [f"Q{i:02d}" for i in range(1, 11)]


def test_comparison_writes_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "artifacts"
    metrics = run_comparison(
        Path("data/evals/day01_queries.json"),
        Path("data/documents"),
        out,
    )
    assert (out / "chunks_200_40.json").is_file()
    assert (out / "chunks_400_80.json").is_file()
    assert (out / "metrics.json").is_file()
    assert (out / "retrieval_report.md").is_file()
    disk = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert disk["winner"] == metrics["winner"]
    assert "Q09" in disk["configurations"]["200_40"]["no_match"]["queries"]
    report = (out / "retrieval_report.md").read_text(encoding="utf-8")
    assert "no_match" in report
    assert "Winner" in report
    a = json.loads((out / "chunks_200_40.json").read_text(encoding="utf-8"))
    chunk = a["chunks"][0]
    source = (Path("data/documents") / chunk["source_file"]).read_text(encoding="utf-8")
    assert source[chunk["char_start"] : chunk["char_end"]] == chunk["text"]
