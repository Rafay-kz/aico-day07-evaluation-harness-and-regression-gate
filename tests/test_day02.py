"""RRF fusion, three retrieval modes, and Day 2 scoring. No network."""

from __future__ import annotations

import json
from pathlib import Path

from aico.evals.day02 import SCORE_FLOORS, evaluate_all, write_artifacts
from aico.retrieval.embed import embed_index
from aico.retrieval.embedding_provider import FakeEmbeddingProvider
from aico.retrieval.hybrid import RRF_K, reciprocal_rank_fusion
from aico.retrieval.ingest import INDEX_FILENAME
from aico.retrieval.search import search


def _chunk(chunk_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "source_file": "t.md",
        "char_start": 0,
        "char_end": len(text),
        "text": text,
    }


def _write_index(tmp_path: Path, chunks: list[dict]) -> Path:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    payload = {
        "meta": {"ingestion_version": "1.0", "chunk_count": len(chunks)},
        "chunks": chunks,
    }
    (index_dir / INDEX_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return index_dir


def test_rrf_hand_worked_example_fuses_to_expected_order() -> None:
    """k=60: A ranks c1,c2,c3 and B ranks c2,c3,c1 → fused order c2, c1, c3."""
    chunks = [_chunk("c1", "one"), _chunk("c2", "two"), _chunk("c3", "three")]
    ranking_a = [
        {**chunks[0], "rank": 1, "score": 9.0},
        {**chunks[1], "rank": 2, "score": 4.0},
        {**chunks[2], "rank": 3, "score": 1.0},
    ]
    ranking_b = [
        {**chunks[1], "rank": 1, "score": 0.9},
        {**chunks[2], "rank": 2, "score": 0.5},
        {**chunks[0], "rank": 3, "score": 0.1},
    ]
    fused = reciprocal_rank_fusion([ranking_a, ranking_b], chunks, top_k=3, k=60)
    assert [hit["chunk_id"] for hit in fused] == ["c2", "c1", "c3"]
    expected_c2 = 1.0 / (60 + 2) + 1.0 / (60 + 1)
    assert fused[0]["score"] == expected_c2
    assert fused[0]["rank"] == 1


def test_rrf_does_not_average_raw_scores() -> None:
    chunks = [_chunk("high-bm25", "x"), _chunk("high-cosine", "y")]
    bm25 = [
        {**chunks[0], "rank": 2, "score": 99.0},
        {**chunks[1], "rank": 1, "score": 0.01},
    ]
    vector = [
        {**chunks[1], "rank": 1, "score": 0.99},
        {**chunks[0], "rank": 2, "score": 0.10},
    ]
    fused = reciprocal_rank_fusion([bm25, vector], chunks, top_k=2, k=RRF_K)
    assert fused[0]["chunk_id"] == "high-cosine"
    assert fused[0]["score"] != (99.0 + 0.10) / 2


def test_modes_return_the_same_record_shape(tmp_path: Path) -> None:
    chunks = [
        _chunk("a:0:1", "public liability cover of five million pounds"),
        _chunk("b:0:1", "inspection window of ten working days"),
    ]
    index_dir = _write_index(tmp_path, chunks)
    vectors_dir = tmp_path / "vectors"
    provider = FakeEmbeddingProvider()
    embed_index(index_dir, vectors_dir, provider)
    required = {"rank", "score", "chunk_id", "source_file", "char_start", "char_end", "text"}
    for mode in ("bm25", "vector", "hybrid"):
        hits = search(
            "five million pounds",
            index_dir,
            top_k=2,
            mode=mode,
            vectors_dir=vectors_dir,
            provider=provider,
        )
        assert hits
        assert required <= set(hits[0])
        assert [hit["rank"] for hit in hits] == list(range(1, len(hits) + 1))


def test_each_mode_ranks_independently(tmp_path: Path) -> None:
    chunks = [
        _chunk("lexical", "assign or novate the agreement"),
        _chunk("other", "packaging must be seasonally appropriate"),
    ]
    index_dir = _write_index(tmp_path, chunks)
    vectors_dir = tmp_path / "vectors"
    provider = FakeEmbeddingProvider()
    embed_index(index_dir, vectors_dir, provider)
    bm25_hits = search(
        "assign novate",
        index_dir,
        2,
        mode="bm25",
        vectors_dir=vectors_dir,
        provider=provider,
    )
    vector_hits = search(
        "assign novate",
        index_dir,
        2,
        mode="vector",
        vectors_dir=vectors_dir,
        provider=provider,
    )
    assert bm25_hits[0]["chunk_id"] == "lexical"
    assert bm25_hits[0]["score"] != vector_hits[0]["score"]


def test_rankings_are_deterministic(tmp_path: Path) -> None:
    chunks = [
        _chunk("a:0:1", "net thirty days from a valid invoice"),
        _chunk("b:0:1", "partial delivery will only be accepted"),
    ]
    index_dir = _write_index(tmp_path, chunks)
    vectors_dir = tmp_path / "vectors"
    provider = FakeEmbeddingProvider()
    embed_index(index_dir, vectors_dir, provider)
    for mode in ("bm25", "vector", "hybrid"):
        first = search(
            "valid invoice",
            index_dir,
            2,
            mode=mode,
            vectors_dir=vectors_dir,
            provider=provider,
        )
        second = search(
            "valid invoice",
            index_dir,
            2,
            mode=mode,
            vectors_dir=vectors_dir,
            provider=provider,
        )
        assert [(h["chunk_id"], h["score"], h["rank"]) for h in first] == [
            (h["chunk_id"], h["score"], h["rank"]) for h in second
        ]


def test_day02_eval_writes_assignment_artifacts(tmp_path: Path) -> None:
    chunks = [
        _chunk("hit:0:1", "public liability cover of at least five million pounds"),
        _chunk("noise:0:1", "packaging must be appropriate to the season"),
    ]
    index_dir = _write_index(tmp_path, chunks)
    vectors_dir = tmp_path / "vectors"
    provider = FakeEmbeddingProvider()
    embed_index(index_dir, vectors_dir, provider)
    queries_path = tmp_path / "queries.json"
    queries_path.write_text(
        json.dumps(
            {
                "scoring": {
                    "scored_queries": ["Q01"],
                    "separately_reported_queries": ["Q09"],
                },
                "queries": [
                    {
                        "query_id": "Q01",
                        "text": "public liability five million pounds",
                        "category": "exact_term",
                        "relevant": [
                            {
                                "doc_id": "DOC-004",
                                "anchor": "public liability cover of at least five million pounds",
                            }
                        ],
                    },
                    {
                        "query_id": "Q09",
                        "text": "what interest rate is charged on a late payment",
                        "category": "no_match",
                        "relevant": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "artifacts"
    metrics = evaluate_all(queries_path, index_dir, vectors_dir, provider)
    write_artifacts(metrics, out)
    assert (out / "metrics.json").is_file()
    assert (out / "mode_comparison.md").is_file()
    assert set(metrics["modes"]) == {"bm25", "vector", "hybrid"}
    assert metrics["score_floors"] == SCORE_FLOORS
    report = (out / "mode_comparison.md").read_text(encoding="utf-8")
    assert "RRF" in report
    assert "bm25" in report
