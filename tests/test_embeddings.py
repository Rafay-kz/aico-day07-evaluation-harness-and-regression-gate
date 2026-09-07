"""Provider interface, cosine similarity, and content-hash cache. No network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aico.retrieval.embed import embed_index
from aico.retrieval.embedding_provider import (
    DimensionMismatchError,
    FakeEmbeddingProvider,
    content_hash,
)
from aico.retrieval.ingest import INDEX_FILENAME
from aico.retrieval.vector_index import VectorCache, cosine_similarity


def _write_index(tmp_path: Path, chunks: list[dict], dataset_version: str = "1.0") -> Path:
    index_dir = tmp_path / "index"
    index_dir.mkdir(exist_ok=True)
    payload = {
        "meta": {"ingestion_version": dataset_version, "chunk_count": len(chunks)},
        "chunks": chunks,
    }
    (index_dir / INDEX_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return index_dir


def _chunk(chunk_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "source_file": "t.md",
        "char_start": 0,
        "char_end": len(text),
        "text": text,
        "content_hash": content_hash(text),
    }


def test_cosine_identical_vectors_are_one() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_are_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_dimension_mismatch_raises_and_does_not_truncate() -> None:
    with pytest.raises(DimensionMismatchError, match="pad or truncate"):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


def test_fake_provider_is_deterministic() -> None:
    provider = FakeEmbeddingProvider()
    first = provider.embed(["insolvent supplier"])
    second = provider.embed(["insolvent supplier"])
    assert first == second
    assert len(first[0]) == provider.dimensions


def test_cache_second_run_makes_zero_provider_calls(tmp_path: Path) -> None:
    chunks = [_chunk("a:0:1", "public liability five million"), _chunk("b:0:1", "net thirty days")]
    index_dir = _write_index(tmp_path, chunks)
    out_dir = tmp_path / "vectors"
    provider = FakeEmbeddingProvider()
    first = embed_index(index_dir, out_dir, provider)
    assert first["embedded"] == 2
    assert first["provider_calls"] == 2
    assert provider.calls == 2
    second = embed_index(index_dir, out_dir, provider)
    assert second["cache_hits"] == 2
    assert second["embedded"] == 0
    assert second["provider_calls"] == 0
    assert provider.calls == 2


def test_editing_one_chunk_reembeds_only_that_chunk(tmp_path: Path) -> None:
    chunks = [
        _chunk("keep:0:1", "unchanged payment terms"),
        _chunk("edit:0:1", "original inspection window"),
    ]
    index_dir = _write_index(tmp_path, chunks)
    out_dir = tmp_path / "vectors"
    provider = FakeEmbeddingProvider()
    embed_index(index_dir, out_dir, provider)
    assert provider.calls == 2

    edited = [
        chunks[0],
        _chunk("edit:0:1", "revised inspection window of ten days"),
    ]
    _write_index(tmp_path, edited)
    stats = embed_index(index_dir, out_dir, provider)
    assert stats["cache_hits"] == 1
    assert stats["embedded"] == 1
    assert stats["provider_calls"] == 1
    assert provider.calls == 3


def test_model_alias_change_invalidates_even_when_text_matches(tmp_path: Path) -> None:
    chunks = [_chunk("a:0:1", "same text both times")]
    index_dir = _write_index(tmp_path, chunks)
    out_dir = tmp_path / "vectors"
    first = FakeEmbeddingProvider(model_alias="text-embedding-3-small")
    embed_index(index_dir, out_dir, first)
    second = FakeEmbeddingProvider(model_alias="text-embedding-3-large")
    stats = embed_index(index_dir, out_dir, second)
    assert stats["cache_hits"] == 0
    assert stats["embedded"] == 1
    assert stats["provider_calls"] == 1


def test_cache_entry_stores_required_fields(tmp_path: Path) -> None:
    chunks = [_chunk("a:0:1", "bank verification before first payment")]
    index_dir = _write_index(tmp_path, chunks)
    out_dir = tmp_path / "vectors"
    provider = FakeEmbeddingProvider()
    embed_index(index_dir, out_dir, provider)
    cache = VectorCache.load(out_dir)
    entry = cache.entries["a:0:1"]
    assert entry["chunk_id"] == "a:0:1"
    assert entry["content_hash"] == content_hash("bank verification before first payment")
    assert entry["model_alias"] == provider.model_alias
    assert entry["dimensions"] == provider.dimensions
    assert entry["dataset_version"] == "1.0"
    assert entry["created_at"]
    assert len(entry["vector"]) == provider.dimensions
