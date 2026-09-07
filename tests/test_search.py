"""Search CLI over an ingest index."""

from __future__ import annotations

from pathlib import Path

import pytest

from aico.retrieval.ingest import ingest
from aico.retrieval.search import search

DOCUMENTS_DIR = Path("data/documents")


def test_search_requires_an_index(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ingest"):
        search("notice period", tmp_path, top_k=5)


def test_corpus_query_ranks_insurance_chunk_first(tmp_path: Path) -> None:
    ingest(DOCUMENTS_DIR, tmp_path, tokens=300, overlap=50)
    hits = search("public liability five million pounds", tmp_path, top_k=5)
    assert hits
    assert hits[0]["rank"] == 1
    blob = hits[0]["text"].lower()
    assert "public liability" in blob
    assert "five million pounds" in blob


def test_search_cli_is_deterministic(tmp_path: Path) -> None:
    ingest(DOCUMENTS_DIR, tmp_path, tokens=300, overlap=50)
    first = search("termination notice period", tmp_path, top_k=5)
    second = search("termination notice period", tmp_path, top_k=5)
    assert [(h["chunk_id"], h["score"]) for h in first] == [
        (h["chunk_id"], h["score"]) for h in second
    ]
    assert [h["rank"] for h in first] == [1, 2, 3, 4, 5]
