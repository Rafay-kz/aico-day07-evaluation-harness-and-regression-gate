from __future__ import annotations

import json
from pathlib import Path

import pytest

from aico.retrieval.chunker import chunk_text
from aico.retrieval.ingest import INDEX_FILENAME, ingest
from aico.retrieval.search import search

DOCUMENTS_DIR = Path("data/documents")
QUERY = "public liability five million pounds"


def test_offset_reconstruction_every_chunk() -> None:
    """original[char_start:char_end] equals the chunk text, for every chunk."""
    for path in sorted(DOCUMENTS_DIR.glob("DOC-*.md")):
        original = path.read_text(encoding="utf-8")
        for chunk in chunk_text(
            original, source_file=path.name, token_size=200, overlap=40
        ):
            assert original[chunk["char_start"] : chunk["char_end"]] == chunk["text"]


def test_overlap_present_on_consecutive_chunks() -> None:
    """The tail of chunk N appears at the head of chunk N+1."""
    text = " ".join(f"Sentence number {i} ends here." for i in range(25))
    chunks = chunk_text(text, source_file="ov.md", token_size=20, overlap=5)
    assert len(chunks) >= 2
    for left, right in zip(chunks, chunks[1:]):
        assert left["char_start"] < right["char_start"] < left["char_end"]
        shared = text[right["char_start"] : left["char_end"]]
        assert shared in left["text"]
        assert shared in right["text"]


def test_unicode_survives_accent_non_latin_and_em_dash() -> None:
    """Accented characters, a non-Latin string and an em dash pass through intact."""
    text = "Café — 東京. The em dash — must stay intact."
    chunks = chunk_text(text, source_file="uni.md", token_size=20, overlap=2)
    blob = " ".join(chunk["text"] for chunk in chunks)
    assert "Café" in blob
    assert "東京" in blob
    assert "—" in blob
    for chunk in chunks:
        assert text[chunk["char_start"] : chunk["char_end"]] == chunk["text"]


def test_empty_and_whitespace_files_produce_zero_chunks(tmp_path: Path) -> None:
    """Empty and whitespace-only files produce zero chunks without crashing."""
    assert chunk_text("", source_file="e.md", token_size=20, overlap=5) == []
    assert chunk_text("  \n\t\n  ", source_file="e.md", token_size=20, overlap=5) == []
    src = tmp_path / "src"
    src.mkdir()
    (src / "empty.md").write_text("", encoding="utf-8")
    (src / "spaces.md").write_text("   \n\n", encoding="utf-8")
    ingest(src, tmp_path / "out", tokens=20, overlap=5)
    payload = json.loads((tmp_path / "out" / INDEX_FILENAME).read_text(encoding="utf-8"))
    assert payload["chunks"] == []


@pytest.mark.parametrize(
    ("tokens", "overlap"),
    [(-1, 0), (0, 0), (10, 10), (10, 11)],
)
def test_invalid_configuration_is_rejected(
    tmp_path: Path, tokens: int, overlap: int
) -> None:
    """Negative tokens, zero tokens and overlap ≥ token size are rejected."""
    with pytest.raises(ValueError):
        chunk_text("hello world", source_file="x.md", token_size=tokens, overlap=overlap)
    with pytest.raises(ValueError):
        ingest(DOCUMENTS_DIR, tmp_path / "out", tokens=tokens, overlap=overlap)


def test_two_identical_runs_same_chunk_ids_and_scores(tmp_path: Path) -> None:
    """Two identical runs produce identical chunk IDs and identical scores."""
    first = tmp_path / "a"
    second = tmp_path / "b"
    ingest(DOCUMENTS_DIR, first, tokens=200, overlap=40)
    ingest(DOCUMENTS_DIR, second, tokens=200, overlap=40)
    hits_a = search(QUERY, first, top_k=5)
    hits_b = search(QUERY, second, top_k=5)
    assert [h["chunk_id"] for h in hits_a] == [h["chunk_id"] for h in hits_b]
    assert [h["score"] for h in hits_a] == [h["score"] for h in hits_b]


def test_unambiguous_query_returns_expected_chunk_first(tmp_path: Path) -> None:
    """A known unambiguous query returns the expected chunk first."""
    ingest(DOCUMENTS_DIR, tmp_path, tokens=300, overlap=50)
    hits = search(QUERY, tmp_path, top_k=5)
    assert hits
    assert hits[0]["rank"] == 1
    text = hits[0]["text"].lower()
    assert "public liability" in text
    assert "five million pounds" in text
