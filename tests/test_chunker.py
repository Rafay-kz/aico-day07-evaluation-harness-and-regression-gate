"""Chunker contract tests.

The offset reconstruction test is the source of truth: it was written before
the chunker. Citation spans later depend on this exact equality.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from aico.retrieval.chunker import chunk_text

DOCUMENTS_DIR = Path("data/documents")

REQUIRED_FIELDS = {
    "chunk_id",
    "source_file",
    "char_start",
    "char_end",
    "token_count",
    "content_hash",
    "ingestion_version",
    "section",
    "text",
}

CORPUS_CONFIGS = [(200, 40), (400, 80), (50, 10)]


def _all_corpus_chunks(token_size: int, overlap: int) -> list[tuple[str, str, dict]]:
    rows = []
    for path in sorted(DOCUMENTS_DIR.glob("DOC-*.md")):
        text = path.read_text(encoding="utf-8")
        for chunk in chunk_text(
            text, source_file=path.name, token_size=token_size, overlap=overlap
        ):
            rows.append((path.name, text, chunk))
    return rows


# --- written before the chunker: this is the real test ---


@pytest.mark.parametrize(("token_size", "overlap"), CORPUS_CONFIGS)
def test_offset_reconstruction_on_corpus(token_size: int, overlap: int) -> None:
    rows = _all_corpus_chunks(token_size, overlap)
    assert rows, "expected chunks from the supplied documents"
    for _name, original, chunk in rows:
        assert original[chunk["char_start"] : chunk["char_end"]] == chunk["text"]


def test_offset_reconstruction_on_fixture() -> None:
    text = "Alpha beta. Gamma delta epsilon. Zeta eta theta iota."
    chunks = chunk_text(text, source_file="fixture.md", token_size=6, overlap=2)
    assert chunks
    for chunk in chunks:
        assert text[chunk["char_start"] : chunk["char_end"]] == chunk["text"]


# --- configuration and shape ---


def test_negative_tokens_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        chunk_text("hello world", source_file="x.md", token_size=-1, overlap=0)


def test_zero_tokens_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        chunk_text("hello world", source_file="x.md", token_size=0, overlap=0)


def test_overlap_equal_to_token_size_rejected() -> None:
    with pytest.raises(ValueError, match="smaller than token size"):
        chunk_text("hello world", source_file="x.md", token_size=10, overlap=10)


def test_overlap_greater_than_token_size_rejected() -> None:
    with pytest.raises(ValueError, match="smaller than token size"):
        chunk_text("hello world", source_file="x.md", token_size=10, overlap=11)


def test_negative_overlap_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("hello world", source_file="x.md", token_size=10, overlap=-1)


def test_empty_and_whitespace_produce_zero_chunks() -> None:
    kwargs = {"source_file": "empty.md", "token_size": 20, "overlap": 5}
    assert chunk_text("", **kwargs) == []
    assert chunk_text("   \n\t  \n", **kwargs) == []


def test_chunk_records_carry_required_fields() -> None:
    text = "Hello world. This is a short document for field checks."
    chunks = chunk_text(text, source_file="fields.md", token_size=8, overlap=2)
    assert chunks
    for chunk in chunks:
        assert REQUIRED_FIELDS <= set(chunk)
        assert chunk["source_file"] == "fields.md"
        assert chunk["token_count"] > 0
        assert chunk["token_count"] <= 8
        assert chunk["content_hash"] == hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()


# --- limits, boundaries, overlap ---


def test_no_chunk_exceeds_token_limit() -> None:
    text = " ".join(f"word{i}" for i in range(80))
    chunks = chunk_text(text, source_file="lim.md", token_size=10, overlap=3)
    assert all(chunk["token_count"] <= 10 for chunk in chunks)


def test_token_size_argument_changes_chunk_count() -> None:
    text = " ".join(f"word{i}" for i in range(100))
    small = chunk_text(text, source_file="n.md", token_size=20, overlap=0)
    large = chunk_text(text, source_file="n.md", token_size=40, overlap=0)
    assert len(small) > len(large)


def test_overlap_argument_increases_chunk_count() -> None:
    text = " ".join(f"word{i}" for i in range(100))
    none = chunk_text(text, source_file="n.md", token_size=20, overlap=0)
    overlapped = chunk_text(text, source_file="n.md", token_size=20, overlap=10)
    assert len(overlapped) > len(none)


def test_overlap_present_in_consecutive_chunks() -> None:
    text = " ".join(f"Sentence number {i} ends here." for i in range(30))
    overlap = 5
    chunks = chunk_text(text, source_file="ov.md", token_size=20, overlap=overlap)
    assert len(chunks) >= 2
    for left, right in zip(chunks, chunks[1:]):
        assert left["char_start"] < right["char_start"] < left["char_end"]
        shared = text[right["char_start"] : left["char_end"]]
        assert shared
        assert shared in left["text"]
        assert shared in right["text"]


def test_words_are_not_split_when_a_boundary_exists() -> None:
    text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda."
    chunks = chunk_text(text, source_file="w.md", token_size=4, overlap=1)
    for chunk in chunks:
        if chunk["char_start"] > 0:
            assert text[chunk["char_start"] - 1].isspace()
        if chunk["char_end"] < len(text):
            assert text[chunk["char_end"]].isspace()


def test_prefers_sentence_boundary_inside_the_limit() -> None:
    text = "First sentence ends here. Second sentence keeps going with extra words."
    chunks = chunk_text(text, source_file="s.md", token_size=8, overlap=2)
    assert chunks[0]["text"] == "First sentence ends here."


def test_long_single_word_is_kept_intact() -> None:
    text = "supercalifragilisticexpialidocious"
    chunks = chunk_text(text, source_file="one.md", token_size=1, overlap=0)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text


# --- unicode, headings, identity ---


def test_unicode_survives() -> None:
    text = "Café — 東京. Next sentence uses an em dash — and must stay intact."
    chunks = chunk_text(text, source_file="uni.md", token_size=20, overlap=2)
    recovered = "".join(
        text[c["char_start"] : c["char_end"]] for c in chunks[:1]
    )
    assert "Café" in recovered
    assert "東京" in recovered
    assert "—" in recovered
    blob = " ".join(c["text"] for c in chunks)
    assert "Café" in blob and "東京" in blob and "—" in blob


def test_section_is_nearest_preceding_heading() -> None:
    text = (
        "# Title\n\n"
        "## Section A\n\n"
        "Hello world from section A with enough words to fill a chunk.\n\n"
        "## Section B\n\n"
        "Goodbye from section B with enough words to fill a chunk."
    )
    chunks = chunk_text(text, source_file="sec.md", token_size=8, overlap=0)
    assert any(c["section"] == "Section A" for c in chunks)
    goodbye = [c for c in chunks if "Goodbye" in c["text"]]
    assert goodbye
    assert goodbye[0]["section"] == "Section B"


def test_chunk_id_is_derived_from_source_and_offsets() -> None:
    text = "Hello world. Another sentence follows after that."
    chunks = chunk_text(text, source_file="id.md", token_size=6, overlap=2)
    chunk = chunks[0]
    assert chunk["source_file"] in chunk["chunk_id"]
    assert str(chunk["char_start"]) in chunk["chunk_id"]
    assert str(chunk["char_end"]) in chunk["chunk_id"]
    assert re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-", chunk["chunk_id"]) is None


def test_determinism_identical_inputs_identical_ids() -> None:
    text = "Repeatable input. Same offsets. Same identifiers."
    first = chunk_text(text, source_file="det.md", token_size=5, overlap=2)
    second = chunk_text(text, source_file="det.md", token_size=5, overlap=2)
    assert [c["chunk_id"] for c in first] == [c["chunk_id"] for c in second]
    assert first == second
