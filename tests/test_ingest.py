"""Ingest CLI: arguments take effect, output is deterministic, offsets still reconstruct."""

from __future__ import annotations

import json
from pathlib import Path

from aico.retrieval.ingest import INDEX_FILENAME, ingest, main

DOCUMENTS_DIR = Path("data/documents")


def _load(out_dir: Path) -> dict:
    return json.loads((out_dir / INDEX_FILENAME).read_text(encoding="utf-8"))


def test_ingest_writes_index_and_reconstructs_offsets(tmp_path: Path) -> None:
    ingest(DOCUMENTS_DIR, tmp_path, tokens=200, overlap=40)
    payload = _load(tmp_path)
    chunks = payload["chunks"]
    assert chunks
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in DOCUMENTS_DIR.glob("DOC-*.md")
    }
    for chunk in chunks:
        original = sources[chunk["source_file"]]
        assert original[chunk["char_start"] : chunk["char_end"]] == chunk["text"]


def test_two_ingest_runs_produce_identical_chunk_ids(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    ingest(DOCUMENTS_DIR, first, tokens=200, overlap=40)
    ingest(DOCUMENTS_DIR, second, tokens=200, overlap=40)
    a = _load(first)
    b = _load(second)
    assert [c["chunk_id"] for c in a["chunks"]] == [c["chunk_id"] for c in b["chunks"]]
    assert a["chunks"] == b["chunks"]


def test_ingest_token_and_overlap_arguments_take_effect(tmp_path: Path) -> None:
    small = tmp_path / "small"
    large = tmp_path / "large"
    overlapped = tmp_path / "ov"
    ingest(DOCUMENTS_DIR, small, tokens=100, overlap=10)
    ingest(DOCUMENTS_DIR, large, tokens=300, overlap=10)
    ingest(DOCUMENTS_DIR, overlapped, tokens=100, overlap=40)
    n_small = len(_load(small)["chunks"])
    n_large = len(_load(large)["chunks"])
    n_ov = len(_load(overlapped)["chunks"])
    assert n_small > n_large
    assert n_ov > n_small


def test_ingest_empty_file_produces_zero_chunks(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "empty.md").write_text("  \n\n\t", encoding="utf-8")
    out = tmp_path / "out"
    ingest(src, out, tokens=20, overlap=5)
    assert _load(out)["chunks"] == []


def test_ingest_unicode_file_roundtrips(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    body = "## Café\n\nTokyo is 東京 — and the dash stays.\n"
    (src / "uni.md").write_text(body, encoding="utf-8")
    out = tmp_path / "out"
    ingest(src, out, tokens=20, overlap=2)
    chunks = _load(out)["chunks"]
    assert chunks
    assert body[chunks[0]["char_start"] : chunks[0]["char_end"]] == chunks[0]["text"]
    blob = " ".join(c["text"] for c in chunks)
    assert "Café" in blob and "東京" in blob and "—" in blob


def test_ingest_cli_accepts_paths_and_sizes(tmp_path: Path) -> None:
    main(
        [
            "--input",
            str(DOCUMENTS_DIR),
            "--out",
            str(tmp_path),
            "--tokens",
            "80",
            "--overlap",
            "15",
        ]
    )
    payload = _load(tmp_path)
    assert payload["meta"]["token_size"] == 80
    assert payload["meta"]["overlap"] == 15
    assert payload["chunks"]
