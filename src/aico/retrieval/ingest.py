from __future__ import annotations

import argparse
import json
from pathlib import Path

from aico.retrieval.chunker import INGESTION_VERSION, TOKEN_METHOD, chunk_text, validate_chunk_params

INDEX_FILENAME = "chunks.json"


def ingest(input_dir: Path, out_dir: Path, tokens: int, overlap: int) -> Path:
    """Read markdown files from input_dir, chunk them, write the index to out_dir."""
    validate_chunk_params(tokens, overlap)
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")

    files = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".md"
    )
    chunks: list[dict] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        chunks.extend(
            chunk_text(
                text,
                source_file=path.name,
                token_size=tokens,
                overlap=overlap,
            )
        )

    payload = {
        "meta": {
            "ingestion_version": INGESTION_VERSION,
            "token_method": TOKEN_METHOD,
            "token_size": tokens,
            "overlap": overlap,
            "source_files": [path.name for path in files],
            "chunk_count": len(chunks),
        },
        "chunks": chunks,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / INDEX_FILENAME
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(chunks)} chunks from {len(files)} files to {out_path} "
        f"(tokens={tokens}, overlap={overlap})"
    )
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chunk documents into a retrieval index.")
    parser.add_argument("--input", required=True, type=Path, help="Folder of source documents")
    parser.add_argument("--out", required=True, type=Path, help="Directory to write the chunk index")
    parser.add_argument("--tokens", required=True, type=int, help="Maximum tokens per chunk")
    parser.add_argument("--overlap", required=True, type=int, help="Overlap in tokens between consecutive chunks")
    args = parser.parse_args(argv)
    ingest(args.input, args.out, args.tokens, args.overlap)


if __name__ == "__main__":
    main()
