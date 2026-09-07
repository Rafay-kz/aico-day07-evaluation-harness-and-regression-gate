from __future__ import annotations

import argparse
import json
from pathlib import Path

from aico.retrieval.embedding_provider import EmbeddingProvider, get_provider
from aico.retrieval.ingest import INDEX_FILENAME
from aico.retrieval.vector_index import VectorCache

DEFAULT_INDEX = Path("data/index")
DEFAULT_OUT = Path("data/vectors")


def load_index(index_dir: Path) -> tuple[list[dict], str]:
    path = Path(index_dir) / INDEX_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"no index at {path}; run ingest first:\n"
            "  python -m aico.retrieval.ingest --input data/documents "
            "--out data/index --tokens 300 --overlap 50"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset_version = str(payload.get("meta", {}).get("ingestion_version", "unknown"))
    return payload["chunks"], dataset_version


def embed_index(
    index_dir: Path,
    out_dir: Path,
    provider: EmbeddingProvider,
) -> dict:
    """Embed any chunks missing from the cache. Unchanged text + same model → zero provider calls."""
    chunks, dataset_version = load_index(index_dir)
    cache = VectorCache.load(out_dir)
    hits = 0
    to_embed: list[dict] = []
    for chunk in chunks:
        if cache.is_valid(chunk["chunk_id"], chunk["text"], provider, dataset_version):
            hits += 1
        else:
            to_embed.append(chunk)

    calls_before = provider.calls
    if to_embed:
        vectors = provider.embed([chunk["text"] for chunk in to_embed])
        for chunk, vector in zip(to_embed, vectors):
            cache.put(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                vector=vector,
                provider=provider,
                dataset_version=dataset_version,
            )
        cache.save(out_dir)

    return {
        "chunks": len(chunks),
        "cache_hits": hits,
        "embedded": len(to_embed),
        "provider_calls": provider.calls - calls_before,
        "dataset_version": dataset_version,
        "model_alias": provider.model_alias,
        "out": str(Path(out_dir)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Embed chunk text and write a persistent vector cache.")
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help="Directory containing the Day 1 chunk index",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Directory to write the vector cache",
    )
    parser.add_argument(
        "--provider",
        choices=("foundry", "fake"),
        default="foundry",
        help="Embedding backend. Tests use fake; live runs use Foundry.",
    )
    args = parser.parse_args(argv)
    stats = embed_index(args.index, args.out, get_provider(args.provider))
    print(
        f"chunks={stats['chunks']} cache_hits={stats['cache_hits']} "
        f"embedded={stats['embedded']} provider_calls={stats['provider_calls']} "
        f"model={stats['model_alias']}"
    )


if __name__ == "__main__":
    main()
