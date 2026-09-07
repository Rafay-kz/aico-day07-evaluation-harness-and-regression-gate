from __future__ import annotations

import argparse
import json
from pathlib import Path

from aico.retrieval.bm25 import BM25Index
from aico.retrieval.embedding_provider import EmbeddingProvider, get_provider
from aico.retrieval.hybrid import RRF_K, reciprocal_rank_fusion
from aico.retrieval.ingest import INDEX_FILENAME
from aico.retrieval.vector_index import VectorCache

_PREVIEW_CHARS = 180
DEFAULT_INDEX = Path("data/index")
DEFAULT_VECTORS = Path("data/vectors")
MODES = ("bm25", "vector", "hybrid")


def load_chunks(index_dir: Path) -> list[dict]:
    path = Path(index_dir) / INDEX_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"no index at {path}; run ingest first:\n"
            "  python -m aico.retrieval.ingest --input data/documents "
            "--out data/index --tokens 300 --overlap 50"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["chunks"]


def load_cache(vectors_dir: Path) -> VectorCache:
    cache = VectorCache.load(vectors_dir)
    if not cache.entries:
        raise FileNotFoundError(
            f"no vector cache at {Path(vectors_dir)}; run embed first:\n"
            "  python -m aico.retrieval.embed --index data/index --out data/vectors"
        )
    return cache


def search(
    query: str,
    index_dir: Path,
    top_k: int,
    mode: str = "bm25",
    vectors_dir: Path | None = None,
    provider: EmbeddingProvider | None = None,
) -> list[dict]:
    """Return top_k hits. Default mode is bm25 so Day 1 commands stay unchanged."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    chunks = load_chunks(index_dir)
    if mode == "bm25":
        return BM25Index(chunks).search(query, top_k=top_k)
    if provider is None:
        provider = get_provider("foundry")
    cache = load_cache(vectors_dir or DEFAULT_VECTORS)
    if mode == "vector":
        return _vector_search(query, chunks, cache, provider, top_k)
    return _hybrid_search(query, chunks, cache, provider, top_k)


def _rank_all_bm25(query: str, chunks: list[dict]) -> list[dict]:
    if not chunks:
        return []
    return BM25Index(chunks).search(query, top_k=len(chunks))


def _vector_search(
    query: str,
    chunks: list[dict],
    cache: VectorCache,
    provider: EmbeddingProvider,
    top_k: int,
) -> list[dict]:
    query_vector = provider.embed([query])[0]
    if len(query_vector) != provider.dimensions:
        raise ValueError(
            f"query embedding has {len(query_vector)} dimensions, "
            f"expected {provider.dimensions}; refusing to pad or truncate"
        )
    return cache.search(query_vector, chunks, top_k=top_k)


def _hybrid_search(
    query: str,
    chunks: list[dict],
    cache: VectorCache,
    provider: EmbeddingProvider,
    top_k: int,
) -> list[dict]:
    bm25_ranked = _rank_all_bm25(query, chunks)
    vector_ranked = _vector_search(query, chunks, cache, provider, top_k=len(chunks) or 1)
    return reciprocal_rank_fusion(
        [bm25_ranked, vector_ranked],
        chunks,
        top_k=top_k,
        k=RRF_K,
    )


def format_hit(hit: dict) -> str:
    preview = " ".join(hit["text"].split())
    if len(preview) > _PREVIEW_CHARS:
        preview = preview[: _PREVIEW_CHARS - 3] + "..."
    return (
        f"{hit['rank']}. score={hit['score']:.4f}  {hit['chunk_id']}\n"
        f"   {hit['source_file']}  [{hit['char_start']}:{hit['char_end']}]\n"
        f"   {preview}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Search chunks with bm25, vector or hybrid ranking.")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="bm25",
        help="Ranking mode. Default bm25 keeps the Day 1 command working.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help="Directory containing the chunk index",
    )
    parser.add_argument(
        "--vectors",
        type=Path,
        default=DEFAULT_VECTORS,
        help="Directory containing the vector cache",
    )
    parser.add_argument(
        "--provider",
        choices=("foundry", "fake"),
        default="foundry",
        help="Embedding backend for vector and hybrid modes",
    )
    args = parser.parse_args(argv)
    provider = None if args.mode == "bm25" else get_provider(args.provider)
    hits = search(
        args.query,
        args.index,
        args.top_k,
        mode=args.mode,
        vectors_dir=args.vectors,
        provider=provider,
    )
    if not hits:
        print("No results.")
        return
    for hit in hits:
        print(format_hit(hit))
        print()


if __name__ == "__main__":
    main()
