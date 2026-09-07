from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from aico.retrieval.embedding_provider import (
    DimensionMismatchError,
    EmbeddingProvider,
    content_hash,
)

CACHE_FILENAME = "cache.json"


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Direction similarity in [-1, 1]. Lengths must match; they are not padded or truncated."""
    if len(left) != len(right):
        raise DimensionMismatchError(
            f"dimension mismatch: {len(left)} vs {len(right)}; refusing to pad or truncate"
        )
    dot = 0.0
    left_norm_sq = 0.0
    right_norm_sq = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm_sq += a * a
        right_norm_sq += b * b
    if left_norm_sq == 0.0 or right_norm_sq == 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm_sq) * math.sqrt(right_norm_sq))


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class VectorCache:
    """On-disk embedding cache. Invalidation is content hash + model alias, not chunk_id alone."""

    def __init__(self, entries: dict[str, dict] | None = None) -> None:
        self.entries: dict[str, dict] = dict(entries or {})

    @classmethod
    def load(cls, cache_dir: Path) -> VectorCache:
        path = Path(cache_dir) / CACHE_FILENAME
        if not path.is_file():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload.get("entries") or {})

    def save(self, cache_dir: Path) -> Path:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / CACHE_FILENAME
        payload = {"entries": self.entries}
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def is_valid(
        self,
        chunk_id: str,
        text: str,
        provider: EmbeddingProvider,
        dataset_version: str,
    ) -> bool:
        entry = self.entries.get(chunk_id)
        if not entry:
            return False
        if entry.get("content_hash") != content_hash(text):
            return False
        if entry.get("model_alias") != provider.model_alias:
            return False
        vector = entry.get("vector") or []
        if entry.get("dimensions") != provider.dimensions or len(vector) != provider.dimensions:
            return False
        if entry.get("dataset_version") != dataset_version:
            return False
        return True

    def put(
        self,
        *,
        chunk_id: str,
        text: str,
        vector: list[float],
        provider: EmbeddingProvider,
        dataset_version: str,
        created_at: str | None = None,
    ) -> dict:
        if len(vector) != provider.dimensions:
            raise DimensionMismatchError(
                f"vector length {len(vector)} does not match provider dimensions "
                f"{provider.dimensions}; refusing to pad or truncate"
            )
        entry = {
            "chunk_id": chunk_id,
            "content_hash": content_hash(text),
            "model_alias": provider.model_alias,
            "dimensions": provider.dimensions,
            "dataset_version": dataset_version,
            "created_at": created_at or _now_utc(),
            "vector": vector,
        }
        self.entries[chunk_id] = entry
        return entry

    def search(
        self,
        query_vector: Sequence[float],
        chunks: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Rank chunks by cosine similarity to the query vector. Stable tie-break on chunk_id."""
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
        scored: list[tuple[float, str, int, dict]] = []
        for offset, chunk in enumerate(chunks):
            entry = self.entries.get(chunk["chunk_id"])
            if not entry:
                continue
            score = cosine_similarity(query_vector, entry["vector"])
            scored.append((score, chunk["chunk_id"], offset, chunk))
        scored.sort(key=lambda row: (-row[0], row[1], row[2]))
        hits: list[dict] = []
        for rank, (score, _chunk_id, _offset, chunk) in enumerate(scored[:top_k], start=1):
            hit = dict(chunk)
            hit["rank"] = rank
            hit["score"] = score
            hits.append(hit)
        return hits
