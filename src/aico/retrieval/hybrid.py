from __future__ import annotations

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[dict]],
    chunks: list[dict],
    top_k: int,
    k: int = RRF_K,
) -> list[dict]:
    """Fuse ranked lists by reciprocal rank. Never averages BM25 and cosine scores.

    score(chunk) = sum over modes of 1 / (k + rank_in_that_mode)
    A mode that did not rank the chunk contributes nothing.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k}")
    if k <= 0:
        raise ValueError(f"RRF k must be a positive integer, got {k}")

    fused: dict[str, float] = {}
    for ranking in rankings:
        for hit in ranking:
            chunk_id = hit["chunk_id"]
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + int(hit["rank"]))

    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    hits: list[dict] = []
    for rank, (chunk_id, score) in enumerate(ordered[:top_k], start=1):
        if chunk_id not in by_id:
            continue
        hit = dict(by_id[chunk_id])
        hit["rank"] = rank
        hit["score"] = score
        hits.append(hit)
    return hits
