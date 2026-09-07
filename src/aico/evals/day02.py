from __future__ import annotations

import argparse
import json
from pathlib import Path

from aico.evals.day01 import _mean, _round_metrics, _score_query
from aico.retrieval.embedding_provider import EmbeddingProvider, get_provider
from aico.retrieval.hybrid import RRF_K
from aico.retrieval.search import search

TOP_K = 5
BM25_SCORE_FLOOR = 2.0
VECTOR_SCORE_FLOOR = 0.35
HYBRID_SCORE_FLOOR = (1.0 / (RRF_K + 1)) + (1.0 / (RRF_K + 8))
SCORE_FLOORS = {
    "bm25": BM25_SCORE_FLOOR,
    "vector": VECTOR_SCORE_FLOOR,
    "hybrid": HYBRID_SCORE_FLOOR,
}
SCORED_CATEGORIES = ("exact_term", "synonym_poor", "multi_chunk", "semantic_only")
MODES = ("bm25", "vector", "hybrid")


def _load_queries(path: Path) -> tuple[list[dict], list[str], list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scoring = payload.get("scoring") or {}
    scored_ids = list(scoring.get("scored_queries") or [])
    separate_ids = list(scoring.get("separately_reported_queries") or [])
    return payload["queries"], scored_ids, separate_ids


def evaluate_mode(
    queries: list[dict],
    *,
    index_dir: Path,
    vectors_dir: Path,
    mode: str,
    provider: EmbeddingProvider | None,
    score_floor: float,
    top_k: int = TOP_K,
    scored_ids: list[str],
    separate_ids: list[str],
) -> dict:
    per_query: list[dict] = []
    for query in queries:
        hits = search(
            query["text"],
            index_dir,
            top_k,
            mode=mode,
            vectors_dir=vectors_dir,
            provider=provider,
        )
        per_query.append(_score_query(query, hits, score_floor=score_floor))
    return _aggregate(
        per_query,
        score_floor=score_floor,
        mode=mode,
        scored_ids=scored_ids,
        separate_ids=separate_ids,
    )


def _aggregate(
    per_query: list[dict],
    *,
    score_floor: float,
    mode: str,
    scored_ids: list[str],
    separate_ids: list[str],
) -> dict:
    scored = [row for row in per_query if row["query_id"] in scored_ids]
    no_match = [row for row in per_query if row["query_id"] in separate_ids]
    by_category: dict[str, dict] = {}
    for name in SCORED_CATEGORIES:
        group = [row for row in scored if row["category"] == name]
        by_category[name] = {
            **_round_metrics(
                _mean([1.0 if row["hit_at_1"] else 0.0 for row in group]),
                _mean([1.0 if row["hit_at_5"] else 0.0 for row in group]),
                _mean([row["reciprocal_rank"] for row in group]),
            ),
            "n": len(group),
        }
        if name == "multi_chunk":
            by_category[name]["full_hit_rate"] = round(
                _mean([1.0 if row["full_hit"] else 0.0 for row in group]), 4
            )

    return {
        "mode": mode,
        "score_floor": score_floor,
        "rrf_k": RRF_K if mode == "hybrid" else None,
        "scored_queries": {
            **_round_metrics(
                _mean([1.0 if row["hit_at_1"] else 0.0 for row in scored]),
                _mean([1.0 if row["hit_at_5"] else 0.0 for row in scored]),
                _mean([row["reciprocal_rank"] for row in scored]),
            ),
            "n": len(scored),
            "ids": scored_ids,
        },
        "by_category": by_category,
        "no_match": {
            "n": len(no_match),
            "ids": separate_ids,
            "correct_count": sum(1 for row in no_match if row["no_match_correct"]),
            "correct_rate": round(
                _mean([1.0 if row["no_match_correct"] else 0.0 for row in no_match]), 4
            ),
            "queries": {
                row["query_id"]: {
                    "correct": row["no_match_correct"],
                    "max_score": round(row["max_score"], 4),
                    "returned_above_floor": row["returned_above_floor"],
                }
                for row in no_match
            },
        },
        "per_query": per_query,
    }


def evaluate_all(
    queries_path: Path,
    index_dir: Path,
    vectors_dir: Path,
    provider: EmbeddingProvider,
    modes: tuple[str, ...] = MODES,
) -> dict:
    queries, scored_ids, separate_ids = _load_queries(queries_path)
    by_mode: dict[str, dict] = {}
    for mode in modes:
        mode_provider = None if mode == "bm25" else provider
        by_mode[mode] = evaluate_mode(
            queries,
            index_dir=index_dir,
            vectors_dir=vectors_dir,
            mode=mode,
            provider=mode_provider,
            score_floor=SCORE_FLOORS[mode],
            scored_ids=scored_ids,
            separate_ids=separate_ids,
        )
    return {
        "rrf_k": RRF_K,
        "score_floors": dict(SCORE_FLOORS),
        "modes": by_mode,
    }


def _metric_row(mode: str, result: dict) -> str:
    sq = result["scored_queries"]
    return (
        f"| {mode} | {sq['hit_at_1']:.4f} | {sq['hit_at_5']:.4f} | "
        f"{sq['mrr']:.4f} | {result['score_floor']:.4f} |"
    )


def _find_vector_win(by_mode: dict[str, dict]) -> dict | None:
    bm25_rows = {row["query_id"]: row for row in by_mode["bm25"]["per_query"]}
    for row in by_mode["vector"]["per_query"]:
        lexical = bm25_rows[row["query_id"]]
        if row["category"] == "no_match":
            continue
        if row["hit_at_5"] and not lexical["hit_at_5"]:
            return {"vector": row, "bm25": lexical}
    return None


def _find_bm25_win(by_mode: dict[str, dict]) -> dict | None:
    vector_rows = {row["query_id"]: row for row in by_mode["vector"]["per_query"]}
    for row in by_mode["bm25"]["per_query"]:
        semantic = vector_rows[row["query_id"]]
        if row["category"] == "no_match":
            continue
        if row["hit_at_5"] and not semantic["hit_at_5"]:
            return {"bm25": row, "vector": semantic}
        if row["hit_at_1"] and not semantic["hit_at_1"] and row["hit_at_5"] and semantic["hit_at_5"]:
            return {"bm25": row, "vector": semantic}
    return None


def _hybrid_vs_singles(by_mode: dict[str, dict]) -> list[str]:
    lines: list[str] = []
    hybrid_sq = by_mode["hybrid"]["scored_queries"]
    for mode in ("bm25", "vector"):
        other = by_mode[mode]["scored_queries"]
        better = hybrid_sq["mrr"] > other["mrr"]
        lines.append(
            f"- vs {mode}: hybrid MRR {hybrid_sq['mrr']:.4f} "
            f"{'>' if better else '<='} {other['mrr']:.4f}"
        )
    for cat in SCORED_CATEGORIES:
        hybrid_cat = by_mode["hybrid"]["by_category"][cat]
        bm25_cat = by_mode["bm25"]["by_category"][cat]
        vector_cat = by_mode["vector"]["by_category"][cat]
        if hybrid_cat["n"] == 0:
            continue
        if hybrid_cat["mrr"] < max(bm25_cat["mrr"], vector_cat["mrr"]):
            winner = "bm25" if bm25_cat["mrr"] >= vector_cat["mrr"] else "vector"
            lines.append(
                f"- hybrid lost on `{cat}` (MRR {hybrid_cat['mrr']:.4f} vs "
                f"{winner} {max(bm25_cat['mrr'], vector_cat['mrr']):.4f})"
            )
    return lines


def render_mode_comparison(metrics: dict) -> str:
    by_mode = metrics["modes"]
    floors = metrics["score_floors"]
    lines: list[str] = [
        "# Day 2 mode comparison",
        "",
        "Same corpus, same chunker, same offsets as Day 1. Labels were not edited.",
        "",
        f"RRF k = **{metrics['rrf_k']}**. Raising k flattens the gap between rank 1 "
        "and rank 10 (gentler fusion). Lowering k makes the top of each list dominate.",
        "",
        "Hybrid fuses **ranks**, never raw scores. BM25 is unbounded and corpus-dependent; "
        "cosine sits in [-1, 1]. Averaging them would let the larger unit win.",
        "",
        "## Score floors",
        "",
        "| Mode | Floor | Why |",
        "|---|---:|---|",
        f"| bm25 | {floors['bm25']:.4f} | Unchanged from Day 1. Exact-term hits score well above 4; below 2 is weak leftover overlap. |",
        f"| vector | {floors['vector']:.4f} | Cosine is a ranking signal, not confidence. A vector index always returns a neighbour; this floor is where we refuse it. |",
        f"| hybrid | {floors['hybrid']:.4f} | `1/(k+1) + 1/(k+8)`. Rank-1 in one list only is not enough — that is what a no_match nearest neighbour looks like. |",
        "",
        "## Three-way metrics (scored queries Q01–Q08, Q11–Q15)",
        "",
        "| Mode | Hit@1 | Hit@5 | MRR | Floor |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        lines.append(_metric_row(mode, by_mode[mode]))

    lines.extend(["", "### Per category", ""])
    lines.append("| Category | n | bm25 Hit@5 / MRR | vector Hit@5 / MRR | hybrid Hit@5 / MRR |")
    lines.append("|---|---:|---|---|---|")
    for cat in SCORED_CATEGORIES:
        n = by_mode["bm25"]["by_category"][cat]["n"]
        cells = []
        for mode in MODES:
            row = by_mode[mode]["by_category"][cat]
            cells.append(f"{row['hit_at_5']:.4f} / {row['mrr']:.4f}")
        lines.append(f"| {cat} | {n} | {cells[0]} | {cells[1]} | {cells[2]} |")

    lines.extend(["", "## no_match (Q09, Q10, Q16) — scored separately", ""])
    lines.append("Correct = zero hits with score above that mode's floor. Not folded into Hit@1 / Hit@5 / MRR.")
    lines.append("")
    lines.append("| Mode | Query | max score | above floor | correct |")
    lines.append("|---|---|---:|---:|---|")
    for mode in MODES:
        for qid, row in by_mode[mode]["no_match"]["queries"].items():
            lines.append(
                f"| {mode} | {qid} | {row['max_score']:.4f} | "
                f"{row['returned_above_floor']} | {row['correct']} |"
            )

    vector_win = _find_vector_win(by_mode)
    lines.extend(["", "## Where vector beat BM25", ""])
    if vector_win:
        row = vector_win["vector"]
        lines.append(
            f"**{row['query_id']}** (`{row['category']}`): {row['text']}"
        )
        lines.append("")
        lines.append(
            f"Vector first-match rank={row['first_match_rank']}; "
            f"BM25 first-match rank={vector_win['bm25']['first_match_rank']!r}."
        )
        lines.append("")
        lines.append(
            "The query wording and the document wording do not overlap enough for BM25. "
            "That is the `semantic_only` point: lexical retrieval had nothing distinctive to score."
        )
    else:
        lines.append("No scored query had a vector Hit@5 where BM25 missed the top 5.")

    bm25_win = _find_bm25_win(by_mode)
    lines.extend(["", "## Where BM25 beat vector", ""])
    if bm25_win:
        row = bm25_win["bm25"]
        lines.append(
            f"**{row['query_id']}** (`{row['category']}`): {row['text']}"
        )
        lines.append("")
        lines.append(
            f"BM25 first-match rank={row['first_match_rank']}; "
            f"vector first-match rank={bm25_win['vector']['first_match_rank']!r}."
        )
        lines.append("")
        lines.append(
            "Shared terms already put the right chunk at the top for BM25. "
            "Cosine ranked a nearby-meaning passage above the exact span."
        )
    else:
        lines.append(
            "No scored query showed a BM25 hit with vector missing. "
            "If this stays empty after a real Foundry run, check the scorer — vector should not win everywhere."
        )

    lines.extend(["", "## Hybrid vs both singles", ""])
    lines.extend(_hybrid_vs_singles(by_mode) or ["- hybrid matched or beat both overall MRR."])
    lines.extend(
        [
            "",
            "## Cache evidence",
            "",
            "Run twice from the project root:",
            "",
            "```",
            "python -m aico.retrieval.embed --index data/index --out data/vectors",
            "```",
            "",
            "First run embeds uncached chunks. Second run over unchanged chunks must print "
            "`provider_calls=0`. Edit one chunk's text, re-run: that chunk and only that chunk "
            "is re-embedded. Invalidation key is the SHA-256 of the chunk **text**, plus model alias.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def format_eval(metrics: dict) -> str:
    lines = [f"RRF k={metrics['rrf_k']}", ""]
    for mode in MODES:
        if mode not in metrics["modes"]:
            continue
        result = metrics["modes"][mode]
        sq = result["scored_queries"]
        lines.append(
            f"{mode:6}  Hit@1={sq['hit_at_1']:.4f}  Hit@5={sq['hit_at_5']:.4f}  "
            f"MRR={sq['mrr']:.4f}  floor={result['score_floor']:.4f}"
        )
        nm = result["no_match"]
        lines.append(
            f"       no_match {nm['correct_count']}/{nm['n']} correct"
        )
    return "\n".join(lines) + "\n"


def write_artifacts(metrics: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if all(mode in metrics["modes"] for mode in MODES):
        (out_dir / "mode_comparison.md").write_text(
            render_mode_comparison(metrics),
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score bm25, vector and hybrid on the Day 2 query set.")
    parser.add_argument("--queries", required=True, type=Path, help="Path to day02_queries.json")
    parser.add_argument("--index", type=Path, default=Path("data/index"))
    parser.add_argument("--vectors", type=Path, default=Path("data/vectors"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/day02"))
    parser.add_argument(
        "--mode",
        choices=("all",) + MODES,
        default="all",
        help="Run one mode or all three",
    )
    parser.add_argument(
        "--provider",
        choices=("foundry", "fake"),
        default="foundry",
    )
    args = parser.parse_args(argv)
    modes = MODES if args.mode == "all" else (args.mode,)
    provider = get_provider(args.provider)
    metrics = evaluate_all(
        args.queries,
        args.index,
        args.vectors,
        provider,
        modes=modes,
    )
    write_artifacts(metrics, args.out)
    print(format_eval(metrics))
    print(f"Wrote artifacts to {args.out}")


if __name__ == "__main__":
    main()
