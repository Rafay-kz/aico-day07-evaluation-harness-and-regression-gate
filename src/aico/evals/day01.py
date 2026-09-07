from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from aico.retrieval.bm25 import B, K1, BM25Index
from aico.retrieval.chunker import TOKEN_METHOD
from aico.retrieval.ingest import INDEX_FILENAME, ingest
from aico.retrieval.search import load_chunks


SCORE_FLOOR = 2.0
TOP_K = 5
COMPARE_CONFIGS = ((200, 40), (400, 80))
SCORED_CATEGORIES = ("exact_term", "synonym_poor", "multi_chunk")


def normalise(text: str) -> str:
    """Assignment matching rule: lowercase, non-alnum → space, collapse, strip."""
    chars: list[str] = []
    for char in text.lower():
        chars.append(char if char.isalnum() or char.isspace() else " ")
    return " ".join("".join(chars).split())


def chunk_matches_anchor(chunk_text: str, anchor: str) -> bool:
    return normalise(anchor) in normalise(chunk_text)


def _anchors_in_hits(hits: list[dict], anchors: list[dict]) -> list[dict]:
    found: list[dict] = []
    for anchor in anchors:
        if any(chunk_matches_anchor(hit["text"], anchor["anchor"]) for hit in hits):
            found.append(anchor)
    return found


def _first_match_rank(hits: list[dict], anchors: list[dict]) -> int | None:
    for hit in hits:
        if any(chunk_matches_anchor(hit["text"], anchor["anchor"]) for anchor in anchors):
            return int(hit["rank"])
    return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _round_metrics(hit1: float, hit5: float, mrr: float) -> dict[str, float]:
    return {
        "hit_at_1": round(hit1, 4),
        "hit_at_5": round(hit5, 4),
        "mrr": round(mrr, 4),
    }


def evaluate_chunks(
    queries: list[dict],
    chunks: list[dict],
    *,
    score_floor: float = SCORE_FLOOR,
    top_k: int = TOP_K,
) -> dict:
    """Score all ten queries over an in-memory chunk list."""
    index = BM25Index(chunks)
    per_query: list[dict] = []
    for query in queries:
        hits = index.search(query["text"], top_k=top_k)
        row = _score_query(query, hits, score_floor=score_floor)
        per_query.append(row)
    return _aggregate(per_query, score_floor=score_floor, chunk_count=len(chunks))


def _score_query(query: dict, hits: list[dict], *, score_floor: float) -> dict:
    anchors = query.get("relevant") or []
    rank = _first_match_rank(hits, anchors) if anchors else None
    matched = _anchors_in_hits(hits, anchors) if anchors else []
    hit_at_1 = rank == 1
    hit_at_5 = rank is not None
    reciprocal = (1.0 / rank) if rank else 0.0
    above_floor = [hit for hit in hits if hit["score"] > score_floor]
    return {
        "query_id": query["query_id"],
        "text": query["text"],
        "category": query["category"],
        "hit_at_1": hit_at_1,
        "hit_at_5": hit_at_5,
        "first_match_rank": rank,
        "reciprocal_rank": reciprocal,
        "matched_anchor_count": len(matched),
        "anchor_count": len(anchors),
        "full_hit": bool(anchors) and len(matched) == len(anchors),
        "max_score": hits[0]["score"] if hits else 0.0,
        "returned_above_floor": len(above_floor),
        "no_match_correct": query["category"] == "no_match" and len(above_floor) == 0,
        "top_hits": [
            {
                "rank": hit["rank"],
                "score": round(hit["score"], 4),
                "chunk_id": hit["chunk_id"],
                "source_file": hit["source_file"],
            }
            for hit in hits
        ],
    }


def _aggregate(per_query: list[dict], *, score_floor: float, chunk_count: int) -> dict:
    scored = [row for row in per_query if row["category"] != "no_match"]
    no_match = [row for row in per_query if row["category"] == "no_match"]
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
            by_category[name]["full_hits"] = {
                row["query_id"]: row["full_hit"] for row in group
            }

    return {
        "score_floor": score_floor,
        "chunk_count": chunk_count,
        "scored_queries": {
            **_round_metrics(
                _mean([1.0 if row["hit_at_1"] else 0.0 for row in scored]),
                _mean([1.0 if row["hit_at_5"] else 0.0 for row in scored]),
                _mean([row["reciprocal_rank"] for row in scored]),
            ),
            "n": len(scored),
        },
        "by_category": by_category,
        "no_match": {
            "n": len(no_match),
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


def run_eval(queries_path: Path, index_dir: Path) -> dict:
    """Run all queries against one ingest index."""
    payload = json.loads(Path(queries_path).read_text(encoding="utf-8"))
    chunks = load_chunks(index_dir)
    result = evaluate_chunks(payload["queries"], chunks)
    result["index_dir"] = str(index_dir)
    return result


def run_comparison(
    queries_path: Path,
    documents_dir: Path,
    out_dir: Path,
) -> dict:
    """Ingest both configs, score them, write artifacts/day01 deliverables."""
    payload = json.loads(Path(queries_path).read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configurations: dict[str, dict] = {}
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)

    for tokens, overlap in COMPARE_CONFIGS:
        name = f"{tokens}_{overlap}"
        index_dir = work / name
        ingest(documents_dir, index_dir, tokens=tokens, overlap=overlap)
        src = index_dir / INDEX_FILENAME
        dest = out_dir / f"chunks_{name}.json"
        shutil.copyfile(src, dest)
        chunks = json.loads(src.read_text(encoding="utf-8"))["chunks"]
        result = evaluate_chunks(payload["queries"], chunks)
        result["tokens"] = tokens
        result["overlap"] = overlap
        result["chunks_path"] = str(dest)
        configurations[name] = result

    winner_name, winner_reason = _pick_winner(configurations)
    metrics = {
        "score_floor": SCORE_FLOOR,
        "token_method": TOKEN_METHOD,
        "k1": K1,
        "b": B,
        "stopwords": "removed",
        "configurations": configurations,
        "winner": winner_name,
        "winner_reason": winner_reason,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "retrieval_report.md").write_text(
        render_report(metrics),
        encoding="utf-8",
    )
    shutil.rmtree(work, ignore_errors=True)
    return metrics


def _pick_winner(configurations: dict[str, dict]) -> tuple[str, str]:
    """Prefer the config that keeps multi-chunk facts together, then Hit@5 / MRR."""
    names = list(configurations)
    best = max(
        names,
        key=lambda name: (
            configurations[name]["by_category"]["multi_chunk"].get("full_hit_rate", 0.0),
            configurations[name]["scored_queries"]["hit_at_5"],
            configurations[name]["scored_queries"]["mrr"],
            configurations[name]["scored_queries"]["hit_at_1"],
        ),
    )
    other = next(name for name in names if name != best)
    best_row = configurations[best]
    other_row = configurations[other]
    q07_best = best_row["by_category"]["multi_chunk"]["full_hits"].get("Q07")
    q07_other = other_row["by_category"]["multi_chunk"]["full_hits"].get("Q07")
    tokens_best = best_row["tokens"]
    if q07_best and not q07_other:
        reason = (
            f"Config {best.replace('_', '/')} wins because {tokens_best}-token chunks "
            "leave fewer competing slices of the word 'notice', so both Q07 anchors "
            "appear in the top 5: ninety days written notice of withdrawal (DOC-001) "
            "and the contract clause that supersedes any notice period in the sourcing "
            "policy (DOC-002). At 200/40, one of those anchors is crowded out of the "
            "top 5 even though Hit@1 still fires on the other."
        )
    elif q07_best and q07_other:
        reason = (
            f"Config {best.replace('_', '/')} wins because both windows can surface "
            "Q07's two anchors, and this configuration still ranks exact-term facts "
            "without splitting a fact from the sentence that qualifies it."
        )
    else:
        reason = (
            f"Config {best.replace('_', '/')} wins on scored Hit@5 "
            f"({best_row['scored_queries']['hit_at_5']} vs "
            f"{other_row['scored_queries']['hit_at_5']}) and MRR "
            f"({best_row['scored_queries']['mrr']} vs "
            f"{other_row['scored_queries']['mrr']}). Q07 still fails a full hit "
            "in both windows — the two anchors live in different documents, and "
            "BM25 on 'notice' alone is not enough to land both in the top 5."
        )
    return best, reason


def _worst_scored_query(per_query: list[dict]) -> dict:
    scored = [row for row in per_query if row["category"] != "no_match"]
    return min(
        scored,
        key=lambda row: (
            1 if row["hit_at_5"] else 0,
            row["reciprocal_rank"],
            1 if row["full_hit"] else 0,
        ),
    )


def render_report(metrics: dict) -> str:
    configs = metrics["configurations"]
    winner = metrics["winner"]
    winner_cfg = configs[winner]
    worst = _worst_scored_query(winner_cfg["per_query"])
    lines: list[str] = [
        "# Day 1 retrieval report",
        "",
        "Lexical BM25 baseline. No embeddings. Labels were not edited.",
        "",
        "## Configurations",
        "",
        "200/40 and 400/80 are the assignment's suggested pair. They sit on "
        "either side of a typical policy section (~150–250 words). 200 is small "
        "enough that an anchor can fall across a cut; 400 is large enough that "
        "a fact and its exception can share a chunk. Overlap is 20% of the "
        "window in both cases, so the comparison is about size, not a different "
        "overlap policy.",
        "",
        f"Token counting: `{metrics['token_method']}` — a token is a maximal run "
        "of non-whitespace characters. Punctuation stays attached until search "
        "strips it. This is not a model tokenizer and not a character count: "
        "`thirty five` is two tokens here and might be one or two subword pieces "
        "in an LLM tokenizer.",
        "",
        "Search tokenisation (query and documents): the same whitespace split, "
        "then lowercase, strip edge punctuation, drop a small English stopword "
        "list. Stopwords are removed because `the` / `of` / `what` appear in "
        "almost every chunk; IDF would already down-weight them, and dropping "
        "them keeps scores on content words. `must` and `not` are kept.",
        "",
        f"BM25 parameters: K1={metrics['k1']}, B={metrics['b']} (named constants).",
        "",
        f"Score floor for no_match: **{metrics['score_floor']}**. Exact-term "
        "rank-1 hits on this corpus score well above 4. Scores below 2.0 are "
        "single weak-term leftovers. The floor is *not* set high enough to "
        "pretend Q09/Q10 returned nothing — those queries retrieve real bait "
        "passages (late payment / late delivery) and are reported as such.",
        "",
        "## Metrics (scored queries Q01–Q08)",
        "",
        "| Config | Chunks | Hit@1 | Hit@5 | MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, cfg in configs.items():
        sq = cfg["scored_queries"]
        lines.append(
            f"| {name.replace('_', '/')} | {cfg['chunk_count']} | "
            f"{sq['hit_at_1']:.4f} | {sq['hit_at_5']:.4f} | {sq['mrr']:.4f} |"
        )
    lines.extend(["", "### Per category", ""])
    for name, cfg in configs.items():
        lines.append(f"**{name.replace('_', '/')}**")
        lines.append("")
        lines.append("| Category | n | Hit@1 | Hit@5 | MRR | Full hit |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for cat, row in cfg["by_category"].items():
            extra = (
                f"{row['full_hit_rate']:.4f}" if cat == "multi_chunk" else "—"
            )
            lines.append(
                f"| {cat} | {row['n']} | {row['hit_at_1']:.4f} | "
                f"{row['hit_at_5']:.4f} | {row['mrr']:.4f} | {extra} |"
            )
        lines.append("")

    lines.extend(["## no_match (Q09–Q10) — scored separately", ""])
    lines.append(
        "Not folded into Hit@1 / Hit@5 / MRR. Correct = zero hits with score "
        f"> {metrics['score_floor']}."
    )
    lines.append("")
    lines.append("| Config | Query | max score | above floor | correct |")
    lines.append("|---|---|---:|---:|---|")
    for name, cfg in configs.items():
        for qid, row in cfg["no_match"]["queries"].items():
            lines.append(
                f"| {name.replace('_', '/')} | {qid} | {row['max_score']:.4f} | "
                f"{row['returned_above_floor']} | {row['correct']} |"
            )
    lines.append("")
    for name, cfg in configs.items():
        nm = cfg["no_match"]
        lines.append(
            f"- {name.replace('_', '/')}: {nm['correct_count']}/{nm['n']} "
            "correctly returned nothing above the floor."
        )

    lines.extend(
        [
            "",
            f"## Winner: {winner.replace('_', '/')}",
            "",
            metrics["winner_reason"],
            "",
            "## Query BM25 handled badly",
            "",
            f"**{worst['query_id']}** ({worst['category']}): {worst['text']}",
            "",
        ]
    )
    if worst["query_id"] == "Q05" or (
        worst["category"] == "synonym_poor" and not worst["hit_at_5"]
    ):
        lines.append(
            "The query never uses the document's words. It says `vendor` and "
            "`hand the agreement over`; DOC-002 says `may not assign or novate "
            "the agreement`. After stopword removal the query is essentially "
            "`vendor hand agreement another company` — none of those distinctive "
            "terms are `assign` or `novate`, which are also rare (high IDF) in "
            "the corpus. BM25 therefore ranks other chunks that share generic "
            "words like `agreement`. Fix without embeddings: expand the query "
            "with `assign` and `novate` at search time. (Q06 happened to Hit@1 "
            "because `order` occurs in the same partial-delivery passage as the "
            "anchor — co-occurrence, not synonym understanding.)"
        )
    elif worst["query_id"] == "Q07" or (
        worst["category"] == "multi_chunk" and not worst["full_hit"]
    ):
        lines.append(
            "The answer lives in two documents. BM25 scores each chunk alone, so "
            "`notice` can promote the 90-day policy chunk or the 60-day contract "
            "chunk without guaranteeing both anchors in the top 5. A larger "
            "window helps keep a fact and its nearby exception together *inside* "
            "one file; it cannot merge DOC-001 and DOC-002. Fix without "
            "embeddings: retrieve extra candidates and diversify by `source_file`, "
            "or expand the query with both `ninety days` and `supersedes`."
        )
    else:
        lines.append(
            f"First match rank={worst['first_match_rank']!r}, Hit@5="
            f"{worst['hit_at_5']}. The top hits share query tokens but not the "
            "anchor span — typical BM25 confusion with a common word."
        )
    lines.extend(
        [
            "",
            "## Per-query detail (winning config)",
            "",
            "| Query | Category | Hit@1 | Hit@5 | Rank | Full hit |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for row in winner_cfg["per_query"]:
        if row["category"] == "no_match":
            continue
        rank = row["first_match_rank"] if row["first_match_rank"] is not None else "—"
        lines.append(
            f"| {row['query_id']} | {row['category']} | {row['hit_at_1']} | "
            f"{row['hit_at_5']} | {rank} | {row['full_hit']} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def format_eval(result: dict) -> str:
    sq = result["scored_queries"]
    lines = [
        f"chunks={result['chunk_count']}  score_floor={result['score_floor']}",
        "",
        "Scored queries Q01–Q08 (no_match excluded)",
        f"  Hit@1  {sq['hit_at_1']:.4f}",
        f"  Hit@5  {sq['hit_at_5']:.4f}",
        f"  MRR    {sq['mrr']:.4f}",
        "",
        "Per category",
    ]
    for cat, row in result["by_category"].items():
        extra = ""
        if cat == "multi_chunk":
            extra = f"  full_hit={row['full_hit_rate']:.4f}"
        lines.append(
            f"  {cat:14} Hit@1={row['hit_at_1']:.4f}  Hit@5={row['hit_at_5']:.4f}  "
            f"MRR={row['mrr']:.4f}{extra}"
        )
    nm = result["no_match"]
    lines.append("")
    lines.append(
        f"no_match Q09–Q10 (inverted): {nm['correct_count']}/{nm['n']} correct "
        f"(nothing above floor {result['score_floor']})"
    )
    for qid, row in nm["queries"].items():
        lines.append(
            f"  {qid}  max_score={row['max_score']:.4f}  "
            f"above_floor={row['returned_above_floor']}  correct={row['correct']}"
        )
    lines.append("")
    lines.append("Per query")
    for row in result["per_query"]:
        if row["category"] == "no_match":
            lines.append(
                f"  {row['query_id']}  {row['category']:14}  "
                f"returned_above_floor={row['returned_above_floor']}"
            )
        else:
            rank = row["first_match_rank"] if row["first_match_rank"] is not None else "-"
            lines.append(
                f"  {row['query_id']}  {row['category']:14}  "
                f"Hit@1={int(row['hit_at_1'])} Hit@5={int(row['hit_at_5'])} "
                f"rank={rank} full_hit={int(row['full_hit'])}"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate lexical retrieval on the Day 1 query set.")
    parser.add_argument("--queries", required=True, type=Path, help="Path to day01_queries.json")
    parser.add_argument("--index", required=True, type=Path, help="Directory containing the chunk index")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="If set, also ingest 200/40 and 400/80 and write artifacts/day01",
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=Path("data/documents"),
        help="Source documents for the two-config comparison",
    )
    args = parser.parse_args(argv)
    result = run_eval(args.queries, args.index)
    print(format_eval(result))
    if args.out is not None:
        metrics = run_comparison(args.queries, args.documents, args.out)
        print(f"Wrote comparison artifacts to {args.out}")
        print(f"Winner: {metrics['winner'].replace('_', '/')}")
        print(metrics["winner_reason"])


if __name__ == "__main__":
    main()
