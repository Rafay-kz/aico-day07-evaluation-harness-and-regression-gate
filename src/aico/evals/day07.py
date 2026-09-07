"""Day 7 evaluation harness and regression gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from aico.contracts.errors import TypedFailure
from aico.evals.dataset import (
    GoldenCase,
    GoldenDataset,
    category_counts,
    iter_holdout_cases,
    iter_tuning_cases,
    load_dataset,
    split_counts,
)
from aico.evals.failure_classifier import classify_failure
from aico.evals.groundedness import (
    EVALUATOR_VERSION,
    EvalLabTransport,
    GroundednessResult,
    evaluate_groundedness,
    groundedness_to_dict,
)
from aico.evals.metrics import (
    citation_is_valid,
    contains_fact,
    first_match_rank,
    hit_at_k,
    mean_hit_at_k,
    mean_mrr,
    prohibited_claim_hit,
    reciprocal_rank,
    refusal_is_correct,
    source_id,
)
from aico.evals.regression import (
    DEFAULT_BASELINE,
    DEFAULT_THRESHOLDS,
    GateDecision,
    apply_gate,
    build_baseline_document,
    load_baseline,
    load_thresholds,
    write_baseline,
)
from aico.platform.config import testing_config
from aico.platform.model_gateway import ModelGateway
from aico.rag.answer_service import (
    AnswerService,
    Blocked,
    Clarify,
    Day2Retriever,
    GroundedAnswer,
    InsufficientEvidence,
)
from aico.rag.prompt_builder import RetrievedChunk
from aico.retrieval.search import search

DEFAULT_INDEX = Path("data/index")
DEFAULT_ARTIFACTS = Path("artifacts/day07")
APPROVED_TOP_K = 5
APPROVED_MODE = "bm25"
WEAK_TOP_K = 1
LAB_CHAT_ALIAS = "eval-lab-chat-v1"
LAB_EMBED_ALIAS = "eval-lab-embed-v1"


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str
    top_k: int
    index_dir: str
    version: str = "day07-bm25-top5"

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "top_k": self.top_k,
            "index_dir": self.index_dir,
            "version": self.version,
        }


@dataclass
class EvaluationRun:
    dataset: GoldenDataset
    retrieval: RetrievalConfig
    rows: list[dict[str, Any]]
    metrics: dict[str, float]
    by_split: dict[str, dict[str, float]]
    by_category: dict[str, dict[str, float]]
    gate: GateDecision
    stability: dict[str, Any]
    report: dict[str, Any]


def outcome_of(result: object) -> str:
    if isinstance(result, GroundedAnswer):
        return "answer"
    if isinstance(result, InsufficientEvidence):
        return "refuse"
    if isinstance(result, Clarify):
        return "clarify"
    if isinstance(result, Blocked):
        return "block"
    if isinstance(result, TypedFailure):
        return "typed_failure"
    return "unknown"


def make_service(retrieval: RetrievalConfig) -> tuple[AnswerService, ModelGateway, EvalLabTransport]:
    transport = EvalLabTransport()
    gateway = ModelGateway(
        transport,
        testing_config(chat_alias=LAB_CHAT_ALIAS, embedding_alias=LAB_EMBED_ALIAS),
        sleep=lambda _seconds: None,
    )
    retriever = Day2Retriever(
        Path(retrieval.index_dir),
        mode=retrieval.mode,
        top_k=retrieval.top_k,
    )
    return AnswerService(gateway, retriever, allow_repair=True), gateway, transport


def evaluate_case(
    case: GoldenCase,
    *,
    retrieval: RetrievalConfig,
    service: AnswerService,
    gateway: ModelGateway,
    k: int,
) -> dict[str, Any]:
    hits = search(
        case.question,
        Path(retrieval.index_dir),
        retrieval.top_k,
        mode=retrieval.mode,
    )
    result = service.answer(case.question)
    observed = outcome_of(result)
    retrieved_ids = [str(hit["chunk_id"]) for hit in hits]
    retrieved_sources = [str(hit.get("source_file") or "") for hit in hits]
    retrieved_chunks = [
        RetrievedChunk(
            chunk_id=str(hit["chunk_id"]),
            text=str(hit["text"]),
            source_file=str(hit.get("source_file") or ""),
        )
        for hit in hits
    ]
    rank = None
    hit = False
    rr = 0.0
    if case.expected_sources:
        hit = hit_at_k(hits, case.expected_sources, k=k)
        rr = reciprocal_rank(hits, case.expected_sources, k=k)
        rank = first_match_rank(hits, case.expected_sources)

    answer_text = ""
    cited_ids: list[str] = []
    citation_valid: bool | None = None
    if isinstance(result, GroundedAnswer):
        answer_text = result.cited_answer.answer
        cited_ids = [item.chunk_id for item in result.cited_answer.citations]
        citation_valid = result.citation_validation.passed
        retrieved_ids = list(result.retrieved_chunk_ids) or retrieved_ids
    elif isinstance(result, InsufficientEvidence):
        answer_text = result.cited_answer.answer
        cited_ids = [item.chunk_id for item in result.cited_answer.citations]
        citation_valid = citation_is_valid(cited_ids, result.retrieved_chunk_ids)
        retrieved_ids = list(result.retrieved_chunk_ids) or retrieved_ids
    elif isinstance(result, TypedFailure):
        answer_text = result.message
        citation_valid = False if result.is_citation_failure() else None

    prohibited = prohibited_claim_hit(answer_text, case.prohibited_claims)
    missing_facts = tuple(
        fact for fact in case.critical_facts if not contains_fact(answer_text, fact)
    )
    facts_in_retrieved = True
    if case.critical_facts:
        joined = " ".join(chunk.text for chunk in retrieved_chunks)
        facts_in_retrieved = all(contains_fact(joined, fact) for fact in case.critical_facts)

    refusal_correct = refusal_is_correct(case.expected_outcome, observed)
    attack_pass = (observed == "block") if case.is_adversarial else None

    grounded: GroundednessResult | None = None
    evaluator_error = None
    if observed in {"answer", "refuse"} and answer_text:
        grounded = evaluate_groundedness(gateway, case.question, answer_text, retrieved_chunks)
        evaluator_error = grounded.error

    observation = {
        "category": case.category,
        "expected_outcome": case.expected_outcome,
        "observed_outcome": observed,
        "retrieval_hit": hit if case.expected_sources else True,
        "citation_valid": citation_valid,
        "evaluator_error": evaluator_error,
        "expected_sources": list(case.expected_sources),
        "prohibited_hits": list(prohibited),
        "missing_critical_facts": list(missing_facts),
        "critical_facts_in_retrieved": facts_in_retrieved,
        "groundedness_score": grounded.score if grounded is not None else 1.0,
    }
    case_failed = _case_failed(
        case,
        refusal_correct=refusal_correct,
        hit=hit,
        citation_valid=citation_valid,
        prohibited=prohibited,
        missing_facts=missing_facts,
        evaluator_error=evaluator_error,
        attack_pass=attack_pass,
    )
    failure_type = classify_failure(observation) if case_failed else None
    reason = _failure_reason(
        case,
        observed=observed,
        hit=hit,
        prohibited=prohibited,
        missing_facts=missing_facts,
        evaluator_error=evaluator_error,
        citation_valid=citation_valid,
    )
    return {
        "case_id": case.case_id,
        "category": case.category,
        "split": case.split,
        "question": case.question,
        "expected_sources": list(case.expected_sources),
        "expected_outcome": case.expected_outcome,
        "observed_outcome": observed,
        "retrieved_ids": retrieved_ids,
        "retrieved_sources": [source_id(item) for item in retrieved_sources],
        "first_match_rank": rank,
        "hit_at_k": hit if case.expected_sources else None,
        "reciprocal_rank": rr if case.expected_sources else None,
        "citation_valid": citation_valid,
        "cited_ids": cited_ids,
        "answer_preview": answer_text[:280],
        "critical_facts_present": not missing_facts,
        "missing_critical_facts": list(missing_facts),
        "critical_facts_in_retrieved": facts_in_retrieved,
        "prohibited_hits": list(prohibited),
        "refusal_correct": refusal_correct,
        "attack_pass": attack_pass,
        "groundedness": groundedness_to_dict(grounded) if grounded is not None else None,
        "evaluator_error": evaluator_error,
        "passed": not case_failed,
        "failure_type": failure_type,
        "failure_reason": reason if case_failed else "",
        "deterministic": {
            "hit_at_k": hit if case.expected_sources else None,
            "reciprocal_rank": rr if case.expected_sources else None,
            "citation_valid": citation_valid,
            "refusal_correct": refusal_correct,
            "attack_pass": attack_pass,
            "prohibited_claim_free": not prohibited,
            "critical_facts_present": not missing_facts if case.expected_outcome == "answer" else None,
        },
        "model_based": {
            "groundedness": groundedness_to_dict(grounded) if grounded is not None else None
        },
    }


def _case_failed(
    case: GoldenCase,
    *,
    refusal_correct: bool,
    hit: bool,
    citation_valid: bool | None,
    prohibited: Sequence[str],
    missing_facts: Sequence[str],
    evaluator_error: str | None,
    attack_pass: bool | None,
) -> bool:
    if evaluator_error:
        return True
    if prohibited:
        return True
    if not refusal_correct:
        return True
    if case.is_adversarial and attack_pass is False:
        return True
    if case.expected_sources and not hit:
        return True
    if case.expected_outcome == "answer" and citation_valid is False:
        return True
    if case.expected_outcome == "answer" and missing_facts:
        return True
    return False


def _failure_reason(
    case: GoldenCase,
    *,
    observed: str,
    hit: bool,
    prohibited: Sequence[str],
    missing_facts: Sequence[str],
    evaluator_error: str | None,
    citation_valid: bool | None,
) -> str:
    if evaluator_error:
        return evaluator_error
    if case.is_adversarial and observed != "block":
        return f"adversarial case observed {observed}, expected block"
    if prohibited:
        return "prohibited claim present: " + ", ".join(prohibited)
    if citation_valid is False:
        return "cited IDs are not a subset of retrieved IDs"
    if case.expected_sources and not hit:
        return "expected sources were not present in retrieved hits"
    if missing_facts:
        return "missing critical facts: " + ", ".join(missing_facts)
    if observed != case.expected_outcome:
        return f"observed {observed}, expected {case.expected_outcome}"
    return "case failed deterministic checks"


def aggregate_metrics(rows: Sequence[dict[str, Any]], k: int) -> dict[str, float]:
    retrieval_rows = [row for row in rows if row["hit_at_k"] is not None]
    citation_rows = [row for row in rows if row["citation_valid"] is not None]
    grounded_rows = [
        row for row in rows if row.get("groundedness") and row["groundedness"].get("error") is None
    ]
    adversarial = [row for row in rows if row["category"] == "adversarial"]
    return {
        "hit_at_k": mean_hit_at_k([bool(row["hit_at_k"]) for row in retrieval_rows]),
        "mrr": mean_mrr([float(row["reciprocal_rank"] or 0.0) for row in retrieval_rows]),
        "citation_validity": mean_hit_at_k(
            [bool(row["citation_valid"]) for row in citation_rows]
        ),
        "refusal_accuracy": mean_hit_at_k([bool(row["refusal_correct"]) for row in rows]),
        "groundedness": mean_mrr(
            [float(row["groundedness"]["score"]) for row in grounded_rows]
        ),
        "attack_pass_rate": mean_hit_at_k(
            [bool(row["attack_pass"]) for row in adversarial]
        ),
        "k": float(k),
        "n": float(len(rows)),
        "n_retrieval": float(len(retrieval_rows)),
        "n_groundedness": float(len(grounded_rows)),
    }


def run_stability(
    dataset: GoldenDataset,
    *,
    retrieval: RetrievalConfig,
    k: int,
    repeats: int | None = None,
) -> dict[str, Any]:
    count = repeats if repeats is not None else max(dataset.stability_repeats, 2)
    selected = [
        dataset.case_by_id(case_id)
        for case_id in dataset.stability_case_ids
        if any(case.case_id == case_id for case in dataset.cases)
    ]
    per_case: list[dict[str, Any]] = []
    for case in selected:
        scores: list[float] = []
        outcomes: list[str] = []
        for _repeat in range(count):
            service, gateway, _transport = make_service(retrieval)
            row = evaluate_case(case, retrieval=retrieval, service=service, gateway=gateway, k=k)
            outcomes.append(str(row["observed_outcome"]))
            grounded = row.get("groundedness") or {}
            if grounded.get("score") is not None:
                scores.append(float(grounded["score"]))
        numeric = {
            "mean": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "min": round(min(scores), 4) if scores else 0.0,
            "max": round(max(scores), 4) if scores else 0.0,
            "range": round((max(scores) - min(scores)), 4) if scores else 0.0,
            "values": scores,
        }
        per_case.append(
            {
                "case_id": case.case_id,
                "repeats": count,
                "outcomes": outcomes,
                "outcome_variation": sorted(set(outcomes)),
                "groundedness": numeric,
            }
        )
    return {
        "repeats": count,
        "case_ids": [case.case_id for case in selected],
        "justification": dataset.stability_justification,
        "cases": per_case,
    }


def run_evaluation(
    *,
    dataset_path: Path = Path("evals/golden_v1.json"),
    thresholds_path: Path = DEFAULT_THRESHOLDS,
    baseline_path: Path = DEFAULT_BASELINE,
    artifacts_dir: Path = DEFAULT_ARTIFACTS,
    index_dir: Path = DEFAULT_INDEX,
    top_k: int = APPROVED_TOP_K,
    mode: str = APPROVED_MODE,
    update_baseline: bool = False,
    write_reports: bool = True,
) -> EvaluationRun:
    dataset = load_dataset(dataset_path)
    thresholds = load_thresholds(thresholds_path)
    retrieval = RetrievalConfig(
        mode=mode,
        top_k=top_k,
        index_dir=str(index_dir),
        version=f"day07-{mode}-top{top_k}",
    )
    service, gateway, _transport = make_service(retrieval)
    rows = [
        evaluate_case(case, retrieval=retrieval, service=service, gateway=gateway, k=dataset.k)
        for case in dataset.cases
    ]
    metrics = aggregate_metrics(rows, k=dataset.k)
    by_split = {
        name: aggregate_metrics([row for row in rows if row["split"] == name], k=dataset.k)
        for name in ("train", "development", "holdout")
    }
    by_category = {
        name: aggregate_metrics([row for row in rows if row["category"] == name], k=dataset.k)
        for name in (
            "answerable",
            "ambiguous",
            "multi_chunk",
            "synonym_heavy",
            "unanswerable",
            "adversarial",
        )
    }
    safety_failures = sum(
        1 for row in rows if row["category"] == "adversarial" and row["attack_pass"] is not True
    )
    baseline = None if update_baseline else load_baseline(baseline_path)
    gate = apply_gate(
        {
            "hit_at_k": metrics["hit_at_k"],
            "mrr": metrics["mrr"],
            "citation_validity": metrics["citation_validity"],
            "refusal_accuracy": metrics["refusal_accuracy"],
            "groundedness": metrics["groundedness"],
            "attack_pass_rate": metrics["attack_pass_rate"],
        },
        thresholds,
        baseline=baseline,
        safety_failures=safety_failures,
    )
    stability = run_stability(dataset, retrieval=retrieval, k=dataset.k)
    report = build_report(
        dataset=dataset,
        retrieval=retrieval,
        rows=rows,
        metrics=metrics,
        by_split=by_split,
        by_category=by_category,
        gate=gate,
        stability=stability,
        safety_failures=safety_failures,
        thresholds=thresholds,
        baseline=baseline if not update_baseline else None,
    )
    if write_reports:
        write_evaluation_artifacts(artifacts_dir, report, rows, stability)
    if update_baseline:
        document = build_baseline_document(
            dataset_version=dataset.version,
            evaluator_version=EVALUATOR_VERSION,
            model_alias=LAB_CHAT_ALIAS,
            retrieval=retrieval.as_dict(),
            metrics={
                "hit_at_k": metrics["hit_at_k"],
                "mrr": metrics["mrr"],
                "citation_validity": metrics["citation_validity"],
                "refusal_accuracy": metrics["refusal_accuracy"],
                "groundedness": metrics["groundedness"],
                "attack_pass_rate": metrics["attack_pass_rate"],
            },
        )
        write_baseline(Path(baseline_path), document)
        report["baseline_updated"] = True
    return EvaluationRun(
        dataset=dataset,
        retrieval=retrieval,
        rows=rows,
        metrics=metrics,
        by_split=by_split,
        by_category=by_category,
        gate=gate,
        stability=stability,
        report=report,
    )


def build_report(
    *,
    dataset: GoldenDataset,
    retrieval: RetrievalConfig,
    rows: Sequence[dict[str, Any]],
    metrics: dict[str, float],
    by_split: dict[str, dict[str, float]],
    by_category: dict[str, dict[str, float]],
    gate: GateDecision,
    stability: dict[str, Any],
    safety_failures: int,
    thresholds,
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    failed = [row for row in rows if not row["passed"]]
    failure_summary: dict[str, int] = {}
    for row in failed:
        kind = str(row["failure_type"] or "prompt")
        failure_summary[kind] = failure_summary.get(kind, 0) + 1
    return {
        "dataset_version": dataset.version,
        "evaluator_version": EVALUATOR_VERSION,
        "model_alias": LAB_CHAT_ALIAS,
        "retrieval": retrieval.as_dict(),
        "total_case_count": len(dataset.cases),
        "split_counts": split_counts(dataset),
        "category_counts": category_counts(dataset),
        "tuning_case_count": len(iter_tuning_cases(dataset)),
        "holdout_case_count": len(iter_holdout_cases(dataset)),
        "deterministic": {
            "hit_at_k": metrics["hit_at_k"],
            "mrr": metrics["mrr"],
            "citation_validity": metrics["citation_validity"],
            "refusal_accuracy": metrics["refusal_accuracy"],
            "attack_pass_rate": metrics["attack_pass_rate"],
            "k": dataset.k,
        },
        "model_based": {
            "groundedness": metrics["groundedness"],
            "evaluator_version": EVALUATOR_VERSION,
            "n": int(metrics["n_groundedness"]),
        },
        "metrics": metrics,
        "by_split": by_split,
        "by_category": by_category,
        "safety_gate": {
            "zero_tolerance": True,
            "adversarial_failures": safety_failures,
            "passed": safety_failures == 0,
        },
        "thresholds": thresholds.metrics,
        "threshold_comparison": list(gate.threshold_failures),
        "baseline_comparison": list(gate.baseline_failures),
        "baseline_metrics": (baseline or {}).get("metrics"),
        "stability_summary": {
            "repeats": stability["repeats"],
            "case_ids": stability["case_ids"],
            "max_groundedness_range": max(
                (item["groundedness"]["range"] for item in stability["cases"]),
                default=0.0,
            ),
        },
        "failed_cases": [
            {
                "case_id": row["case_id"],
                "split": row["split"],
                "category": row["category"],
                "failure_type": row["failure_type"],
                "reason": row["failure_reason"],
            }
            for row in failed
        ],
        "failure_classification_summary": failure_summary,
        "final_gate_verdict": "PASS" if gate.passed else "FAIL",
        "gate_reasons": list(gate.reasons),
        "exit_code": gate.exit_code,
        "holdout_separation": {
            "holdout_excluded_from_tuning": True,
            "tuning_splits": ["train", "development"],
            "holdout_metrics": by_split["holdout"],
        },
    }


def write_evaluation_artifacts(
    artifacts_dir: Path,
    report: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    stability: dict[str, Any],
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["cases"] = list(rows)
    (artifacts_dir / "evaluation_report.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (artifacts_dir / "evaluation_report.md").write_text(
        render_evaluation_markdown(report),
        encoding="utf-8",
    )
    (artifacts_dir / "failure_classification.md").write_text(
        render_failure_markdown(report, rows),
        encoding="utf-8",
    )
    (artifacts_dir / "stability_report.md").write_text(
        render_stability_markdown(stability),
        encoding="utf-8",
    )


def render_evaluation_markdown(report: dict[str, Any]) -> str:
    det = report["deterministic"]
    model = report["model_based"]
    lines = [
        "# Day 7 evaluation report",
        "",
        f"Dataset: `{report['dataset_version']}` ({report['total_case_count']} cases)",
        f"Evaluator: `{report['evaluator_version']}` via `{report['model_alias']}`",
        f"Retrieval: `{report['retrieval']['mode']}` top_k={report['retrieval']['top_k']}",
        f"Final gate verdict: **{report['final_gate_verdict']}**",
        "",
        "## Split and category counts",
        "",
        f"Splits: {report['split_counts']}",
        f"Categories: {report['category_counts']}",
        "",
        "Holdout is reported separately and is not used to tune thresholds, prompts, or labels.",
        "",
        "## Deterministic checks",
        "",
        "These are exact checks. They are not mixed into a single model score.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Hit@{int(det['k'])} | {det['hit_at_k']:.4f} |",
        f"| MRR | {det['mrr']:.4f} |",
        f"| Citation validity | {det['citation_validity']:.4f} |",
        f"| Refusal accuracy | {det['refusal_accuracy']:.4f} |",
        f"| Attack pass rate | {det['attack_pass_rate']:.4f} |",
        "",
        "## Model-based groundedness",
        "",
        f"Evaluator version `{model['evaluator_version']}` scored {model['n']} answers.",
        f"Mean groundedness: **{model['groundedness']:.4f}**.",
        "",
        "The evaluator cannot rewrite the system answer.",
        "",
        "## Train / development / holdout",
        "",
        "| Split | Hit@K | MRR | Citation | Refusal | Groundedness | Attack |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["by_split"].items():
        lines.append(
            f"| {name} | {row['hit_at_k']:.4f} | {row['mrr']:.4f} | "
            f"{row['citation_validity']:.4f} | {row['refusal_accuracy']:.4f} | "
            f"{row['groundedness']:.4f} | {row['attack_pass_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Safety gate",
            "",
            f"Zero tolerance. Adversarial failures: {report['safety_gate']['adversarial_failures']}. "
            f"Safety passed: {report['safety_gate']['passed']}.",
            "",
            "## Threshold comparison",
            "",
        ]
    )
    if report["threshold_comparison"]:
        lines.extend(f"- {item}" for item in report["threshold_comparison"])
    else:
        lines.append("- All threshold comparisons passed.")
    lines.extend(["", "## Baseline comparison", ""])
    if report["baseline_metrics"] is None:
        lines.append("- No reviewed baseline loaded (update path or missing file).")
    elif report["baseline_comparison"]:
        lines.extend(f"- {item}" for item in report["baseline_comparison"])
    else:
        lines.append("- Candidate met the reviewed baseline.")
    lines.extend(
        [
            "",
            "## Stability summary",
            "",
            f"Repeats: {report['stability_summary']['repeats']} on "
            f"{', '.join(report['stability_summary']['case_ids'])}.",
            f"Max groundedness range: {report['stability_summary']['max_groundedness_range']:.4f}.",
            "",
            "## Failed cases",
            "",
        ]
    )
    if not report["failed_cases"]:
        lines.append("None.")
    else:
        lines.append("| Case | Split | Category | Type | Reason |")
        lines.append("|---|---|---|---|---|")
        for row in report["failed_cases"]:
            lines.append(
                f"| {row['case_id']} | {row['split']} | {row['category']} | "
                f"{row['failure_type']} | {row['reason']} |"
            )
    lines.extend(
        [
            "",
            "## Failure classification summary",
            "",
            json.dumps(report["failure_classification_summary"], indent=2),
            "",
            f"## Final gate verdict: {report['final_gate_verdict']}",
            "",
        ]
    )
    if report["gate_reasons"]:
        lines.extend(f"- {item}" for item in report["gate_reasons"])
    return "\n".join(lines) + "\n"


def render_failure_markdown(report: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Day 7 failure classification",
        "",
        "Every failed case has exactly one primary type: "
        "chunking, retrieval, prompt, citation, refusal, or evaluator.",
        "",
    ]
    failed = [row for row in rows if not row["passed"]]
    if not failed:
        lines.append("No failed cases in this run.")
        return "\n".join(lines) + "\n"
    lines.append("| Case ID | Split | Category | Observed failure | Primary type | Reason |")
    lines.append("|---|---|---|---|---|---|")
    for row in failed:
        lines.append(
            f"| {row['case_id']} | {row['split']} | {row['category']} | "
            f"{row['observed_outcome']} | {row['failure_type']} | {row['failure_reason']} |"
        )
    lines.extend(["", "Summary:", "", json.dumps(report["failure_classification_summary"], indent=2), ""])
    return "\n".join(lines) + "\n"


def render_stability_markdown(stability: dict[str, Any]) -> str:
    lines = [
        "# Day 7 stability report",
        "",
        f"Selected cases: {', '.join(stability['case_ids'])}",
        f"Repetition count: **{stability['repeats']}**",
        "",
        stability.get("justification") or "",
        "",
        "| Case | Repeats | Outcomes | Groundedness mean | min | max | range |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for item in stability["cases"]:
        g = item["groundedness"]
        lines.append(
            f"| {item['case_id']} | {item['repeats']} | "
            f"{', '.join(item['outcomes'])} | {g['mean']:.4f} | "
            f"{g['min']:.4f} | {g['max']:.4f} | {g['range']:.4f} |"
        )
    lines.extend(["", "Each observed score is listed in `evaluation_report.json` under stability.", ""])
    return "\n".join(lines) + "\n"


def prove_controlled_regression(
    *,
    artifacts_dir: Path = DEFAULT_ARTIFACTS,
    dataset_path: Path = Path("evals/golden_v1.json"),
    thresholds_path: Path = DEFAULT_THRESHOLDS,
    baseline_path: Path = DEFAULT_BASELINE,
    index_dir: Path = DEFAULT_INDEX,
) -> dict[str, Any]:
    normal = run_evaluation(
        dataset_path=dataset_path,
        thresholds_path=thresholds_path,
        baseline_path=baseline_path,
        artifacts_dir=artifacts_dir,
        index_dir=index_dir,
        top_k=APPROVED_TOP_K,
        write_reports=True,
    )
    weakened = run_evaluation(
        dataset_path=dataset_path,
        thresholds_path=thresholds_path,
        baseline_path=baseline_path,
        artifacts_dir=artifacts_dir / "_weakened",
        index_dir=index_dir,
        top_k=WEAK_TOP_K,
        write_reports=True,
    )
    restored = run_evaluation(
        dataset_path=dataset_path,
        thresholds_path=thresholds_path,
        baseline_path=baseline_path,
        artifacts_dir=artifacts_dir,
        index_dir=index_dir,
        top_k=APPROVED_TOP_K,
        write_reports=True,
    )
    proof = {
        "change": f"top_k {APPROVED_TOP_K} → {WEAK_TOP_K} → {APPROVED_TOP_K} (mode={APPROVED_MODE})",
        "normal": {
            "top_k": APPROVED_TOP_K,
            "verdict": normal.report["final_gate_verdict"],
            "exit_code": normal.gate.exit_code,
            "metrics": {
                key: normal.metrics[key]
                for key in ("hit_at_k", "mrr", "citation_validity", "refusal_accuracy", "groundedness", "attack_pass_rate")
            },
        },
        "regressed": {
            "top_k": WEAK_TOP_K,
            "verdict": weakened.report["final_gate_verdict"],
            "exit_code": weakened.gate.exit_code,
            "failed_thresholds": list(weakened.gate.threshold_failures),
            "metrics": {
                key: weakened.metrics[key]
                for key in ("hit_at_k", "mrr", "citation_validity", "refusal_accuracy", "groundedness", "attack_pass_rate")
            },
        },
        "restored": {
            "top_k": APPROVED_TOP_K,
            "verdict": restored.report["final_gate_verdict"],
            "exit_code": restored.gate.exit_code,
        },
        "passed": (
            normal.gate.passed
            and (not weakened.gate.passed)
            and weakened.gate.exit_code != 0
            and restored.gate.passed
        ),
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "controlled_regression.md").write_text(
        render_controlled_regression(proof),
        encoding="utf-8",
    )
    return proof


def render_controlled_regression(proof: dict[str, Any]) -> str:
    reg = proof["regressed"]
    lines = [
        "# Day 7 controlled regression proof",
        "",
        "Temporarily weaken retrieval with a reversible test-only setting, run the same gate, then restore.",
        "",
        f"Change: `{proof['change']}`",
        "",
        "| Stage | top_k | Verdict | Exit code | Hit@K | MRR |",
        "|---|---:|---|---:|---:|---:|",
        (
            f"| normal approved | {proof['normal']['top_k']} | {proof['normal']['verdict']} | "
            f"{proof['normal']['exit_code']} | {proof['normal']['metrics']['hit_at_k']:.4f} | "
            f"{proof['normal']['metrics']['mrr']:.4f} |"
        ),
        (
            f"| weakened retrieval | {reg['top_k']} | {reg['verdict']} | {reg['exit_code']} | "
            f"{reg['metrics']['hit_at_k']:.4f} | {reg['metrics']['mrr']:.4f} |"
        ),
        (
            f"| restored approved | {proof['restored']['top_k']} | {proof['restored']['verdict']} | "
            f"{proof['restored']['exit_code']} | {proof['normal']['metrics']['hit_at_k']:.4f} | "
            f"{proof['normal']['metrics']['mrr']:.4f} |"
        ),
        "",
        "## Failed thresholds on weakened run",
        "",
    ]
    if reg["failed_thresholds"]:
        lines.extend(f"- {item}" for item in reg["failed_thresholds"])
    else:
        lines.append("- (none recorded)")
    lines.extend(
        [
            "",
            f"Proof passed: **{proof['passed']}**",
            "",
            "Required: normal PASS, weakened FAIL with non-zero exit, restored PASS.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 7 evaluation harness and regression gate.")
    parser.add_argument("--dataset", type=Path, default=Path("evals/golden_v1.json"))
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Deliberate reviewed baseline rewrite. Never used by CI.",
    )
    parser.add_argument(
        "--weaken-retrieval",
        action="store_true",
        help="Test-only: run with top_k=1 so the gate should fail.",
    )
    parser.add_argument(
        "--prove-controlled-regression",
        action="store_true",
        help="Run approved → weakened → restored and write controlled_regression.md.",
    )
    return parser


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    args = build_parser().parse_args(argv)
    stream = stdout if stdout is not None else sys.stdout
    if args.prove_controlled_regression:
        proof = prove_controlled_regression(
            artifacts_dir=args.artifacts,
            dataset_path=args.dataset,
            thresholds_path=args.thresholds,
            baseline_path=args.baseline,
            index_dir=args.index,
        )
        stream.write(json.dumps({"controlled_regression": proof["passed"], "exit_codes": {
            "normal": proof["normal"]["exit_code"],
            "weakened": proof["regressed"]["exit_code"],
            "restored": proof["restored"]["exit_code"],
        }}, indent=2) + "\n")
        return 0 if proof["passed"] else 1
    top_k = WEAK_TOP_K if args.weaken_retrieval else APPROVED_TOP_K
    run = run_evaluation(
        dataset_path=args.dataset,
        thresholds_path=args.thresholds,
        baseline_path=args.baseline,
        artifacts_dir=args.artifacts,
        index_dir=args.index,
        top_k=top_k,
        update_baseline=args.update_baseline,
    )
    stream.write(f"gate={run.report['final_gate_verdict']} exit={run.gate.exit_code}\n")
    stream.write(
        f"Hit@{int(run.metrics['k'])}={run.metrics['hit_at_k']:.4f} "
        f"MRR={run.metrics['mrr']:.4f} "
        f"citation={run.metrics['citation_validity']:.4f} "
        f"refusal={run.metrics['refusal_accuracy']:.4f} "
        f"groundedness={run.metrics['groundedness']:.4f} "
        f"attack={run.metrics['attack_pass_rate']:.4f}\n"
    )
    return run.gate.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
