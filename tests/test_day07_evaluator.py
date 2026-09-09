"""Low-groundedness classification, evaluator errors, and transport failures."""

from __future__ import annotations

from aico.evals.dataset import GoldenCase
from aico.evals.day07 import GROUNDEDNESS_FAIL_SCORE, _case_failed, aggregate_metrics
from aico.evals.failure_classifier import classify_failure
from aico.evals.groundedness import EVALUATOR_MARKER, EvalLabTransport, evaluate_groundedness
from aico.evals.regression import apply_gate, load_thresholds
from aico.platform.config import testing_config as gateway_testing_config
from aico.platform.model_gateway import ModelGateway
from aico.rag.prompt_builder import RetrievedChunk


def _answerable_case() -> GoldenCase:
    return GoldenCase(
        case_id="G-TEST",
        category="answerable",
        split="train",
        question="What is the cover?",
        expected_sources=("DOC-004",),
        answerability="answerable",
        critical_facts=("five million pounds",),
        prohibited_claims=(),
        expected_outcome="answer",
    )


def test_low_groundedness_is_a_failed_classified_case() -> None:
    failed = _case_failed(
        _answerable_case(),
        refusal_correct=True,
        hit=True,
        citation_valid=True,
        prohibited=(),
        missing_facts=(),
        evaluator_error=None,
        attack_pass=None,
        groundedness_score=0.0,
        grounded_evaluated=True,
    )
    assert failed is True
    assert (
        classify_failure(
            {
                "category": "answerable",
                "expected_outcome": "answer",
                "observed_outcome": "answer",
                "retrieval_hit": True,
                "citation_valid": True,
                "groundedness_score": 0.0,
            }
        )
        == "prompt"
    )
    assert GROUNDEDNESS_FAIL_SCORE <= 0.5


def test_evaluator_error_is_included_in_groundedness_mean_and_fails_gate() -> None:
    rows = [
        {
            "hit_at_k": True,
            "reciprocal_rank": 1.0,
            "citation_valid": True,
            "refusal_correct": True,
            "attack_pass": None,
            "category": "answerable",
            "evaluator_error": None,
            "groundedness": {"score": 1.0, "error": None},
        },
        {
            "hit_at_k": True,
            "reciprocal_rank": 1.0,
            "citation_valid": True,
            "refusal_correct": True,
            "attack_pass": None,
            "category": "answerable",
            "evaluator_error": "evaluator JSON parse failed",
            "groundedness": {"score": 0.0, "error": "evaluator JSON parse failed"},
        },
    ]
    metrics = aggregate_metrics(rows, k=5)
    assert metrics["groundedness"] == 0.5
    assert metrics["n_evaluator_errors"] == 1.0
    thresholds = load_thresholds()
    perfect = {name: 1.0 for name in thresholds.metrics}
    decision = apply_gate(
        perfect,
        thresholds,
        baseline=None,
        safety_failures=0,
        evaluator_failures=1,
    )
    assert decision.passed is False
    assert decision.evaluator_failed is True
    assert decision.exit_code != 0


class _BoomEvalTransport(EvalLabTransport):
    def chat(self, messages, *, model_alias, timeout_seconds, cancellation):
        joined = "\n".join(message.content for message in messages)
        if EVALUATOR_MARKER in joined:
            raise RuntimeError("forced evaluator outage")
        return super().chat(
            messages,
            model_alias=model_alias,
            timeout_seconds=timeout_seconds,
            cancellation=cancellation,
        )


def test_gateway_exception_becomes_an_evaluator_row() -> None:
    gateway = ModelGateway(
        _BoomEvalTransport(),
        gateway_testing_config(chat_alias="eval-lab-chat-v1"),
        sleep=lambda _seconds: None,
    )
    result = evaluate_groundedness(
        gateway,
        "What cover is required?",
        "Suppliers must hold public liability cover of five million pounds.",
        [RetrievedChunk(chunk_id="c1", text="five million pounds", source_file="DOC-004")],
    )
    assert result.error is not None
    assert "evaluator transport failed" in result.error
    assert result.failed is True
