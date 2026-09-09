from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from aico.contracts.errors import TypedFailure
from aico.contracts.models import SCHEMA_VERSION_V1, CitedAnswer, ConfidenceLabel, Status
from aico.contracts.service import ingest_cited_answer
from aico.observability.logging import log_operation
from aico.observability.telemetry import PipelineStats, get_tracer, span_attributes
from aico.platform.model_gateway import CancellationToken, ChatRequest, ModelGateway
from aico.rag.citation_validator import (
    CitationValidationResult,
    validate_answer_citations,
)
from aico.rag.evidence_guard import unsupported_poisoned_claims
from aico.rag.prompt_builder import RetrievedChunk, build_messages
from aico.retrieval.search import search
from aico.security.input_policy import PolicyDecision, evaluate_policy
from aico.security.normalization import normalize

PolicyFn = Callable[[str], PolicyDecision]


class Retriever(Protocol):
    def retrieve(self, question: str) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class GroundedAnswer:
    question: str
    cited_answer: CitedAnswer
    retrieved_chunk_ids: tuple[str, ...]
    citation_validation: CitationValidationResult


@dataclass(frozen=True)
class InsufficientEvidence:
    question: str
    cited_answer: CitedAnswer
    retrieved_chunk_ids: tuple[str, ...]
    explanation: str
    invented_fact: bool = False
    invented_citation: bool = False


@dataclass(frozen=True)
class Clarify:
    question: str
    policy: PolicyDecision


@dataclass(frozen=True)
class Blocked:
    question: str
    policy: PolicyDecision


AnswerResult = GroundedAnswer | InsufficientEvidence | Clarify | Blocked | TypedFailure


class FixedRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = list(chunks)
        self.calls: list[str] = []

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        self.calls.append(question)
        return list(self.chunks)


class Day2Retriever:
    def __init__(
        self,
        index_dir: Path = Path("data/index"),
        *,
        mode: str = "bm25",
        top_k: int = 5,
        vectors_dir: Path | None = None,
        provider=None,
    ) -> None:
        self.index_dir = index_dir
        self.mode = mode
        self.top_k = top_k
        self.vectors_dir = vectors_dir
        self.provider = provider
        self.calls: list[str] = []

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        self.calls.append(question)
        hits = search(
            question,
            self.index_dir,
            self.top_k,
            mode=self.mode,
            vectors_dir=self.vectors_dir,
            provider=self.provider,
        )
        return [
            RetrievedChunk(
                chunk_id=hit["chunk_id"],
                text=hit["text"],
                source_file=hit["source_file"],
            )
            for hit in hits
        ]


class AnswerService:
    def __init__(
        self,
        gateway: ModelGateway,
        retriever: Retriever,
        *,
        allow_repair: bool = True,
        policy: PolicyFn = evaluate_policy,
    ) -> None:
        self._gateway = gateway
        self._retriever = retriever
        self._allow_repair = allow_repair
        self._policy = policy

    def answer(
        self,
        question: str,
        cancellation: CancellationToken | None = None,
        *,
        correlation_id: str = "",
        request_id: str = "",
        stats: PipelineStats | None = None,
    ) -> AnswerResult:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        tracer = get_tracer()
        collected = stats if stats is not None else PipelineStats()
        attrs = span_attributes(request_id=request_id, correlation_id=correlation_id)

        normalize(question)  # always before policy; evaluate_policy also normalizes
        with tracer.start_as_current_span("policy", attributes=attrs):
            started = time.perf_counter()
            policy = self._policy(question)
            log_operation(
                stage="policy",
                outcome=policy.outcome,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        if policy.outcome == "block":
            return Blocked(question=question, policy=policy)
        if policy.outcome == "clarify":
            return Clarify(question=question, policy=policy)

        if cancellation is not None:
            cancellation.raise_if_cancelled()
        with tracer.start_as_current_span("retrieval", attributes=attrs):
            started = time.perf_counter()
            chunks = self._retriever.retrieve(question)
            collected.retrieval_latency_ms = (time.perf_counter() - started) * 1000.0
            collected.cache_hit = bool(getattr(self._retriever, "last_cache_hit", False))
            log_operation(
                stage="retrieval",
                outcome="ok",
                latency_ms=collected.retrieval_latency_ms,
            )
        retrieved_ids = tuple(chunk.chunk_id for chunk in chunks)
        messages = build_messages(question, chunks)

        if cancellation is not None:
            cancellation.raise_if_cancelled()
        with tracer.start_as_current_span("model_gateway", attributes=attrs):
            chat = self._gateway.chat(
                ChatRequest(messages=messages, cancellation=cancellation)
            )
            collected.gateway_latency_ms = chat.metadata.latency_ms
            collected.retry_count = chat.metadata.retry_count
            collected.model_alias = chat.metadata.model_alias
            collected.token_input = chat.metadata.token_usage.input_tokens
            collected.token_output = chat.metadata.token_usage.output_tokens
            collected.token_total = chat.metadata.token_usage.total_tokens
            log_operation(
                stage="model_gateway",
                outcome="ok",
                latency_ms=collected.gateway_latency_ms,
            )

        with tracer.start_as_current_span("validation", attributes=attrs):
            started = time.perf_counter()
            typed = ingest_cited_answer(
                chat.text,
                gateway=self._gateway,
                allow_repair=self._allow_repair,
                cancellation=cancellation,
            )
            if isinstance(typed, TypedFailure):
                log_operation(
                    stage="validation",
                    outcome="failure",
                    error_category=typed.category.value,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )
                return typed

            citation_result = validate_answer_citations(typed, retrieved_ids)
            if not citation_result.passed:
                log_operation(
                    stage="validation",
                    outcome="failure",
                    error_category="citation",
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )
                # Fail closed: do not drop forged IDs and keep the answer.
                return citation_result.to_failure()
            poison_reason = unsupported_poisoned_claims(typed.answer, chunks)
            if typed.status == Status.answered and poison_reason:
                log_operation(
                    stage="validation",
                    outcome="failure",
                    error_category="unsupported_claim",
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )
                return InsufficientEvidence(
                    question=question,
                    cited_answer=CitedAnswer(
                        schema_version=SCHEMA_VERSION_V1,
                        status=Status.insufficient_evidence,
                        answer="INSUFFICIENT_EVIDENCE: retrieved evidence does not support the claimed facts.",
                        citations=[],
                        confidence_label=ConfidenceLabel.low,
                    ),
                    retrieved_chunk_ids=retrieved_ids,
                    explanation=poison_reason,
                    invented_fact=True,
                    invented_citation=False,
                )
            log_operation(
                stage="validation",
                outcome="ok",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        if typed.status == Status.insufficient_evidence:
            return InsufficientEvidence(
                question=question,
                cited_answer=typed,
                retrieved_chunk_ids=retrieved_ids,
                explanation=typed.answer,
                invented_fact=False,
                invented_citation=False,
            )

        return GroundedAnswer(
            question=question,
            cited_answer=typed,
            retrieved_chunk_ids=retrieved_ids,
            citation_validation=citation_result,
        )
