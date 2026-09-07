"""Model-based groundedness evaluation through the existing Model Gateway."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from aico.contracts.models import SCHEMA_VERSION_V1, CitedAnswer, Status
from aico.platform.fake_transport import FakeTransport
from aico.platform.model_gateway import ChatMessage, ChatRequest, ModelGateway, TokenUsage, TransportChatResponse
from aico.rag.prompt_builder import EVIDENCE_LABEL, USER_LABEL, RetrievedChunk
from aico.retrieval.bm25 import tokenize_for_search

EVALUATOR_VERSION = "aico-groundedness-v1"
EVALUATOR_MARKER = "AICO_GROUNDEDNESS_EVALUATOR_V1"
SPECIFIC_HINTS = frozenset(
    {
        "rate",
        "vat",
        "ceo",
        "chief",
        "executive",
        "manchester",
        "warehouse",
    }
)
GENERIC_TERMS = frozenset(
    {
        "meridian",
        "procurement",
        "ltd",
        "limited",
        "organisation",
        "organization",
        "document",
        "doc",
        "policy",
    }
)

EVALUATOR_INSTRUCTIONS = f"""{EVALUATOR_MARKER}
You are a groundedness grader, version {EVALUATOR_VERSION}.

Score whether the SYSTEM ANSWER is supported by RETRIEVED EVIDENCE.
Do not rewrite, improve, or replace the system answer.
Do not answer the user question yourself.

Return only JSON with fields:
- evaluator_version (exactly "{EVALUATOR_VERSION}")
- score (number 0 to 1)
- label ("grounded", "partial", or "ungrounded")
- claim_count (integer)
- supported_claim_count (integer)
- reason (short string)
- rewrites_answer (always false)
"""


@dataclass(frozen=True)
class GroundednessResult:
    score: float
    label: str
    claim_count: int
    supported_claim_count: int
    reason: str
    evaluator_version: str
    model_alias: str
    raw_text: str
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


class EvalLabTransport(FakeTransport):
    """Deterministic lab transport used for answering and grading through ModelGateway.

    Callers still go through ModelGateway. This transport never talks to Foundry.
    """

    def __init__(self) -> None:
        super().__init__(embed_dimensions=8, chat_text="{}")

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model_alias: str,
        timeout_seconds: float,
        cancellation,
    ) -> TransportChatResponse:
        self._prepare(timeout_seconds, cancellation)
        self.chat_calls += 1
        self.last_chat_messages = list(messages)
        joined = "\n".join(message.content for message in messages)
        if EVALUATOR_MARKER in joined:
            text = grade_groundedness(messages)
        else:
            text = answer_from_evidence(messages)
        self.chat_text = text
        return TransportChatResponse(text=text, token_usage=TokenUsage())


def build_groundedness_messages(
    question: str,
    answer_text: str,
    chunks: list[RetrievedChunk],
) -> list[ChatMessage]:
    evidence = "\n\n".join(
        f"[{chunk.chunk_id}] source={chunk.source_file}\n{chunk.text}" for chunk in chunks
    ) or "(no chunks retrieved)"
    user = (
        f"QUESTION\n{question}\n\n"
        f"SYSTEM ANSWER\n{answer_text}\n\n"
        f"RETRIEVED EVIDENCE\n{evidence}"
    )
    return [
        ChatMessage(role="system", content=EVALUATOR_INSTRUCTIONS),
        ChatMessage(role="user", content=user),
    ]


def evaluate_groundedness(
    gateway: ModelGateway,
    question: str,
    answer_text: str,
    chunks: list[RetrievedChunk],
) -> GroundednessResult:
    messages = build_groundedness_messages(question, answer_text, chunks)
    chat = gateway.chat(ChatRequest(messages=messages))
    return parse_groundedness(chat.text, model_alias=chat.metadata.model_alias)


def parse_groundedness(text: str, *, model_alias: str) -> GroundednessResult:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return _evaluator_error(text, model_alias, f"evaluator JSON parse failed: {exc}")
    if not isinstance(payload, dict):
        return _evaluator_error(text, model_alias, "evaluator result is not an object")
    if payload.get("rewrites_answer") is True:
        return _evaluator_error(text, model_alias, "evaluator attempted to rewrite the answer")
    try:
        score = float(payload["score"])
        label = str(payload["label"])
        claim_count = int(payload["claim_count"])
        supported = int(payload["supported_claim_count"])
        version = str(payload.get("evaluator_version") or "")
    except (KeyError, TypeError, ValueError) as exc:
        return _evaluator_error(text, model_alias, f"evaluator fields missing: {exc}")
    if version != EVALUATOR_VERSION:
        return _evaluator_error(text, model_alias, f"unexpected evaluator version {version!r}")
    if not 0.0 <= score <= 1.0:
        return _evaluator_error(text, model_alias, "evaluator score out of range")
    return GroundednessResult(
        score=round(score, 4),
        label=label,
        claim_count=claim_count,
        supported_claim_count=supported,
        reason=str(payload.get("reason") or ""),
        evaluator_version=version,
        model_alias=model_alias,
        raw_text=text,
    )


def answer_from_evidence(messages: list[ChatMessage]) -> str:
    question, chunks = _parse_answer_prompt(messages)
    selected = _supporting_spans(question, chunks)
    if not selected:
        return json.dumps(
            {
                "schema_version": SCHEMA_VERSION_V1,
                "status": Status.insufficient_evidence.value,
                "answer": "INSUFFICIENT_EVIDENCE: retrieved evidence does not support a specific answer.",
                "citations": [],
                "confidence_label": "low",
            }
        )
    answer_text = " ".join(span.strip() for _, span in selected)
    citations = []
    seen: set[str] = set()
    for chunk, _span in selected:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        citations.append({"chunk_id": chunk.chunk_id, "source_file": chunk.source_file})
    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION_V1,
            "status": Status.answered.value,
            "answer": answer_text,
            "citations": citations,
            "confidence_label": "medium",
        }
    )


def grade_groundedness(messages: list[ChatMessage]) -> str:
    question, answer_text, chunks = _parse_grade_prompt(messages)
    claims = _split_claims(answer_text)
    if answer_text.upper().startswith("INSUFFICIENT_EVIDENCE"):
        supported = len(claims)
        score = 1.0
        label = "grounded"
        reason = "refusal is treated as grounded when the answer does not add unsupported facts"
    elif not claims:
        supported = 0
        score = 0.0
        label = "ungrounded"
        reason = "no scorable claims"
    else:
        evidence_terms = set()
        evidence_text = " ".join(chunk.text for chunk in chunks)
        evidence_terms.update(tokenize_for_search(evidence_text))
        supported = 0
        for claim in claims:
            if _claim_supported(claim, evidence_text, evidence_terms):
                supported += 1
        score = supported / len(claims)
        if score >= 0.99:
            label = "grounded"
        elif score >= 0.5:
            label = "partial"
        else:
            label = "ungrounded"
        reason = f"{supported}/{len(claims)} claims supported by retrieved evidence"
    payload = {
        "evaluator_version": EVALUATOR_VERSION,
        "score": round(score, 4),
        "label": label,
        "claim_count": len(claims),
        "supported_claim_count": supported,
        "reason": reason,
        "rewrites_answer": False,
        "question_echo": question[:80],
    }
    return json.dumps(payload)


def cited_answer_text(answer: CitedAnswer) -> str:
    return answer.answer


def _evaluator_error(text: str, model_alias: str, error: str) -> GroundednessResult:
    return GroundednessResult(
        score=0.0,
        label="ungrounded",
        claim_count=0,
        supported_claim_count=0,
        reason=error,
        evaluator_version=EVALUATOR_VERSION,
        model_alias=model_alias,
        raw_text=text,
        error=error,
    )


def _parse_answer_prompt(messages: list[ChatMessage]) -> tuple[str, list[RetrievedChunk]]:
    user = next((message.content for message in messages if message.role == "user"), "")
    question = ""
    if f"{USER_LABEL} (untrusted)" in user:
        after = user.split(f"{USER_LABEL} (untrusted)", 1)[1]
        question = after.split(f"{EVIDENCE_LABEL}", 1)[0].strip()
    chunks = _parse_chunks(user)
    return question, chunks


def _parse_grade_prompt(messages: list[ChatMessage]) -> tuple[str, str, list[RetrievedChunk]]:
    user = next((message.content for message in messages if message.role == "user"), "")
    question = _section(user, "QUESTION")
    answer_text = _section(user, "SYSTEM ANSWER")
    evidence = _section(user, "RETRIEVED EVIDENCE")
    chunks = _parse_chunks(evidence)
    return question, answer_text, chunks


def _section(text: str, heading: str) -> str:
    pattern = rf"{re.escape(heading)}\n(.*?)(?=\n[A-Z][A-Z ]+\n|\Z)"
    match = re.search(pattern, text, flags=re.S)
    return match.group(1).strip() if match else ""


def _parse_chunks(text: str) -> list[RetrievedChunk]:
    blocks = re.findall(
        r"\[([^\]]+)\] source=(\S+)\n(.*?)(?=\n\[[^\]]+\] source=|\Z)",
        text,
        flags=re.S,
    )
    return [
        RetrievedChunk(chunk_id=chunk_id, source_file=source, text=body.strip())
        for chunk_id, source, body in blocks
        if body.strip() and body.strip() != "(no chunks retrieved)"
    ]


def _supporting_spans(
    question: str,
    chunks: list[RetrievedChunk],
) -> list[tuple[RetrievedChunk, str]]:
    required = set(tokenize_for_search(question)) - GENERIC_TERMS
    if not required:
        required = set(tokenize_for_search(question))
    specific = set(tokenize_for_search(question)) & SPECIFIC_HINTS
    evidence_terms: set[str] = set()
    for chunk in chunks:
        evidence_terms.update(tokenize_for_search(chunk.text))
    if specific - evidence_terms:
        return []
    scored: list[tuple[int, RetrievedChunk, str]] = []
    for chunk in chunks:
        terms = set(tokenize_for_search(chunk.text))
        overlap = len(required & terms)
        if overlap:
            scored.append((overlap, chunk, chunk.text))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return []
    used_terms: set[str] = set()
    spans: list[tuple[RetrievedChunk, str]] = []
    for _overlap, chunk, text in scored:
        spans.append((chunk, text))
        used_terms.update(set(tokenize_for_search(text)) & required)
        if len(spans) >= 2:
            break
    coverage = len(used_terms) / len(required) if required else 0.0
    if coverage <= 0.5 or len(used_terms) < 2:
        return []
    return spans


def _claim_supported(claim: str, evidence_text: str, evidence_terms: set[str]) -> bool:
    folded_claim = " ".join(claim.lower().split())
    folded_evidence = " ".join(evidence_text.lower().split())
    if folded_claim and folded_claim in folded_evidence:
        return True
    terms = set(tokenize_for_search(claim)) - GENERIC_TERMS
    if not terms:
        return True
    return (len(terms & evidence_terms) / len(terms)) >= 0.5


def _split_claims(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if len(part.strip()) > 8]


def groundedness_to_dict(result: GroundednessResult) -> dict[str, Any]:
    return {
        "score": result.score,
        "label": result.label,
        "claim_count": result.claim_count,
        "supported_claim_count": result.supported_claim_count,
        "reason": result.reason,
        "evaluator_version": result.evaluator_version,
        "model_alias": result.model_alias,
        "error": result.error,
    }
