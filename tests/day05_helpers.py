"""Helpers for Day 5 deterministic tests. No cloud calls."""

from __future__ import annotations

import json
from pathlib import Path

from aico.contracts.models import SCHEMA_VERSION_V1
from aico.platform.config import testing_config
from aico.platform.fake_transport import FakeTransport
from aico.platform.model_gateway import ModelGateway
from aico.rag.answer_service import AnswerService, FixedRetriever
from aico.rag.prompt_builder import RetrievedChunk

ANSWER_CASES = Path("tests/fixtures/day05/answer_cases.json")
CITATION_CASES = Path("tests/fixtures/day05/citation_cases.json")
ATTACK_FIXTURES = Path("tests/fixtures/day05/attacks/attack_fixtures.json")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def answer_cases() -> dict[str, dict]:
    payload = load_json(ANSWER_CASES)
    return {item["id"]: item for item in payload["cases"]}


def citation_cases() -> dict[str, dict]:
    payload = load_json(CITATION_CASES)
    return {item["id"]: item for item in payload["cases"]}


def attack_fixtures() -> dict[str, dict]:
    payload = load_json(ATTACK_FIXTURES)
    return {item["id"]: item for item in payload["fixtures"]}


def chunks_from_case(case: dict) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=item["chunk_id"],
            text=item["text"],
            source_file="synthetic",
        )
        for item in case["retrieved"]
    ]


def cited_answer_json(
    *,
    status: str,
    answer: str,
    citations: list[dict] | None = None,
    confidence_label: str = "medium",
) -> str:
    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION_V1,
            "status": status,
            "answer": answer,
            "citations": citations or [],
            "confidence_label": confidence_label,
        }
    )


def make_service(
    chat_texts: list[str],
    chunks: list[RetrievedChunk],
    *,
    allow_repair: bool = True,
) -> tuple[AnswerService, FakeTransport, FixedRetriever]:
    transport = FakeTransport(chat_texts=chat_texts)
    gateway = ModelGateway(transport, testing_config(), sleep=lambda _seconds: None)
    retriever = FixedRetriever(chunks)
    service = AnswerService(gateway, retriever, allow_repair=allow_repair)
    return service, transport, retriever
