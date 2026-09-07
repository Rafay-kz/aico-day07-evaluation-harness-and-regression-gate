"""Day 6 API contract, OpenAPI, protections, correlation, and error envelope."""

from __future__ import annotations

import json

from aico.api.contracts import AskRequest, AskResponse
from aico.contracts.models import CitedAnswer
from aico.rag.answer_service import GroundedAnswer as RagGroundedAnswer
from tests.day06_helpers import (
    API_CASES,
    SUPPORTED_QUESTION,
    ask_headers,
    load_cases,
    make_client,
)


def test_openapi_documents_ask_with_public_models() -> None:
    client, _, _ = make_client()
    schema = client.get("/openapi.json").json()
    assert "/ask" in schema["paths"]
    ask = schema["paths"]["/ask"]["post"]
    assert "AskRequest" in json.dumps(ask)
    assert "AskResponse" in json.dumps(schema)
    components = schema["components"]["schemas"]
    assert "AskRequest" in components
    assert "AskResponse" in components
    assert "ApiErrorBody" in components
    dumped = json.dumps(schema)
    assert "GroundedAnswer" not in dumped
    assert "CitedAnswer" not in dumped
    assert "invented_fact" not in dumped
    assert "citation_validation" not in dumped


def test_api_contracts_are_not_internal_domain_models() -> None:
    assert AskRequest is not CitedAnswer
    assert AskResponse is not CitedAnswer
    assert AskResponse is not RagGroundedAnswer
    assert "schema_version" not in AskResponse.model_fields
    assert "question" in AskRequest.model_fields
    assert "outcome" in AskResponse.model_fields


def test_valid_ask_returns_typed_public_response() -> None:
    case = load_cases(API_CASES)["API-001"]
    client, service, transport = make_client()
    response = client.post("/ask", json=case["payload"], headers=ask_headers())
    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "answered"
    assert body["answer"]
    assert body["request_id"]
    assert body["correlation_id"]
    assert body["citations"][0]["chunk_id"] == "CHUNK-101"
    assert service.calls == [case["payload"]["question"]]
    assert transport.chat_calls == 1
    AskResponse.model_validate(body)


def test_unsupported_content_type_is_4xx_before_pipeline() -> None:
    case = load_cases(API_CASES)["API-002"]
    client, service, transport = make_client()
    response = client.post(
        "/ask",
        content=str(case["payload"]),
        headers={"Content-Type": case["content_type"]},
    )
    assert 400 <= response.status_code < 500
    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_media_type"
    assert service.calls == []
    assert transport.chat_calls == 0


def test_oversize_request_is_4xx_before_pipeline() -> None:
    client, service, transport = make_client(max_request_bytes=64)
    payload = {"question": "x" * 200}
    response = client.post("/ask", json=payload, headers=ask_headers())
    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"
    assert "request_id" in response.json()
    assert "correlation_id" in response.json()
    assert service.calls == []
    assert transport.chat_calls == 0


def test_invalid_request_is_4xx_with_error_envelope() -> None:
    case = load_cases(API_CASES)["API-004"]
    client, service, _ = make_client()
    response = client.post("/ask", json=case["payload"], headers=ask_headers())
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "invalid_request"
    assert body["message"]
    assert body["request_id"]
    assert body["correlation_id"]
    assert "traceback" not in body
    assert service.calls == []


def test_missing_correlation_id_is_generated() -> None:
    case = load_cases(API_CASES)["API-005"]
    client, _, _ = make_client()
    response = client.post("/ask", json=case["payload"], headers=ask_headers())
    assert response.status_code == 200
    body = response.json()
    assert body["correlation_id"]
    assert response.headers["X-Correlation-ID"] == body["correlation_id"]
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_incoming_correlation_id_is_preserved() -> None:
    client, _, _ = make_client()
    response = client.post(
        "/ask",
        json={"question": SUPPORTED_QUESTION},
        headers=ask_headers(**{"X-Correlation-ID": "corr-fixed-001", "X-Request-ID": "req-fixed-001"}),
    )
    assert response.status_code == 200
    assert response.json()["correlation_id"] == "corr-fixed-001"
    assert response.json()["request_id"] == "req-fixed-001"
    assert response.headers["X-Correlation-ID"] == "corr-fixed-001"
