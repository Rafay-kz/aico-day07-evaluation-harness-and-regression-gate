"""Trusted identity is claims-only. Body identity cannot override it."""

from __future__ import annotations

from aico.api.dependencies import TrustedClaims
from tests.day06_helpers import IDENTITY_CASES, VALID_CLAIMS, ask_headers, load_cases, make_client


def test_valid_trusted_identity_allows_request() -> None:
    case = load_cases(IDENTITY_CASES)["ID-001"]
    claims = TrustedClaims(**case["trusted_claims"])
    client, service, _ = make_client(claims=claims)
    response = client.post("/ask", json={"question": "Synthetic test question"}, headers=ask_headers())
    assert response.status_code == 200
    assert service.calls


def test_missing_tenant_is_rejected() -> None:
    case = load_cases(IDENTITY_CASES)["ID-002"]
    client, service, transport = make_client(claims=TrustedClaims(**case["trusted_claims"]))
    response = client.post("/ask", json={"question": "Synthetic test question"}, headers=ask_headers())
    assert response.status_code == 401
    assert response.json()["code"] == "identity"
    assert "tenant" in response.json()["message"]
    assert service.calls == []
    assert transport.chat_calls == 0


def test_missing_user_is_rejected() -> None:
    case = load_cases(IDENTITY_CASES)["ID-003"]
    client, service, transport = make_client(claims=TrustedClaims(**case["trusted_claims"]))
    response = client.post("/ask", json={"question": "Synthetic test question"}, headers=ask_headers())
    assert response.status_code == 401
    assert response.json()["code"] == "identity"
    assert "user" in response.json()["message"]
    assert service.calls == []
    assert transport.chat_calls == 0


def test_empty_identity_is_rejected() -> None:
    case = load_cases(IDENTITY_CASES)["ID-004"]
    client, service, transport = make_client(claims=TrustedClaims(**case["trusted_claims"]))
    response = client.post("/ask", json={"question": "Synthetic test question"}, headers=ask_headers())
    assert response.status_code == 401
    assert response.json()["code"] == "identity"
    assert service.calls == []
    assert transport.chat_calls == 0


def test_body_identity_cannot_override_trusted_claims() -> None:
    case = load_cases(IDENTITY_CASES)["ID-005"]
    client, _, _ = make_client(claims=TrustedClaims(**case["trusted_claims"]))
    payload = {
        "question": "Synthetic test question",
        **case["request_body_extra_identity"],
    }
    response = client.post("/ask", json=payload, headers=ask_headers())
    assert response.status_code == 200
    body = response.json()
    dumped = json_dump(body)
    assert "TENANT-ATTACK" not in dumped
    assert "USER-ATTACK" not in dumped
    assert VALID_CLAIMS.tenant_id not in dumped
    assert VALID_CLAIMS.user_id not in dumped


def json_dump(payload: dict) -> str:
    from json import dumps

    return dumps(payload)
