from __future__ import annotations

import os
from dataclasses import dataclass, field

from aico.api.errors import ApiError
from aico.observability.metrics import MetricsRegistry, get_metrics
from aico.platform.model_gateway import ModelGateway
from aico.rag.answer_service import AnswerService, Retriever
from aico.security.input_policy import PolicyDecision, evaluate_policy


@dataclass(frozen=True)
class TrustedClaims:
    tenant_id: str | None = None
    user_id: str | None = None


@dataclass
class ApiSettings:
    max_request_bytes: int = 8192
    json_content_type: str = "application/json"


@dataclass
class AppContainer:
    answer_service: AnswerService
    health: object
    settings: ApiSettings = field(default_factory=ApiSettings)
    gateway: ModelGateway | None = None
    retriever: Retriever | None = None
    policy: object = evaluate_policy
    metrics: MetricsRegistry = field(default_factory=get_metrics)


def env_trusted_claims() -> TrustedClaims:
    """Lab stand-in for verified authentication claims. Not request-body identity."""
    tenant = (os.environ.get("AICO_TRUSTED_TENANT_ID") or "").strip() or None
    user = (os.environ.get("AICO_TRUSTED_USER_ID") or "").strip() or None
    return TrustedClaims(tenant_id=tenant, user_id=user)


def get_trusted_claims() -> TrustedClaims:
    return env_trusted_claims()


def require_trusted_identity(claims: TrustedClaims) -> TrustedClaims:
    if not claims.tenant_id and not claims.user_id:
        raise ApiError(
            401,
            "identity",
            "trusted tenant and user claims are required",
        )
    if not claims.tenant_id:
        raise ApiError(401, "identity", "trusted tenant claim is required")
    if not claims.user_id:
        raise ApiError(401, "identity", "trusted user claim is required")
    return claims


def default_policy(question: str) -> PolicyDecision:
    return evaluate_policy(question)
