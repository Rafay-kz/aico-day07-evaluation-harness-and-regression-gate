from __future__ import annotations

import asyncio
import json
import time
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from aico.api.contracts import ApiErrorBody, AskRequest, AskResponse, PublicCitation
from aico.api.dependencies import (
    ApiSettings,
    AppContainer,
    TrustedClaims,
    get_trusted_claims,
    require_trusted_identity,
)
from aico.api.errors import ApiError, error_response
from aico.api.health import (
    HealthService,
    configuration_probe,
    gateway_configured_probe,
    retrieval_index_probe,
)
from aico.contracts.errors import TypedFailure
from aico.observability.logging import bind_request_context, log_operation
from aico.observability.metrics import get_metrics
from aico.observability.telemetry import (
    PipelineStats,
    get_tracer,
    setup_telemetry,
    span_attributes,
)
from aico.platform.config import testing_config
from aico.platform.errors import OperationCancelled
from aico.platform.fake_transport import FakeTransport
from aico.platform.model_gateway import CancellationToken, ModelGateway
from aico.rag.answer_service import (
    AnswerService,
    Blocked,
    Clarify,
    FixedRetriever,
    GroundedAnswer,
    InsufficientEvidence,
)
from aico.rag.prompt_builder import RetrievedChunk
from aico.security.input_policy import evaluate_policy

JSON_CONTENT_TYPE = "application/json"


def _ids_from(request: Request) -> tuple[str, str]:
    request_id = getattr(request.state, "request_id", "") or str(uuid4())
    correlation_id = getattr(request.state, "correlation_id", "") or str(uuid4())
    return request_id, correlation_id


def _header_map(scope: dict) -> dict[str, str]:
    return {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in scope.get("headers", [])
    }


async def _send_json(
    send,
    status_code: int,
    payload: dict,
    *,
    request_id: str,
    correlation_id: str,
) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"x-request-id", request_id.encode("latin-1")),
                (b"x-correlation-id", correlation_id.encode("latin-1")),
                (b"content-length", str(len(body)).encode("latin-1")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class RequestBoundaryMiddleware:
    """ASGI middleware for IDs, Content-Type, and size. Avoids BaseHTTPMiddleware."""

    def __init__(
        self,
        app,
        *,
        max_request_bytes: int,
        json_content_type: str = JSON_CONTENT_TYPE,
    ) -> None:
        self.app = app
        self.max_request_bytes = max_request_bytes
        self.json_content_type = json_content_type

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = _header_map(scope)
        request_id = headers.get("x-request-id") or str(uuid4())
        correlation_id = headers.get("x-correlation-id") or str(uuid4())
        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id
        scope["state"]["correlation_id"] = correlation_id
        bind_request_context(request_id=request_id, correlation_id=correlation_id)

        async def send_with_ids(message):
            if message["type"] == "http.response.start":
                existing = list(message.get("headers", []))
                existing.append((b"x-request-id", request_id.encode("latin-1")))
                existing.append((b"x-correlation-id", correlation_id.encode("latin-1")))
                message = dict(message)
                message["headers"] = existing
            await send(message)

        path = scope.get("path", "")
        method = scope.get("method", "")
        if method == "POST" and path.rstrip("/") == "/ask":
            content_type = (headers.get("content-type") or "").split(";")[0].strip().lower()
            if content_type != self.json_content_type:
                log_operation(
                    stage="api",
                    outcome="rejected",
                    error_category="unsupported_media_type",
                )
                await _send_json(
                    send_with_ids,
                    415,
                    {
                        "code": "unsupported_media_type",
                        "message": "POST /ask requires application/json",
                        "request_id": request_id,
                        "correlation_id": correlation_id,
                    },
                    request_id=request_id,
                    correlation_id=correlation_id,
                )
                return
            content_length = headers.get("content-length")
            if content_length is not None:
                try:
                    length = int(content_length)
                except ValueError:
                    length = 0
                if length > self.max_request_bytes:
                    log_operation(
                        stage="api",
                        outcome="rejected",
                        error_category="payload_too_large",
                    )
                    await _send_json(
                        send_with_ids,
                        413,
                        {
                            "code": "payload_too_large",
                            "message": "request exceeds the configured size limit",
                            "request_id": request_id,
                            "correlation_id": correlation_id,
                        },
                        request_id=request_id,
                        correlation_id=correlation_id,
                    )
                    return

            chunks: list[bytes] = []
            more_body = True
            while more_body:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                if message["type"] != "http.request":
                    continue
                chunks.append(message.get("body", b"") or b"")
                more_body = bool(message.get("more_body", False))
                if sum(len(part) for part in chunks) > self.max_request_bytes:
                    log_operation(
                        stage="api",
                        outcome="rejected",
                        error_category="payload_too_large",
                    )
                    await _send_json(
                        send_with_ids,
                        413,
                        {
                            "code": "payload_too_large",
                            "message": "request exceeds the configured size limit",
                            "request_id": request_id,
                            "correlation_id": correlation_id,
                        },
                        request_id=request_id,
                        correlation_id=correlation_id,
                    )
                    return
            body = b"".join(chunks)
            replayed = False

            async def replay_receive():
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            await self.app(scope, replay_receive, send_with_ids)
            return

        await self.app(scope, receive, send_with_ids)


def _map_result(
    result: GroundedAnswer | InsufficientEvidence | Clarify | Blocked | TypedFailure,
    *,
    request_id: str,
    correlation_id: str,
) -> AskResponse:
    if isinstance(result, GroundedAnswer):
        cited = result.cited_answer
        return AskResponse(
            request_id=request_id,
            correlation_id=correlation_id,
            outcome="answered",
            answer=cited.answer,
            citations=[
                PublicCitation(chunk_id=item.chunk_id, source_file=item.source_file)
                for item in cited.citations
            ],
            confidence_label=cited.confidence_label.value,
        )
    if isinstance(result, InsufficientEvidence):
        cited = result.cited_answer
        return AskResponse(
            request_id=request_id,
            correlation_id=correlation_id,
            outcome="insufficient_evidence",
            answer=cited.answer,
            citations=[],
            confidence_label=cited.confidence_label.value,
        )
    if isinstance(result, Clarify):
        return AskResponse(
            request_id=request_id,
            correlation_id=correlation_id,
            outcome="clarify",
            answer=result.policy.reason,
            citations=[],
        )
    if isinstance(result, Blocked):
        return AskResponse(
            request_id=request_id,
            correlation_id=correlation_id,
            outcome="blocked",
            answer=result.policy.reason,
            citations=[],
        )
    raise ApiError(
        422,
        result.category.value,
        "the grounded answer contract was rejected",
        request_id=request_id,
        correlation_id=correlation_id,
    )


_LAB_ANSWER = "Synthetic supplier payment terms are net 30."
_LAB_CHAT_JSON = json.dumps(
    {
        "schema_version": "1.0",
        "status": "answered",
        "answer": _LAB_ANSWER,
        "citations": [{"chunk_id": "CHUNK-101", "source_file": "synthetic"}],
        "confidence_label": "high",
    }
)


def build_lab_container() -> AppContainer:
    # Live uvicorn uses a fake model. It must return valid Day 4 JSON, not
    # "gateway-chat-ok", or /ask fails parse after a successful HTTP call.
    transport = FakeTransport(chat_text=_LAB_CHAT_JSON)
    gateway = ModelGateway(transport, testing_config(), sleep=lambda _seconds: None)
    retriever = FixedRetriever(
        [
            RetrievedChunk(
                chunk_id="CHUNK-101",
                text=_LAB_ANSWER,
                source_file="synthetic",
            )
        ]
    )
    service = AnswerService(gateway, retriever, policy=evaluate_policy)
    health = HealthService(
        {
            "retrieval": retrieval_index_probe(),
            "model_gateway": gateway_configured_probe(gateway),
            "configuration": configuration_probe(True),
        }
    )
    return AppContainer(
        answer_service=service,
        health=health,
        settings=ApiSettings(),
        gateway=gateway,
        retriever=retriever,
        policy=evaluate_policy,
        metrics=get_metrics(),
    )


def create_app(container: AppContainer | None = None) -> FastAPI:
    setup_telemetry()
    resolved = container or build_lab_container()
    application = FastAPI(
        title="AICO Ask API",
        version="0.6.0",
        description=(
            "Typed HTTP boundary over the Day 5 grounded RAG pipeline. "
            "Identity comes only from trusted claims. Telemetry is redacted."
        ),
    )
    application.state.container = resolved
    application.add_middleware(
        RequestBoundaryMiddleware,
        max_request_bytes=resolved.settings.max_request_bytes,
        json_content_type=resolved.settings.json_content_type,
    )

    @application.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        request_id, correlation_id = _ids_from(request)
        request_id = exc.request_id or request_id
        correlation_id = exc.correlation_id or correlation_id
        log_operation(
            stage="api",
            outcome="error",
            error_category=exc.code,
        )
        return error_response(
            exc.status_code,
            code=exc.code,
            message=exc.safe_message,
            request_id=request_id,
            correlation_id=correlation_id,
        )

    @application.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, _exc: RequestValidationError) -> JSONResponse:
        request_id, correlation_id = _ids_from(request)
        log_operation(stage="api", outcome="rejected", error_category="invalid_request")
        return error_response(
            422,
            code="invalid_request",
            message="request body does not match the public /ask contract",
            request_id=request_id,
            correlation_id=correlation_id,
        )

    @application.exception_handler(OperationCancelled)
    async def cancelled_handler(request: Request, _exc: OperationCancelled) -> JSONResponse:
        request_id, correlation_id = _ids_from(request)
        log_operation(stage="api", outcome="cancelled", error_category="cancelled")
        return error_response(
            499,
            code="cancelled",
            message="request was cancelled before completion",
            request_id=request_id,
            correlation_id=correlation_id,
        )

    @application.exception_handler(Exception)
    async def unhandled_handler(request: Request, _exc: Exception) -> JSONResponse:
        request_id, correlation_id = _ids_from(request)
        log_operation(stage="api", outcome="error", error_category="internal")
        return error_response(
            500,
            code="internal",
            message="the request failed",
            request_id=request_id,
            correlation_id=correlation_id,
        )

    @application.get("/health/live")
    def live(request: Request) -> dict:
        return request.app.state.container.health.liveness()

    @application.get("/health/ready")
    def ready(request: Request) -> JSONResponse:
        is_ready, body = request.app.state.container.health.ready()
        request_id, correlation_id = _ids_from(request)
        return JSONResponse(
            status_code=200 if is_ready else 503,
            content=body,
            headers={
                "X-Request-ID": request_id,
                "X-Correlation-ID": correlation_id,
            },
        )

    @application.get("/health/dependencies")
    def dependencies(request: Request) -> dict:
        return request.app.state.container.health.dependencies()

    @application.post(
        "/ask",
        response_model=AskResponse,
        responses={
            401: {"model": ApiErrorBody, "description": "Missing or invalid trusted identity"},
            413: {"model": ApiErrorBody, "description": "Request exceeds size limit"},
            415: {"model": ApiErrorBody, "description": "Unsupported Content-Type"},
            422: {"model": ApiErrorBody, "description": "Invalid public request contract"},
            499: {"model": ApiErrorBody, "description": "Client cancelled the request"},
        },
    )
    async def ask(
        request: Request,
        payload: AskRequest,
        claims: TrustedClaims = Depends(get_trusted_claims),
    ) -> AskResponse:
        request_id, correlation_id = _ids_from(request)
        require_trusted_identity(claims)
        token = CancellationToken()
        stats = PipelineStats()
        container: AppContainer = request.app.state.container
        started = time.perf_counter()
        tracer = get_tracer()

        async def _watch_disconnect() -> None:
            while not token.cancelled:
                if await request.is_disconnected():
                    token.cancel()
                    return
                await asyncio.sleep(0.01)

        watcher = asyncio.create_task(_watch_disconnect())
        log_operation(stage="api", outcome="started")
        try:
            with tracer.start_as_current_span(
                "api",
                attributes=span_attributes(
                    request_id=request_id,
                    correlation_id=correlation_id,
                ),
            ):
                result = await asyncio.to_thread(
                    container.answer_service.answer,
                    payload.question,
                    cancellation=token,
                    correlation_id=correlation_id,
                    request_id=request_id,
                    stats=stats,
                )
                with tracer.start_as_current_span(
                    "response_composition",
                    attributes=span_attributes(
                        request_id=request_id,
                        correlation_id=correlation_id,
                    ),
                ):
                    response = _map_result(
                        result,
                        request_id=request_id,
                        correlation_id=correlation_id,
                    )
        except OperationCancelled:
            raise
        finally:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass

        latency_ms = (time.perf_counter() - started) * 1000.0
        token_total = stats.token_total
        container.metrics.record_request(
            latency_ms=latency_ms,
            outcome=response.outcome,
            retrieval_latency_ms=stats.retrieval_latency_ms or None,
            gateway_latency_ms=stats.gateway_latency_ms or None,
            token_usage=token_total,
            retry_count=stats.retry_count,
            cache_hit=stats.cache_hit,
        )
        log_operation(stage="api", outcome=response.outcome, latency_ms=latency_ms)
        return response

    return application


app = create_app()
