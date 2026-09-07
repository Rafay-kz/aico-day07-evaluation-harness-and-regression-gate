"""Typed Model Gateway: the only application-facing chat and embed contract."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, Protocol, TypeVar

from aico.platform.config import GatewayConfig, load_config, testing_config
from aico.platform.errors import (
    BadRequestError,
    FallbackBlockedError,
    GatewayError,
    OperationCancelled,
)
from aico.platform.retry import backoff_seconds
from aico.platform.routing import failed_fallback_checks

LOGGER = logging.getLogger("aico.platform.model_gateway")

T = TypeVar("T")
SleepFn = Callable[[float], None]


@dataclass
class CancellationToken:
    """Callers set this to stop an in-flight gateway operation."""

    _cancelled: bool = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise OperationCancelled()


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class GatewayMetadata:
    """Sanitized operational facts. Must never contain prompts, completions or secrets."""

    model_alias: str
    token_usage: TokenUsage
    latency_ms: float
    retry_count: int
    budget_status: str


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class EmbedRequest:
    texts: list[str]
    timeout_seconds: float | None = None
    cancellation: CancellationToken | None = None


@dataclass(frozen=True)
class ChatRequest:
    messages: list[ChatMessage]
    timeout_seconds: float | None = None
    cancellation: CancellationToken | None = None


@dataclass(frozen=True)
class EmbedResult:
    vectors: list[list[float]]
    metadata: GatewayMetadata


@dataclass(frozen=True)
class ChatResult:
    text: str
    metadata: GatewayMetadata


@dataclass(frozen=True)
class TransportEmbedResponse:
    vectors: list[list[float]]
    token_usage: TokenUsage


@dataclass(frozen=True)
class TransportChatResponse:
    text: str
    token_usage: TokenUsage


class ModelTransport(Protocol):
    """Provider adapter. The only objects that may talk to a model SDK."""

    def embed(
        self,
        texts: list[str],
        *,
        model_alias: str,
        timeout_seconds: float,
        cancellation: CancellationToken | None,
    ) -> TransportEmbedResponse: ...

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model_alias: str,
        timeout_seconds: float,
        cancellation: CancellationToken | None,
    ) -> TransportChatResponse: ...


class ModelGateway:
    """Single platform boundary for chat and embedding traffic."""

    def __init__(
        self,
        transport: ModelTransport,
        config: GatewayConfig,
        *,
        fallback_transport: ModelTransport | None = None,
        sleep: SleepFn | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._transport = transport
        self._fallback_transport = fallback_transport
        self._config = config
        self._sleep = sleep or time.sleep
        self._rng = rng or random.Random()

    @property
    def chat_alias(self) -> str:
        return self._config.chat_alias

    @property
    def embedding_alias(self) -> str:
        return self._config.embedding_alias

    @property
    def embedding_dimensions(self) -> int:
        return self._config.embedding_dimensions

    def embed(self, request: EmbedRequest) -> EmbedResult:
        if request.cancellation is not None:
            request.cancellation.raise_if_cancelled()
        if request.texts and len(request.texts) > self._config.max_items_per_call:
            raise BadRequestError(
                f"embedding batch size {len(request.texts)} exceeds budget "
                f"{self._config.max_items_per_call}"
            )
        timeout = self._timeout(request.timeout_seconds)
        started = time.perf_counter()
        try:
            response, retry_count, route = self._execute(
                operation="embed",
                cancellation=request.cancellation,
                primary=lambda: self._transport.embed(
                    request.texts,
                    model_alias=self._config.embedding_alias,
                    timeout_seconds=timeout,
                    cancellation=request.cancellation,
                ),
                fallback=(
                    None
                    if self._fallback_transport is None
                    else lambda: self._fallback_transport.embed(
                        request.texts,
                        model_alias=self._config.embedding_alias,
                        timeout_seconds=timeout,
                        cancellation=request.cancellation,
                    )
                ),
            )
        except GatewayError as exc:
            self._log(
                operation="embed",
                outcome=exc.category,
                model_alias=self._config.embedding_alias,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                retry_count=getattr(exc, "retry_count", 0),
                route="primary",
            )
            raise
        latency_ms = (time.perf_counter() - started) * 1000.0
        metadata = self._metadata(
            model_alias=self._config.embedding_alias,
            token_usage=response.token_usage,
            latency_ms=latency_ms,
            retry_count=retry_count,
            budget_status=self._embed_budget_status(len(request.texts)),
        )
        self._log(
            operation="embed",
            outcome="success",
            model_alias=metadata.model_alias,
            latency_ms=metadata.latency_ms,
            retry_count=metadata.retry_count,
            route=route,
            budget_status=metadata.budget_status,
        )
        return EmbedResult(vectors=response.vectors, metadata=metadata)

    def chat(self, request: ChatRequest) -> ChatResult:
        if request.cancellation is not None:
            request.cancellation.raise_if_cancelled()
        timeout = self._timeout(request.timeout_seconds)
        started = time.perf_counter()
        try:
            response, retry_count, route = self._execute(
                operation="chat",
                cancellation=request.cancellation,
                primary=lambda: self._transport.chat(
                    request.messages,
                    model_alias=self._config.chat_alias,
                    timeout_seconds=timeout,
                    cancellation=request.cancellation,
                ),
                fallback=(
                    None
                    if self._fallback_transport is None
                    else lambda: self._fallback_transport.chat(
                        request.messages,
                        model_alias=self._config.chat_alias,
                        timeout_seconds=timeout,
                        cancellation=request.cancellation,
                    )
                ),
            )
        except GatewayError as exc:
            self._log(
                operation="chat",
                outcome=exc.category,
                model_alias=self._config.chat_alias,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                retry_count=getattr(exc, "retry_count", 0),
                route="primary",
            )
            raise
        latency_ms = (time.perf_counter() - started) * 1000.0
        metadata = self._metadata(
            model_alias=self._config.chat_alias,
            token_usage=response.token_usage,
            latency_ms=latency_ms,
            retry_count=retry_count,
            budget_status="within_budget" if self._config.budget_available else "exceeded",
        )
        self._log(
            operation="chat",
            outcome="success",
            model_alias=metadata.model_alias,
            latency_ms=metadata.latency_ms,
            retry_count=metadata.retry_count,
            route=route,
            budget_status=metadata.budget_status,
        )
        return ChatResult(text=response.text, metadata=metadata)

    def _execute(
        self,
        *,
        operation: str,
        cancellation: CancellationToken | None,
        primary: Callable[[], T],
        fallback: Callable[[], T] | None,
    ) -> tuple[T, int, str]:
        del operation
        try:
            result, retries = self._attempt_route(primary, cancellation)
            return result, retries, "primary"
        except GatewayError as exc:
            if not exc.retryable:
                raise
            if not self._config.fallback_enabled or fallback is None:
                raise
            failed = failed_fallback_checks(
                self._config,
                budget_ok=self._config.budget_available,
            )
            if failed:
                raise FallbackBlockedError(failed_checks=failed) from exc
            result, fallback_retries = self._attempt_route(fallback, cancellation)
            primary_failed = getattr(exc, "retry_count", self._config.retry.max_attempts)
            return result, primary_failed + fallback_retries, "fallback"

    def _attempt_route(
        self,
        call: Callable[[], T],
        cancellation: CancellationToken | None,
    ) -> tuple[T, int]:
        max_attempts = self._config.retry.max_attempts
        if max_attempts < 1:
            raise BadRequestError("retry max_attempts must be at least 1")
        last_error: GatewayError | None = None
        for attempt in range(max_attempts):
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            try:
                return call(), attempt
            except GatewayError as exc:
                last_error = exc
                if not exc.retryable:
                    raise
                if attempt + 1 >= max_attempts:
                    exc.retry_count = attempt + 1
                    raise
                delay = backoff_seconds(attempt, self._config.retry, self._rng)
                if cancellation is not None:
                    cancellation.raise_if_cancelled()
                self._sleep(delay)
        assert last_error is not None
        raise last_error

    def _timeout(self, requested: float | None) -> float:
        return self._config.timeout_seconds if requested is None else requested

    def _embed_budget_status(self, item_count: int) -> str:
        if item_count <= self._config.max_items_per_call and self._config.budget_available:
            return "within_budget"
        return "exceeded"

    @staticmethod
    def _metadata(
        *,
        model_alias: str,
        token_usage: TokenUsage,
        latency_ms: float,
        retry_count: int,
        budget_status: str,
    ) -> GatewayMetadata:
        return GatewayMetadata(
            model_alias=model_alias,
            token_usage=token_usage,
            latency_ms=latency_ms,
            retry_count=retry_count,
            budget_status=budget_status,
        )

    @staticmethod
    def _log(
        *,
        operation: str,
        outcome: str,
        model_alias: str,
        latency_ms: float,
        retry_count: int,
        route: str,
        budget_status: str = "-",
    ) -> None:
        LOGGER.info(
            "operation=%s model_alias=%s latency_ms=%.2f retry_count=%s outcome=%s route=%s budget_status=%s",
            operation,
            model_alias,
            latency_ms,
            retry_count,
            outcome,
            route,
            budget_status,
        )


def build_foundry_gateway(config: GatewayConfig | None = None) -> ModelGateway:
    """Live Microsoft Foundry gateway. SDK import stays inside the adapter."""
    from aico.platform.foundry_adapter import FoundryAdapter

    resolved = config if config is not None else load_config()
    return ModelGateway(FoundryAdapter(resolved), resolved)


def build_fake_gateway(config: GatewayConfig | None = None) -> ModelGateway:
    from aico.platform.fake_transport import FakeTransport

    resolved = config if config is not None else testing_config()
    return ModelGateway(
        FakeTransport(embed_dimensions=resolved.embedding_dimensions),
        resolved,
        sleep=lambda _seconds: None,
    )
