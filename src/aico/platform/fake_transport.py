"""Deterministic in-process transport. No network. Used by tests and FakeEmbeddingProvider."""

from __future__ import annotations

import hashlib
import math
import time

from aico.platform.errors import (
    AuthenticationError,
    BadRequestError,
    GatewayTimeout,
    OperationCancelled,
    RateLimitError,
    ServerError,
    error_for_category,
)
from aico.platform.model_gateway import (
    CancellationToken,
    ChatMessage,
    TokenUsage,
    TransportChatResponse,
    TransportEmbedResponse,
)

FAKE_CHAT_TEXT = "gateway-chat-ok"

_OUTCOME_ERRORS = {
    "timeout": GatewayTimeout,
    "rate_limit": RateLimitError,
    "authentication": AuthenticationError,
    "bad_request": BadRequestError,
    "server_error": ServerError,
}


def vector_from_text(text: str, dimensions: int) -> list[float]:
    """Stable unit vector derived from the text. Same text always yields the same vector."""
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    while len(values) < dimensions:
        for byte in seed:
            values.append((byte / 127.5) - 1.0)
            if len(values) >= dimensions:
                break
        seed = hashlib.sha256(seed).digest()
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


class FakeTransport:
    """Pretend Foundry. Callers never import a model SDK to use this."""

    def __init__(
        self,
        outcomes: list[str] | None = None,
        *,
        embed_dimensions: int = 8,
        chat_text: str = FAKE_CHAT_TEXT,
        chat_texts: list[str] | None = None,
        wait_for_cancel: bool = False,
    ) -> None:
        self._outcomes = list(outcomes) if outcomes is not None else None
        self._last_outcome = "success"
        self.embed_dimensions = embed_dimensions
        self.chat_text = chat_text
        self._chat_queue = list(chat_texts) if chat_texts is not None else []
        self.wait_for_cancel = wait_for_cancel
        self.embed_calls = 0
        self.chat_calls = 0
        self.last_timeout_seconds: float | None = None
        self.last_chat_messages: list[ChatMessage] | None = None

    def embed(
        self,
        texts: list[str],
        *,
        model_alias: str,
        timeout_seconds: float,
        cancellation: CancellationToken | None,
    ) -> TransportEmbedResponse:
        self._prepare(timeout_seconds, cancellation)
        self.embed_calls += 1
        self._raise_outcome()
        return TransportEmbedResponse(
            vectors=[vector_from_text(text, self.embed_dimensions) for text in texts],
            token_usage=TokenUsage(),
        )

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model_alias: str,
        timeout_seconds: float,
        cancellation: CancellationToken | None,
    ) -> TransportChatResponse:
        self._prepare(timeout_seconds, cancellation)
        self.chat_calls += 1
        self.last_chat_messages = list(messages)
        self._raise_outcome()
        if self._chat_queue:
            self.chat_text = self._chat_queue.pop(0)
        return TransportChatResponse(text=self.chat_text, token_usage=TokenUsage())

    def _prepare(self, timeout_seconds: float, cancellation: CancellationToken | None) -> None:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        self.last_timeout_seconds = timeout_seconds
        if self.wait_for_cancel:
            if cancellation is None:
                raise BadRequestError("wait_for_cancel transport requires a cancellation token")
            while not cancellation.cancelled:
                time.sleep(0.01)
            raise OperationCancelled()

    def _next_outcome(self) -> str:
        if self._outcomes is None:
            return "success"
        if self._outcomes:
            self._last_outcome = self._outcomes.pop(0)
        return self._last_outcome

    def _raise_outcome(self) -> None:
        outcome = self._next_outcome()
        if outcome == "success":
            return
        cls = _OUTCOME_ERRORS.get(outcome)
        if cls is None:
            raise error_for_category("server_error")
        raise cls()
