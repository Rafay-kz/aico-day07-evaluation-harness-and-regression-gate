from __future__ import annotations

import os

from aico.platform.config import GatewayConfig, load_dotenv
from aico.platform.errors import (
    AuthenticationError,
    BadRequestError,
    ConfigurationError,
    GatewayTimeout,
    RateLimitError,
    ServerError,
)
from aico.platform.model_gateway import (
    CancellationToken,
    ChatMessage,
    TokenUsage,
    TransportChatResponse,
    TransportEmbedResponse,
)

FOUNDRY_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"


def _openai_v1_base_url(endpoint: str) -> str:
    """Normalize Foundry endpoints, including a pasted /openai/v1/responses URL."""
    base = endpoint.rstrip("/")
    if base.endswith("/openai/v1/responses"):
        base = base[: -len("/responses")]
    if base.endswith("/openai/v1"):
        return base + "/"
    return base + "/openai/v1/"


class FoundryAdapter:
    """Talks to Microsoft Foundry with DefaultAzureCredential. Maps failures into gateway errors."""

    def __init__(self, config: GatewayConfig) -> None:
        load_dotenv()
        self._config = config
        endpoint = config.endpoint
        if not endpoint:
            raise ConfigurationError(
                f"Foundry adapter needs {config.endpoint_env} in the environment; "
                "do not put secrets or endpoints in YAML"
            )
        from openai import OpenAI

        api_key = (os.environ.get("AICO_FOUNDRY_API_KEY") or "").strip()
        if api_key:
            self._client = OpenAI(
                base_url=_openai_v1_base_url(endpoint),
                api_key=api_key,
                timeout=config.timeout_seconds,
            )
            return
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            FOUNDRY_TOKEN_SCOPE,
        )
        self._client = OpenAI(
            base_url=_openai_v1_base_url(endpoint),
            api_key=token_provider,
            timeout=config.timeout_seconds,
        )

    def embed(
        self,
        texts: list[str],
        *,
        model_alias: str,
        timeout_seconds: float,
        cancellation: CancellationToken | None,
    ) -> TransportEmbedResponse:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        if not texts:
            return TransportEmbedResponse(vectors=[], token_usage=TokenUsage())
        try:
            response = self._client.embeddings.create(
                model=model_alias,
                input=texts,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            raise _normalize(exc) from exc
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        ordered = sorted(response.data, key=lambda item: item.index)
        usage = _usage_from(getattr(response, "usage", None))
        return TransportEmbedResponse(
            vectors=[list(item.embedding) for item in ordered],
            token_usage=usage,
        )

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model_alias: str,
        timeout_seconds: float,
        cancellation: CancellationToken | None,
    ) -> TransportChatResponse:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        try:
            response = self._client.responses.create(
                model=model_alias,
                input=[{"role": item.role, "content": item.content} for item in messages],
                timeout=timeout_seconds,
            )
        except Exception as exc:
            raise _normalize(exc) from exc
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        return TransportChatResponse(
            text=_response_text(response),
            token_usage=_usage_from(getattr(response, "usage", None)),
        )


def _response_text(response: object) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            value = getattr(content, "text", None)
            if isinstance(value, str):
                parts.append(value)
    return "".join(parts)


def _usage_from(usage: object | None) -> TokenUsage:
    if usage is None:
        return TokenUsage()
    input_tokens = getattr(usage, "input_tokens", None)
    if not isinstance(input_tokens, int):
        input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if not isinstance(output_tokens, int):
        output_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    return TokenUsage(
        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
        output_tokens=output_tokens if isinstance(output_tokens, int) else None,
        total_tokens=total_tokens if isinstance(total_tokens, int) else None,
    )


def _normalize(exc: Exception):
    """Map provider exceptions to gateway categories. Never leak SDK types upward."""
    name = type(exc).__name__
    message = str(exc) or name
    module = type(exc).__module__ or ""
    status = getattr(exc, "status_code", None)

    if name in {"APITimeoutError", "TimeoutError"} or "timeout" in name.lower():
        return GatewayTimeout(message)
    if name in {"RateLimitError"} or status == 429:
        return RateLimitError(message)
    if name in {"AuthenticationError", "PermissionDeniedError"} or status in {401, 403}:
        return AuthenticationError(message)
    if name in {"BadRequestError", "UnprocessableEntityError"} or status in {400, 422}:
        return BadRequestError(message)
    if name in {"InternalServerError"} or (isinstance(status, int) and status >= 500):
        return ServerError(message)
    if "openai" in module:
        return ServerError(message)
    return ServerError(message)
