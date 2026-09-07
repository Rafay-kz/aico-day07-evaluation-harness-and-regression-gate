"""Platform boundary for chat and embedding traffic."""

from aico.platform.model_gateway import (
    CancellationToken,
    ChatMessage,
    ChatRequest,
    ChatResult,
    EmbedRequest,
    EmbedResult,
    GatewayMetadata,
    ModelGateway,
    TokenUsage,
    build_fake_gateway,
    build_foundry_gateway,
)

__all__ = [
    "CancellationToken",
    "ChatMessage",
    "ChatRequest",
    "ChatResult",
    "EmbedRequest",
    "EmbedResult",
    "GatewayMetadata",
    "ModelGateway",
    "TokenUsage",
    "build_fake_gateway",
    "build_foundry_gateway",
]
