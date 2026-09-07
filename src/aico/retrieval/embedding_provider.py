"""Day 2 embedding interface. Live traffic goes through the Model Gateway; no model SDK here."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from aico.platform.config import testing_config
from aico.platform.model_gateway import EmbedRequest, ModelGateway, build_foundry_gateway

DEFAULT_MODEL_ALIAS = "text-embedding-3-small"
TEXT_EMBEDDING_3_SMALL_DIMENSIONS = 1536
FAKE_MODEL_ALIAS = "fake-embed-v1"
FAKE_DIMENSIONS = 8


class DimensionMismatchError(ValueError):
    """Raised when two vectors (or a vector and the configured model) differ in length."""


class EmbeddingProvider(ABC):
    """One contract for embedding. Search and cache depend on this, not on an SDK."""

    model_alias: str
    dimensions: int
    calls: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input string, in the same order."""


def content_hash(text: str) -> str:
    """SHA-256 of the chunk text. This is the cache invalidation key, not chunk_id."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class GatewayEmbeddingProvider(EmbeddingProvider):
    """Day 2 retrieval face of the Model Gateway."""

    def __init__(self, gateway: ModelGateway, *, dimensions: int) -> None:
        self._gateway = gateway
        self.model_alias = gateway.embedding_alias
        self.dimensions = dimensions
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = self._gateway.embed(EmbedRequest(texts=texts))
        self.calls += len(texts)
        for vector in result.vectors:
            if len(vector) != self.dimensions:
                raise DimensionMismatchError(
                    f"provider returned {len(vector)} dimensions, "
                    f"expected {self.dimensions} for model {self.model_alias}; "
                    "refusing to pad or truncate"
                )
        return result.vectors


class FakeEmbeddingProvider(GatewayEmbeddingProvider):
    """Deterministic provider for tests. Never touches the network."""

    def __init__(
        self,
        model_alias: str = FAKE_MODEL_ALIAS,
        dimensions: int = FAKE_DIMENSIONS,
    ) -> None:
        from aico.platform.fake_transport import FakeTransport

        gateway = ModelGateway(
            FakeTransport(embed_dimensions=dimensions),
            testing_config(embedding_alias=model_alias, embedding_dimensions=dimensions),
            sleep=lambda _seconds: None,
        )
        super().__init__(gateway, dimensions=dimensions)


def get_provider(kind: str = "foundry") -> EmbeddingProvider:
    if kind == "fake":
        return FakeEmbeddingProvider()
    if kind == "foundry":
        gateway = build_foundry_gateway()
        return GatewayEmbeddingProvider(gateway, dimensions=gateway.embedding_dimensions)
    raise ValueError(f"unknown embedding provider {kind!r}; use 'foundry' or 'fake'")
