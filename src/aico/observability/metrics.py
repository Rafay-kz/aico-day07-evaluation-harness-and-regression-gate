"""In-process metrics. Labels are low-cardinality operational categories only."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class MetricsRegistry:
    request_latency_ms: list[float] = field(default_factory=list)
    retrieval_latency_ms: list[float] = field(default_factory=list)
    gateway_latency_ms: list[float] = field(default_factory=list)
    token_usage_total: int = 0
    retry_count_total: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    outcomes: Counter[str] = field(default_factory=Counter)

    def record_request(
        self,
        *,
        latency_ms: float,
        outcome: str,
        retrieval_latency_ms: float | None = None,
        gateway_latency_ms: float | None = None,
        token_usage: int | None = None,
        retry_count: int = 0,
        cache_hit: bool | None = None,
    ) -> None:
        self.request_latency_ms.append(latency_ms)
        self.outcomes[outcome] += 1
        if retrieval_latency_ms is not None:
            self.retrieval_latency_ms.append(retrieval_latency_ms)
        if gateway_latency_ms is not None:
            self.gateway_latency_ms.append(gateway_latency_ms)
        if token_usage is not None:
            self.token_usage_total += token_usage
        self.retry_count_total += retry_count
        if cache_hit is True:
            self.cache_hits += 1
        elif cache_hit is False:
            self.cache_misses += 1


_metrics = MetricsRegistry()


def get_metrics() -> MetricsRegistry:
    return _metrics


def reset_metrics() -> None:
    global _metrics
    _metrics = MetricsRegistry()
