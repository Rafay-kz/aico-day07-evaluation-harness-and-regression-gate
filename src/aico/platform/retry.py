"""Bounded exponential backoff with optional full jitter."""

from __future__ import annotations

import random
from collections.abc import Callable

from aico.platform.config import RetrySettings


def backoff_seconds(
    failed_attempt_index: int,
    settings: RetrySettings,
    rng: random.Random,
) -> float:
    """Delay after failure `failed_attempt_index` (0 after the first failure)."""
    uncapped = settings.base_delay_ms * (2 ** failed_attempt_index)
    capped = min(uncapped, settings.max_delay_ms)
    if settings.jitter:
        capped = rng.random() * capped
    return capped / 1000.0


def default_rng() -> random.Random:
    return random.Random()


SleepFn = Callable[[float], None]
