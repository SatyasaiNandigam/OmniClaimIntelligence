"""Bounded retry with exponential backoff (T6.6).

    policy = RetryPolicy(max_attempts=4, base_delay_s=0.5, factor=2.0, max_delay_s=10.0)
    result, attempts = call_with_retry(lambda: claims.create_task(req), policy)           # raises RetriesExhausted, or the original error if it is not retryable

Only a failure of the SYSTEM that may pass is retried: `AdapterTimeout` (and any other `AdapterError` the caller marks retryable). `AdapterNotConfigured`, a programming
error or a business refusal is not retried: trying again cannot help. The delay before attempt n+1 is base * factor**(n-1), capped; the sleeper is injectable, so tests do not wait.
A retry reuses the SAME idempotency key (the caller's closure does), which is what makes retrying a write safe.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from claimtower.integrations.base import AdapterError, AdapterNotConfigured, AdapterTimeout

T = TypeVar("T")


class RetriesExhausted(AdapterError):
    def __init__(self, attempts: int, last: Exception) -> None:
        super().__init__(f"gave up after {attempts} attempt(s); last error: {type(last).__name__}: {last}")
        self.attempts, self.last = attempts, last


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay_s: float = 0.5
    factor: float = 2.0
    max_delay_s: float = 10.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1 or self.base_delay_s < 0 or self.factor < 1 or self.max_delay_s < 0:
            raise ValueError("invalid retry policy")

    def delay(self, failed_attempt: int) -> float:
        """Seconds to wait after the `failed_attempt`-th failure (1-based)."""
        return min(self.base_delay_s * self.factor ** (failed_attempt - 1), self.max_delay_s)


def is_retryable(e: Exception) -> bool:
    return isinstance(e, AdapterTimeout) or (isinstance(e, AdapterError) and not isinstance(e, AdapterNotConfigured | RetriesExhausted) and getattr(e, "retryable", False))


def call_with_retry(fn: Callable[[], T], policy: RetryPolicy = RetryPolicy(), *, sleeper: Callable[[float], None] = time.sleep,
                    retryable: Callable[[Exception], bool] = is_retryable, on_attempt: Callable[[int], Any] | None = None) -> tuple[T, int]:
    """(result, attempts). A non-retryable error propagates as it is; running out of attempts raises RetriesExhausted."""
    last: Exception | None = None
    for n in range(1, policy.max_attempts + 1):
        if on_attempt:
            on_attempt(n)
        try:
            return fn(), n
        except Exception as e:
            if not retryable(e):
                raise
            last = e
            if n < policy.max_attempts:
                sleeper(policy.delay(n))
    assert last is not None
    raise RetriesExhausted(policy.max_attempts, last)


__all__ = ["RetriesExhausted", "RetryPolicy", "call_with_retry", "is_retryable"]
