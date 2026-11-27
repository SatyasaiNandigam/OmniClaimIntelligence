"""Reliable calls to the external systems (T6.6): idempotent, retried within bounds, dead-lettered when nothing helps.

    d = Dispatcher(IdempotencyStore(), DeadLetterQueue(), RetryPolicy())
    out = d.execute("create_task", key, claim_id, lambda: claims.create_task(req), payload=req)
    out.status      "done" | "replayed" | "dead_lettered"

  * a key whose call already completed is answered from the store: the system is NOT called again ("replayed")
  * a retryable failure (a timeout: the call may or may not have taken effect) is retried with the SAME key, after backoff, up to `max_attempts`
  * a failure retrying cannot fix (a poison message, an unconfigured live adapter) is not retried: it goes to the dead-letter queue at once
  * when attempts run out the call goes to the dead-letter queue; the key stays "failed" until a person repairs it and `requeue`s it
Nothing is raised to the caller for a system failure: the outcome says what happened, so one failing task cannot stop a claim's other steps.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from claimtower.integrations.deadletter import DeadLetter, DeadLetterQueue
from claimtower.integrations.idempotency import IdempotencyStore
from claimtower.integrations.retry import RetriesExhausted, RetryPolicy, call_with_retry


@dataclass
class Outcome:
    status: Literal["done", "replayed", "dead_lettered"]
    key: str
    operation: str
    result: Any = None
    attempts: int = 0
    dead_letter: DeadLetter | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in ("done", "replayed")


class Dispatcher:
    def __init__(self, store: IdempotencyStore | None = None, dead_letters: DeadLetterQueue | None = None, policy: RetryPolicy | None = None,
                 sleeper: Callable[[float], None] = time.sleep) -> None:
        self.store = store or IdempotencyStore()
        self.dead_letters = dead_letters or DeadLetterQueue()
        self.policy = policy or RetryPolicy()
        self._sleep = sleeper

    def execute(self, operation: str, key: str, claim_id: str, call: Callable[[], Any], payload: Any = None) -> Outcome:
        entry = self.store.begin(key, operation)
        if entry.state == "done":
            return Outcome("replayed", key, operation, entry.result, entry.attempts)
        made = [0]

        def counted(_n: int) -> None:
            made[0] += 1
            self.store.attempt(key)

        try:
            result, attempts = call_with_retry(call, self.policy, sleeper=self._sleep, on_attempt=counted)
        except RetriesExhausted as e:
            return self._dead(operation, key, claim_id, payload, str(e), e.attempts, "retries_exhausted")
        except Exception as e:                                                     # not retryable: retrying cannot help
            return self._dead(operation, key, claim_id, payload, f"{type(e).__name__}: {e}", made[0], "non_retryable")
        self.store.complete(key, result)
        return Outcome("done", key, operation, result, attempts)

    def _dead(self, operation: str, key: str, claim_id: str, payload: Any, error: str, attempts: int, reason: str) -> Outcome:
        self.store.fail(key, error)
        item = self.dead_letters.add(operation=operation, key=key, claim_id=claim_id, payload=payload, error=error, attempts=attempts, reason=reason)  # type: ignore[arg-type]
        return Outcome("dead_lettered", key, operation, None, attempts, item, error)

    def requeue(self, entry_id: str, call: Callable[[], Any], *, by: str) -> Outcome:
        """After a person fixed the cause: try the dead-lettered call again (same key). A success closes the entry as repaired."""
        item = self.dead_letters.get(entry_id)
        if item.status != "open":
            raise ValueError(f"{entry_id} is already {item.status}")
        self.store.reopen(item.key)
        out = self.execute(item.operation, item.key, item.claim_id, call, item.payload)
        if out.ok:
            self.dead_letters.resolve(entry_id, "repaired", by=by, note=f"requeued: {out.status}")
        return out


__all__ = ["Dispatcher", "Outcome"]
