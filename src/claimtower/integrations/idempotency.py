"""Unique external references (T6.6): the client side of "a retry must never create a second claim, task or payment" (guardrail 12).

    store = IdempotencyStore()
    entry = store.begin("CLM-1:pay:1", "submit_payment_instruction")        # first time: state "started"; later: the existing entry
    store.complete("CLM-1:pay:1", result)                                    # remember the answer: the call is never made again
    store.fail("CLM-1:pay:1", "why")                                         # the attempt ended in a failure: the key may be tried again

The adapters are idempotent on the key themselves (that is the contract), and this store makes the CALLER idempotent too: a key whose call completed is answered from the
store without calling the system at all, a key that is still in flight is not started twice, and every key is kept as the audit record of what was asked. In memory for the
prototype; the interface is what a table with a unique constraint on `key` would implement.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

State = Literal["started", "done", "failed"]


@dataclass
class IdempotencyEntry:
    key: str
    operation: str
    state: State = "started"
    result: Any = None
    attempts: int = 0
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    new: bool = True                                    # True only for the call that created the entry


class IdempotencyStore:
    def __init__(self) -> None:
        self._entries: dict[str, IdempotencyEntry] = {}
        self._lock = threading.RLock()

    def begin(self, key: str, operation: str) -> IdempotencyEntry:
        """The entry for `key`. A key is bound to ONE operation: reusing it for another is a bug and raises."""
        with self._lock:
            e = self._entries.get(key)
            if e is None:
                e = self._entries[key] = IdempotencyEntry(key, operation)
                return e
            if e.operation != operation:
                raise ValueError(f"idempotency key '{key}' was used for '{e.operation}', not '{operation}'")
            return IdempotencyEntry(e.key, e.operation, e.state, e.result, e.attempts, e.error, e.created_at, new=False)

    def attempt(self, key: str) -> int:
        with self._lock:
            self._entries[key].attempts += 1
            return self._entries[key].attempts

    def complete(self, key: str, result: Any) -> None:
        with self._lock:
            e = self._entries[key]
            e.state, e.result, e.error = "done", result, None

    def fail(self, key: str, error: str) -> None:
        with self._lock:
            e = self._entries[key]
            e.state, e.error = "failed", error

    def reopen(self, key: str) -> None:
        """A person repaired the cause: let the key be tried again (dead-letter requeue)."""
        with self._lock:
            e = self._entries[key]
            if e.state == "failed":
                e.state, e.error = "started", None

    def get(self, key: str) -> IdempotencyEntry | None:
        with self._lock:
            return self._entries.get(key)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._entries)


__all__ = ["IdempotencyEntry", "IdempotencyStore"]
