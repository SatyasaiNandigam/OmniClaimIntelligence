"""Dead-letter / manual-repair queue (T6.6): where a call lands when retrying cannot help, so a person can look at it. Nothing is silently dropped.

    dlq = DeadLetterQueue()
    entry = dlq.add(operation="create_task", key="CLM-1:create_task:x", claim_id="CLM-1", payload={...}, error="...", attempts=4, reason="retries_exhausted")
    dlq.open()                                   # what still needs a person
    dlq.resolve(entry.entry_id, "repaired", by="rev-9", note="...")        # or "discarded"; requeue is the caller's job (Dispatcher.requeue)

Reasons: `retries_exhausted` (the system stayed unreachable), `non_retryable` (a failure retrying cannot fix: a poison message, a refusal, an unconfigured live adapter).
The queue is in memory for the prototype; the interface is what a table would implement.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

Reason = Literal["retries_exhausted", "non_retryable"]
Status = Literal["open", "repaired", "discarded"]


@dataclass
class DeadLetter:
    entry_id: str
    operation: str
    key: str
    claim_id: str
    payload: Any
    error: str
    attempts: int
    reason: Reason
    status: Status = "open"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    resolved_by: str | None = None
    note: str = ""


class DeadLetterQueue:
    def __init__(self) -> None:
        self._items: list[DeadLetter] = []
        self._lock = threading.RLock()

    def add(self, *, operation: str, key: str, claim_id: str, payload: Any, error: str, attempts: int, reason: Reason) -> DeadLetter:
        """One open entry per key: a key that fails again updates its open entry instead of piling up."""
        with self._lock:
            for e in self._items:
                if e.key == key and e.status == "open":
                    e.error, e.attempts, e.reason = error, e.attempts + attempts, reason
                    return e
            item = DeadLetter(f"DLQ-{len(self._items) + 1:04d}", operation, key, claim_id, payload, error, attempts, reason)
            self._items.append(item)
            return item

    def open(self, claim_id: str | None = None) -> list[DeadLetter]:
        with self._lock:
            return [e for e in self._items if e.status == "open" and (claim_id is None or e.claim_id == claim_id)]

    def all(self) -> list[DeadLetter]:
        with self._lock:
            return list(self._items)

    def get(self, entry_id: str) -> DeadLetter:
        with self._lock:
            return next(e for e in self._items if e.entry_id == entry_id)

    def resolve(self, entry_id: str, status: Literal["repaired", "discarded"], *, by: str, note: str = "") -> DeadLetter:
        if not by.strip():
            raise ValueError("a dead letter is resolved by a named person")
        with self._lock:
            e = self.get(entry_id)
            if e.status != "open":
                raise ValueError(f"{entry_id} is already {e.status}")
            e.status, e.resolved_by, e.note = status, by.strip(), note
            return e


__all__ = ["DeadLetter", "DeadLetterQueue"]
