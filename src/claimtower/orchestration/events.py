"""Run events (T6.1): what the graph did, claim by claim, in order. The audit trail the API and the UI will read.

Every node writes one event when it finishes or fails, the supervisor writes one for each routing decision, and the human gate writes one for each
decision (who, in what role, what, why). The log lives next to the graph, not in ClaimState: the checkpointed state has no room for it, and the
persistent audit trail (T8.3) will keep it properly. In the prototype it is in memory, so it does not survive a restart (the checkpointer does not either).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal

EventKind = Literal["run_started", "agent_finished", "agent_failed", "ownership_violation", "routed", "human_requested", "human_decision",
                    "human_decision_refused", "commit", "reassessment", "halted", "node_skipped", "action_denied"]


@dataclass(frozen=True)
class RunEvent:
    seq: int
    ts: datetime
    claim_id: str
    kind: EventKind
    node: str
    message: str
    agent_id: str | None = None
    mock: bool = False                                  # the node ran a MOCK ("yet to implement") agent
    state_version: int | None = None
    data: dict[str, Any] = field(default_factory=dict)


class EventLog:
    def __init__(self, clock: Callable[[], datetime] | None = None, sink: Callable[[RunEvent], None] | None = None) -> None:
        self.sink = sink                                  # every event is also handed to the audit trail
        self._events: list[RunEvent] = []
        self._last = 0                                    # the last sequence number given
        self._lock = threading.Lock()
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def add(self, claim_id: str, kind: EventKind, node: str, message: str, *, agent_id: str | None = None, mock: bool = False,
            state_version: int | None = None, **data: Any) -> RunEvent:
        with self._lock:
            self._last += 1
            ev = RunEvent(self._last, self._clock(), claim_id, kind, node, message, agent_id, mock, state_version, dict(data))
            self._events.append(ev)
            if self.sink is not None:
                self.sink(ev)
        return ev

    def restore(self, events: list[RunEvent]) -> None:
        """After a restart: take back the events of an earlier run (from the audit trail), in order, with their own sequence numbers. The sink is NOT called: they are already recorded."""
        with self._lock:
            self._events = sorted(events, key=lambda e: e.seq)
            self._last = max((e.seq for e in self._events), default=0)

    def add_once(self, key: str, claim_id: str, kind: EventKind, node: str, message: str, **kw: Any) -> RunEvent | None:
        """Like `add`, but a second call with the same key logs nothing. For nodes LangGraph re-runs from their first line (the human gate, on every resume)."""
        with self._lock:
            if any(e.data.get("key") == key for e in self._events):
                return None
        return self.add(claim_id, kind, node, message, key=key, **kw)

    def for_claim(self, claim_id: str) -> list[RunEvent]:
        return [e for e in self._events if e.claim_id == claim_id]

    def of_kind(self, claim_id: str, *kinds: EventKind) -> list[RunEvent]:
        return [e for e in self.for_claim(claim_id) if e.kind in kinds]

    def of_kind_all(self, *kinds: EventKind) -> list[RunEvent]:
        """The events of these kinds for every claim, in order."""
        return [e for e in self._events if e.kind in kinds]

    def nodes_run(self, claim_id: str) -> list[str]:
        """The nodes that finished, in order (mock or real)."""
        return [e.node for e in self.of_kind(claim_id, "agent_finished")]

    def last_failure(self, claim_id: str) -> RunEvent | None:
        failures = self.of_kind(claim_id, "agent_failed")
        return failures[-1] if failures else None

    def __len__(self) -> int:
        return len(self._events)
