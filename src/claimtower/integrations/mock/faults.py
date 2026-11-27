"""Failure injection for the mock adapters (T6.5): latency, a timeout, a duplicate message, a stale answer. For tests and the demo; never on by default.

    plan = FaultPlan({"create_task": ["timeout", "ok"], "get_policy_version": ["stale"]})
    claims = FaultInjector(MockClaimsAdapter(), plan)
    claims.create_task(request)            # 1st call: raises AdapterTimeout (the call did NOT reach the system); 2nd call: goes through

Faults are consumed one per call of that method, in order; once the list is used up the call is plain. Kinds:
  ok          no fault
  timeout     raise AdapterTimeout BEFORE the call (it did not take effect)
  timeout_after  make the call, then raise AdapterTimeout (it DID take effect: the caller does not know, and a retry with the same key must not create a second record)
  duplicate   deliver the same call twice and return the SECOND answer (a duplicate message: an idempotent system answers `created=False` / `duplicate_ignored`)
  stale       return the answer this method gave on its previous call, if any (a stale snapshot); the first call is plain
  slow        sleep `latency_s` first (the sleeper is injectable, so tests do not wait)
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from claimtower.integrations.base import AdapterTimeout

KINDS = {"ok", "timeout", "timeout_after", "duplicate", "stale", "slow"}


@dataclass
class FaultPlan:
    faults: dict[str, list[str]] = field(default_factory=dict)
    latency_s: float = 0.0

    def __post_init__(self) -> None:
        bad = {k for ks in self.faults.values() for k in ks} - KINDS
        if bad:
            raise ValueError(f"unknown fault kind(s) {sorted(bad)}; known: {sorted(KINDS)}")
        self.faults = {m: list(ks) for m, ks in self.faults.items()}

    def next(self, method: str) -> str:
        queue = self.faults.get(method)
        return queue.pop(0) if queue else "ok"


class FaultInjector:
    """Wraps an adapter; methods named in the plan misbehave as planned, everything else passes straight through."""

    def __init__(self, adapter: Any, plan: FaultPlan, sleeper: Callable[[float], None] = time.sleep) -> None:
        self._adapter, self._plan, self._sleep = adapter, plan, sleeper
        self._last: dict[str, Any] = {}
        self.injected: list[tuple[str, str]] = []                # (method, fault) actually applied

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._adapter, name)
        if not callable(target) or name.startswith("_"):
            return target

        def call(*args: Any, **kwargs: Any) -> Any:
            fault = self._plan.next(name)
            if fault != "ok":
                self.injected.append((name, fault))
            if fault == "timeout":
                raise AdapterTimeout(f"{name}: timed out before the system answered (injected)")
            if fault == "slow":
                self._sleep(self._plan.latency_s)
            if fault == "stale" and name in self._last:
                return self._last[name]
            if fault == "duplicate":
                target(*args, **kwargs)
            result = target(*args, **kwargs)
            self._last[name] = result
            if fault == "timeout_after":
                raise AdapterTimeout(f"{name}: timed out after the system had acted (injected)")
            return result

        return call

    @property
    def mock(self) -> bool:
        return bool(getattr(self._adapter, "mock", True))


__all__ = ["FaultInjector", "FaultPlan", "KINDS"]
