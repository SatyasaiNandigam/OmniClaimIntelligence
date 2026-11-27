"""Guards (T8.4): the autonomy matrix and the role order, enforced in code at the points where something is done, not only written in YAML.

    guard = AutonomyGuard(settings.autonomy, approval_order=settings.rbac.approval_order, sink=on_denial)
    guard.require("release_payment", Actor("human", "rev-9", "supervisor"))         # passes
    guard.require("release_payment", Actor("agent", "action"))                      # raises NotAuthorised and records a Denial

Rules (configs/autonomy_matrix.yaml; anything not listed is DENIED, there is no default permission):
  authorised_human_only   only a human whose role is named for the operation, or ranks at least as high as the lowest named role in the approval order, may do it. No agent ever may.
  controlled_api          an agent may do it only through the controlled call (mode "controlled_api"), never directly
  draft_only              an agent may produce a draft; it may not send or carry it out
  human_reviewed          an agent may recommend; a person decides (the gate)
  autonomous              an agent may do it
A denial is returned or raised with its reason, appended to `guard.denials` and handed to the `sink` (the run log): a blocked attempt leaves a trace. `GuardedRecon` wraps the payment system
so an agent cannot reach it: `submit_payment_instruction` needs a human actor who may release payments.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from claimtower.core.config import AutonomyMatrix


@dataclass(frozen=True)
class Actor:
    type: Literal["agent", "human", "system"]
    id: str
    role: str | None = None


@dataclass(frozen=True)
class Denial:
    operation: str
    actor: Actor
    reason: str
    claim_id: str = ""
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


class NotAuthorised(PermissionError):
    def __init__(self, denial: Denial) -> None:
        super().__init__(denial.reason)
        self.denial = denial


class AutonomyGuard:
    def __init__(self, matrix: AutonomyMatrix, approval_order: tuple[str, ...] | list[str] = (), sink: Callable[[Denial], None] | None = None) -> None:
        self.matrix = matrix
        self.rank = {r: i + 1 for i, r in enumerate(approval_order)}
        self.sink = sink
        self.denials: list[Denial] = []

    def _human_may(self, roles: list[str], actor: Actor) -> bool:
        if actor.type != "human" or not actor.role or not actor.id.strip():
            return False
        if actor.role in roles:
            return True
        lowest = min((self.rank[r] for r in roles if r in self.rank), default=None)
        return lowest is not None and self.rank.get(actor.role, 0) >= lowest

    def decide(self, operation: str, actor: Actor, *, mode: Literal["draft", "controlled_api", "execute"] | None = None) -> tuple[bool, str]:
        """(allowed, reason). No side effects: use `require` to enforce and record."""
        op = self.matrix.operations.get(operation)
        if op is None:
            return False, f"'{operation}' is not in the autonomy matrix: nothing that is not listed is allowed"
        if op.mode == "authorised_human_only":
            if actor.type != "human":
                return False, f"'{operation}' is for an authorised person only; an {actor.type} ('{actor.id}') may not do it"
            if not self._human_may(op.roles, actor):
                return False, f"'{operation}' needs the role {' or '.join(op.roles)} (or higher); '{actor.role}' ({actor.id}) may not"
            return True, "authorised"
        if actor.type == "human":
            return True, "a person may do what an agent may"
        want = mode or "draft"
        if want == "execute" and op.mode != "autonomous":
            return False, f"'{operation}' is {op.mode}: an agent may not carry it out directly"
        if want == "controlled_api" and op.mode not in ("controlled_api", "autonomous"):
            return False, f"'{operation}' is {op.mode}: an agent may not call a system for it"
        return True, f"allowed as {op.mode}"

    def require(self, operation: str, actor: Actor, *, claim_id: str = "", mode: Literal["draft", "controlled_api", "execute"] | None = None) -> None:
        ok, why = self.decide(operation, actor, mode=mode)
        if not ok:
            d = Denial(operation, actor, why, claim_id)
            self.denials.append(d)
            if self.sink is not None:
                self.sink(d)
            raise NotAuthorised(d)

    def guarded(self, operation: str, *, mode: Literal["draft", "controlled_api", "execute"] | None = None) -> Callable:
        """Decorator for a tool call: the call must be made with `actor=` (an Actor); without one it is treated as an unidentified agent and denied."""
        def deco(fn: Callable) -> Callable:
            def wrapper(*args: Any, actor: Actor | None = None, claim_id: str = "", **kwargs: Any) -> Any:
                self.require(operation, actor or Actor("agent", "unidentified"), claim_id=claim_id, mode=mode)
                return fn(*args, **kwargs)
            wrapper.__name__ = getattr(fn, "__name__", "guarded")
            return wrapper
        return deco


class GuardedRecon:
    """The payment system behind the guard: only a human who may release payments can instruct one. Everything else passes through unchanged."""

    def __init__(self, recon: Any, guard: AutonomyGuard) -> None:
        self._recon, self._guard = recon, guard

    def submit_payment_instruction(self, request: Any, idempotency_key: str, *, actor: Actor | None = None) -> Any:
        self._guard.require("release_payment", actor or Actor("agent", "unidentified"), claim_id=request.claim_id)
        return self._recon.submit_payment_instruction(request, idempotency_key)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._recon, name)


__all__ = ["Actor", "AutonomyGuard", "Denial", "GuardedRecon", "NotAuthorised"]
