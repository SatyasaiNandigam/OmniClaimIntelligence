"""State patching, ownership enforcement and versioning (T0.3, BUILD_PLAN section 5 + A.1).

* Every agent may write only the ClaimState fields it owns (OWNERSHIP).
* Every committed patch increments `state_version` and is recorded in a VersionStore,
  so any version can be replayed and two versions diffed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from claimtower.core.ids import new_id
from claimtower.core.schemas import AuditEvent, ClaimState


class OwnershipViolation(Exception):
    """An agent tried to write a field it does not own (or a protected/unknown field)."""


class VersionNotFound(KeyError):
    """Requested claim/version has no recorded history."""


class StatePatch(BaseModel):
    agent_id: str
    updates: dict[str, Any]          # {field_name: new_value}; only owned fields allowed
    rationale: str = ""


# agent_id -> ClaimState fields it may write (BUILD_PLAN section 5; mirrors docs/contracts/README.md)
OWNERSHIP: dict[str, frozenset[str]] = {
    "intake": frozenset({"claim_header", "intake_checklist", "duplicate_candidates"}),
    "policy": frozenset({"policy_snapshot"}),
    "evidence": frozenset({"documents", "facts", "image_findings"}),
    "gap": frozenset({"open_questions", "contradictions", "missing_evidence", "counterfactuals"}),
    "coverage": frozenset({"coverage_issues"}),
    "settlement": frozenset({"settlement"}),
    "action": frozenset({"proposed_actions", "drafts"}),
    "audit": frozenset({"qa_report", "audit_ref"}),
    "supervisor": frozenset({"routing", "human_approval_required", "resolution_status"}),
}
# claim_id, state_version, correlation_id are written only by this module / claim creation.


def owns_for(agent_id: str) -> frozenset[str]:
    try:
        return OWNERSHIP[agent_id]
    except KeyError:
        raise OwnershipViolation(f"unknown agent '{agent_id}' (not in OWNERSHIP)") from None


def apply_patch(state: ClaimState, patch: StatePatch, owns: frozenset[str] | None = None) -> ClaimState:
    """Pure: return a new, validated state with version + 1. Raises OwnershipViolation on illegal writes."""
    owns = owns_for(patch.agent_id) if owns is None else owns
    legal_fields = set(ClaimState.model_fields)
    illegal = {f for f in patch.updates if f not in owns or f not in legal_fields}
    if illegal:
        raise OwnershipViolation(f"{patch.agent_id} tried to write {sorted(illegal)}")
    merged = {**{f: getattr(state, f) for f in ClaimState.model_fields}, **patch.updates,
              "state_version": state.state_version + 1}
    return ClaimState.model_validate(merged)       # re-validates: bad patch data fails here


class VersionRecord(BaseModel):
    version: int
    agent_id: str
    rationale: str = ""
    delta: dict[str, Any]            # changed fields only, JSON-mode
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class _History(BaseModel):
    base_version: int
    base: dict[str, Any]             # JSON-mode snapshot before the first recorded patch
    records: list[VersionRecord] = []
    events: list[AuditEvent] = []


class InMemoryVersionStore:
    """Append-only history per claim. A DB-backed store can replace this behind the same methods."""

    def __init__(self) -> None:
        self._h: dict[str, _History] = {}

    def record(self, before: ClaimState, after: ClaimState, patch: StatePatch) -> AuditEvent:
        h = self._h.setdefault(before.claim_id, _History(base_version=before.state_version,
                                                         base=before.model_dump(mode="json")))
        if h.records and h.records[-1].version != before.state_version:
            raise ValueError(f"non-contiguous history: have v{h.records[-1].version}, patch applied on v{before.state_version}")
        dumped = after.model_dump(mode="json")
        delta = {f: dumped[f] for f in patch.updates}
        delta["state_version"] = after.state_version
        rec = VersionRecord(version=after.state_version, agent_id=patch.agent_id,
                            rationale=patch.rationale, delta=delta)
        h.records.append(rec)
        event = AuditEvent(
            event_id=new_id("EVT"), ts=rec.ts, actor_type="system" if patch.agent_id == "supervisor" else "agent",
            actor_id=patch.agent_id, action=f"patch:{','.join(sorted(patch.updates))}",
            state_version_before=before.state_version, state_version_after=after.state_version,
            outputs_ref=f"{after.claim_id}@v{after.state_version}",
        )
        h.events.append(event)
        return event

    def versions(self, claim_id: str) -> list[int]:
        h = self._get(claim_id)
        return [h.base_version] + [r.version for r in h.records]

    def history(self, claim_id: str) -> list[VersionRecord]:
        return list(self._get(claim_id).records)

    def audit_events(self, claim_id: str) -> list[AuditEvent]:
        return list(self._get(claim_id).events)

    def replay(self, claim_id: str, version: int) -> ClaimState:
        """Reconstruct the state as of `version` from the base snapshot + recorded deltas."""
        h = self._get(claim_id)
        if version not in self.versions(claim_id):
            raise VersionNotFound(f"{claim_id} has no v{version}; known: {self.versions(claim_id)}")
        data = dict(h.base)
        for rec in h.records:
            if rec.version > version:
                break
            data.update(rec.delta)
        return ClaimState.model_validate(data)

    def diff(self, claim_id: str, from_version: int, to_version: int) -> dict[str, tuple[Any, Any]]:
        """{field: (old, new)} for fields that differ between two versions (JSON-mode values)."""
        a = self.replay(claim_id, from_version).model_dump(mode="json")
        b = self.replay(claim_id, to_version).model_dump(mode="json")
        return {f: (a[f], b[f]) for f in a if a[f] != b[f] and f != "state_version"}

    def _get(self, claim_id: str) -> _History:
        try:
            return self._h[claim_id]
        except KeyError:
            raise VersionNotFound(f"no history for claim {claim_id}") from None


def commit_patch(state: ClaimState, patch: StatePatch, store: InMemoryVersionStore) -> ClaimState:
    """Apply (ownership-checked) and record in the store. This is what the supervisor calls."""
    new = apply_patch(state, patch)
    store.record(state, new, patch)
    return new
