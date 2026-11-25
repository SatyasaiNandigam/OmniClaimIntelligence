"""What changed between two versions of a ClaimState (T7.1 `GET /claims/{id}/diff`, and the run log of T6.2).

Pure and generic: it compares the contract fields, by identifier for the lists (facts, questions, ...), and reports added / removed / changed items and
every open question whose status moved. It knows nothing about the agents that produced the change.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from claimtower.core.schemas import ClaimState

# list field -> how an item is identified
KEYED: dict[str, Callable[[Any], str]] = {
    "documents": lambda x: x.document_id,
    "facts": lambda x: x.fact_id,
    "image_findings": lambda x: x.finding_id,
    "intake_checklist": lambda x: x.item_id,
    "duplicate_candidates": lambda x: x.candidate_id,
    "open_questions": lambda x: x.question_id,
    "contradictions": lambda x: x.contradiction_id,
    "missing_evidence": lambda x: x.minimum_evidence_request.addressee_role,
    "counterfactuals": lambda x: f"{x.question_id}/{x.scenario_id}",
    "coverage_issues": lambda x: x.issue_id,
    "proposed_actions": lambda x: x.action_id,
    "drafts": lambda x: x.draft_id,
}
SCALARS = ("claim_header", "policy_snapshot", "settlement", "routing", "qa_report", "audit_ref", "human_approval_required", "resolution_status")


class FieldChange(BaseModel):
    field: str
    added: list[str] = []
    removed: list[str] = []
    changed: list[str] = []
    before: str | None = None            # scalar fields only
    after: str | None = None


class StatusChange(BaseModel):
    question_id: str
    before: str | None
    after: str | None


class StateDiff(BaseModel):
    from_version: int
    to_version: int
    changes: list[FieldChange] = []
    status_changes: list[StatusChange] = []

    @property
    def empty(self) -> bool:
        return not self.changes

    def describe(self) -> str:
        parts = []
        for c in self.changes:
            bits = [f"+{len(c.added)}" if c.added else "", f"-{len(c.removed)}" if c.removed else "", f"~{len(c.changed)}" if c.changed else ""]
            parts.append(f"{c.field} {' '.join(b for b in bits if b)}" if c.field in KEYED else f"{c.field}: {c.before} -> {c.after}")
        parts += [f"{s.question_id} {s.before or '(new)'} -> {s.after or '(gone)'}" for s in self.status_changes]
        return "; ".join(parts) or "no change"


def _scalar(name: str, value: Any) -> Any:
    if name == "policy_snapshot" and value is not None:
        return value.model_copy(update={"retrieved_at": None})            # re-fetching an identical snapshot is not a change
    return value


def _short(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, BaseModel):
        for key in ("version", "tier", "status", "passed", "claim_id"):
            if hasattr(value, key):
                return f"{type(value).__name__}({key}={getattr(value, key)})"
        return type(value).__name__
    return str(value)


def diff_states(a: ClaimState, b: ClaimState) -> StateDiff:
    out = StateDiff(from_version=a.state_version, to_version=b.state_version)
    for name, key in KEYED.items():
        before = {key(x): x for x in getattr(a, name)}
        after = {key(x): x for x in getattr(b, name)}
        added, removed = sorted(set(after) - set(before)), sorted(set(before) - set(after))
        changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
        if added or removed or changed:
            out.changes.append(FieldChange(field=name, added=added, removed=removed, changed=changed))
    for name in SCALARS:
        x, y = _scalar(name, getattr(a, name)), _scalar(name, getattr(b, name))
        if x != y:
            out.changes.append(FieldChange(field=name, before=_short(getattr(a, name)), after=_short(getattr(b, name))))
    qa, qb = {q.question_id: q.status for q in a.open_questions}, {q.question_id: q.status for q in b.open_questions}
    out.status_changes = [StatusChange(question_id=q, before=qa.get(q), after=qb.get(q)) for q in [*qa, *[k for k in qb if k not in qa]] if qa.get(q) != qb.get(q)]
    return out
