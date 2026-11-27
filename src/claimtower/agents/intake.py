"""Intake agent (T6.7): the first notice of loss -> the claim header, the consolidated checklist of what is still needed, and the registration in the claims system.

Owns exactly `claim_header`, `intake_checklist` and `duplicate_candidates` (core.state.OWNERSHIP); it leaves the duplicate candidates alone (duplicate detection is T6.8).

  header     what was supplied with the claim stays as it is; a field that is MISSING is filled from the claim's first-notice-of-loss document (the document is classified and
             its fields extracted exactly as the Evidence agent does, so a model fallback applies when one is configured). A supplied value that disagrees with the FNOL is
             kept and reported for review, never silently replaced. The loss type defaults to the pack's (an assumption stated in the rationale). `registered_at` is set once.
  checklist  ONE consolidated list, so the claimant is asked once: the required header fields still missing (a missing loss date is tracked, not blocking), the required
             documents of the pack (received when a stored document has that type), and the policy checks (exact match in the policy system; in force on the loss date).
  policy     an EXACT lookup in the policy system by policy number and loss date. No guessing: a number that is not found, or a policy not in force on the loss date, leaves the
             check unreceived, which sends the claim to an operator (the planner and the routing rules read it); nothing is decided here.
  register   the claim is created in the claims system through the dispatcher (idempotent on `<claim>:create_claim`, retried, dead-lettered). A claim that cannot be registered
             stops the flow with the reason (the supervisor parks it for a person): a claim must not go on unregistered.
Running again changes nothing: the same key replays, and the timestamp is kept.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any, ClassVar

from claimtower.core.config import IntakeChecklistCfg
from claimtower.core.schemas import ClaimCreateRequest, ClaimHeader, ClaimState, IntakeChecklistItem
from claimtower.core.state import OWNERSHIP, StatePatch
from claimtower.integrations.base import AdapterError
from claimtower.integrations.dispatcher import Dispatcher
from claimtower.ingestion.store import DocumentStore

FNOL_FIELDS = {"claimant_name": "claimant_name", "policy_ref": "policy_number", "loss_date": "loss_date", "loss_description": "loss_description"}


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


class IntakeAgent:
    agent_id: ClassVar[str] = "intake"
    owns: ClassVar[frozenset[str]] = OWNERSHIP["intake"]

    def __init__(self, cfg: IntakeChecklistCfg, pas: Any | None, claims: Any | None, dispatcher: Dispatcher | None, *, store: DocumentStore | None = None,
                 classifier: Any | None = None, extractor: Any | None = None, clock: Callable[[], datetime] | None = None) -> None:
        self.cfg, self.pas, self.claims, self.dispatcher = cfg, pas, claims, dispatcher
        self.store, self.classifier, self.extractor = store, classifier, extractor
        self._clock = clock or (lambda: datetime.now(UTC))

    # ------------------------------------------------------------------ the first notice of loss
    def _fnol_fields(self, claim_id: str) -> tuple[dict[str, str], list[str]]:
        """{header field: value} from the claim's FNOL document(s), and the document ids read. The first value found for a field wins."""
        if self.store is None or self.classifier is None or self.extractor is None:
            return {}, []
        values: dict[str, str] = {}
        used: list[str] = []
        for rec in self.store.records(claim_id):
            if rec.detected_type == "unknown" and self.store.stored(rec.document_id).type_method != "human":
                rec = self.store.classify(rec.document_id, self.classifier)
            if rec.detected_type != "fnol":
                continue
            result = self.store.extract(rec.document_id, self.extractor)
            used.append(rec.document_id)
            for head, field in FNOL_FIELDS.items():
                f = result.field(field)
                if f is not None and (f.normalised or f.value) and head not in values:
                    values[head] = f.normalised or f.value
        return values, used

    def build_header(self, state: ClaimState) -> tuple[ClaimHeader, list[str], list[str]]:
        """(header, notes, conflicts)."""
        h = state.claim_header or ClaimHeader(claim_id=state.claim_id)
        fnol, used = self._fnol_fields(state.claim_id)
        notes = [f"read the first notice of loss: {', '.join(used)}"] if used else []
        update: dict[str, Any] = {}
        conflicts: list[str] = []
        for name, value in fnol.items():
            have = getattr(h, name)
            if have in (None, ""):
                update[name] = date.fromisoformat(value) if name == "loss_date" else value
                notes.append(f"{name} taken from the notice of loss")
            elif _norm(str(have)) != _norm(value):
                conflicts.append(f"{name}: supplied '{have}', the notice of loss says '{value}' (the supplied value is kept)")
        if not h.loss_type and self.cfg.loss_type:
            update["loss_type"] = self.cfg.loss_type
            notes.append(f"loss type assumed '{self.cfg.loss_type}' (the pack's line of business)")
        if h.registered_at is None:
            update["registered_at"] = self._clock()
        return h.model_copy(update=update), notes, conflicts

    # ------------------------------------------------------------------ checklist
    def checklist(self, h: ClaimHeader, claim_id: str) -> list[IntakeChecklistItem]:
        items: list[IntakeChecklistItem] = []
        for spec in self.cfg.required_fields:
            have = getattr(h, spec["name"], None)
            items.append(IntakeChecklistItem(item_id=f"field:{spec['name']}", description=spec["description"], required=True, received=bool(have)))
        records = self.store.records(claim_id) if self.store is not None else None
        if records is not None:
            for spec in self.cfg.required_documents:
                doc = next((r for r in records if r.detected_type == spec["type"]), None)
                items.append(IntakeChecklistItem(item_id=f"doc:{spec['type']}", description=spec["description"], required=True, received=doc is not None,
                                                 document_id=doc.document_id if doc else None))
        if self.pas is not None and h.policy_ref and h.loss_date:
            s = self.pas.get_status(h.policy_ref, h.loss_date)
            found = s.status != "not_found"
            items.append(IntakeChecklistItem(item_id="policy:exact_match", required=True, received=found, description=(
                f"Policy {h.policy_ref} exists in the policy system" if found else f"Policy {h.policy_ref} was not found in the policy system: an operator confirms the number")))
            if found:
                ok = s.status in ("in_force", "reinstated")
                items.append(IntakeChecklistItem(item_id="policy:in_force", required=True, received=ok, description=(
                    f"Policy {h.policy_ref} is in force on {h.loss_date}" if ok else f"Policy {h.policy_ref} is {s.status} on {h.loss_date}: an operator reviews the claim")))
        return items

    # ------------------------------------------------------------------ run
    def run(self, state: ClaimState) -> StatePatch:
        header, notes, conflicts = self.build_header(state)
        if state.claim_header is not None and state.claim_header.registered_at is not None:
            header = header.model_copy(update={"registered_at": state.claim_header.registered_at})
        items = self.checklist(header, state.claim_id)
        reg = "no claims system configured: not registered"
        if self.claims is not None and self.dispatcher is not None:
            key = f"{state.claim_id}:create_claim"
            req = ClaimCreateRequest(claim_header=header, idempotency_key=key, correlation_id=state.correlation_id)
            out = self.dispatcher.execute("create_claim", key, state.claim_id, lambda: self.claims.create_claim(req), header)
            if not out.ok:
                raise AdapterError(f"the claim could not be registered in the claims system ({out.error}); it is in the dead-letter queue")
            reg = "registered in the claims system" if out.status == "done" and out.result.created else "already registered (replayed)"
        missing = [i for i in items if i.required and not i.received]
        parts = [f"{reg}; checklist: {len(items) - len(missing)} of {len(items)} received"]
        if missing:
            parts.append("STILL NEEDED (asked once, together): " + "; ".join(i.description for i in missing))
        if any(i.item_id.startswith("policy:") and not i.received for i in items):
            parts.append("policy check failed: an operator takes it from here (nothing is decided)")
        parts += notes
        if conflicts:
            parts.append("REVIEW, supplied values differ from the notice of loss: " + "; ".join(conflicts))
        return StatePatch(agent_id=self.agent_id, updates={"claim_header": header, "intake_checklist": items, "duplicate_candidates": list(state.duplicate_candidates)},
                          rationale=" | ".join(parts))


__all__ = ["IntakeAgent"]
