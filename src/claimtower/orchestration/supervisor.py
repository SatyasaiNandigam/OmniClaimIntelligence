"""The supervisor (T6.1, T6.3): runs each agent safely, decides where the flow goes next, and holds the human approval gate. Deterministic: no model.

Three jobs, each a plain function of the state so they are testable without LangGraph:

  1. NODE WRAPPER. `Supervisor.node(slot)` runs one agent, enforces ownership (core.state.apply_patch), bumps the version and logs an event.
     * an OwnershipViolation is a bug: it is logged and RAISED (the run stops loudly);
     * any other failure (the PAS is down, a policy wording is missing, a model is unreachable ...) is logged with its reason and the claim is parked
       with resolution_status `awaiting_human`: the flow halts, it never loops silently and never continues without that agent's output.
  2. ROUTING. After a pre-routing node the flow halts if the claim was parked; after the coverage node a claim with no policy in force skips
     settlement (there is nothing to assess against); after the human gate only an approval goes on to the commit step.
  3. HUMAN GATE. The graph pauses with a payload (what the reviewer needs to decide), and resumes with a decision: approve, reject or
     request_more_evidence, by a named reviewer whose role must reach the required approver role (SAMPLE hierarchy below). The system only RECOMMENDS:
     approving here never pays or denies anything (that is the commit step, which is a mock until the adapters exist). A decision that is not valid is
     refused and the gate asks again.

LangGraph note (ADR-0001): a node that calls interrupt() is re-run from its first line on resume, so everything before interrupt() is free of side effects.
"""
from __future__ import annotations

from typing import Any, Literal

from langgraph.types import interrupt
from pydantic import BaseModel, field_validator

from decimal import Decimal, InvalidOperation

from claimtower.audit.guards import Actor, AutonomyGuard, Denial, NotAuthorised
from claimtower.core.config import SETTLEMENT_INPUTS
from claimtower.core.schemas import ClaimState
from claimtower.core.state import OwnershipViolation, StatePatch, apply_patch
from claimtower.orchestration.dependencies import changed_fields, input_digest
from claimtower.orchestration.events import EventLog
from claimtower.orchestration.overrides import OverrideStore
from claimtower.orchestration.registry import AgentRegistry, AgentSlot

Decision = Literal["approve", "edit", "reject", "request_more_evidence"]
DECISION_STATUS: dict[str, str] = {"approve": "approved", "edit": "assessing", "reject": "closed", "request_more_evidence": "awaiting_evidence"}
ROLE_RANK: dict[str, int] = {"adjuster": 1, "senior_adjuster": 2, "supervisor": 3}          # SAMPLE: who may decide what a lower role may
HALT = "halt"


class HumanDecision(BaseModel):
    """What a reviewer sends to the gate. Reject and request_more_evidence need a reason."""

    decision: Decision
    reviewer_id: str
    reviewer_role: str
    reason: str = ""
    edits: dict[str, str] = {}                  # "edit" only: settlement input name -> the amount the reviewer sets (a person changes an INPUT, never the payable)

    @field_validator("reviewer_id", "reviewer_role")
    @classmethod
    def _named(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("the reviewer must be named: a decision without a person is not a decision")
        return v.strip()


def policy_in_force(state: ClaimState) -> bool:
    """False when the policy system answered and nothing is in force on the loss date (the Policy agent's convention: snapshot None, claim has a policy
    number and a date). True while there is a snapshot, and also while it is not yet known (no policy number or loss date yet)."""
    h = state.claim_header
    return state.policy_snapshot is not None or h is None or not (h.policy_ref and h.loss_date)


class Supervisor:
    def __init__(self, registry: AgentRegistry, events: EventLog, *, known_roles: tuple[str, ...] | None = None, commit_service: Any | None = None, overrides: OverrideStore | None = None,
                 approval_order: tuple[str, ...] | None = None, autonomy: Any | None = None, skip_unchanged: bool = True, low_confidence_below: float | None = None,
                 trail: Any | None = None, llm: Any | None = None) -> None:
        self.trail, self.llm = trail, llm
        self.low_confidence_below = low_confidence_below
        order = approval_order or tuple(ROLE_RANK)
        self.rank = {r: i + 1 for i, r in enumerate(order)}                       # later in the order = more authority (rbac.yaml `approval_order`)
        self.registry, self.events, self.known_roles, self.commit_service = registry, events, known_roles or order, commit_service
        self.overrides, self.autonomy, self.skip_unchanged = overrides or OverrideStore(), autonomy, skip_unchanged
        self.guard = AutonomyGuard(autonomy, order, sink=self.log_denial) if autonomy is not None else None      # the autonomy matrix, enforced (T8.4)
        self.force: dict[str, str] = {}                                           # claim -> the node a person asked to re-run: it is never skipped

    def log_denial(self, d: Denial) -> None:
        """A blocked attempt leaves a trace in the run log (logged once however often the gate node is replayed)."""
        self.events.add_once(f"denied:{d.claim_id}:{d.operation}:{d.actor.id}:{d.reason}", d.claim_id, "action_denied", "guard",
                             f"blocked: {d.actor.type} '{d.actor.id}' tried '{d.operation}': {d.reason}", agent_id="guard", operation=d.operation, actor_type=d.actor.type,
                             actor_id=d.actor.id, actor_role=d.actor.role, reason=d.reason)

    def record(self, claim: str, patch: StatePatch, new: ClaimState, *, mock: bool = False, inputs: Any = None, model_calls: list[dict[str, Any]] | None = None) -> None:
        """One state change into the audit trail (T8.3): who, what was written, the version it made, and the model calls behind it. Recorded once per version and author."""
        if self.trail is None:
            return
        last = self.trail.records(claim, "state_patch")
        if last and last[-1].data["version"] == new.state_version and last[-1].data["agent_id"] == patch.agent_id:
            return                                                       # a node LangGraph replayed: the same change is not written twice
        self.trail.append(claim, "state_patch", {"agent_id": patch.agent_id, "version": new.state_version, "updates": patch.updates, "rationale": patch.rationale, "mock": mock,
                                                 "inputs": inputs, "model_calls": model_calls or []})

    def _calls_since(self, n: int) -> list[dict[str, Any]]:
        calls = getattr(self.llm, "calls", None) or []
        return [{"kind": c.kind, "model": c.model, "prompt": c.prompt_name, "prompt_version": c.prompt_version, "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
                 "replayed": c.replayed} for c in calls[n:]]

    # ------------------------------------------------------------------ nodes
    def start(self, state: ClaimState) -> dict[str, Any]:
        """The first node of every (re)assessment: the claim is being assessed again, so any earlier gate state is cleared."""
        patch = StatePatch(agent_id="supervisor", updates={"resolution_status": "assessing"})
        new = apply_patch(state, patch)
        if self.trail is not None and not self.trail.records(state.claim_id, "claim_created"):
            self.trail.append(state.claim_id, "claim_created", {"state": state})                   # the base of any replay: what the claim started with
        self.record(state.claim_id, patch, new)
        self.events.add(state.claim_id, "run_started", "start", f"assessment started (state v{new.state_version})", agent_id="supervisor", state_version=new.state_version)
        return {**patch.updates, "state_version": new.state_version}

    def node(self, slot: AgentSlot):
        def run(state: ClaimState) -> dict[str, Any]:
            claim = state.claim_id
            extra = self.overrides.fingerprint(claim) if slot.node == "settlement" and self.overrides.get(claim) else None
            digest = input_digest(slot.node, state, extra)
            previous = self._last_digest(claim, slot.node) if digest is not None else None
            why = changed_fields(previous, digest) if digest is not None else []
            forced = self.force.get(claim) == slot.node
            if forced:
                del self.force[claim]
            if self.skip_unchanged and digest is not None and previous is not None and not why and not forced:
                self.events.add(claim, "node_skipped", slot.node, f"skipped (unchanged): the {slot.node} node reads {', '.join(sorted(digest)) or 'nothing from the claim'}, "
                                "and none of it changed since its last run", agent_id=slot.agent.agent_id, mock=slot.is_mock, state_version=state.state_version, inputs=digest)
                return {}
            calls_before = len(getattr(self.llm, "calls", None) or [])
            try:
                patch = slot.agent.run(state)
                if patch.agent_id != slot.agent.agent_id:
                    raise OwnershipViolation(f"{slot.node}: the agent returned a patch as '{patch.agent_id}'")
                new = apply_patch(state, patch)                                  # ownership + validation of the patched state
            except OwnershipViolation as e:
                self.events.add(claim, "ownership_violation", slot.node, str(e), agent_id=slot.agent.agent_id, mock=slot.is_mock)
                raise
            except Exception as e:                                               # PAS down, missing wording, model unreachable ...
                reason = f"{type(e).__name__}: {e}"
                parked = StatePatch(agent_id="supervisor", updates={"resolution_status": "awaiting_human"})
                new = apply_patch(state, parked)
                self.record(claim, parked, new, model_calls=self._calls_since(calls_before))
                self.events.add(claim, "agent_failed", slot.node, f"the {slot.node} agent failed, the claim is parked for a human: {reason}",
                                agent_id=slot.agent.agent_id, mock=slot.is_mock, state_version=new.state_version, reason=reason)
                return {**parked.updates, "state_version": new.state_version}
            self.record(claim, patch, new, mock=slot.is_mock, inputs=input_digest(slot.node, new, extra), model_calls=self._calls_since(calls_before))
            self.events.add(claim, "agent_finished", slot.node, patch.rationale or "(no rationale)", agent_id=slot.agent.agent_id, mock=slot.is_mock,
                            state_version=new.state_version, fields=sorted(patch.updates), inputs=input_digest(slot.node, new, extra), rerun_reason=why if previous is not None else None)
            qa = patch.updates.get("qa_report") if slot.node == "audit" else None
            if qa is not None and not qa.passed:                                 # a failing quality report blocks the release to the human gate (T8.5)
                reason = "quality checks failed, the claim is not released to the gate: " + "; ".join(f"{f.check}: {f.detail}" for f in qa.failures[:5])
                parked = StatePatch(agent_id="supervisor", updates={"resolution_status": "awaiting_human"})
                blocked = apply_patch(new, parked)
                self.record(claim, parked, blocked)
                self.events.add(claim, "agent_failed", slot.node, reason, agent_id=slot.agent.agent_id, mock=slot.is_mock, state_version=blocked.state_version, reason=reason,
                                qa_failures=[f.check for f in qa.failures])
                return {**patch.updates, **parked.updates, "state_version": blocked.state_version}
            return {**patch.updates, "state_version": new.state_version}

        run.__name__ = slot.node
        return run

    def _last_digest(self, claim: str, node: str) -> dict[str, str] | None:
        """The input digest of the node's last SUCCESSFUL run (a later failure of the node means: run it again)."""
        for e in reversed(self.events.for_claim(claim)):
            if e.node == node and e.kind == "agent_failed":
                return None
            if e.node == node and e.kind == "agent_finished":
                return e.data.get("inputs")
        return None

    # ------------------------------------------------------------------ routing (conditional edges)
    def after_agent(self, state: ClaimState) -> str:
        """Continue, or halt when an agent failed and parked the claim (only the routing step sets awaiting_human on a healthy claim, and it is the last)."""
        return HALT if state.resolution_status == "awaiting_human" else "continue"

    def after_coverage(self, state: ClaimState) -> str:
        """No policy in force: coverage has recorded the one row that says so, and there is nothing to settle."""
        if state.resolution_status == "awaiting_human":
            return HALT
        if not policy_in_force(state):
            self.events.add(state.claim_id, "routed", "coverage", "no policy in force on the loss date: settlement is skipped", agent_id="supervisor",
                            state_version=state.state_version, skipped=["settlement"])
            return "skip"
        return "continue"

    def after_routing(self, state: ClaimState) -> str:
        """The routing step sets `awaiting_human` on a healthy claim ("ready for the gate"), so a failure is read from the log: was the last thing that
        happened for this claim a failed agent?"""
        last = self.events.for_claim(state.claim_id)
        return HALT if last and last[-1].kind == "agent_failed" else "continue"

    def after_gate(self, state: ClaimState) -> str:
        """approve -> commit; edit -> the settlement is calculated again with the reviewer's inputs and the gate opens again; anything else ends the run."""
        return {"approved": "commit", "assessing": "edit"}.get(state.resolution_status, "end")

    # ------------------------------------------------------------------ the human gate
    def gate_payload(self, state: ClaimState) -> dict[str, Any]:
        r = state.routing
        return {
            "claim_id": state.claim_id, "state_version": state.state_version, "resolution_status": state.resolution_status,
            "tier": r.tier if r else None, "reason_codes": list(r.reason_codes) if r else [],
            "required_approver_role": r.required_approver_role if r else "adjuster",
            "open_questions": {q.question_id: q.status for q in state.open_questions},
            "contradictions": len(state.contradictions), "evidence_requests": [m.minimum_evidence_request.addressee_role for m in state.missing_evidence],
            "coverage_issues": {i.issue_id: i.status for i in state.coverage_issues},
            "settlement": {"status": state.settlement.status, "payable": str(state.settlement.payable) if state.settlement and state.settlement.payable is not None else None}
            if state.settlement else None,
            "mock_agents": self.registry.mocks,
            "allowed_decisions": list(DECISION_STATUS),
            "overrides": {k: str(v) for k, v in self.overrides.get(state.claim_id).items()},
            "low_confidence_facts": sorted(f.fact_id for f in state.facts if self.low_confidence_below is not None and f.extraction_confidence < self.low_confidence_below
                                           and f.verification_status == "unverified"),
            "note": "The system only recommends. Approving here pays or denies nothing; mock agents are not implemented and their output is illustrative.",
        }

    def validate(self, raw: Any, state: ClaimState) -> HumanDecision | str:
        """The decision, or the reason it is refused."""
        try:
            d = HumanDecision.model_validate(raw)
        except Exception as e:
            return f"not a valid decision: {e}"
        if d.reviewer_role not in self.known_roles:
            return f"unknown role '{d.reviewer_role}'; roles are {', '.join(self.known_roles)}"
        need = state.routing.required_approver_role if state.routing else "adjuster"
        if self.rank.get(d.reviewer_role, 0) < self.rank.get(need, 1):
            return f"the role '{d.reviewer_role}' may not decide this claim: it needs '{need}' or higher"
        if d.decision == "reject" and self.guard is not None:                   # rejecting is denying: the autonomy matrix names who may (deny_claim), enforced by the guard
            try:
                self.guard.require("deny_claim", Actor("human", d.reviewer_id, d.reviewer_role), claim_id=state.claim_id)
            except NotAuthorised as e:
                return f"rejecting a claim is not allowed: {e} (autonomy matrix: deny_claim)"
        if d.decision != "approve" and not d.reason.strip():
            return f"a '{d.decision}' decision needs a reason"
        if d.decision == "edit":
            if not d.edits:
                return "an 'edit' decision needs the settlement inputs to change (edits)"
            bad = sorted(set(d.edits) - SETTLEMENT_INPUTS)
            if bad:
                return f"cannot edit {bad}: the settlement inputs are {sorted(SETTLEMENT_INPUTS)}"
            for k, v in d.edits.items():
                try:
                    if not Decimal(v).is_finite() or Decimal(v) < 0:
                        raise InvalidOperation
                except InvalidOperation:
                    return f"'{v}' is not a valid amount for {k}: it must be a number that is not negative"
        elif d.edits:
            return "only an 'edit' decision carries edits"
        return d

    def human_gate(self, state: ClaimState) -> dict[str, Any]:
        payload = self.gate_payload(state)                                      # pure: this node is re-run from here on resume
        attempt = 0
        while True:
            raw = interrupt(payload)
            checked = self.validate(raw, state)
            if isinstance(checked, HumanDecision):
                break
            # the node replays every earlier answer on each resume: the refusal of attempt n is logged once
            self.events.add_once(f"refused:{state.claim_id}:{state.state_version}:{attempt}", state.claim_id, "human_decision_refused", "human_gate", checked,
                                 agent_id="supervisor", state_version=state.state_version)
            payload = {**payload, "error": checked}
            attempt += 1
        if checked.decision == "edit":                                          # a person changes INPUTS; the engine computes again (replayed edits are recorded once)
            self.overrides.set(state.claim_id, {k: Decimal(v) for k, v in checked.edits.items()}, by=checked.reviewer_id, reason=checked.reason)
        patch = StatePatch(agent_id="supervisor", updates={"resolution_status": DECISION_STATUS[checked.decision]})
        new = apply_patch(state, patch)
        self.record(state.claim_id, patch, new)
        self.events.add(state.claim_id, "human_decision", "human_gate",
                        f"{checked.reviewer_id} ({checked.reviewer_role}): {checked.decision}" + (f" - {checked.reason}" if checked.reason else ""),
                        agent_id="supervisor", state_version=new.state_version, decision=checked.decision, reviewer_id=checked.reviewer_id,
                        reviewer_role=checked.reviewer_role, reason=checked.reason, edits=dict(checked.edits))
        return {**patch.updates, "state_version": new.state_version}

    def commit(self, state: ClaimState) -> dict[str, Any]:
        """After an approval: the claim, its controlled tasks and (if the approver may release it) the payment instruction go through the adapters (orchestration/commit.py:
        idempotent, retried, dead-lettered). Without a commit service this is a MOCK that sends nothing."""
        decisions = self.events.of_kind(state.claim_id, "human_decision")
        if not decisions or decisions[-1].data.get("decision") != "approve":         # the only way here is the gate, and this is the guard if that ever changed
            raise RuntimeError(f"claim {state.claim_id}: no recorded human approval: nothing is committed")
        if self.commit_service is not None:
            decision = decisions[-1]
            result = self.commit_service.commit(state, reviewer_id=decision.data["reviewer_id"], reviewer_role=decision.data["reviewer_role"], approval_ref=f"EVT-{decision.seq}")
            patch = StatePatch(agent_id="action", updates={"proposed_actions": result.actions})
            new = apply_patch(state, patch)
            self.record(state.claim_id, patch, new)
            self.events.add(state.claim_id, "commit", "commit", f"approval committed through the adapters (mock systems): {result.summary()}", agent_id="supervisor",
                            mock=result.mock, state_version=new.state_version, dead_lettered=[o.key for o in result.dead_lettered], payment=result.payment,
                            calls=[(o.operation, o.key, o.status) for o in result.outcomes])
            return {**patch.updates, "state_version": new.state_version}
        self.events.add(state.claim_id, "commit", "commit", f"{'YET TO IMPLEMENT'} (mock, T6.5-T6.6): the approval would be committed to the claims system and "
                        "the payment instructed to reconciliation; nothing was sent", agent_id="supervisor", mock=True, state_version=state.state_version)
        return {}
