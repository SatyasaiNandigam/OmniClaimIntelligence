"""The supervisor (T6.1, T6.3): runs each agent safely, decides where the flow goes next, and holds the human approval gate. Deterministic: no model.

Three jobs, each a plain function of the state so they are testable without LangGraph:

  1. NODE WRAPPER. `Supervisor.node(slot)` runs one agent, enforces ownership (core.state.apply_patch), bumps the version and logs an event.
     * an OwnershipViolation is a bug: it is logged and RAISED (the run stops loudly);
     * any other failure (the PAS is down, a policy wording is missing, a model is unreachable ...) is logged with its reason and the claim is parked
       with resolution_status `awaiting_human`: the flow halts, it never loops silently and never continues without that agent's output.
  2. ROUTING. After a pre-routing node the flow halts if the claim was parked; after the gap node a claim with no policy in force skips coverage and
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

from claimtower.core.schemas import ClaimState
from claimtower.core.state import OwnershipViolation, StatePatch, apply_patch
from claimtower.orchestration.events import EventLog
from claimtower.orchestration.registry import AgentRegistry, AgentSlot

Decision = Literal["approve", "reject", "request_more_evidence"]
DECISION_STATUS: dict[str, str] = {"approve": "approved", "reject": "closed", "request_more_evidence": "awaiting_evidence"}
ROLE_RANK: dict[str, int] = {"adjuster": 1, "senior_adjuster": 2, "supervisor": 3}          # SAMPLE: who may decide what a lower role may
HALT = "halt"


class HumanDecision(BaseModel):
    """What a reviewer sends to the gate. Reject and request_more_evidence need a reason."""

    decision: Decision
    reviewer_id: str
    reviewer_role: str
    reason: str = ""

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
    def __init__(self, registry: AgentRegistry, events: EventLog, *, known_roles: tuple[str, ...] = tuple(ROLE_RANK)) -> None:
        self.registry, self.events, self.known_roles = registry, events, known_roles

    # ------------------------------------------------------------------ nodes
    def start(self, state: ClaimState) -> dict[str, Any]:
        """The first node of every (re)assessment: the claim is being assessed again, so any earlier gate state is cleared."""
        patch = StatePatch(agent_id="supervisor", updates={"resolution_status": "assessing"})
        new = apply_patch(state, patch)
        self.events.add(state.claim_id, "run_started", "start", f"assessment started (state v{new.state_version})", agent_id="supervisor", state_version=new.state_version)
        return {**patch.updates, "state_version": new.state_version}

    def node(self, slot: AgentSlot):
        def run(state: ClaimState) -> dict[str, Any]:
            claim = state.claim_id
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
                self.events.add(claim, "agent_failed", slot.node, f"the {slot.node} agent failed, the claim is parked for a human: {reason}",
                                agent_id=slot.agent.agent_id, mock=slot.is_mock, state_version=new.state_version, reason=reason)
                return {**parked.updates, "state_version": new.state_version}
            self.events.add(claim, "agent_finished", slot.node, patch.rationale or "(no rationale)", agent_id=slot.agent.agent_id, mock=slot.is_mock,
                            state_version=new.state_version, fields=sorted(patch.updates))
            return {**patch.updates, "state_version": new.state_version}

        run.__name__ = slot.node
        return run

    # ------------------------------------------------------------------ routing (conditional edges)
    def after_agent(self, state: ClaimState) -> str:
        """Continue, or halt when an agent failed and parked the claim (only the routing step sets awaiting_human on a healthy claim, and it is the last)."""
        return HALT if state.resolution_status == "awaiting_human" else "continue"

    def after_gap(self, state: ClaimState) -> str:
        if state.resolution_status == "awaiting_human":
            return HALT
        if not policy_in_force(state):
            self.events.add(state.claim_id, "routed", "gap", "no policy in force on the loss date: coverage and settlement are skipped", agent_id="supervisor",
                            state_version=state.state_version, skipped=["coverage", "settlement"])
            return "skip"
        return "continue"

    def after_routing(self, state: ClaimState) -> str:
        """The routing step sets `awaiting_human` on a healthy claim ("ready for the gate"), so a failure is read from the log: was the last thing that
        happened for this claim a failed agent?"""
        last = self.events.for_claim(state.claim_id)
        return HALT if last and last[-1].kind == "agent_failed" else "continue"

    def after_gate(self, state: ClaimState) -> str:
        return "commit" if state.resolution_status == "approved" else "end"

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
        if ROLE_RANK.get(d.reviewer_role, 0) < ROLE_RANK.get(need, 1):
            return f"the role '{d.reviewer_role}' may not decide this claim: it needs '{need}' or higher"
        if d.decision != "approve" and not d.reason.strip():
            return f"a '{d.decision}' decision needs a reason"
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
        patch = StatePatch(agent_id="supervisor", updates={"resolution_status": DECISION_STATUS[checked.decision]})
        new = apply_patch(state, patch)
        self.events.add(state.claim_id, "human_decision", "human_gate",
                        f"{checked.reviewer_id} ({checked.reviewer_role}): {checked.decision}" + (f" - {checked.reason}" if checked.reason else ""),
                        agent_id="supervisor", state_version=new.state_version, decision=checked.decision, reviewer_id=checked.reviewer_id,
                        reviewer_role=checked.reviewer_role, reason=checked.reason)
        return {**patch.updates, "state_version": new.state_version}

    def commit(self, state: ClaimState) -> dict[str, Any]:
        """MOCK. After an approval the Claims and Recon adapters will be instructed (T6.5-T6.6). Not implemented: nothing is sent anywhere."""
        self.events.add(state.claim_id, "commit", "commit", f"{'YET TO IMPLEMENT'} (mock, T6.5-T6.6): the approval would be committed to the claims system and "
                        "the payment instructed to reconciliation; nothing was sent", agent_id="supervisor", mock=True, state_version=state.state_version)
        return {}
