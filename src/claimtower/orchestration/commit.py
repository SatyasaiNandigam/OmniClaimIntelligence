"""The commit step (T6.4 / T6.6): after a recorded human approval the claim, its tasks and (only if authorised) the payment instruction go to the external systems.

    service = CommitService(adapters, dispatcher, planner, settings.autonomy)
    result = service.commit(state, reviewer_id="rev-9", reviewer_role="supervisor", approval_ref="EVT-12")

Every call goes through the Dispatcher: idempotent on its key, retried within bounds, dead-lettered when nothing helps, so a retry or a re-run never creates a second claim,
task or payment, and one failing call never stops the others. What is sent:
  * the claim (`create_claim`) and its status `approved` (`update_status`)
  * each proposed `controlled_api` action that is still `proposed`: a task through the claims system (a draft is NEVER sent: it waits for a person)
  * a payment instruction ONLY when the settlement is calculated and the approving reviewer's role may release payment (configs/autonomy_matrix.yaml `release_payment`);
    otherwise the result says why no payment was instructed
All adapters in the prototype are mocks, so nothing real is sent: every outcome carries the `mock` flag of the adapter that answered.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from claimtower.actions.planner import ActionPlanner, PlannedAction
from claimtower.audit.guards import Actor, AutonomyGuard, NotAuthorised
from claimtower.core.config import AutonomyMatrix
from claimtower.core.schemas import ClaimCreateRequest, ClaimHeader, ClaimState, ClaimStatusUpdate, PaymentInstruction, ProposedAction, TaskRequest
from claimtower.integrations import Adapters
from claimtower.integrations.dispatcher import Dispatcher, Outcome


# the matrix operation each planned action is carried out under
OPERATION_OF = {"create_task": "create_investigation_task", "route_to_operator": "route_to_operator", "register_claim": "register_claim",
                "hold_settlement_until_endorsement_received": "hold_settlement"}


@dataclass
class CommitResult:
    outcomes: list[Outcome] = field(default_factory=list)
    actions: list[ProposedAction] = field(default_factory=list)
    payment: str = ""                                          # what happened to the payment, in words
    mock: bool = True

    @property
    def dead_lettered(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.status == "dead_lettered"]

    def summary(self) -> str:
        done = sum(o.status == "done" for o in self.outcomes)
        replay = sum(o.status == "replayed" for o in self.outcomes)
        bits = [f"{done} call(s) made, {replay} already done earlier, {len(self.dead_lettered)} in the dead-letter queue", f"payment: {self.payment}"]
        return "; ".join(bits)


class CommitService:
    def __init__(self, adapters: Adapters, dispatcher: Dispatcher, planner: ActionPlanner, matrix: AutonomyMatrix, guard: AutonomyGuard | None = None) -> None:
        self.adapters, self.dispatcher, self.planner, self.matrix = adapters, dispatcher, planner, matrix
        self.guard = guard or AutonomyGuard(matrix)

    def commit(self, state: ClaimState, *, reviewer_id: str, reviewer_role: str, approval_ref: str) -> CommitResult:
        cid, corr = state.claim_id, state.correlation_id
        res = CommitResult(mock=any(getattr(a, "mock", True) for a in self.adapters))
        d, a = self.dispatcher, self.adapters
        header = state.claim_header or ClaimHeader(claim_id=cid)
        res.outcomes.append(d.execute("create_claim", f"{cid}:create_claim", cid, lambda: a.claims.create_claim(
            ClaimCreateRequest(claim_header=header, idempotency_key=f"{cid}:create_claim", correlation_id=corr)), header))
        res.outcomes.append(d.execute("update_status", f"{cid}:status:approved", cid, lambda: a.claims.update_status(
            ClaimStatusUpdate(claim_id=cid, status="approved", reason=f"approved by {reviewer_id}", idempotency_key=f"{cid}:status:approved", correlation_id=corr)), None))
        planned = {p.key: p for p in self.planner.plan(state)}
        for act in state.proposed_actions:
            if act.status != "proposed" or act.execution_mode != "controlled_api":
                res.actions.append(act)                          # drafts stay drafts; anything a person already moved on stays as it is
                continue
            try:                                            # an action may only be carried out the way the autonomy matrix allows (T8.4)
                self.guard.require(OPERATION_OF.get(act.action_type, act.action_type), Actor("agent", "commit"), claim_id=cid, mode="controlled_api")
            except NotAuthorised:
                res.actions.append(act.model_copy(update={"status": "failed"}))
                continue
            if act.action_type == "register_claim":
                out = res.outcomes[0]                           # the claim was created above under its own key
            else:
                req = self._task(act, planned.get(act.idempotency_key), state)
                out = d.execute("create_task", act.idempotency_key, cid, lambda r=req: a.claims.create_task(r), req)
                res.outcomes.append(out)
            res.actions.append(act.model_copy(update={"status": "executed" if out.ok else "failed"}))
        res.payment = self._payment(state, reviewer_id, reviewer_role, approval_ref, res)
        return res

    def _assignee(self, act: ProposedAction) -> str:
        """The role that receives a task: the task role of the pack whose task name this action carries, else the adjuster (internal steps)."""
        if act.action_type == "create_task":
            return next((role for role, spec in self.planner.t.task_roles.items() if act.responsible_party.startswith(spec["task"])), "adjuster")
        return "adjuster"

    def _task(self, act: ProposedAction, planned: PlannedAction | None, state: ClaimState) -> TaskRequest:
        title = (planned.title if planned and planned.title else act.action_type.replace("_", " "))
        items = planned.items if planned else []
        desc = f"{title} for claim {state.claim_id}." + ("\nNeeded:\n" + "\n".join(f"- {i}" for i in items) if items else "")
        return TaskRequest(claim_id=state.claim_id, task_type=act.responsible_party if act.action_type == "create_task" else act.action_type,
                           assignee_role=self._assignee(act), description=desc, idempotency_key=act.idempotency_key, correlation_id=state.correlation_id)

    def _payment(self, state: ClaimState, reviewer_id: str, reviewer_role: str, approval_ref: str, res: CommitResult) -> str:
        s = state.settlement
        if s is None or s.status != "calculated" or not s.payable or s.payable <= 0:
            return "not instructed (no calculated payable amount)"
        actor = Actor("human", reviewer_id, reviewer_role)
        allowed, why = self.guard.decide("release_payment", actor)
        if not allowed:
            roles = self.matrix.operations["release_payment"].roles
            return f"not instructed: releasing a payment needs a reviewer with the role {' or '.join(roles)}; the approval was by {reviewer_role}"
        key = f"{state.claim_id}:pay:1"
        req = PaymentInstruction(claim_id=state.claim_id, amount=s.payable, currency=s.currency, payee_ref=f"PAYEE-{state.claim_id}", approved_by=reviewer_id,
                                 approval_ref=approval_ref, idempotency_key=key, correlation_id=state.correlation_id)
        out = self.dispatcher.execute("submit_payment_instruction", key, state.claim_id, lambda: self.adapters.recon.submit_payment_instruction(req, key, actor=actor), req)
        res.outcomes.append(out)
        return f"instruction {out.result.recon_id} {out.result.status}" if out.ok else "instruction could not be delivered (dead-letter queue)"


__all__ = ["CommitResult", "CommitService"]
