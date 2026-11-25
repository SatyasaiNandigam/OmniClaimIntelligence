"""Coverage, settlement, routing, actions, QA and audit contracts."""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import (
    ActorType, Contract, ExecutionMode, IssueStatus, Money, Tier, datetime,
)
from .policy import ClauseCitation


class CoverageIssue(Contract):
    issue_id: str
    issue: str
    status: IssueStatus
    clause_citations: list[ClauseCitation] = []   # >=1 required unless abstaining (enforced by QA agent)
    fact_ids: list[str] = []
    question_ids: list[str] = []
    note: str | None = None                       # e.g. both readings for an ambiguous exclusion


class SettlementStepTrace(Contract):
    rule_id: str
    input_ref: str | None = None
    amount_before: Money
    amount_after: Money


class SettlementInputs(Contract):
    """Typed input of the deterministic engine (the LLM extracts and cites these; it never computes).

    `unavailable_inputs` lists inputs that APPLY to this claim but cannot be determined yet (e.g. a sublimit endorsement
    referenced by the schedule whose wording is missing, scenario S-05). The engine must report them as missing and
    never calculate without them - this is what `optional: true` in the rule file cannot express.
    """

    assessed_covered_loss: Money | None = None
    non_covered_items_total: Money | None = None
    applicable_deductible: Money | None = None
    water_damage_sublimit: Money | None = None
    remaining_policy_limit: Money | None = None
    prior_payments_total: Money | None = None
    unavailable_inputs: list[str] = []
    input_citations: dict[str, list[str]] = {}    # input name -> fact_ids / clause_ids it was taken from


class SettlementResult(Contract):
    status: Literal["calculated", "cannot_calculate"]
    reserve_forecast: Money | None = None
    assessed_loss: Money | None = None
    payable: Money | None = None
    steps: list[SettlementStepTrace] = []
    missing_inputs: list[str] = []
    rule_version: str | None = None
    currency: str = "INR"
    explanation: str | None = None                # generated from trace, re-verified against engine


class RoutingDecision(Contract):
    tier: Tier
    reason_codes: list[str] = Field(min_length=1)
    required_approver_role: str
    rules_version: str


class ProposedAction(Contract):
    action_id: str
    action_type: str
    responsible_party: str
    execution_mode: ExecutionMode
    status: Literal["proposed", "approved", "executed", "failed", "cancelled"] = "proposed"
    idempotency_key: str
    payload_ref: str | None = None


class Draft(Contract):
    draft_id: str
    kind: Literal["information_request", "status_message", "acknowledgement", "other"]
    addressee_role: str
    subject: str
    body: str
    status: Literal["draft", "approved", "sent"] = "draft"


class QAFailure(Contract):
    check: str
    detail: str
    ref: str | None = None


class QAReport(Contract):
    passed: bool
    failures: list[QAFailure] = []


class AuditEvent(Contract):
    event_id: str
    ts: datetime
    actor_type: ActorType
    actor_id: str
    action: str
    state_version_before: int
    state_version_after: int
    inputs_ref: str | None = None
    outputs_ref: str | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    override_reason: str | None = None


class HumanDecision(Contract):
    """Payload of the `human_decision` event (T6.3)."""

    claim_id: str
    decision: Literal["approve", "edit", "reject", "request_more_evidence"]
    reviewer_id: str
    reviewer_role: str
    reason: str | None = None                    # mandatory for edit/reject (checked by the gate)
    decided_at: datetime | None = None
