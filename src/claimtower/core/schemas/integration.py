"""Integration contracts: request/response models for the PAS, Claims and Recon adapters (BUILD_PLAN T6.5, guardrails 10 and 12).

Every response carries `mock` (always labelled, guardrail 10) and the `correlation_id` that ties PAS / Claims / Recon / audit together.
Every write carries an `idempotency_key` so a retry never creates a second claim, task or payment (guardrail 12).
"""
from __future__ import annotations

from typing import Literal

from .common import Contract, Money, ResolutionStatus, date, datetime
from .intake import ClaimHeader


class IntegrationResponse(Contract):
    mock: bool = True
    correlation_id: str = ""
    source_system: str = ""
    retrieved_at: datetime | None = None


# ----------------------------------------------------------------------------- PAS (read-only)
class EndorsementRef(Contract):
    endorsement_id: str
    effective_from: date
    effective_to: date | None = None


class PolicyStatusResponse(IntegrationResponse):
    policy_no: str
    as_of: date
    status: Literal["in_force", "lapsed", "cancelled", "reinstated", "not_found"]
    status_effective_from: date | None = None


class PolicyVersionResponse(PolicyStatusResponse):
    version: str | None = None                       # None when status is not_found
    effective_from: date | None = None
    effective_to: date | None = None
    base_wording_id: str | None = None
    endorsements: list[EndorsementRef] = []          # as listed in the schedule on `as_of`, whether or not the wording is available
    response_ref: str | None = None                  # stored as PolicySnapshot.source_system_response_ref


# ----------------------------------------------------------------------------- Claims
class ClaimCreateRequest(Contract):
    claim_header: ClaimHeader
    idempotency_key: str
    correlation_id: str = ""


class ClaimStatusUpdate(Contract):
    claim_id: str
    status: ResolutionStatus
    reason: str | None = None
    idempotency_key: str
    correlation_id: str = ""


class ClaimAck(IntegrationResponse):
    claim_id: str
    status: ResolutionStatus
    created: bool = True                             # False when the idempotency key replayed an earlier call


class TaskRequest(Contract):
    claim_id: str
    task_type: str
    assignee_role: str
    description: str
    idempotency_key: str
    correlation_id: str = ""


class TaskAck(IntegrationResponse):
    task_id: str
    claim_id: str
    created: bool = True


class ClaimRecord(IntegrationResponse):
    claim_id: str
    status: ResolutionStatus
    claim_header: ClaimHeader | None = None


# ----------------------------------------------------------------------------- Recon (payment instruction / read-back)
class PaymentInstruction(Contract):
    claim_id: str
    amount: Money
    currency: str = "INR"
    payee_ref: str
    approved_by: str                                 # recorded authorised human (no path to payment without one)
    approval_ref: str                                # audit event id of the human decision
    idempotency_key: str
    correlation_id: str = ""


class ReconAck(IntegrationResponse):
    recon_id: str
    status: Literal["accepted", "duplicate_ignored", "rejected"]
    idempotency_key: str


class MatchStatus(IntegrationResponse):
    recon_id: str
    state: Literal["matched", "unmatched", "pending"]
    detail: str | None = None


class ReconBreak(IntegrationResponse):
    recon_id: str
    claim_id: str
    reason: str
    amount: Money | None = None
