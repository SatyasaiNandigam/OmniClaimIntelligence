"""Adapter protocols for the three external systems (BUILD_PLAN T6.5). CONTRACT - frozen at M0.

Mock implementations live in integrations/mock/, real ones in integrations/live/; `core.registry.resolve_adapter` picks one from the
mock flags in configs/app.yaml. Rules every implementation must follow:
  * every response sets `mock` truthfully (guardrail 10)
  * every write is idempotent on `idempotency_key` - a replay returns the original result with `created=False` /
    `status="duplicate_ignored"` and never creates a second claim, task or payment (guardrail 12)
  * PASClient is read-only.
"""
from __future__ import annotations

from datetime import date
from typing import Protocol

from claimtower.core.schemas import (
    ClaimAck, ClaimCreateRequest, ClaimRecord, ClaimStatusUpdate, MatchStatus, PaymentInstruction,
    PolicyStatusResponse, PolicyVersionResponse, ReconAck, ReconBreak, TaskAck, TaskRequest,
)


class PASClient(Protocol):
    """Policy Administration System - read-only."""

    def get_policy_version(self, policy_no: str, as_of: date) -> PolicyVersionResponse: ...

    def get_status(self, policy_no: str, as_of: date) -> PolicyStatusResponse: ...


class ClaimsClient(Protocol):
    """Claims system of record."""

    def create_claim(self, request: ClaimCreateRequest) -> ClaimAck: ...

    def update_status(self, update: ClaimStatusUpdate) -> ClaimAck: ...

    def create_task(self, request: TaskRequest) -> TaskAck: ...

    def get_claim(self, claim_id: str) -> ClaimRecord: ...


class ReconClient(Protocol):
    """Payment reconciliation. Only ever called after a recorded authorised human decision."""

    def submit_payment_instruction(self, request: PaymentInstruction, idempotency_key: str) -> ReconAck: ...

    def get_match_status(self, recon_id: str) -> MatchStatus: ...

    def list_breaks(self, claim_id: str | None = None) -> list[ReconBreak]: ...
