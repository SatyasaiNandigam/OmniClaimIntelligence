"""Request and response models of the API (T7.1). The claim itself is the frozen `ClaimState` contract; everything here is the envelope around it."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from claimtower.core.schemas import ClaimState, DocumentRecord
from claimtower.core.state_diff import StateDiff
from claimtower.orchestration.graph import RunStatus, VersionInfo


class ClaimCreate(BaseModel):
    """Create a claim and run it to the human gate. Everything is optional: with no id one is generated, with no header the intake agent (a mock today)
    returns the labelled fixture of a known synthetic claim id, or leaves the header as given."""

    claim_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,63}$", examples=["CLM-0001"])
    claimant_name: str | None = None
    policy_ref: str | None = Field(default=None, examples=["POL-1001"])
    loss_date: date | None = None
    loss_type: str | None = Field(default=None, examples=["water_damage"])
    loss_description: str | None = None
    channel: Literal["portal", "call_centre", "email", "agent", "other"] = "other"
    use_synthetic_documents: bool = Field(default=False, description="Load the SYNTHETIC documents of data/synthetic/claims/<claim_id> into the claim (demo).")


class ClaimView(BaseModel):
    """A claim: where it stands (phase, gate, which agents are mocks) and its full state at the current version."""

    claim_id: str
    state_version: int
    status: RunStatus
    state: ClaimState


class ClaimSummary(BaseModel):
    claim_id: str
    phase: str
    state_version: int | None
    resolution_status: str | None
    tier: str | None = None
    open_questions: dict[str, int] = {}                 # status -> count
    mock_agents: list[str] = []


class DocumentUploadResult(BaseModel):
    document: DocumentRecord
    reassessment: RunStatus | None = Field(default=None, description="The claim re-assessed from the evidence node (null when the claim was not re-run).")
    note: str = ""


class EventView(BaseModel):
    seq: int
    ts: datetime
    kind: str
    node: str
    message: str
    agent_id: str | None = None
    mock: bool = False
    state_version: int | None = None
    data: dict[str, Any] = {}


class AuditView(BaseModel):
    claim_id: str
    events: list[EventView]
    human_decisions: list[EventView]
    agents: list[dict[str, str]]
    note: str = "In-memory event log of the prototype: it does not survive a restart. The Quality and Audit agent (T8.5) will persist it."


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "claimtower"
    prototype: bool = True
    synthetic_data_only: bool = True
    mock_agents: list[str]
    implemented_agents: list[str]
    claims: int
    checkpointer: str = "memory"


class ErrorDetail(BaseModel):
    detail: str | dict[str, Any]


__all__ = ["AuditView", "ClaimCreate", "ClaimSummary", "ClaimView", "DocumentUploadResult", "ErrorDetail", "EventView", "Health", "StateDiff", "VersionInfo"]
