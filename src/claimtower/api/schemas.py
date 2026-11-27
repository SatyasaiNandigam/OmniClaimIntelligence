"""Request and response models of the API (T7.1). The claim itself is the frozen `ClaimState` contract; everything here is the envelope around it."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from claimtower.core.schemas import ClaimState, DocumentRecord
from claimtower.core.state_diff import StateDiff
from claimtower.integrations.mock.data import PasPolicy
from claimtower.orchestration.graph import RunStatus, VersionInfo


class ClaimCreate(BaseModel):
    """Create a claim and run it to the human gate. Everything is optional: with no id one is generated, with no header the intake agent reads the
    claim's first notice of loss (a stored document) and fills what is missing; what you supply is kept."""

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
    reassessment: RunStatus | None = Field(default=None, description="The claim re-assessed from the intake node (null when the claim was not re-run).")
    note: str = ""


class SkippedFile(BaseModel):
    filename: str
    reason: str


class ClaimWithDocumentsResult(BaseModel):
    """The claim created from its documents in one call: the claim run once to the human gate, the documents stored, and any file that was skipped and why."""

    claim: ClaimView
    stored: list[DocumentRecord] = Field(description="The documents stored with the claim (their types are decided when the claim is assessed).")
    skipped: list[SkippedFile] = Field(default=[], description="Files that were not stored (empty, too large, unreadable) and why. They do not fail the claim.")


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
    note: str = "In-memory event log of the prototype: it does not survive a restart. A persistent audit trail (T8.3) will keep it."


class PageView(BaseModel):
    number: int
    source: Literal["text_layer", "ocr", "none"]
    needs_ocr: bool
    text: str = ""


class FieldView(BaseModel):
    name: str
    value: str
    normalised: str | None = None
    label: str = ""
    page: int
    region: dict[str, float] | None = None
    evidence_text: str
    confidence: float
    low_confidence: bool
    method: str = "rule"
    flags: list[str] = []


class DocumentDetails(BaseModel):
    """What the system read in one document (T7.3): how it was typed and why, the pages and where their text came from, and the fields extracted with page, region and confidence."""

    document_id: str
    filename: str
    detected_type: str
    type_confidence: float
    type_method: str | None = None
    type_reason: str | None = None
    type_evidence_snippet: str | None = None
    type_page: int | None = None
    media_type: str | None = None
    kind: str | None = None
    page_count: int
    uploaded_by: str | None = None
    source_channel: str | None = None
    uploaded_at: datetime | None = None
    same_content_as: list[str] = []
    ocr_engine: str | None = None
    ocr_errors: list[str] = []
    load_error: str | None = None
    pages: list[PageView] = []
    fields: list[FieldView] = []
    missing_required: list[str] = []
    warnings: list[str] = []
    low_confidence_below: float | None = None


class PolicyDocumentView(BaseModel):
    """One policy document in the library. The id and version were read from INSIDE the document (`id_from`), not taken from the file name."""

    doc_id: str
    title: str | None = None
    kind: str
    version: str | None = None
    clauses: int
    unclassified: int = 0
    warnings: list[str] = []
    source: Literal["built-in", "uploaded"]
    filename: str
    sha256: str
    id_from: Literal["document", "model", "given", "file_name"]
    uploaded_by: str | None = None
    uploaded_at: str | None = None
    replaces_built_in: bool = False
    id_evidence: str = ""
    details: dict[str, str] = {}


class PolicyDocumentResult(BaseModel):
    outcome: Literal["added", "replaced", "unchanged"]
    document: PolicyDocumentView
    note: str = ""


class PolicyBatchItem(BaseModel):
    filename: str
    outcome: Literal["added", "replaced", "unchanged", "failed"]
    document: PolicyDocumentView | None = None
    status: int = Field(200, description="What the single-file route would have answered for this file: 200 / 201 ok, 409 conflict, 422 unreadable.")
    error: str = ""


class PolicyBatchResult(BaseModel):
    """One result per file, in the order sent. The batch itself succeeds even when some files fail: look at each `outcome`."""

    results: list[PolicyBatchItem]
    added: int = 0
    replaced: int = 0
    unchanged: int = 0
    failed: int = 0
    note: str = ""


class PasPolicyView(BaseModel):
    """One policy of the mock policy administration system (PAS)."""

    policy_no: str
    holder_name: str
    versions: int
    effective_from: date | None = None
    effective_to: date | None = None
    base_wording_id: str = ""
    endorsements: list[str] = []
    schedule: dict[str, str] = {}
    source: Literal["synthetic", "registered"]
    note: str = ""


class ScheduleCheckView(BaseModel):
    field: str
    value: str
    found: bool
    note: str = ""


class ScheduleReadResult(BaseModel):
    """What the system read from a policy schedule. Nothing is stored: a person looks at `policy` and its checks, may correct it, and registers it with POST /pas/policies."""

    filename: str
    policy: PasPolicy | None = None
    checks: list[ScheduleCheckView] = []
    problems: list[str] = Field([], description="Why there is no record (nothing could be confirmed from the document).")
    warnings: list[str] = Field([], description="What to look at before registering, e.g. the base wording is not named.")
    existing: Literal["none", "synthetic", "registered"] = "none"
    read_by: Literal["model", "rules"] = "model"


class PasRegisterResult(BaseModel):
    outcome: Literal["added", "replaced"]
    policy_no: str
    note: str = ""


class PolicyReloadResult(BaseModel):
    total: int
    added: list[str] = []
    removed: list[str] = []
    changed: list[str] = []
    problems: list[str] = []


class TrailRecordView(BaseModel):
    seq: int
    ts: str
    kind: str
    summary: str
    hash: str


class TrailView(BaseModel):
    """The persistent audit trail of one claim (T8.3): append-only, hash-chained, enough to reconstruct the decision."""

    claim_id: str
    records: list[TrailRecordView]
    chain_ok: bool
    problems: list[str] = []
    head: str


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "claimtower"
    prototype: bool = True
    synthetic_data_only: bool = True
    mock_agents: list[str]
    implemented_agents: list[str]
    claims: int
    checkpointer: str = "memory"
    low_confidence_below: float | None = None          # an extraction confidence under this is marked for a person to check


class ErrorDetail(BaseModel):
    detail: str | dict[str, Any]


__all__ = ["AuditView", "ClaimCreate", "ClaimSummary", "ClaimView", "DocumentUploadResult", "ErrorDetail", "EventView", "Health", "StateDiff", "VersionInfo"]
