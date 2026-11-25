"""Intake-owned and document contracts: ClaimHeader, DocumentRecord, DuplicateCandidate."""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import Confidence, Contract, date, datetime


class ClaimHeader(Contract):
    claim_id: str
    claimant_name: str | None = None
    policy_ref: str | None = None          # as given by the claimant; exact PAS match is separate
    loss_date: date | None = None          # may be missing at FNOL (S-17): tracked, not blocking
    loss_type: str | None = None           # e.g. "water_damage"
    loss_description: str | None = None
    channel: Literal["portal", "call_centre", "email", "agent", "other"] = "other"
    registered_at: datetime | None = None


class IntakeChecklistItem(Contract):
    item_id: str
    description: str
    required: bool = True
    received: bool = False
    document_id: str | None = None


class DocumentRecord(Contract):
    document_id: str
    claim_id: str
    filename: str
    sha256: str
    page_count: int = Field(default=1, ge=1)
    detected_type: str = "unknown"         # from pack document_types.yaml; "unknown" -> human
    type_confidence: Confidence = 0.0
    type_evidence_snippet: str | None = None
    source_channel: str | None = None
    uploaded_by: str | None = None
    uploaded_at: datetime | None = None
    needs_ocr: bool = False
    blob_path: str | None = None


class DuplicateCandidate(Contract):
    """A *possible* duplicate. Never fraud, never auto-merged (guardrail 3)."""

    candidate_id: str
    duplicate_type: Literal["fnol", "invoice", "payment", "document"]
    linked_claim_id: str | None = None
    matched_fields: list[str] = []
    differing_fields: list[str] = []
    prior_outcome: str | None = None
    disposition: Literal["needs_review", "related_but_distinct", "confirmed_duplicate"] = "needs_review"
    explanation: str = ""
