"""ClaimState - the persistent, versioned contract shared by every agent (BUILD_PLAN section 6.1)."""
from __future__ import annotations

from pydantic import Field

from .common import Contract, ResolutionStatus
from .decision import (
    CoverageIssue, Draft, ProposedAction, QAReport, RoutingDecision, SettlementResult,
)
from .evidence import (
    Contradiction, CounterfactualScenario, Fact, ImageFinding, MissingEvidence, OpenQuestion,
)
from .intake import ClaimHeader, DocumentRecord, DuplicateCandidate, IntakeChecklistItem
from .policy import PolicySnapshot


class ClaimState(Contract):
    claim_id: str
    state_version: int = Field(default=0, ge=0)   # increments on every committed patch
    correlation_id: str = ""                      # ties PAS / Claims / Recon / audit together

    # Intake
    claim_header: ClaimHeader | None = None
    intake_checklist: list[IntakeChecklistItem] = Field(default_factory=list)
    duplicate_candidates: list[DuplicateCandidate] = Field(default_factory=list)
    # Policy
    policy_snapshot: PolicySnapshot | None = None
    # Evidence
    documents: list[DocumentRecord] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    image_findings: list[ImageFinding] = Field(default_factory=list)
    # Gap analysis (hero)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    missing_evidence: list[MissingEvidence] = Field(default_factory=list)
    counterfactuals: list[CounterfactualScenario] = Field(default_factory=list)
    # Coverage / settlement
    coverage_issues: list[CoverageIssue] = Field(default_factory=list)
    settlement: SettlementResult | None = None
    # Action planning
    proposed_actions: list[ProposedAction] = Field(default_factory=list)
    drafts: list[Draft] = Field(default_factory=list)
    # QA / audit
    qa_report: QAReport | None = None
    audit_ref: str = ""
    # Supervisor (deterministic)
    routing: RoutingDecision | None = None
    human_approval_required: bool = True
    resolution_status: ResolutionStatus = "registered"
