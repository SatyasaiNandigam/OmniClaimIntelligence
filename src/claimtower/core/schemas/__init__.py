"""CONTRACT - changes after M0 need Lead approval.

Pydantic models for ClaimState and everything it contains. Exported to docs/contracts/ by
scripts/export_schemas.py (CI fails on drift). Field owners: docs/contracts/README.md.
"""
from .claim_state import ClaimState
from .common import BBox, Contract
from .decision import (
    AuditEvent, CoverageIssue, Draft, HumanDecision, ProposedAction, QAFailure, QAReport,
    RoutingDecision, SettlementInputs, SettlementResult, SettlementStepTrace,
)
from .evidence import (
    Contradiction, CounterfactualScenario, Fact, ImageFinding, MinimumEvidenceRequest,
    MissingEvidence, OpenQuestion,
)
from .integration import (
    ClaimAck, ClaimCreateRequest, ClaimRecord, ClaimStatusUpdate, EndorsementRef, IntegrationResponse,
    MatchStatus, PaymentInstruction, PolicyStatusResponse, PolicyVersionResponse, ReconAck, ReconBreak,
    TaskAck, TaskRequest,
)
from .intake import ClaimHeader, DocumentRecord, DuplicateCandidate, IntakeChecklistItem
from .policy import (
    ClauseCitation, ClauseRelation, Endorsement, NotInForce, PolicyClause, PolicySnapshot,
)

# Models exported as standalone JSON Schemas (ClaimState embeds the rest via $defs).
EXPORTED = [
    ClaimState, Fact, OpenQuestion, Contradiction, MissingEvidence, CounterfactualScenario,
    CoverageIssue, PolicySnapshot, ClauseCitation, NotInForce, SettlementResult,
    RoutingDecision, ProposedAction, AuditEvent, HumanDecision, SettlementInputs,
    # integration contracts (adapter protocols in integrations/base.py)
    PolicyStatusResponse, PolicyVersionResponse, ClaimCreateRequest, ClaimStatusUpdate, ClaimAck,
    TaskRequest, TaskAck, ClaimRecord, PaymentInstruction, ReconAck, MatchStatus, ReconBreak,
]

__all__ = [m.__name__ for m in EXPORTED] + [
    "BBox", "Contract", "ClaimHeader", "DocumentRecord", "DuplicateCandidate", "IntakeChecklistItem",
    "ImageFinding", "MinimumEvidenceRequest", "PolicyClause", "ClauseRelation", "Endorsement",
    "Draft", "QAReport", "QAFailure", "SettlementStepTrace", "EndorsementRef", "IntegrationResponse", "EXPORTED",
]
