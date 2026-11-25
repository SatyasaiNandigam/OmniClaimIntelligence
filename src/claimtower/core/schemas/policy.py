"""Policy contracts (owner: Policy agent). PolicySnapshot = contract in force on the date of loss."""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import Contract, date, datetime

ClauseType = Literal["coverage", "exclusion", "definition", "condition", "limit", "sublimit", "deductible"]
RelationType = Literal["modifies", "excepts", "defines", "conditions"]


class PolicyClause(Contract):
    clause_id: str                          # stable ID, referenced by ground truth
    clause_type: ClauseType
    heading: str | None = None
    text: str                               # verbatim span
    document: str                           # source document (base wording / endorsement id)
    page: int | None = None
    span: tuple[int, int] | None = None     # char offsets on the page


class ClauseRelation(Contract):
    source_clause_id: str
    target_clause_id: str
    relation: RelationType


class Endorsement(Contract):
    endorsement_id: str
    title: str
    effective_from: date
    effective_to: date | None = None
    clause_ids: list[str] = []


class PolicySnapshot(Contract):
    policy_id: str
    version: str
    effective_from: date
    effective_to: date | None = None
    pas_status: Literal["in_force", "lapsed", "cancelled", "reinstated"] = "in_force"
    clauses: list[PolicyClause] = []
    endorsements: list[Endorsement] = []
    relations: list[ClauseRelation] = []
    precedence_rules: list[str] = []        # stated per policy, never assumed silently
    source_system_response_ref: str | None = None
    retrieved_at: datetime | None = None
    mock: bool = True                       # guardrail 10


class NotInForce(Contract):
    """Returned by get_snapshot when no policy version applies on the loss date."""

    policy_id: str
    loss_date: date
    reason: str


class ClauseCitation(Contract):
    clause_id: str
    document: str
    page: int | None = None
    span: tuple[int, int] | None = None
    text: str
    modified_by: list[str] = Field(default_factory=list)   # endorsement clause IDs returned with the base clause
