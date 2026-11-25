"""Evidence contracts: Evidence agent (facts, image findings) and Gap agent (questions, gaps, counterfactuals)."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .common import (
    BBox, Confidence, Contract, GapType, QuestionCategory, QuestionStatus, Severity,
)


class Fact(Contract):
    """Provenance is mandatory. evidence_text must be a verbatim substring of the page (checked in T2.4)."""

    fact_id: str
    claim_id: str
    proposition: str
    source_document_id: str
    page: int | None = None
    region: BBox | None = None
    evidence_text: str = Field(min_length=1)
    evidence_type: Literal["supporting", "contradicting", "neutral"]
    related_question_ids: list[str] = []
    extraction_confidence: Confidence       # NOT probability of coverage
    verification_status: Literal["unverified", "human_verified", "rejected"] = "unverified"
    extractor_version: str = "0"


class ImageFinding(Contract):
    finding_id: str
    document_id: str
    observation: str                        # visible damage only; never cause/date/coverage
    region: BBox | None = None
    confidence: Confidence
    is_visible_fact: bool = True            # False => inference (must be flagged)
    limitation_note: str | None = None
    quality_verdict: Literal["usable", "poor", "unrelated", "unknown"] = "unknown"
    suggested_next_photo: str | None = None


class OpenQuestion(Contract):
    question_id: str
    text: str
    category: QuestionCategory
    status: QuestionStatus = "unverified"
    establishing_criteria: list[str] = []   # "what would establish this"
    linked_clause_ids: list[str] = []
    linked_fact_ids: list[str] = []


class Contradiction(Contract):
    contradiction_id: str
    question_id: str
    fact_ids_a: list[str] = Field(min_length=1)
    fact_ids_b: list[str] = Field(min_length=1)    # no contradiction without two cited facts
    kind: Literal["date", "amount", "cause", "extent", "other"] = "other"
    severity: Severity
    explanation: str


class MinimumEvidenceRequest(Contract):
    addressee_role: str                     # who to ask (plumber, claimant, surveyor ...)
    items: list[str] = Field(min_length=1)  # what exactly
    must_answer_sub_questions: list[str] = []
    maps_to_criteria: list[str] = []        # self-check: each item maps to a T4.4 criterion


class MissingEvidence(Contract):
    question_id: str
    gap_type: GapType
    minimum_evidence_request: MinimumEvidenceRequest
    resolves_question_ids: list[str] = []


class CounterfactualScenario(Contract):
    scenario_id: Literal["A", "B", "C"]
    question_id: str | None = None
    assumed_finding: str
    affected_issue_ids: list[str] = []
    implication_text: str                   # never predicts approve/deny
    next_step: str

    @model_validator(mode="after")
    def _no_outcome_prediction(self) -> "CounterfactualScenario":
        banned = ("will be approved", "will be denied", "will be rejected", "claim approved", "claim denied")
        text = f"{self.implication_text} {self.next_step}".lower()
        hit = [b for b in banned if b in text]
        if hit:
            raise ValueError(f"counterfactual must not predict an outcome: {hit}")
        return self
