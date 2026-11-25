"""Typed loader for data/scenarios/scenarios.yaml and data/ground_truth/*.json (T1.2).

Ground truth reuses the T0.2 contracts (Fact, Contradiction, MissingEvidence, CounterfactualScenario,
RoutingDecision) so labels cannot drift from what agents must produce. The T8.1 eval runner imports this.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import model_validator

from claimtower.core.schemas import (
    ClaimHeader, Contradiction, CounterfactualScenario, Fact, MissingEvidence, RoutingDecision,
)
from claimtower.core.schemas.common import Contract, IssueStatus, QuestionCategory, QuestionStatus

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_FILE = ROOT / "data" / "scenarios" / "scenarios.yaml"
GROUND_TRUTH_DIR = ROOT / "data" / "ground_truth"


# ------------------------------------------------------------------ scenario inputs
class CatalogueClause(Contract):
    clause_id: str
    clause_type: str
    document: str
    purpose: str


class ScenarioDocument(Contract):
    document_id: str
    filename: str
    doc_type: str
    received_on: str
    uploaded_by: str
    pages: list[str]


class Scenario(Contract):
    id: str
    title: str
    tests: list[str]
    claim_id: str
    claim: dict
    pas_expectation: dict
    documents: list[ScenarioDocument]
    events: list[dict]

    def header(self) -> ClaimHeader:
        return ClaimHeader(claim_id=self.claim_id, **self.claim)

    def page_text(self, document_id: str, page: int) -> str:
        doc = next(d for d in self.documents if d.document_id == document_id)
        return doc.pages[page - 1]


class ScenarioFile(Contract):
    schema_version: int
    synthetic: Literal[True]
    note: str
    clause_catalogue: list[CatalogueClause]
    scenarios: list[Scenario]


# ------------------------------------------------------------------ ground truth
class GTQuestion(Contract):
    question_id: str
    text: str
    category: QuestionCategory
    status: QuestionStatus
    establishing_criteria: list[str]
    linked_clause_ids: list[str] = []
    linked_fact_ids: list[str] = []
    expected_answer: str


class GTIssue(Contract):
    issue_id: str
    issue: str
    status: IssueStatus
    clause_ids: list[str]
    fact_ids: list[str] = []
    note: str | None = None

    @model_validator(mode="after")
    def _cited(self) -> "GTIssue":
        if not self.clause_ids:
            raise ValueError(f"{self.issue_id}: every issue needs >=1 clause citation")
        return self


class GTSnapshot(Contract):
    status: Literal["in_force", "not_in_force"]
    effective_clause_ids: list[str]
    not_effective_clause_ids: list[str]
    missing_documents: list[str]
    precedence_note: str
    reason: str | None = None


class GTSettlement(Contract):
    status: Literal["calculated", "cannot_calculate"]
    inputs: dict[str, str]
    missing_inputs: list[str]
    payable: str | None
    unavailable_inputs: list[str] = []   # inputs that apply but cannot be determined yet (S-05)
    engine_check: bool = True


class GroundTruth(Contract):
    scenario_id: str
    claim_id: str
    synthetic: Literal[True]
    facts: list[Fact]
    questions: list[GTQuestion]
    contradictions: list[Contradiction]
    missing_evidence: list[MissingEvidence]
    counterfactuals: list[CounterfactualScenario]
    issues: list[GTIssue]
    snapshot: GTSnapshot
    routing: RoutingDecision
    routing_features: dict[str, str | bool | float]   # expected derived features that feed routing_rules.yaml
    settlement: GTSettlement | None
    expected_actions: list[str]
    human_approval_required: Literal[True]
    notes: list[str]


def load_scenarios(path: Path = SCENARIOS_FILE) -> ScenarioFile:
    return ScenarioFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_ground_truth(scenario_id: str, directory: Path = GROUND_TRUTH_DIR) -> GroundTruth:
    return GroundTruth.model_validate_json((directory / f"{scenario_id}.json").read_text(encoding="utf-8"))


def load_all_ground_truth(directory: Path = GROUND_TRUTH_DIR) -> dict[str, GroundTruth]:
    return {p.stem: load_ground_truth(p.stem, directory) for p in sorted(directory.glob("S-*.json"))}
