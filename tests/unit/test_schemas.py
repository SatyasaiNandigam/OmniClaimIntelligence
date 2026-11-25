import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from claimtower.core.schemas import (
    ClaimState, Contradiction, CounterfactualScenario, Fact, RoutingDecision,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "docs/contracts/examples/valid_claim_state_s01.json"


def test_valid_example_loads_and_round_trips():
    state = ClaimState.model_validate_json(EXAMPLE.read_text(encoding="utf-8"))
    assert state.claim_id == "CLM-0001"
    assert ClaimState.model_validate_json(state.model_dump_json()) == state


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        ClaimState(claim_id="c1", bogus=1)


def test_fact_requires_evidence_text_and_bounded_confidence():
    base = dict(fact_id="F", claim_id="c", proposition="p", source_document_id="d", evidence_type="neutral")
    with pytest.raises(ValidationError):
        Fact(**base, evidence_text="", extraction_confidence=0.5)
    with pytest.raises(ValidationError):
        Fact(**base, evidence_text="x", extraction_confidence=1.5)


def test_contradiction_needs_two_sides():
    with pytest.raises(ValidationError):
        Contradiction(contradiction_id="c", question_id="q", fact_ids_a=["F1"], fact_ids_b=[],
                      severity="low", explanation="x")


def test_counterfactual_cannot_predict_outcome():
    with pytest.raises(ValidationError):
        CounterfactualScenario(scenario_id="A", assumed_finding="x",
                               implication_text="The claim will be approved.", next_step="n")


def test_routing_requires_reason_codes():
    with pytest.raises(ValidationError):
        RoutingDecision(tier="simple", reason_codes=[], required_approver_role="adjuster", rules_version="v")


def test_exported_schemas_are_up_to_date():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/export_schemas.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
