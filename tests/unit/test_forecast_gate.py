"""While the repair cost is not validated no payable is calculated: the deterministic engine gives a forecast range instead (never a payment)."""
from datetime import date
from decimal import Decimal

import pytest

from claimtower.agents.settlement import SettlementAgent
from claimtower.composition import SETTLEMENT_RULES
from claimtower.core.config import load_settings
from claimtower.core.schemas import ClaimHeader, ClaimState, Fact, OpenQuestion, PolicySnapshot
from claimtower.core.schemas.policy import PolicyClause
from claimtower.integrations.mock.claims_mock import MockPaymentsLedger
from claimtower.integrations.mock.pas_mock import MockPasAdapter
from claimtower.settlement.engine import SettlementCalculator
from claimtower.settlement.explain import verify
from claimtower.settlement.rules_loader import rules_by_version

CALC = SettlementCalculator(rules_by_version(load_settings().pack.calculation))
AGENT = SettlementAgent(CALC, SETTLEMENT_RULES, MockPasAdapter(), MockPaymentsLedger())


def fact(fid, p):
    return Fact(fact_id=fid, claim_id="C", proposition=p, source_document_id="D", page=1, evidence_text=p, evidence_type="neutral", related_question_ids=["Q-AMOUNT"],
                extraction_confidence=0.9)


SUBLIMIT = PolicyClause(clause_id="E-01/5.2", clause_type="sublimit", heading="Clause 5.2", text="the maximum We will pay is INR 300,000 in aggregate.", document="E-01", page=1, span=(0, 10))


def state(status, with_estimate=True, sublimit=True):
    facts = [fact("a", "Surveyor assessed the repair cost at INR 254,260")] + ([fact("e", "Estimate total is 450,000")] if with_estimate else [])
    q = OpenQuestion(question_id="Q-AMOUNT", text="t", category="amount", status=status, linked_fact_ids=[f.fact_id for f in facts])
    snap = PolicySnapshot(policy_id="POL-1001", version="v1", effective_from=date(2026, 1, 1), clauses=[SUBLIMIT] if sublimit else [])
    return ClaimState(claim_id="C", claim_header=ClaimHeader(claim_id="C", policy_ref="POL-1001", loss_date=date(2026, 5, 2)), facts=facts, open_questions=[q], policy_snapshot=snap)


@pytest.mark.parametrize("status", ["partially_established", "possible", "missing", "requires_validation"])
def test_an_unvalidated_repair_cost_holds_the_payable_and_shows_a_range_the_sublimit_caps(status):
    r = AGENT.run(state(status)).updates["settlement"]
    assert r.status == "cannot_calculate" and r.payable is None and r.missing_inputs == ["validated_repair_cost"]
    assert (r.forecast_low, r.forecast_high) == (Decimal(244260), Decimal(300000))          # 254,260 less the 10,000 deductible; 450,000 less 10,000 capped by the 300,000 sublimit
    assert "between INR 244,260 and INR 300,000" in r.explanation and "No payable amount is shown" in r.explanation and verify(r.explanation, r) == []


def test_the_range_is_one_figure_when_nothing_higher_was_claimed():
    r = AGENT.run(state("partially_established", with_estimate=False)).updates["settlement"]
    assert (r.forecast_low, r.forecast_high) == (Decimal(244260), Decimal(244260)) and "about INR 244,260" in r.explanation


def test_an_established_repair_cost_is_calculated_as_before():
    r = AGENT.run(state("established")).updates["settlement"]
    assert r.status == "calculated" and r.payable == Decimal(244260) and r.forecast_low is None


def test_unverified_means_the_judge_did_not_run_so_the_calculation_is_unchanged():
    r = AGENT.run(state("unverified")).updates["settlement"]
    assert r.status == "calculated" and r.payable == Decimal(244260)
