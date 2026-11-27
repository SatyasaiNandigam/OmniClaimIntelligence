"""T5.5 - the Settlement agent: extract -> compute -> explain. Matches the ground truth on every scenario; never invents a number; the explanation is re-verified."""
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from claimtower.agents.settlement import ExplanationRewriter, SettlementAgent, SettlementExplanationError, make_settlement_agent
from claimtower.agents.settlement_inputs import AmountItem, AmountsAnswer, Extracted, LLMAmountReader, combine, rule_amounts, sublimit_from_snapshot
from claimtower.composition import SETTLEMENT_RULES
from claimtower.core.config import load_settings
from claimtower.core.schemas import ClaimHeader, ClaimState, Contradiction, Fact, OpenQuestion, SettlementResult, SettlementStepTrace
from claimtower.core.state import OwnershipViolation, StatePatch, apply_patch
from claimtower.integrations.mock.claims_mock import MockPaymentsLedger
from claimtower.integrations.mock.pas_mock import MockPasAdapter
from claimtower.settlement.engine import SettlementCalculator
from claimtower.settlement.explain import explain, verify
from claimtower.settlement.rules_loader import rules_by_version

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()
CALC = SettlementCalculator(rules_by_version(SETTINGS.pack.calculation))


def agent(**kw) -> SettlementAgent:
    return SettlementAgent(CALC, SETTLEMENT_RULES, MockPasAdapter(), MockPaymentsLedger(), **kw)


def fixture(sid) -> ClaimState:
    return ClaimState(**json.loads((ROOT / f"data/fixtures/{sid}/claim_state.json").read_text(encoding="utf-8")))


def truth(sid) -> dict:
    return json.loads((ROOT / f"data/ground_truth/{sid}.json").read_text(encoding="utf-8"))["settlement"]


def fact(fid, p, e=None, q="Q-AMOUNT") -> Fact:
    return Fact(fact_id=fid, claim_id="C", proposition=p, source_document_id="D", page=1, evidence_text=e or p, evidence_type="neutral", related_question_ids=[q],
                extraction_confidence=0.9)


def state_with(*facts, **kw) -> ClaimState:
    q = OpenQuestion(question_id="Q-AMOUNT", text="t", category="amount", status="unverified", linked_fact_ids=[f.fact_id for f in facts])
    return ClaimState(claim_id="C", claim_header=ClaimHeader(claim_id="C", policy_ref="POL-1001", loss_date=date(2026, 5, 2)), facts=list(facts), open_questions=[q], **kw)


# ---------------------------------------------------------------------------------------------------------------- the labelled scenarios
@pytest.mark.parametrize("sid", ["S-01", "S-02", "S-04", "S-05"])
def test_every_scenario_matches_the_ground_truth_without_a_model(sid):
    gt = truth(sid)
    r = agent().run(fixture(sid)).updates["settlement"]
    assert r.status == gt["status"] and r.missing_inputs == gt["missing_inputs"]
    assert r.payable == (Decimal(gt["payable"]) if gt["payable"] else None)
    assert verify(r.explanation, r) == []


def test_the_engine_inputs_are_the_labelled_ones():
    inputs, parts, _ = agent().extract(fixture("S-02"))
    gt = truth("S-02")["inputs"]
    assert {k: str(v) for k, v in inputs.model_dump().items() if k in gt} == gt
    assert inputs.input_citations["assessed_covered_loss"] == ["F-S02-06"] and inputs.input_citations["water_damage_sublimit"] == ["CL-SUB-WATER-E1"]
    assert parts["non_covered_items_total"].how == "rules"


def test_an_estimate_is_not_an_assessment_so_the_input_is_missing_not_guessed():
    r = agent().run(fixture("S-01")).updates["settlement"]
    assert r.status == "cannot_calculate" and r.payable is None and r.reserve_forecast is None and r.missing_inputs == ["assessed_covered_loss", "non_covered_items_total"]


def test_a_missing_sublimit_wording_is_unavailable_never_zero_and_gives_a_forecast_not_a_payable():
    r = agent().run(fixture("S-05")).updates["settlement"]
    assert r.missing_inputs == ["water_damage_sublimit"] and r.payable is None and r.reserve_forecast == Decimal(100000) and r.assessed_loss == Decimal(110000)
    assert "Forecast only" in r.explanation and "cannot_calculate_final_settlement_yet" in r.explanation and "Payable:" not in r.explanation


def test_the_agent_owns_settlement_only_and_is_pure():
    a = agent()
    assert (a.agent_id, set(a.owns)) == ("settlement", {"settlement"})
    s = fixture("S-02")
    first, second = a.run(s), a.run(s)
    assert first.updates == second.updates and set(first.updates) == {"settlement"}
    assert apply_patch(s, first).settlement.payable == Decimal(80000)
    with pytest.raises(OwnershipViolation):
        apply_patch(s, StatePatch(agent_id="settlement", updates={"facts": []}))


# ---------------------------------------------------------------------------------------------------------------- the rules
def test_rules_read_the_proposition_so_an_evidence_line_with_two_amounts_is_not_confused():
    a, n = rule_amounts([fact("a", "Surveyor assessed the repair cost at INR 98,000", "Assessed repair cost: INR 98,000, of which INR 8,000 is an upgrade not caused by the loss"),
                         fact("b", "Estimate line 'Cabinet upgrade' is INR 8,000", "Cabinet upgrade: INR 8,000")])
    assert (a.value, a.cites, n.value, n.cites) == (Decimal(98000), ["a"], Decimal(8000), ["b"])


def test_two_different_assessed_amounts_make_the_input_unavailable():
    inputs, parts, _ = agent().extract(state_with(fact("a", "Surveyor assessed the repair cost at INR 90,000"), fact("b", "Loss adjuster assessed the repair cost at INR 110,000")))
    assert inputs.assessed_covered_loss is None and "assessed_covered_loss" in inputs.unavailable_inputs and "differ" in parts["assessed_covered_loss"].note


def test_a_contradiction_about_the_amount_makes_the_assessed_loss_unavailable():
    c = Contradiction(contradiction_id="X", question_id="Q-AMOUNT", fact_ids_a=["a"], fact_ids_b=["b"], kind="amount", severity="high", explanation="amounts differ")
    s = state_with(fact("a", "Surveyor assessed the repair cost at INR 90,000"), contradictions=[c])
    inputs, _, _ = agent().extract(s)
    assert inputs.assessed_covered_loss is None and "assessed_covered_loss" in inputs.unavailable_inputs


def test_no_stated_non_covered_item_means_zero_only_when_an_assessment_exists_and_says_so():
    inputs, parts, _ = agent().extract(state_with(fact("a", "Surveyor assessed the repair cost at INR 90,000")))
    assert inputs.non_covered_items_total == Decimal(0) and parts["non_covered_items_total"].how == "assumed"
    inputs, _, _ = agent().extract(state_with(fact("a", "Contractor estimate total is INR 90,000")))
    assert inputs.non_covered_items_total is None


def test_the_schedule_and_payments_come_from_the_policy_and_claims_systems():
    inputs, parts, _ = agent().extract(state_with(fact("a", "Surveyor assessed the repair cost at INR 90,000")))
    assert (inputs.applicable_deductible, inputs.remaining_policy_limit, inputs.prior_payments_total) == (Decimal(10000), Decimal(2000000), Decimal(0))
    assert parts["applicable_deductible"].how == "schedule" and parts["prior_payments_total"].how == "payments"
    none = SettlementAgent(CALC, SETTLEMENT_RULES, None, None)
    r = none.run(state_with(fact("a", "Surveyor assessed the repair cost at INR 90,000"))).updates["settlement"]
    assert r.status == "cannot_calculate" and {"applicable_deductible", "remaining_policy_limit", "prior_payments_total"} <= set(r.missing_inputs)


def test_the_limit_is_reduced_by_payments_on_other_claims_not_this_one():
    class Ledger:
        def paid_on_claim(self, claim_id):
            return Decimal(30000)

        def paid_against_policy(self, policy_no):
            return Decimal(50000)

    a = SettlementAgent(CALC, SETTLEMENT_RULES, MockPasAdapter(), Ledger())
    inputs, _, _ = a.extract(state_with(fact("a", "Surveyor assessed the repair cost at INR 90,000")))
    assert inputs.remaining_policy_limit == Decimal(2000000 - 20000) and inputs.prior_payments_total == Decimal(30000)


def test_the_sublimit_is_read_from_the_clause_and_is_absent_when_the_policy_has_none():
    s = fixture("S-02")
    assert sublimit_from_snapshot(s.policy_snapshot, s.open_questions).value == Decimal(100000)
    bare = s.policy_snapshot.model_copy(update={"clauses": [c for c in s.policy_snapshot.clauses if c.clause_type != "sublimit"]})
    e = sublimit_from_snapshot(bare, [q for q in s.open_questions if q.question_id != "Q-SUBLIMIT"])
    assert e.value is None and not e.unavailable


# ---------------------------------------------------------------------------------------------------------------- the model's reading of the amounts
class FakeLLM:
    def __init__(self, answer=None, fail=None, text=None):
        self.answer, self.fail, self.text = answer, fail, text

    def structured(self, cls, prompt, **kw):
        if self.fail:
            raise self.fail
        if cls.__name__ == "Rewrite":
            return cls(text=self.text)
        return self.answer


def reader(ans, fail=None):
    return LLMAmountReader(FakeLLM(ans, fail), SETTINGS.pack)


FACTS = [fact("a", "The surveyor values the repair at INR 64,500", "Valuation: INR 64,500"), fact("b", "Repaint of an unaffected room, INR 4,500, is outside the loss", "Repaint: INR 4,500")]


def test_a_grounded_model_reading_is_accepted_and_found_by_the_model_only_is_flagged():
    ans = AmountsAnswer(assessed=AmountItem(amount="64,500", fact_id="a", quote="Valuation: INR 64,500"), non_covered=[AmountItem(amount="4,500", fact_id="b", quote="INR 4,500")])
    a, n, refused = reader(ans).read(FACTS)
    assert (a.value, n.value, refused) == (Decimal(64500), Decimal(4500), None)
    inputs, parts, _ = make_settlement_agent(SETTINGS.pack, CALC, SETTLEMENT_RULES, MockPasAdapter(), MockPaymentsLedger(), FakeLLM(ans)).extract(state_with(*FACTS))
    assert inputs.assessed_covered_loss == Decimal(64500) and parts["assessed_covered_loss"].how == "model" and parts["assessed_covered_loss"].review


@pytest.mark.parametrize("item", [AmountItem(amount="99,000", fact_id="a", quote="Valuation: INR 64,500"),         # an amount the fact does not state
                                  AmountItem(amount="64,500", fact_id="zzz", quote="INR 64,500"),                     # a fact that was not offered
                                  AmountItem(amount="64,500", fact_id="a", quote="the surveyor said so"),              # a quote that is not in the fact
                                  AmountItem(amount="sixty", fact_id="a", quote="Valuation")])
def test_an_ungrounded_model_amount_is_refused_and_leaves_the_input_missing(item):
    a, n, refused = reader(AmountsAnswer(assessed=item)).read(FACTS)
    assert a.value is None and refused


def test_the_model_being_down_leaves_the_rules_untouched():
    a = make_settlement_agent(SETTINGS.pack, CALC, SETTLEMENT_RULES, MockPasAdapter(), MockPaymentsLedger(), FakeLLM(fail=RuntimeError("down")))
    inputs, _, notes = a.extract(state_with(fact("x", "Surveyor assessed the repair cost at INR 90,000")))
    assert inputs.assessed_covered_loss == Decimal(90000) and any("down" in n for n in notes)


def test_rules_and_the_model_must_agree_a_disagreement_is_unavailable_not_a_guess():
    r, m = Extracted(Decimal(90000), ["a"], "rules"), Extracted(Decimal(95000), ["a"], "model")
    c = combine(r, m, "assessed loss")
    assert c.value is None and c.unavailable and "90,000" in c.note and "95,000" in c.note
    assert combine(r, Extracted(Decimal(90000), ["a"], "model"), "x").how == "rules+model"
    assert combine(Extracted(), m, "x").review and combine(r, Extracted(), "x") is r


# ---------------------------------------------------------------------------------------------------------------- the explanation is re-verified
def calculated() -> SettlementResult:
    return agent().run(fixture("S-02")).updates["settlement"]


def test_a_text_with_the_wrong_payable_fails_verification_so_the_build_fails():
    r = calculated()
    assert verify(r.explanation, r) == []
    tampered = r.explanation.replace("Payable: INR 80,000", "Payable: INR 90,000")
    assert any("differs from the engine's" in p for p in verify(tampered, r))
    assert any("does not state the payable" in p for p in verify("The settlement was done.", r))
    assert any("not in the engine output" in p for p in verify(r.explanation + " An extra INR 12,345 was added.", r))


def test_a_cannot_calculate_text_may_not_show_a_payable():
    r = agent().run(fixture("S-01")).updates["settlement"]
    assert verify(r.explanation, r) == []
    assert any("payable is shown" in p for p in verify("cannot_calculate: Payable: INR 5,000", r))
    assert any("missing inputs are not listed" in p for p in verify("cannot_calculate_final_settlement_yet", r))


def test_a_model_reworded_explanation_is_used_only_if_it_verifies():
    good = "The payable is Payable: INR 80,000 after the deductible. " + calculated().explanation
    a = agent(rewriter=ExplanationRewriter(FakeLLM(text=good), SETTINGS.pack))
    assert a.run(fixture("S-02")).updates["settlement"].explanation == good
    bad = agent(rewriter=ExplanationRewriter(FakeLLM(text="Payable: INR 99,999."), SETTINGS.pack))
    p = bad.run(fixture("S-02"))
    assert p.updates["settlement"].explanation == calculated().explanation and "dropped" in p.rationale


def test_a_broken_template_raises_instead_of_showing_a_wrong_total(monkeypatch):
    import claimtower.agents.settlement as mod
    monkeypatch.setattr(mod, "explain", lambda *a, **k: "Payable: INR 1.")
    with pytest.raises(SettlementExplanationError):
        agent().run(fixture("S-02"))


def test_explain_shows_no_decimals_for_whole_rupees_and_every_amount_comes_from_the_trace():
    r = SettlementResult(status="calculated", assessed_loss=Decimal(100), payable=Decimal(90), rule_version="v",
                         steps=[SettlementStepTrace(rule_id="start", input_ref="assessed_covered_loss", amount_before=Decimal(0), amount_after=Decimal(100)),
                                SettlementStepTrace(rule_id="less_deductible", input_ref="applicable_deductible", amount_before=Decimal(100), amount_after=Decimal(90))])
    t = explain(r, {"start": "Loss"}, None, {"applicable_deductible": ["sched"]})
    assert "INR 90" in t and "INR 90.00" not in t and "[sched]" in t and verify(t, r) == []


# ---------------------------------------------------------------------------------------------------------------- registered in the graph
def test_the_default_registry_runs_the_real_settlement_agent():
    from claimtower.composition import build_registry
    reg = build_registry(SETTINGS)
    assert "settlement" in reg.implemented and reg["settlement"].agent.agent_id == "settlement"


def test_a_low_severity_amount_difference_is_noted_but_does_not_block_the_calculation():
    low = Contradiction(contradiction_id="X", question_id="Q-AMOUNT", fact_ids_a=["a"], fact_ids_b=["b"], kind="amount", severity="low", explanation="the estimate is 10% above the assessment")
    s = state_with(fact("a", "Surveyor assessed the repair cost at INR 90,000"), contradictions=[low])
    inputs, _, _ = agent().extract(s)
    assert inputs.assessed_covered_loss == Decimal(90000) and "assessed_covered_loss" not in inputs.unavailable_inputs
