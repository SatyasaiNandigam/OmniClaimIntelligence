"""An endorsement in force on the loss date can amend the deductible: the model picks the clause, the pick is checked, and the schedule's figure is never used silently."""
from datetime import date
from decimal import Decimal

from claimtower.agents.settlement_inputs import DeductibleAnswer, DeductibleReader, deductible_candidates
from claimtower.core.schemas import PolicySnapshot
from claimtower.core.schemas.policy import Endorsement, PolicyClause


def clause(cid, text):
    return PolicyClause(clause_id=cid, clause_type="deductible", heading=text[:40], text=text, document="E-02", page=1, span=(0, len(text)))


NEW = clause("E-02/a", "for claims under Clause 2.1 in respect of loss occurring on or after 01 September 2026, the Deductible is INR 25,000 per claim.")
OLD = clause("E-02/b", "For loss occurring before 01 September 2026 the Deductible stated in Clause 5.1 (INR 10,000) continues to apply.")
SNAP = PolicySnapshot(policy_id="P", version="v1", effective_from=date(2026, 1, 1), clauses=[NEW, OLD],
                      endorsements=[Endorsement(endorsement_id="E-02", title="t", effective_from=date(2026, 9, 1), clause_ids=["E-02/a", "E-02/b"])])


def test_only_deductible_clauses_of_endorsements_in_force_are_candidates():
    assert [c.clause_id for c in deductible_candidates(SNAP, date(2026, 10, 14))] == ["E-02/a", "E-02/b"]
    assert deductible_candidates(SNAP, date(2026, 8, 31)) == [] and deductible_candidates(None, date(2026, 10, 14)) == []


def test_a_checked_pick_gives_the_amount_with_its_clause_and_a_review_flag():
    e = DeductibleReader.check(DeductibleAnswer(amount="25,000", clause_id="E-02/a", quote="the Deductible is INR 25,000 per claim"), [NEW, OLD])
    assert e.value == Decimal(25000) and e.cites == ["E-02/a"] and e.review and not e.unavailable


def test_an_unchecked_pick_is_unavailable_never_the_schedule_figure():
    for bad in (DeductibleAnswer(amount="25,000", clause_id="E-02/zz", quote="the Deductible is INR 25,000 per claim"),
                DeductibleAnswer(amount="30,000", clause_id="E-02/a", quote="the Deductible is INR 25,000 per claim"),
                DeductibleAnswer(amount="25,000", clause_id="E-02/a", quote="words the clause never said INR 25,000"),
                DeductibleAnswer(note="no clause applies")):
        e = DeductibleReader.check(bad, [NEW, OLD])
        assert e.value is None and e.unavailable


# ---------------------------------------------------------------------------------------------------------------- through the settlement agent
def _state():
    from claimtower.core.schemas import ClaimHeader, ClaimState, Fact, OpenQuestion
    f = Fact(fact_id="a", claim_id="C", proposition="Surveyor assessed the repair cost at INR 254,260", source_document_id="D", page=1, evidence_text="x",
             evidence_type="neutral", related_question_ids=["Q-AMOUNT"], extraction_confidence=0.9)
    q = OpenQuestion(question_id="Q-AMOUNT", text="t", category="amount", status="unverified", linked_fact_ids=["a"])
    return ClaimState(claim_id="C", claim_header=ClaimHeader(claim_id="C", policy_ref="POL-1001", loss_date=date(2026, 10, 14)), facts=[f], open_questions=[q], policy_snapshot=SNAP)


def test_the_settlement_uses_the_endorsed_deductible_and_not_the_schedule_figure():
    from claimtower.agents.settlement import SettlementAgent
    from claimtower.composition import SETTLEMENT_RULES
    from claimtower.core.config import load_settings
    from claimtower.integrations.mock.claims_mock import MockPaymentsLedger
    from claimtower.integrations.mock.pas_mock import MockPasAdapter
    from claimtower.settlement.engine import SettlementCalculator
    from claimtower.settlement.rules_loader import rules_by_version

    calc = SettlementCalculator(rules_by_version(load_settings().pack.calculation))

    class Reader:
        def read(self, cands, loss_date, standard):
            return DeductibleReader.check(DeductibleAnswer(amount="25,000", clause_id="E-02/a", quote="the Deductible is INR 25,000 per claim"), cands)

    agent = SettlementAgent(calc, SETTLEMENT_RULES, MockPasAdapter(), MockPaymentsLedger(), deductible_reader=Reader())
    inputs, parts, _ = agent.extract(_state())
    assert inputs.applicable_deductible == Decimal(25000) and parts["applicable_deductible"].how == "model" and parts["applicable_deductible"].cites == ["E-02/a"]
    without = SettlementAgent(calc, SETTLEMENT_RULES, MockPasAdapter(), MockPaymentsLedger())          # no model: never the schedule's 10,000
    i2, _, _ = without.extract(_state())
    assert i2.applicable_deductible is None and "applicable_deductible" in i2.unavailable_inputs


# ---------------------------------------------------------------------------------------------------------------- the non-covered amounts
def test_an_estimate_or_invoice_line_is_never_non_covered_unless_a_fact_says_so_and_the_total_cannot_exceed_the_assessment():
    from claimtower.agents.settlement_inputs import AmountItem, AmountsAnswer, LLMAmountReader
    from claimtower.core.schemas import Fact

    def fact(fid, p):
        return Fact(fact_id=fid, claim_id="C", proposition=p, source_document_id="D", page=1, evidence_text=p, evidence_type="neutral", related_question_ids=["Q-AMOUNT"], extraction_confidence=0.9)

    facts = [fact("A", "Surveyor assessed the repair cost at INR 254,260"), fact("F-D-item-5", "Estimate line 'Supply and lay engineered wood flooring' is INR 176,700"),
             fact("U", "Upgrade to premium tiles, not caused by the loss, INR 20,000")]
    ans = AmountsAnswer(assessed=AmountItem(amount="254,260", fact_id="A", quote="Surveyor assessed the repair cost at INR 254,260"),
                        non_covered=[AmountItem(amount="176,700", fact_id="F-D-item-5", quote="is INR 176,700"), AmountItem(amount="20,000", fact_id="U", quote="INR 20,000")])
    a, n, refused = LLMAmountReader.check(ans, facts)
    assert a.value == Decimal(254260) and n.value == Decimal(20000) and "is a price" in (refused or "")
    big = AmountsAnswer(assessed=ans.assessed, non_covered=[AmountItem(amount="20,000", fact_id="U", quote="INR 20,000")] * 14)
    _, n2, refused2 = LLMAmountReader.check(big, facts)
    assert n2.value is None and "exceed the assessed loss" in (refused2 or "")


def test_an_amount_the_assessment_says_is_not_included_is_never_deducted_again():
    from claimtower.agents.settlement_inputs import AmountItem, AmountsAnswer, LLMAmountReader
    from claimtower.core.schemas import Fact

    line = "The assessed loss is INR 254,260. The cost of repairing the failed pipe section (plumber's invoice INR 35,000) is not included."
    assessed = Fact(fact_id="F-D-assessed_repair_cost", claim_id="C", proposition="Surveyor assessed the repair cost at INR 254,260", source_document_id="D", page=1, evidence_text=line,
                    evidence_type="neutral", related_question_ids=["Q-AMOUNT"], extraction_confidence=0.9)
    ans = AmountsAnswer(assessed=AmountItem(amount="254,260", fact_id=assessed.fact_id, quote="The assessed loss is INR 254,260"),
                        non_covered=[AmountItem(amount="35,000", fact_id=assessed.fact_id, quote="plumber's invoice INR 35,000) is not included")])
    a, n, refused = LLMAmountReader.check(ans, [assessed])
    assert a.value == Decimal(254260) and n.value is None and "already left out" in (refused or "")


def test_the_settlement_does_not_deduct_what_the_assessment_already_leaves_out():
    from claimtower.agents.settlement import SettlementAgent
    from claimtower.agents.settlement_inputs import AmountItem, AmountsAnswer, LLMAmountReader
    from claimtower.composition import SETTLEMENT_RULES
    from claimtower.core.config import load_settings
    from claimtower.core.schemas import ClaimHeader, ClaimState, Fact, OpenQuestion
    from claimtower.integrations.mock.claims_mock import MockPaymentsLedger
    from claimtower.integrations.mock.pas_mock import MockPasAdapter
    from claimtower.settlement.engine import SettlementCalculator
    from claimtower.settlement.rules_loader import rules_by_version

    line = "The assessed loss is INR 254,260. The cost of repairing the failed pipe section (plumber's invoice INR 35,000) is not included."
    f = Fact(fact_id="F-D-assessed_repair_cost", claim_id="C", proposition="Surveyor assessed the repair cost at INR 254,260", source_document_id="D", page=1, evidence_text=line,
             evidence_type="neutral", related_question_ids=["Q-AMOUNT"], extraction_confidence=0.9)
    q = OpenQuestion(question_id="Q-AMOUNT", text="t", category="amount", status="established", linked_fact_ids=[f.fact_id])
    state = ClaimState(claim_id="C", claim_header=ClaimHeader(claim_id="C", policy_ref="POL-1001", loss_date=date(2026, 5, 2)), facts=[f], open_questions=[q])
    wrong = AmountsAnswer(assessed=AmountItem(amount="254,260", fact_id=f.fact_id, quote="The assessed loss is INR 254,260"),
                          non_covered=[AmountItem(amount="35,000", fact_id=f.fact_id, quote="plumber's invoice INR 35,000) is not included")])

    class Reader(LLMAmountReader):
        def __init__(self):
            pass

        def read(self, facts):
            return LLMAmountReader.check(wrong, facts)

    calc = SettlementCalculator(rules_by_version(load_settings().pack.calculation))
    result = SettlementAgent(calc, SETTLEMENT_RULES, MockPasAdapter(), MockPaymentsLedger(), reader=Reader()).run(state).updates["settlement"]
    assert result.status == "calculated" and result.payable == Decimal(244260)          # 254,260 less the 10,000 deductible of POL-1001: nothing else deducted


def test_an_amount_the_assessment_says_is_not_included_is_never_deducted_from_it():
    from claimtower.agents.settlement_inputs import AmountItem, AmountsAnswer, LLMAmountReader
    from claimtower.core.schemas import Fact

    def fact(fid, p, e=None):
        return Fact(fact_id=fid, claim_id="C", proposition=p, source_document_id="D", page=1, evidence_text=e or p, evidence_type="neutral", related_question_ids=["Q-AMOUNT"], extraction_confidence=0.9)

    line = "The assessed loss for reinstatement is INR 254,260. The cost of repairing the failed pipe section (plumber's invoice INR 35,000) is not included."
    facts = [fact("A", "Surveyor assessed the repair cost at INR 254,260", line), fact("U", "Gold taps, not caused by the loss, INR 20,000")]
    ans = AmountsAnswer(assessed=AmountItem(amount="254,260", fact_id="A", quote="The assessed loss for reinstatement is INR 254,260"),
                        non_covered=[AmountItem(amount="35,000", fact_id="A", quote="plumber's invoice INR 35,000) is not included"), AmountItem(amount="20,000", fact_id="U", quote="INR 20,000")])
    a, n, refused = LLMAmountReader.check(ans, facts)
    assert a.value == Decimal(254260) and n.value == Decimal(20000)                 # the 35,000 stated by the assessment's own fact is refused; a separate fact's real deduction stays
    assert "already left out" in (refused or "") and n.cites == ["U"]
