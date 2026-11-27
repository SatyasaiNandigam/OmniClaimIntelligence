"""T4.3 - contradiction detector: rules for dates and amounts, a checked model for narrative conflicts, never a conflict without two cited sides."""
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from claimtower.core.config import ContradictionConfig, load_settings
from claimtower.core.schemas import ClaimHeader, Contradiction, Fact, PolicySnapshot
from claimtower.evidence.contradiction import ContradictionDetector, contradiction_id
from claimtower.evidence.linker import FactQuestionLinker
from claimtower.evidence.llm_contradiction import ConflictAnswer, ConflictItem, LLMConflictJudge, make_detector
from claimtower.evidence.questions import QuestionRegister
from claimtower.evidence.values import parse_amount, parse_date

ROOT = Path(__file__).resolve().parents[2]
PACK = load_settings().pack
TEMPLATES = PACK.question_templates
CFG = TEMPLATES.contradictions
SNAP = PolicySnapshot(**json.loads((ROOT / "data/fixtures/S-01/policy.json").read_text(encoding="utf-8"))["policy_snapshot"])
QUESTIONS = QuestionRegister(TEMPLATES).build(ClaimHeader(claim_id="C"), SNAP).questions


def fact(fid, text, tags=(), **kw):
    return Fact(fact_id=fid, claim_id="C", proposition=text, source_document_id="D1", page=1, evidence_text=text, evidence_type="neutral",
                related_question_ids=list(tags), extraction_confidence=0.9, **kw)


def rules_only():
    return ContradictionDetector(CFG)


# ------------------------------------------------------------------------------------------------ reading values
@pytest.mark.parametrize("text,want", [("14 March 2026", date(2026, 3, 14)), ("14 Mar 2026", date(2026, 3, 14)), ("March 14, 2026", date(2026, 3, 14)),
                                       ("2026-03-14", date(2026, 3, 14)), ("14/03/2026", date(2026, 3, 14)), ("1st April 2026", date(2026, 4, 1)),
                                       ("31 February 2026", None), ("sometime in March", None)])
def test_dates_are_read_day_first_and_a_bad_date_is_none(text, want):
    assert parse_date(text) == want


@pytest.mark.parametrize("text,want", [("INR 180,000", Decimal(180000)), ("Rs. 1,80,000", Decimal(180000)), ("₹98,000.50", Decimal("98000.50")),
                                       ("1.5 lakh", Decimal(150000)), ("INR 2 crore", Decimal(20000000)), ("about the same", None)])
def test_amounts_are_read_with_indian_grouping_and_units(text, want):
    assert parse_amount(text) == want


# ------------------------------------------------------------------------------------------------ rules
def test_two_different_loss_dates_are_a_high_date_contradiction_naming_both_facts():
    facts = [fact("F1", "Reported date of loss is 14 March 2026", ["Q-INFORCE"]), fact("F2", "Surveyor reports: date of loss was 16 March 2026")]
    r = rules_only().detect(facts, QUESTIONS)
    (c,) = r.contradictions
    assert (c.kind, c.severity, c.question_id) == ("date", "high", "Q-INFORCE") and c.fact_ids_a == ["F1"] and c.fact_ids_b == ["F2"]
    assert r.details[c.contradiction_id].source == "rule" and r.details[c.contradiction_id].rule_id == "loss_date"
    assert "2026-03-14" in c.explanation and "2026-03-16" in c.explanation


def test_the_same_date_written_two_ways_is_not_a_contradiction():
    facts = [fact("F1", "Reported date of loss is 14 March 2026"), fact("F2", "Surveyor reports: date of loss 14/03/2026")]
    assert rules_only().detect(facts, QUESTIONS).contradictions == []


@pytest.mark.parametrize("invoice,expect", [("1 March 2026", True), ("14 March 2026", False), ("20 March 2026", False), ("not a date", False)])
def test_an_invoice_dated_before_the_loss_is_a_contradiction(invoice, expect):
    facts = [fact("F1", "Reported date of loss is 14 March 2026"), fact("F2", f"Invoice is dated {invoice}")]
    got = rules_only().detect(facts, QUESTIONS).contradictions
    assert bool(got) is expect
    if expect:
        assert (got[0].kind, got[0].question_id, got[0].severity) == ("date", "Q-AMOUNT", "medium")


def test_a_surveyor_visit_or_plumber_inspection_before_the_loss_is_flagged_on_its_own_question():
    facts = [fact("F1", "Reported date of loss is 14 March 2026"), fact("F2", "Surveyor visited the site on 2 March 2026"),
             fact("F3", "Plumber inspected the property on 10 March 2026")]
    got = {c.question_id for c in rules_only().detect(facts, QUESTIONS).contradictions}
    assert got == {"Q-EXTENT", "Q-START"}


@pytest.mark.parametrize("claimed,assessed,severity", [("INR 180,000", "INR 98,000", "medium"), ("INR 100,000", "INR 105,000", None),
                                                       ("INR 100,000", "INR 90,000", "low"), ("INR 100,000", "INR 50,000", "high"),
                                                       ("Rs. 1,80,000", "INR 180,000", None), ("1.5 lakh", "INR 100,000", "medium")])
def test_an_amount_gap_gets_its_severity_from_the_bands(claimed, assessed, severity):
    facts = [fact("F1", f"Claimant estimates the loss at {claimed}"), fact("F2", f"Surveyor assessed the repair cost at {assessed}")]
    got = rules_only().detect(facts, QUESTIONS).contradictions
    if severity is None:
        assert got == []
    else:
        assert (got[0].kind, got[0].severity, got[0].question_id) == ("amount", severity, "Q-AMOUNT") and "differ by" in got[0].explanation


def test_an_invoice_and_an_estimate_are_different_jobs_so_a_gap_is_noted_never_blocking():
    facts = [fact("F1", "Invoice total is INR 120,000"), fact("F2", "Estimate total is INR 100,000")]
    (c,) = rules_only().detect(facts, QUESTIONS).contradictions
    assert c.severity == "low"                         # 17% gap: noted only (real documents: the plumber's invoice and the repair estimate are not the same job)


def test_unreadable_values_and_rejected_facts_are_never_compared():
    facts = [fact("F1", "Claimant estimates the loss at a fair amount"), fact("F2", "Surveyor assessed the repair cost at INR 98,000"),
             fact("F3", "Claimant estimates the loss at INR 500,000", verification_status="rejected")]
    assert rules_only().detect(facts, QUESTIONS).contradictions == []


def test_a_rule_whose_question_is_not_in_the_register_is_skipped():
    only_inforce = [q for q in QUESTIONS if q.question_id == "Q-INFORCE"]
    facts = [fact("F1", "Claimant estimates the loss at INR 180,000"), fact("F2", "Surveyor assessed the repair cost at INR 98,000")]
    assert rules_only().detect(facts, only_inforce).contradictions == []


def test_contradiction_ids_are_stable_and_order_free():
    assert contradiction_id("Q", "date", ["a"], ["b"]) == contradiction_id("Q", "date", ["b"], ["a"]) != contradiction_id("Q", "amount", ["a"], ["b"])


def test_the_contract_itself_refuses_a_side_without_a_fact():
    with pytest.raises(ValueError):
        Contradiction(contradiction_id="C-1", question_id="Q-SUDDEN", fact_ids_a=["F1"], fact_ids_b=[], severity="high", explanation="x")


def test_rule_config_is_validated():
    base = CFG.model_dump()
    with pytest.raises(ValueError, match="value"):
        ContradictionConfig(**{**base, "date_rules": [{"id": "x", "question": "Q-INFORCE", "severity": "high", "sources": ["no group"]}]})
    with pytest.raises(ValueError, match="increasing"):
        ContradictionConfig(**{**base, "amount_rules": [{"id": "y", "question": "Q-AMOUNT", "a": ["(?P<value>a)"], "b": ["(?P<value>b)"],
                                                          "bands": [{"min_gap_pct": 20, "severity": "low"}, {"min_gap_pct": 10, "severity": "high"}]}]})
    with pytest.raises(ValueError, match="min_confidence"):
        ContradictionConfig(min_confidence=0.9, confidence_cap=0.5)


def test_pack_switches_the_model_part_on():
    assert CFG.enabled is True and make_detector(PACK, FakeLLM()).judge is not None and make_detector(PACK, None).judge is None


# ------------------------------------------------------------------------------------------------ model: narrative conflicts
S1 = fact("F1", "Claimant reports the kitchen supply pipe burst suddenly", ["Q-SUDDEN"])
S2 = fact("F2", "Plumber reports: significant corrosion was present at the joint of the kitchen supply pipe", ["Q-SUDDEN", "Q-START"])
S3 = fact("F3", "Surveyor reports: moisture was elevated in the kitchen, hallway and adjoining bedroom wall", ["Q-SUDDEN", "Q-EXTENT"])
FACTS = [S1, S2, S3]


def item(**over):
    base = dict(fact_ids_a=["F1"], fact_ids_b=["F2", "F3"], kind="cause", severity="high", quote_a="burst suddenly", quote_b="significant corrosion",
                explanation="A sudden burst cannot also be long-term corrosion.", confidence=0.9)
    base.update(over)
    return ConflictItem(**base)


class FakeLLM:
    def __init__(self, by_question=None, fail=None):
        self.by_question, self.fail, self.prompts = by_question or {}, fail, []

    def structured(self, model_cls, prompt, **kw):
        self.prompts.append(prompt.text)
        if self.fail:
            raise self.fail
        for qid, items in self.by_question.items():
            if f"{qid}: " in prompt.text:
                return ConflictAnswer(conflicts=items)
        return ConflictAnswer()


def detector(by_question=None, fail=None, **cfg):
    c = CFG.model_copy(update=cfg)
    return ContradictionDetector(c, LLMConflictJudge(FakeLLM(by_question, fail), PACK, c))


def test_a_grounded_narrative_conflict_is_returned_with_both_sides_and_capped_confidence():
    r = detector({"Q-SUDDEN": [item(confidence=0.99)]}).detect(FACTS, QUESTIONS)
    (c,) = r.contradictions
    assert (c.question_id, c.kind, c.severity, c.fact_ids_a, c.fact_ids_b) == ("Q-SUDDEN", "cause", "high", ["F1"], ["F2", "F3"])
    d = r.details[c.contradiction_id]
    assert d.source == "model" and d.confidence == CFG.confidence_cap and d.quote_a == "burst suddenly" and d.quote_b == "significant corrosion"


@pytest.mark.parametrize("over,why", [
    (dict(fact_ids_b=[]), "each side"), (dict(fact_ids_a=[]), "each side"),
    (dict(fact_ids_b=["F1", "F2"]), "both sides"),
    (dict(fact_ids_b=["F9"]), "not offered"),
    (dict(kind="vibes"), "allowed kind"), (dict(severity="none"), "allowed severity"), (dict(severity="critical"), "allowed severity"),
    (dict(quote_a="burst slowly"), "verbatim"), (dict(quote_b=""), "verbatim"),
    (dict(quote_a="significant corrosion"), "verbatim"),              # a quote must come from a fact on ITS side
    (dict(explanation=""), "explanation"), (dict(explanation="x" * 500), "explanation"),
    (dict(confidence=0.3), "below"),
])
def test_unsupported_conflicts_are_refused_with_the_reason(over, why):
    r = detector({"Q-SUDDEN": [item(**over)]}).detect(FACTS, QUESTIONS)
    assert r.contradictions == [] and why in r.refused[0].reason


def test_a_conflict_is_kept_once_and_at_most_max_per_question():
    twin = item(fact_ids_a=["F2", "F3"], fact_ids_b=["F1"], quote_a="significant corrosion", quote_b="burst suddenly", confidence=0.7)
    r = detector({"Q-SUDDEN": [item(), twin]}).detect(FACTS, QUESTIONS)
    assert len(r.contradictions) == 1 and "repeats" in r.refused[0].reason
    many = [item(fact_ids_b=[f], quote_b=q, confidence=c) for f, q, c in [("F2", "significant corrosion", 0.9), ("F3", "moisture was elevated", 0.8)]]
    r = detector({"Q-SUDDEN": many}, max_per_question=1).detect(FACTS, QUESTIONS)
    assert len(r.contradictions) == 1 and "over the limit" in r.refused[0].reason


def test_only_narrative_questions_with_two_linked_facts_go_to_the_model():
    llm = FakeLLM()
    c = CFG
    ContradictionDetector(c, LLMConflictJudge(llm, PACK, c)).detect(FACTS, QUESTIONS)
    asked = [p for p in llm.prompts]
    assert len(asked) == 1 and "Q-SUDDEN" in asked[0]                 # Q-START and Q-EXTENT have one fact each; Q-AMOUNT / Q-INFORCE are not narrative
    assert "F1" in asked[0] and "F3" in asked[0] and "high:" in asked[0]


def test_model_outage_keeps_the_rule_contradictions_and_warns():
    facts = [*FACTS, fact("F4", "Claimant estimates the loss at INR 180,000"), fact("F5", "Surveyor assessed the repair cost at INR 98,000")]
    r = detector(fail=RuntimeError("down")).detect(facts, QUESTIONS)
    assert [c.kind for c in r.contradictions] == ["amount"] and any("not checked" in w and "down" in w for w in r.warnings)


def test_facts_are_linked_through_the_t42_links_when_given_else_through_their_tags():
    links = FactQuestionLinker(TEMPLATES.fact_links).link(FACTS, QUESTIONS).links
    llm = FakeLLM()
    d = ContradictionDetector(CFG, LLMConflictJudge(llm, PACK, CFG))
    d.detect(FACTS, QUESTIONS, links)
    d.detect(FACTS, QUESTIONS, [x for x in links if x.fact_id != "F3"])        # F3 no longer linked: only two facts remain, still asked
    d.detect(FACTS, QUESTIONS, [x for x in links if x.fact_id == "F1"])        # one fact: not asked
    assert len(llm.prompts) == 2


def test_the_model_is_not_asked_when_switched_off():
    llm = FakeLLM()
    off = CFG.model_copy(update={"enabled": False})
    ContradictionDetector(off, LLMConflictJudge(llm, PACK, off)).detect(FACTS, QUESTIONS)
    assert llm.prompts == []


# ------------------------------------------------------------------------------------------------ earlier conflicts and retractions
def _earlier():
    from claimtower.evidence.llm_contradiction import ConflictAnswerWithResolved  # noqa: F401
    r = detector({"Q-SUDDEN": [item()]}).detect(FACTS, QUESTIONS)
    return r.contradictions


class _RecLLM:
    def __init__(self, answer):
        self.answer, self.classes, self.prompts = answer, [], []

    def structured(self, model_cls, prompt, **kw):
        self.classes.append(model_cls.__name__)
        self.prompts.append(prompt.text)
        return self.answer


NEW = fact("F9", "Plumber reports: the corrosion is surface rust only", ["Q-SUDDEN"])


def _resolve(**over):
    from claimtower.evidence.llm_contradiction import ConflictAnswerWithResolved, ResolvedItem
    cid = _earlier()[0].contradiction_id
    base = dict(contradiction_id=cid, resolving_fact_ids=["F9"], evidence_quote="surface rust only", explanation="the later report explains the corrosion away")
    base.update(over)
    return ConflictAnswerWithResolved(resolved=[ResolvedItem(**base)])


def _detect(answer, changed=("F9",), facts=(*FACTS, NEW)):
    llm = _RecLLM(answer)
    c = CFG.model_copy()
    r = ContradictionDetector(c, LLMConflictJudge(llm, PACK, c)).detect(list(facts), QUESTIONS, previous=_earlier(), changed_fact_ids=changed, only={"Q-SUDDEN"})
    return r, llm


def test_an_earlier_conflict_the_model_does_not_repeat_is_kept():
    r, llm = _detect(ConflictAnswer())
    assert [c.contradiction_id for c in r.contradictions] == [_earlier()[0].contradiction_id] and r.retained and r.retracted == []
    assert llm.classes == ["ConflictAnswerWithResolved"] and "Earlier conflicts on this question" in llm.prompts[0] and "New facts since then: F9" in llm.prompts[0]


def test_a_checked_retraction_removes_the_earlier_conflict_and_is_recorded():
    r, _ = _detect(_resolve())
    assert r.contradictions == [] and r.retracted[0].fact_ids == ("F9",) and "explains the corrosion away" in r.retracted[0].explanation


@pytest.mark.parametrize("over", [dict(contradiction_id="C-NOPE"), dict(resolving_fact_ids=["F1"]), dict(resolving_fact_ids=["F99"]),
                                  dict(evidence_quote="all is well"), dict(evidence_quote=""), dict(explanation="")])
def test_a_retraction_that_is_not_grounded_in_a_new_fact_is_ignored_and_the_conflict_stays(over):
    r, _ = _detect(_resolve(**over))
    assert len(r.contradictions) == 1 and r.retracted == []


def test_an_earlier_conflict_whose_fact_changed_is_not_carried_over_and_the_first_assessment_keeps_its_schema():
    r, llm = _detect(ConflictAnswer(), changed=("F2",))
    assert r.retained == [] and llm.classes == ["ConflictAnswer"] and "Earlier conflicts" not in llm.prompts[0]


def test_an_estimate_far_above_the_assessment_is_noted_never_blocking():
    facts = [fact("F1", "Estimate total is INR 450,000"), fact("F2", "Surveyor assessed the repair cost at INR 254,260")]
    (c,) = rules_only().detect(facts, QUESTIONS).contradictions
    assert c.severity == "low"
