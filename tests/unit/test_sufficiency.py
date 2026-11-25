"""T4.4 - sufficiency classifier: criteria -> decision table -> status and gap type (contradictory / missing / insufficient), abstaining when unclear."""
import json
from pathlib import Path

import pytest

from claimtower.core.config import SufficiencyConfig, load_settings
from claimtower.core.schemas import ClaimHeader, Contradiction, Fact, NotInForce, PolicySnapshot
from claimtower.evidence.linker import Link
from claimtower.evidence.llm_sufficiency import LLMCriterionJudge, VerdictAnswer, VerdictItem, make_classifier
from claimtower.evidence.questions import QuestionRegister
from claimtower.evidence.sufficiency import SufficiencyClassifier

ROOT = Path(__file__).resolve().parents[2]
PACK = load_settings().pack
TEMPLATES = PACK.question_templates
CFG = TEMPLATES.sufficiency


def snapshot(scenario="S-01"):
    return PolicySnapshot(**json.loads((ROOT / f"data/fixtures/{scenario}/policy.json").read_text(encoding="utf-8"))["policy_snapshot"])


SNAP = snapshot("S-01")
QUESTIONS = QuestionRegister(TEMPLATES).build(ClaimHeader(claim_id="C"), SNAP).questions


def fact(fid, text, **kw):
    return Fact(fact_id=fid, claim_id="C", proposition=text, source_document_id="D1", page=1, evidence_text=text, evidence_type="neutral",
                extraction_confidence=0.9, **kw)


P1 = fact("P1", "Plumber reports: the joint failed suddenly and showed a fresh fracture surface")
P2 = fact("P2", "Plumber reports: no corrosion or earlier leakage was found; the fitting failed on the day of loss")
C1 = fact("C1", "Claimant reports the pipe burst suddenly")


def link(f, q, d="neutral"):
    return Link(f, q, d, "x", "template")


def verdict(i, v, ids=(), quote="", conf=0.9, why="because"):
    return VerdictItem(criterion_index=i, verdict=v, fact_ids=list(ids), evidence_quote=quote, rationale=why, confidence=conf)


class FakeLLM:
    def __init__(self, by_question=None, fail=None):
        self.by_question, self.fail, self.prompts = by_question or {}, fail, []

    def structured(self, model_cls, prompt, **kw):
        self.prompts.append(prompt.text)
        if self.fail:
            raise self.fail
        for qid, items in self.by_question.items():
            if f"{qid}: " in prompt.text.split("Criteria to judge")[0]:
                return VerdictAnswer(criteria=items)
        return VerdictAnswer()


def classify(by_question=None, facts=(), links=(), contradictions=(), snap=SNAP, fail=None, judge=True, **cfg):
    c = CFG.model_copy(update=cfg)
    j = LLMCriterionJudge(FakeLLM(by_question, fail), PACK, c) if judge else None
    return SufficiencyClassifier(c, j).classify(QUESTIONS, list(facts), list(links), list(contradictions), snap)


SUDDEN = [verdict(0, "met", ["P1"], "failed suddenly"), verdict(1, "met", ["P2"], "failed on the day of loss")]


def contradiction(sev="high", qid="Q-SUDDEN"):
    return Contradiction(contradiction_id="C-1", question_id=qid, fact_ids_a=["C1"], fact_ids_b=["P2"], kind="cause", severity=sev, explanation="x")


# ------------------------------------------------------------------------------------------------ the decision table
def test_every_criterion_met_is_established_with_no_gap():
    r = classify({"Q-SUDDEN": SUDDEN}, [P1, P2], [link("P1", "Q-SUDDEN"), link("P2", "Q-SUDDEN")])
    a = r.get("Q-SUDDEN")
    assert (a.status, a.gap_type, a.row_id, a.abstained) == ("established", None, "R3-established", False)
    assert [c.verdict for c in a.criteria] == ["met", "met"] and a.criteria[0].fact_ids == ("P1",) and a.criteria[0].quote == "failed suddenly"
    assert a.unresolved() == []


def test_a_blocking_contradiction_beats_everything_else_even_with_every_criterion_met():
    r = classify({"Q-SUDDEN": SUDDEN}, [P1, P2, C1], [link("P1", "Q-SUDDEN"), link("P2", "Q-SUDDEN")], [contradiction("high")])
    a = r.get("Q-SUDDEN")
    assert (a.status, a.gap_type, a.row_id) == ("requires_validation", "contradictory", "R1-contradicted") and a.contradiction_ids == ("C-1",)


def test_a_low_severity_contradiction_is_noted_but_does_not_block():
    r = classify({"Q-SUDDEN": SUDDEN}, [P1, P2], [link("P1", "Q-SUDDEN"), link("P2", "Q-SUDDEN")], [contradiction("low")])
    a = r.get("Q-SUDDEN")
    assert a.status == "established" and a.noted_contradiction_ids == ("C-1",) and a.contradiction_ids == ()


def test_some_criteria_met_is_partially_established_and_insufficient():
    r = classify({"Q-SUDDEN": [verdict(0, "met", ["P1"], "failed suddenly"), verdict(1, "unmet")]}, [P1], [link("P1", "Q-SUDDEN")])
    a = r.get("Q-SUDDEN")
    assert (a.status, a.gap_type, a.row_id) == ("partially_established", "insufficient", "R4-partial")
    assert [c.index for c in a.unresolved()] == [1]


def test_partly_met_with_a_supporting_fact_is_possible_without_one_it_is_unverified():
    v = {"Q-SUDDEN": [verdict(0, "partly", ["C1"], "burst suddenly"), verdict(1, "unmet")]}
    supporting = classify(v, [C1], [link("C1", "Q-SUDDEN", "supporting")]).get("Q-SUDDEN")
    neutral = classify(v, [C1], [link("C1", "Q-SUDDEN", "neutral")]).get("Q-SUDDEN")
    assert (supporting.status, supporting.gap_type, supporting.row_id) == ("possible", "insufficient", "R5-possible")
    assert (neutral.status, neutral.gap_type, neutral.row_id) == ("unverified", "insufficient", "R6-insufficient")


def test_nothing_bearing_on_the_question_is_missing_and_the_model_is_not_asked():
    llm = FakeLLM()
    c = SufficiencyClassifier(CFG, LLMCriterionJudge(llm, PACK, CFG))
    r = c.classify(QUESTIONS, [P1], [link("P1", "Q-SUDDEN")], [], SNAP)
    a = r.get("Q-START")
    assert (a.status, a.gap_type, a.row_id) == ("missing", "missing", "R7-missing") and {x.verdict for x in a.criteria} == {"unmet"}
    assert all("Q-START" not in p.split("Criteria to judge")[0] for p in llm.prompts)       # only Q-SUDDEN had facts to judge


def test_every_criterion_unmet_after_judging_is_missing_too():
    r = classify({"Q-START": [verdict(0, "unmet"), verdict(1, "unmet")]}, [P1], [link("P1", "Q-START")])
    assert (r.get("Q-START").status, r.get("Q-START").gap_type) == ("missing", "missing")


# ------------------------------------------------------------------------------------------------ abstaining
def test_an_unclear_criterion_makes_the_question_abstain_without_guessing_a_gap_type():
    r = classify({"Q-SUDDEN": [verdict(0, "met", ["P1"], "failed suddenly"), verdict(1, "unclear")]}, [P1], [link("P1", "Q-SUDDEN")])
    a = r.get("Q-SUDDEN")
    assert (a.status, a.gap_type, a.abstained, a.row_id) == ("unverified", None, True, "R2-abstain")


@pytest.mark.parametrize("bad,why", [
    (verdict(0, "met", ["P1"], "failed suddenly", conf=0.3), "below"),
    (verdict(0, "met", [], "failed suddenly"), "cites no offered fact"),
    (verdict(0, "met", ["P9"], "failed suddenly"), "cites no offered fact"),
    (verdict(0, "met", ["P1"], "burst slowly"), "verbatim"),
    (verdict(0, "partly", ["P1"], ""), "verbatim"),
    (verdict(0, "perhaps", ["P1"], "failed suddenly"), "allowed verdict"),
    (verdict(0, "met", ["P1"], "failed suddenly", why=""), "rationale"),
    (verdict(7, "met", ["P1"], "failed suddenly"), "not asked"),
])
def test_a_refused_verdict_leaves_the_criterion_unclear_so_the_question_abstains(bad, why):
    r = classify({"Q-SUDDEN": [bad, verdict(1, "unmet")]}, [P1], [link("P1", "Q-SUDDEN")])
    assert why in r.refused[0].reason
    a = r.get("Q-SUDDEN")
    assert a.abstained and a.criteria[0].verdict == "unclear"


def test_a_criterion_the_model_skipped_is_unclear():
    a = classify({"Q-SUDDEN": [verdict(0, "met", ["P1"], "failed suddenly")]}, [P1], [link("P1", "Q-SUDDEN")]).get("Q-SUDDEN")
    assert a.criteria[1].verdict == "unclear" and "did not judge" in a.criteria[1].rationale and a.abstained


def test_model_outage_never_makes_a_question_look_established():
    r = classify(facts=[P1], links=[link("P1", "Q-SUDDEN")], fail=RuntimeError("down"))
    a = r.get("Q-SUDDEN")
    assert a.abstained and a.status == "unverified" and any("down" in w for w in r.warnings)
    assert r.get("Q-START").status == "missing"                 # no fact bears on it: still decided without the model


def test_no_judge_at_all_is_the_same_as_an_outage():
    a = classify(facts=[P1], links=[link("P1", "Q-SUDDEN")], judge=False).get("Q-SUDDEN")
    assert a.abstained and all(c.verdict == "unclear" for c in a.criteria)


def test_confidence_of_a_model_verdict_is_capped():
    a = classify({"Q-SUDDEN": [verdict(0, "met", ["P1"], "failed suddenly", conf=0.99), verdict(1, "unmet")]}, [P1], [link("P1", "Q-SUDDEN")]).get("Q-SUDDEN")
    assert a.criteria[0].confidence == CFG.confidence_cap


# ------------------------------------------------------------------------------------------------ policy criteria
@pytest.mark.parametrize("snap,verdict_,status", [(SNAP, "met", "established"), (NotInForce(policy_id="P", loss_date="2026-03-14", reason="lapsed"), "met", "established"),
                                                   (None, "unmet", "missing")])
def test_was_the_policy_in_force_is_met_whenever_the_policy_system_answered(snap, verdict_, status):
    a = classify(snap=snap).get("Q-INFORCE")
    assert (a.criteria[0].verdict, a.criteria[0].source, a.status) == (verdict_, "policy", status)


def test_the_sublimit_question_needs_sublimit_wording_in_the_snapshot():
    assert classify().get("Q-SUBLIMIT").status == "established"
    s5 = snapshot("S-05")                                     # END-01 is referenced but its wording is missing
    q5 = QuestionRegister(TEMPLATES).build(ClaimHeader(claim_id="C"), s5).questions
    a = SufficiencyClassifier(CFG).classify(q5, [], [], [], s5).get("Q-SUBLIMIT")
    assert (a.status, a.gap_type) == ("missing", "missing") and "no sublimit clause" in a.criteria[0].rationale


# ------------------------------------------------------------------------------------------------ output helpers and config
def test_results_apply_to_the_questions_and_list_the_gaps():
    r = classify({"Q-SUDDEN": SUDDEN}, [P1, P2, C1], [link("P1", "Q-SUDDEN"), link("P2", "Q-SUDDEN")], [contradiction()])
    out = {q.question_id: q for q in r.apply(QUESTIONS)}
    assert out["Q-SUDDEN"].status == "requires_validation" and out["Q-INFORCE"].status == "established" and out["Q-EXTENT"].text == QUESTIONS[3].text
    assert r.gaps()["Q-SUDDEN"] == "contradictory" and r.gaps()["Q-START"] == "missing" and "Q-INFORCE" not in r.gaps()


def test_the_pack_is_on_and_validates_its_table():
    assert CFG.enabled and make_classifier(PACK, FakeLLM()).judge is not None and make_classifier(PACK, None).judge is None
    base = CFG.model_dump()
    with pytest.raises(ValueError, match="no conditions"):
        SufficiencyConfig(**{**base, "table": base["table"][:-1]})
    with pytest.raises(ValueError, match="unknown condition"):
        SufficiencyConfig(**{**base, "table": [{"id": "x", "when": {"bogus": True}, "status": "missing"}, {"id": "y", "status": "missing"}]})
    with pytest.raises(ValueError, match="min_confidence"):
        SufficiencyConfig(**{**base, "min_confidence": 0.9, "confidence_cap": 0.5})


def test_a_criteria_rule_must_name_a_real_criterion():
    from claimtower.core.config import QuestionTemplates
    bad = CFG.model_dump()
    bad["criteria_rules"] = [{"question": "Q-INFORCE", "criterion": "not a real criterion", "met_if": "policy_answer_known"}]
    with pytest.raises(ValueError, match="no criterion"):
        QuestionTemplates(questions=TEMPLATES.questions, sufficiency=SufficiencyConfig(**bad))
