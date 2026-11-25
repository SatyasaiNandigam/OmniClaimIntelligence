"""T4.1 - open-question register: template questions are the floor, model questions are checked additions."""
import json
from pathlib import Path

import pytest

from claimtower.core.config import ModelQuestionsConfig, QuestionTemplates, load_settings
from claimtower.core.schemas import ClaimHeader, Fact, NotInForce, PolicySnapshot
from claimtower.evidence.llm_questions import (
    DuplicateVerdict, LLMQuestionProposer, ProposalAnswer, ProposedItem, make_register, question_id,
)
from claimtower.evidence.questions import ProposalContext, QuestionRegister, overlap

ROOT = Path(__file__).resolve().parents[2]
PACK = load_settings().pack
TEMPLATES = PACK.question_templates
HEADER = ClaimHeader(claim_id="CLM-1", loss_date="2026-03-14", loss_type="water_damage",
                     loss_description="Pipe leak in the kitchen; the house had been empty for five weeks.")


def snapshot(scenario: str = "S-01") -> PolicySnapshot:
    return PolicySnapshot(**json.loads((ROOT / f"data/fixtures/{scenario}/policy.json").read_text(encoding="utf-8"))["policy_snapshot"])


def fact(fid: str, proposition: str, evidence: str | None = None) -> Fact:
    return Fact(fact_id=fid, claim_id="CLM-1", proposition=proposition, source_document_id="DOC-1", page=1,
                evidence_text=evidence or proposition, evidence_type="neutral", extraction_confidence=0.9)


FACTS = [fact("F-1", "Claimant reports: the house had been empty for five weeks", "the house had been empty for five weeks"),
         fact("F-2", "Plumber reports: corrosion at the joint")]


class FakeLLM:
    """Answers the proposal call with `items`, and the duplicate-check call from `verdicts` (phrase of the proposed question -> verdict)."""

    def __init__(self, items=None, fail=None, verdicts=None, judge_fail=None):
        self.items, self.fail, self.verdicts, self.judge_fail, self.prompts = items or [], fail, verdicts or {}, judge_fail, []

    def structured(self, model_cls, prompt, **kw):
        self.prompts.append(prompt.text)
        if model_cls is DuplicateVerdict:
            if self.judge_fail:
                raise self.judge_fail
            proposed = prompt.text.split("Proposed question:")[1]
            for key, v in self.verdicts.items():
                if key in proposed:
                    return DuplicateVerdict(**v)
            return DuplicateVerdict(verdict="new")
        if self.fail:
            raise self.fail
        return ProposalAnswer(questions=self.items)


def good(**over) -> ProposedItem:
    base = dict(text="Was the building unoccupied for more than 30 days before the loss?", category="coverage_condition",
                clause_ids=["CL-COND-UNOCC-E4"], fact_ids=["F-1"], evidence_quote="empty for five weeks",
                establishing_criteria=["Occupancy history for the 30 days before the loss"], confidence=0.8)
    base.update(over)
    return ProposedItem(**base)


def snap_with_unocc() -> PolicySnapshot:
    s = snapshot("S-01")
    extra = {"clause_id": "CL-COND-UNOCC-E4", "clause_type": "condition", "heading": "1. Unoccupied buildings", "document": "END-04",
             "text": "We do not cover escape of water if the Buildings have been unoccupied for more than 30 consecutive days before the loss."}
    return s.model_copy(update={"clauses": [*s.clauses, type(s.clauses[0])(**extra)]})


def ctx(snap=None, facts=FACTS):
    snap = snap or snap_with_unocc()
    reg = QuestionRegister(TEMPLATES)
    return ProposalContext(HEADER, facts, snap.clauses, reg.build(HEADER, snap).questions)


def proposer(items=None, fail=None, verdicts=None, judge_fail=None, **cfg):
    return LLMQuestionProposer(FakeLLM(items, fail, verdicts, judge_fail), PACK, TEMPLATES.model_questions.model_copy(update=cfg))


# ------------------------------------------------------------------------------------------------ template questions
def test_templates_give_the_six_questions_with_clause_links():
    r = QuestionRegister(TEMPLATES).build(HEADER, snapshot("S-01"))
    assert [q.question_id for q in r.questions] == ["Q-INFORCE", "Q-SUDDEN", "Q-START", "Q-EXTENT", "Q-AMOUNT", "Q-SUBLIMIT"]
    assert all(q.status == "unverified" and r.origin[q.question_id] == "template" for q in r.questions)
    assert r.get("Q-INFORCE").linked_clause_ids == ["CL-COND-PERIOD"]
    assert r.get("Q-SUDDEN").linked_clause_ids == ["CL-COV-WATER", "CL-DEF-SUDDEN", "CL-EXC-GRADUAL"]   # cover, definition, exclusion
    assert r.get("Q-START").linked_clause_ids == ["CL-EXC-GRADUAL"]
    assert r.get("Q-SUBLIMIT").linked_clause_ids == ["CL-SUB-WATER-E1"]
    assert r.get("Q-EXTENT").linked_clause_ids == [] and r.get("Q-AMOUNT").linked_clause_ids == []
    assert r.get("Q-SUDDEN").establishing_criteria                  # "what would establish this" travels with the question


def test_sublimit_question_falls_back_to_the_cover_when_the_endorsement_is_missing():
    r = QuestionRegister(TEMPLATES).build(HEADER, snapshot("S-05"))           # END-01 listed but not available
    assert r.get("Q-SUBLIMIT").linked_clause_ids == ["CL-COV-WATER"]


@pytest.mark.parametrize("snap", [None, NotInForce(policy_id="POL-1", loss_date="2026-03-14", reason="lapsed")])
def test_without_a_policy_only_the_in_force_question_is_asked_and_the_gap_is_stated(snap):
    r = QuestionRegister(TEMPLATES).build(HEADER, snap)
    assert [q.question_id for q in r.questions] == ["Q-INFORCE"] and r.get("Q-INFORCE").linked_clause_ids == []
    assert any("not asked" in w and "Q-SUDDEN" in w for w in r.warnings)


def test_clauses_typed_by_a_model_are_flagged_for_review():
    r = QuestionRegister(TEMPLATES).build(HEADER, snapshot("S-01"), review_clause_ids={"CL-EXC-GRADUAL"})
    assert r.review_links == {"Q-SUDDEN": ["CL-EXC-GRADUAL"], "Q-START": ["CL-EXC-GRADUAL"]}
    assert any(w.startswith("REVIEW") for w in r.warnings)


def test_no_proposer_means_no_model_questions_even_when_enabled():
    assert QuestionRegister(TEMPLATES).proposer is None
    assert make_register(PACK, None).proposer is None


def test_config_refuses_a_proposal_threshold_nothing_could_reach():
    with pytest.raises(ValueError, match="min_confidence"):
        ModelQuestionsConfig(min_confidence=0.9, confidence_cap=0.5)
    with pytest.raises(ValueError, match="pattern"):
        ModelQuestionsConfig(forbidden_patterns=["("])


def test_pack_switches_model_questions_on():
    assert TEMPLATES.model_questions.enabled is True


# ------------------------------------------------------------------------------------------------ checks on model proposals
def test_a_grounded_new_question_is_accepted_with_capped_confidence_and_a_stable_id():
    p = proposer([good(confidence=0.99)])
    out = p.check(ProposalAnswer(questions=[good(confidence=0.99)]), ctx())
    assert out.rejected == [] and len(out.accepted) == 1
    q = out.accepted[0]
    assert q.confidence == TEMPLATES.model_questions.confidence_cap and q.question_id == question_id(q.text)
    assert q.question_id.startswith("Q-M-") and q.clause_ids == ("CL-COND-UNOCC-E4",) and q.fact_ids == ("F-1",)


@pytest.mark.parametrize("over,why", [
    (dict(clause_ids=["CL-NOPE"]), "do not exist"),
    (dict(fact_ids=["F-99"]), "do not exist"),
    (dict(clause_ids=[], fact_ids=[]), "cites no clause"),
    (dict(evidence_quote="vacant for a month"), "verbatim"),
    (dict(evidence_quote="a"), "verbatim"),
    (dict(category="vibes"), "allowed category"),
    (dict(establishing_criteria=[]), "criteria"),
    (dict(establishing_criteria=["x"] * 9), "criteria"),
    (dict(confidence=0.3), "below"),
    (dict(text="Should the claim be rejected because the house was empty?"), "pre-judges"),
    (dict(text="Will the claim be approved?"), "pre-judges"),
    (dict(text="Hm?"), "too short"),
    (dict(text="Please send the plumber's invoice."), "not phrased as a question"),
    (dict(text="x" * 300 + "?"), "longer"),
    (dict(text="Was the water damage sudden and accidental rather than gradual?"), "repeats"),
])
def test_unsupported_proposals_are_refused_with_the_reason(over, why):
    out = proposer().check(ProposalAnswer(questions=[good(**over)]), ctx())
    assert out.accepted == [] and why in out.rejected[0].reason


def test_quote_may_come_from_the_clause_text():
    out = proposer().check(ProposalAnswer(questions=[good(fact_ids=[], evidence_quote="unoccupied for more than 30 consecutive days")]), ctx())
    assert len(out.accepted) == 1


def test_near_duplicates_of_each_other_are_kept_once():
    twin = good(text="Was the building unoccupied for over 30 days before the loss?", confidence=0.7)
    out = proposer().check(ProposalAnswer(questions=[good(), twin]), ctx())
    assert len(out.accepted) == 1 and "repeats" in out.rejected[0].reason


def test_at_most_max_questions_are_kept_highest_confidence_first():
    items = [good(text=t, confidence=c) for t, c in [("Did the boiler service lapse during the policy year?", 0.7),
                                                      ("Who holds the plumber's registration certificate?", 0.9),
                                                      ("Where is the stopcock for the affected flat located?", 0.8)]]
    out = proposer(max_questions=2).check(ProposalAnswer(questions=items), ctx())
    assert [q.confidence for q in out.accepted] == [0.85, 0.8] and "over the limit" in out.rejected[0].reason


def test_overlap_counts_content_words_only():
    assert overlap("Was the policy in force?", "Was the policy in force on the date of loss?") == 1.0
    assert overlap("Was the roof inspected?", "Was the policy in force?") == 0.0


# ------------------------------------------------------------------------------------------------ register with a model
def test_register_adds_checked_model_questions_after_the_templates():
    reg = QuestionRegister(TEMPLATES, proposer([good(), good(text="Was a plumber's statement obtained?", clause_ids=["NOPE"])]))
    r = reg.build(HEADER, snap_with_unocc(), FACTS)
    ids = [q.question_id for q in r.questions]
    assert ids[:6] == ["Q-INFORCE", "Q-SUDDEN", "Q-START", "Q-EXTENT", "Q-AMOUNT", "Q-SUBLIMIT"] and len(ids) == 7
    mq = r.get(ids[6])
    assert r.origin[mq.question_id] == "model" and mq.linked_clause_ids == ["CL-COND-UNOCC-E4"] and mq.status == "unverified"
    assert mq.establishing_criteria == ["Occupancy history for the 30 days before the loss"]
    assert r.proposed[mq.question_id].evidence_quote == "empty for five weeks"
    assert len(r.rejected) == 1


def test_prompt_shows_the_claim_the_facts_the_clauses_and_the_standard_questions():
    llm = FakeLLM([])
    QuestionRegister(TEMPLATES, LLMQuestionProposer(llm, PACK)).build(HEADER, snap_with_unocc(), FACTS)
    text = llm.prompts[0]
    for needle in ["empty for five weeks", "F-1", "CL-COND-UNOCC-E4", "Q-SUDDEN", "2026-03-14", "coverage_condition"]:
        assert needle in text


def test_model_outage_leaves_the_template_questions_and_says_so():
    r = QuestionRegister(TEMPLATES, proposer(fail=RuntimeError("down"))).build(HEADER, snap_with_unocc(), FACTS)
    assert len(r.questions) == 6 and all(v == "template" for v in r.origin.values())
    assert any("unavailable" in w and "down" in w for w in r.warnings)


def test_model_is_not_asked_without_a_policy_or_when_switched_off():
    llm = FakeLLM([good()])
    on = QuestionRegister(TEMPLATES, LLMQuestionProposer(llm, PACK))
    on.build(HEADER, None, FACTS)
    on.build(HEADER, NotInForce(policy_id="P", loss_date="2026-03-14", reason="lapsed"), FACTS)
    assert llm.prompts == []
    off = QuestionTemplates(questions=TEMPLATES.questions, model_questions=TEMPLATES.model_questions.model_copy(update={"enabled": False}))
    QuestionRegister(off, LLMQuestionProposer(llm, PACK)).build(HEADER, snap_with_unocc(), FACTS)
    assert llm.prompts == []


def test_make_register_uses_the_model_when_a_client_is_given():
    assert make_register(PACK, FakeLLM()).proposer is not None


def test_the_model_cannot_remove_or_reword_a_template_question():
    r = QuestionRegister(TEMPLATES, proposer([good(text="Was the policy in force on the date of loss?")])).build(HEADER, snap_with_unocc(), FACTS)
    assert r.get("Q-INFORCE").text == TEMPLATES.questions[0].text and len(r.questions) == 6 and "repeats" in r.rejected[0].reason


# ------------------------------------------------------------------------------------------------ the second look (paraphrased repeats)
PARAPHRASE = good(text="Are there earlier repair records for the kitchen pipe?", clause_ids=["CL-EXC-GRADUAL"], evidence_quote="gradual leakage")


def test_a_repeat_in_other_words_is_refused_by_the_second_look_with_the_standard_question_named():
    p = proposer([PARAPHRASE, good()], verdicts={"earlier repair records": {"verdict": "duplicate", "duplicate_of": "Q-START", "reason": "same evidence"}})
    out = p.propose(ctx())
    assert [q.clause_ids for q in out.accepted] == [("CL-COND-UNOCC-E4",)]
    assert "Q-START" in out.rejected[0].reason and "same evidence" in out.rejected[0].reason


def test_a_duplicate_verdict_that_names_nothing_still_refuses_the_proposal():
    out = proposer([good()], verdicts={"unoccupied": {"verdict": "duplicate", "duplicate_of": "none"}}).propose(ctx())
    assert out.accepted == [] and "(unnamed)" in out.rejected[0].reason


def test_an_unreadable_verdict_refuses_the_proposal():
    out = proposer([good()], verdicts={"unoccupied": {"verdict": "maybe"}}).propose(ctx())
    assert out.accepted == [] and "no usable verdict" in out.rejected[0].reason


def test_if_the_second_look_cannot_run_the_proposal_is_not_accepted_unchecked():
    out = proposer([good()], judge_fail=RuntimeError("down")).propose(ctx())
    assert out.accepted == [] and out.unavailable and "could not be checked" in out.rejected[0].reason


def test_second_look_can_be_switched_off():
    out = proposer([PARAPHRASE], duplicate_check=False, verdicts={"earlier": {"verdict": "duplicate", "duplicate_of": "Q-START"}}).propose(ctx())
    assert len(out.accepted) == 1


def test_second_look_shows_the_standard_questions_with_their_criteria():
    p = proposer([good()])
    p.propose(ctx())
    judge_prompt = p._llm.prompts[-1]
    assert "Q-START" in judge_prompt and "Technical evidence of when" in judge_prompt and "unoccupied for more than 30 days" in judge_prompt
