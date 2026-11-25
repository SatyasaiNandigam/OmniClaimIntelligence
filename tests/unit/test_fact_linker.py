"""T4.2 - fact <-> question linker: tagged links are the floor, model judgements set direction and are checked."""
import json
from pathlib import Path

import pytest

from claimtower.core.config import FactLinkConfig, load_settings
from claimtower.core.schemas import ClaimHeader, Fact, PolicySnapshot
from claimtower.evidence.linker import FactQuestionLinker, apply_to_questions, fact_direction
from claimtower.evidence.llm_linking import LLMLinkJudge, LinkAnswer, LinkItem, make_linker
from claimtower.evidence.questions import QuestionRegister

ROOT = Path(__file__).resolve().parents[2]
PACK = load_settings().pack
TEMPLATES = PACK.question_templates
CFG = TEMPLATES.fact_links
SNAP = PolicySnapshot(**json.loads((ROOT / "data/fixtures/S-01/policy.json").read_text(encoding="utf-8"))["policy_snapshot"])
QUESTIONS = QuestionRegister(TEMPLATES).build(ClaimHeader(claim_id="C"), SNAP).questions


def fact(fid, text, tags=(), **kw):
    return Fact(fact_id=fid, claim_id="C", proposition=text, source_document_id="D1", page=1, evidence_text=text, evidence_type="neutral",
                related_question_ids=list(tags), extraction_confidence=0.9, **kw)


F1 = fact("F1", "Claimant reports the pipe burst suddenly", ["Q-SUDDEN"])
F2 = fact("F2", "Significant corrosion was present at the joint", ["Q-SUDDEN", "Q-START"])
F3 = fact("F3", "Contractor estimate total is INR 180,000", ["Q-AMOUNT"])
FACTS = [F1, F2, F3]


def item(fid, qid, d, quote, conf=0.9, why="because the fact says so"):
    return LinkItem(fact_id=fid, question_id=qid, direction=d, evidence_quote=quote, rationale=why, confidence=conf)


class FakeLLM:
    def __init__(self, items=None, fail=None):
        self.items, self.fail, self.prompts = items or [], fail, []

    def structured(self, model_cls, prompt, **kw):
        self.prompts.append(prompt.text)
        if self.fail:
            raise self.fail
        return LinkAnswer(links=self.items)


def linker(items=None, fail=None, **cfg):
    c = CFG.model_copy(update=cfg)
    return FactQuestionLinker(c, LLMLinkJudge(FakeLLM(items, fail), PACK, c))


# ------------------------------------------------------------------------------------------------ the floor: tagged links
def test_without_a_model_every_tagged_link_is_kept_neutral_and_says_so():
    r = FactQuestionLinker(CFG).link(FACTS, QUESTIONS)
    assert r.pairs() == {("F1", "Q-SUDDEN"), ("F2", "Q-SUDDEN"), ("F2", "Q-START"), ("F3", "Q-AMOUNT")}
    assert all(x.direction == "neutral" and x.source == "template" and "not judged" in x.rationale for x in r.links)


def test_tags_naming_unknown_questions_are_ignored_and_rejected_facts_are_not_linked():
    facts = [fact("F1", "x" * 10, ["Q-NOPE", "Q-SUDDEN"]), fact("F2", "rejected fact", ["Q-SUDDEN"], verification_status="rejected")]
    r = FactQuestionLinker(CFG).link(facts, QUESTIONS)
    assert r.pairs() == {("F1", "Q-SUDDEN")}


def test_config_refuses_a_threshold_nothing_could_reach():
    with pytest.raises(ValueError, match="min_confidence"):
        FactLinkConfig(min_confidence=0.9, confidence_cap=0.5)


def test_pack_switches_fact_links_on():
    assert CFG.enabled is True and make_linker(PACK, FakeLLM()).judge is not None and make_linker(PACK, None).judge is None


# ------------------------------------------------------------------------------------------------ the model sets direction
def test_model_sets_direction_keeping_the_quote_and_capping_confidence():
    r = linker([item("F1", "Q-SUDDEN", "supporting", "burst suddenly", 0.99),
                item("F2", "Q-SUDDEN", "contradicting", "Significant corrosion"),
                item("F2", "Q-START", "contradicting", "corrosion was present")]).link(FACTS, QUESTIONS)
    by = {(x.fact_id, x.question_id): x for x in r.links}
    assert by[("F1", "Q-SUDDEN")].direction == "supporting" and by[("F1", "Q-SUDDEN")].confidence == CFG.confidence_cap
    assert by[("F1", "Q-SUDDEN")].source == "template+model" and by[("F1", "Q-SUDDEN")].quote == "burst suddenly"
    assert by[("F2", "Q-START")].direction == "contradicting"
    assert by[("F3", "Q-AMOUNT")].direction == "neutral" and by[("F3", "Q-AMOUNT")].source == "template"      # not judged: stays as tagged
    assert fact_direction(r.links, "F2") == "contradicting" and fact_direction(r.links, "F1") == "supporting"


def test_model_may_add_a_link_the_tags_missed():
    r = linker([item("F2", "Q-EXTENT", "neutral", "joint")]).link(FACTS, QUESTIONS)
    added = [x for x in r.links if x.question_id == "Q-EXTENT"]
    assert len(added) == 1 and added[0].source == "model" and ("F2", "Q-SUDDEN") in r.pairs()


def test_model_may_drop_a_wrong_tag_and_the_drop_is_recorded():
    r = linker([item("F2", "Q-START", "unrelated", "", why="a corrosion finding says nothing about when it began")]).link(FACTS, QUESTIONS)
    assert ("F2", "Q-START") not in r.pairs() and ("F2", "Q-SUDDEN") in r.pairs()
    assert r.dropped[0].question_id == "Q-START" and "nothing about when" in r.dropped[0].rationale


def test_unrelated_for_a_pair_nobody_suggested_changes_nothing():
    r = linker([item("F3", "Q-EXTENT", "unrelated", "")]).link(FACTS, QUESTIONS)
    assert r.pairs() == {("F1", "Q-SUDDEN"), ("F2", "Q-SUDDEN"), ("F2", "Q-START"), ("F3", "Q-AMOUNT")} and r.dropped == []


@pytest.mark.parametrize("bad,why", [
    (item("F9", "Q-SUDDEN", "supporting", "x" * 5), "does not exist"),
    (item("F1", "Q-NOPE", "supporting", "burst suddenly"), "does not exist"),
    (item("F1", "Q-SUDDEN", "mostly yes", "burst suddenly"), "allowed direction"),
    (item("F1", "Q-SUDDEN", "supporting", "burst slowly"), "verbatim"),
    (item("F1", "Q-SUDDEN", "supporting", ""), "verbatim"),
    (item("F1", "Q-SUDDEN", "supporting", "burst suddenly", conf=0.3), "below"),
    (item("F1", "Q-SUDDEN", "supporting", "burst suddenly", why=""), "rationale"),
    (item("F1", "Q-SUDDEN", "supporting", "burst suddenly", why="y" * 400), "rationale"),
])
def test_unsupported_judgements_are_refused_and_the_tagged_link_stays_neutral(bad, why):
    r = linker([bad]).link(FACTS, QUESTIONS)
    assert why in r.refused[0].reason
    assert {x.direction for x in r.links} == {"neutral"}


def test_a_pair_is_judged_once_the_most_confident_wins():
    r = linker([item("F1", "Q-SUDDEN", "contradicting", "burst suddenly", 0.7), item("F1", "Q-SUDDEN", "supporting", "burst suddenly", 0.9)]).link(FACTS, QUESTIONS)
    assert r.for_fact("F1")[0].direction == "supporting" and "already judged" in r.refused[0].reason


def test_a_fact_keeps_at_most_max_links_per_fact():
    items = [item("F2", q, "neutral", "joint", 0.9 - i / 100) for i, q in enumerate(["Q-SUDDEN", "Q-START", "Q-EXTENT", "Q-AMOUNT"])]
    r = linker(items, max_links_per_fact=2).link([F2], QUESTIONS)
    assert len(r.for_fact("F2")) == 2 and any("more than 2" in x.reason for x in r.refused)


# ------------------------------------------------------------------------------------------------ outage and batching
def test_model_outage_keeps_the_tags_and_warns():
    r = linker(fail=RuntimeError("down")).link(FACTS, QUESTIONS)
    assert len(r.links) == 4 and all(x.direction == "neutral" for x in r.links)
    assert any("not judged" in w and "down" in w for w in r.warnings)


def test_facts_go_to_the_model_in_batches_and_the_prompt_shows_questions_and_tags():
    llm = FakeLLM()
    c = CFG.model_copy(update={"batch_size": 2})
    FactQuestionLinker(c, LLMLinkJudge(llm, PACK, c)).link(FACTS, QUESTIONS)
    assert len(llm.prompts) == 2
    assert "Q-START" in llm.prompts[0] and "Technical evidence of when" in llm.prompts[0] and "F1" in llm.prompts[0] and "suggested questions: Q-SUDDEN" in llm.prompts[0]
    assert "F3" in llm.prompts[1] and "F1" not in llm.prompts[1]


def test_model_is_not_asked_when_there_is_nothing_to_judge_or_it_is_switched_off():
    llm = FakeLLM()
    FactQuestionLinker(CFG, LLMLinkJudge(llm, PACK, CFG)).link([], QUESTIONS)
    FactQuestionLinker(CFG, LLMLinkJudge(llm, PACK, CFG)).link(FACTS, [])
    off = CFG.model_copy(update={"enabled": False})
    FactQuestionLinker(off, LLMLinkJudge(llm, PACK, off)).link(FACTS, QUESTIONS)
    assert llm.prompts == []


def test_links_are_applied_to_the_questions_without_touching_other_fields():
    r = FactQuestionLinker(CFG).link(FACTS, QUESTIONS)
    out = {q.question_id: q for q in apply_to_questions(QUESTIONS, r.links)}
    assert out["Q-SUDDEN"].linked_fact_ids == ["F1", "F2"] and out["Q-START"].linked_fact_ids == ["F2"] and out["Q-EXTENT"].linked_fact_ids == []
    assert out["Q-SUDDEN"].linked_clause_ids == next(q for q in QUESTIONS if q.question_id == "Q-SUDDEN").linked_clause_ids
