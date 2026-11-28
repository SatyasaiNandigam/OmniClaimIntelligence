"""A question about HOW something happened can be settled on the evidence as a whole (config `purpose`), checked and never trusted; every other question keeps the decision table's answer."""
from claimtower.core.config import PurposeConfig
from claimtower.evidence.llm_sufficiency import LLMCriterionJudge, PurposeAnswer, VerdictAnswer
from claimtower.evidence.sufficiency import PurposeVerdict, SufficiencyClassifier
from tests.unit.test_sufficiency import CFG, PACK, QUESTIONS, SNAP, C1, P1, P2, FakeLLM, contradiction, link, verdict  # noqa: F401

PURPOSE = PurposeConfig(enabled=True, categories=["timing", "cause"], min_confidence=0.75)
PARTLY = {"Q-START": [verdict(0, "partly", ["P2"], "no corrosion or earlier leakage"), verdict(1, "unmet")]}          # the listed examples are not met word for word


class Judge(LLMCriterionJudge):
    """The criteria judged by a canned answer, the purpose question by a canned PurposeAnswer (or an error)."""

    def __init__(self, purpose, by_question=None):
        self.purpose = purpose
        super().__init__(FakeLLM(by_question), PACK, CFG.model_copy(update={"purpose": PURPOSE}))

        class Both:
            def structured(inner, cls, prompt, **kw):
                if cls is PurposeAnswer:
                    if isinstance(self.purpose, Exception):
                        raise self.purpose
                    return self.purpose
                return self._canned.structured(cls, prompt, **kw)
        self._canned, self._llm = FakeLLM(by_question), Both()


def run(purpose, by_question=PARTLY, facts=(P1, P2), links=None, contradictions=(), cfg_purpose=PURPOSE, **kw):
    cfg = CFG.model_copy(update={"purpose": cfg_purpose})
    j = Judge(purpose, by_question)
    j.cfg = cfg
    links = links if links is not None else [link("P1", "Q-START"), link("P2", "Q-START", "supporting")]
    return SufficiencyClassifier(cfg, j).classify(QUESTIONS, list(facts), links, list(contradictions), SNAP).get(kw.get("qid", "Q-START"))


SETTLED = PurposeAnswer(settled=True, fact_ids=["P2"], evidence_quote="no corrosion or earlier leakage was found", rationale="The plumber found a fresh failure with no earlier leakage.", confidence=0.9)


def test_without_the_second_look_the_partly_met_question_stays_possible_with_a_gap():
    a = run(SETTLED, cfg_purpose=PurposeConfig())
    assert (a.status, a.gap_type, a.settled_on_evidence) == ("possible", "insufficient", False)


def test_facts_that_answer_what_the_question_is_for_settle_it_with_the_reason_and_the_cited_fact():
    a = run(SETTLED)
    assert (a.status, a.gap_type, a.row_id, a.settled_on_evidence) == ("established", None, "P1-settled-on-evidence", True)
    assert a.settled_fact_ids == ("P2",) and a.settled_quote.startswith("no corrosion") and "fresh failure" in a.reason and a.unresolved()     # the listed examples stay as they were judged


def test_it_never_settles_what_the_checks_refuse():
    for answer in (SETTLED.model_copy(update={"settled": False}), SETTLED.model_copy(update={"confidence": 0.5}), SETTLED.model_copy(update={"evidence_quote": "words the fact never said"}),
                   SETTLED.model_copy(update={"fact_ids": ["P9"]}), SETTLED.model_copy(update={"fact_ids": []}), SETTLED.model_copy(update={"rationale": ""}), RuntimeError("model down")):
        a = run(answer)
        assert (a.status, a.gap_type, a.settled_on_evidence) == ("possible", "insufficient", False), answer


def test_a_fact_pointing_the_other_way_or_a_blocking_contradiction_stops_it():
    assert run(SETTLED, links=[link("P1", "Q-START", "contradicting"), link("P2", "Q-START", "supporting")]).status == "possible"
    a = run(SETTLED, contradictions=[contradiction("high", "Q-START")])
    assert (a.status, a.row_id) == ("requires_validation", "R1-contradicted")


def test_only_a_partly_evidenced_question_of_a_listed_category_qualifies():
    assert run(SETTLED, cfg_purpose=PurposeConfig(enabled=True, categories=["amount"])).status == "possible"          # its category is not listed
    assert run(SETTLED, by_question={"Q-START": [verdict(0, "unmet"), verdict(1, "unmet")]}, facts=(P2,), links=[link("P2", "Q-START")]).status == "missing"          # nothing partly met: nothing to settle on
    assert run(SETTLED, facts=(), links=[]).status == "missing"                                               # no fact: nothing to settle on
    unclear = run(SETTLED, by_question={"Q-START": [verdict(0, "unclear"), verdict(1, "unclear")]}, facts=(P2,), links=[link("P2", "Q-START")])
    assert unclear.abstained and unclear.settled_on_evidence is False                                         # the model could not judge: no second guess
    assert run(SETTLED, qid="Q-AMOUNT").settled_on_evidence is False and run(SETTLED, qid="Q-EXTENT").settled_on_evidence is False


def test_the_judge_reports_a_checked_verdict():
    j = Judge(SETTLED)
    v = j.settle(QUESTIONS[2], [], [P2], {"P2": "supporting"}, {"P2": "plumber_report"})
    assert isinstance(v, PurposeVerdict) and v.settled and v.fact_ids == ("P2",) and v.confidence == 0.9
    assert isinstance(VerdictAnswer(), VerdictAnswer)
    down = Judge(RuntimeError("x"))
    assert down.settle(QUESTIONS[2], [], [P2], {}, {}) is None
