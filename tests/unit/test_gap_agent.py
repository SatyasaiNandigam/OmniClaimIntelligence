"""T4.7 - Gap agent and incremental reassessment: new evidence re-runs only the questions it touches; the rest is carried over unchanged."""
import json
from pathlib import Path

import pytest

from claimtower.agents.gap import GapAgent, GapMemory, make_gap_agent
from claimtower.core.config import load_settings
from claimtower.core.schemas import ClaimState, DocumentRecord, Fact, PolicySnapshot
from claimtower.core.state import OwnershipViolation, StatePatch, apply_patch
from claimtower.evidence.contradiction import ConflictOutcome, ContradictionDetector, ModelConflict, Retraction
from claimtower.evidence.counterfactual import CounterfactualEngine
from claimtower.evidence.counterfactual import WriterOutcome as ScenarioOutcome
from claimtower.evidence.incremental import affected_questions, diff_gap, new_or_changed_facts, snapshot_changed
from claimtower.evidence.linker import FactQuestionLinker, JudgeOutcome as LinkOutcome, LinkVerdict
from claimtower.evidence.min_evidence import MinimumEvidenceGenerator, WriterOutcome as RequestOutcome
from claimtower.evidence.questions import ProposalOutcome, QuestionRegister
from claimtower.evidence.sufficiency import CriterionAssessment, JudgeOutcome as CriterionOutcome, SufficiencyClassifier

ROOT = Path(__file__).resolve().parents[2]
PACK = load_settings().pack
T = PACK.question_templates


def fixture_state(scenario="S-01") -> ClaimState:
    return ClaimState(**json.loads((ROOT / f"data/fixtures/{scenario}/claim_state.json").read_text(encoding="utf-8")))


# ------------------------------------------------------------------------------------------------ scripted stand-ins for the model
class Log:
    def __init__(self):
        self.links, self.conflicts, self.criteria, self.requests, self.scenarios, self.proposals = [], [], [], [], [], 0


class LinkJudge:
    """supporting for 'suddenly', contradicting for corrosion / long-term, otherwise neutral; only judges the pairs it is shown."""

    def __init__(self, log):
        self.log = log

    def judge(self, facts, questions, tagged):
        self.log.links += [f.fact_id for f in facts]
        verdicts = []
        for f in facts:
            t = f.proposition.lower()
            d = "contradicting" if ("corrosion" in t or "long-term" in t) and "surface" not in t else "supporting" if "suddenly" in t else "neutral"
            verdicts += [LinkVerdict(f.fact_id, q, d, "scripted", 0.9, f.evidence_text[:10]) for q in tagged.get(f.fact_id, [])]
        return LinkOutcome(verdicts=verdicts)


class ConflictJudge:
    """Finds 'sudden claim vs corrosion' on Q-SUDDEN. `forget`: on a later call that shows earlier conflicts it reports nothing (a model that did not repeat itself)."""

    def __init__(self, log, forget=False):
        self.log, self.forget = log, forget

    def judge(self, question, facts, earlier=(), new_fact_ids=()):
        self.log.conflicts.append(question.question_id)
        resolving = [f for f in facts if "surface rust only" in f.proposition and f.fact_id in set(new_fact_ids)]
        if earlier and resolving:
            return ConflictOutcome(retractions=[Retraction(c.contradiction_id, (resolving[0].fact_id,), "surface rust only", "the later report explains the corrosion away")
                                                for c in earlier])
        if earlier and self.forget:
            return ConflictOutcome()
        sudden = [f for f in facts if "burst suddenly" in f.proposition]
        slow = [f for f in facts if ("corrosion" in f.proposition.lower() and "surface" not in f.proposition.lower())]
        resolved = any("surface rust only" in f.proposition for f in facts)                  # the later report explains the corrosion away
        if question.question_id == "Q-SUDDEN" and sudden and slow and not resolved:
            return ConflictOutcome([ModelConflict("Q-SUDDEN", (sudden[0].fact_id,), (slow[0].fact_id,), "cause", "high", "sudden vs corrosion",
                                                  sudden[0].proposition[:12], slow[0].proposition[:12], 0.9)])
        return ConflictOutcome()


class CriterionJudge:
    """A plumber fact that says 'suddenly' meets the cause criteria; one that says 'predates' meets the start-date criterion; nothing else is met."""

    def __init__(self, log):
        self.log = log

    def judge(self, question, criteria, facts, directions, sources):
        self.log.criteria.append(question.question_id)
        plumber = [f for f in facts if f.proposition.lower().startswith("plumber")]
        out = {}
        for i, text in criteria.items():
            ok = bool(plumber) and (("suddenly" in " ".join(f.proposition for f in plumber) and question.question_id == "Q-SUDDEN")
                                    or ("predates" in " ".join(f.proposition for f in plumber) and question.question_id == "Q-START" and "Technical" in text))
            out[i] = CriterionAssessment(i, text, "met" if ok else "unmet", "model", "scripted", tuple(f.fact_id for f in plumber[:1]) if ok else (),
                                         plumber[0].proposition[:10] if ok else "", 0.9)
        return CriterionOutcome(verdicts=out)


class RequestWriter:
    def __init__(self, log):
        self.log = log

    def write(self, ctx):
        self.log.requests.append(ctx.role)
        return RequestOutcome()


class ScenarioWriter:
    def __init__(self, log):
        self.log = log

    def write(self, ctx):
        self.log.scenarios.append(ctx.question.question_id)
        return ScenarioOutcome()


class Proposer:
    def __init__(self, log):
        self.log = log

    def propose(self, ctx):
        self.log.proposals += 1
        return ProposalOutcome()


def agent(log=None, forget=False):
    log = log or Log()
    t = T.model_copy()
    ag = GapAgent(register=QuestionRegister(t, Proposer(log)), register_rules=QuestionRegister(t),
                  linker=FactQuestionLinker(t.fact_links, LinkJudge(log)), detector=ContradictionDetector(t.contradictions, ConflictJudge(log, forget)),
                  classifier=SufficiencyClassifier(t.sufficiency, CriterionJudge(log)), generator=MinimumEvidenceGenerator(t, RequestWriter(log)),
                  engine=CounterfactualEngine(t, ScenarioWriter(log)), modes={q.id: q.criteria_mode for q in t.questions},
                  policy_questions=[cr.question for cr in t.sufficiency.criteria_rules])
    return ag, log


def step(ag, state):
    """Run the agent and apply its patch, as the orchestrator does."""
    patch = ag.run(state)
    assert isinstance(patch, StatePatch) and patch.agent_id == "gap"
    return apply_patch(state, patch)


def with_supplementary_report(state: ClaimState) -> ClaimState:
    doc = DocumentRecord(document_id="DOC-S01-5", claim_id=state.claim_id, filename="plumber_supplementary.pdf", sha256="0" * 64, detected_type="plumber_report")
    def f(n, text, tags):
        return Fact(fact_id=f"F-S01-{n}", claim_id=state.claim_id, proposition=text, source_document_id="DOC-S01-5", page=1, evidence_text=text,
                    evidence_type="neutral", related_question_ids=tags, extraction_confidence=0.9)
    new = [f("07", "Plumber supplementary report: the corrosion is surface rust only and the joint failed suddenly", ["Q-SUDDEN", "Q-START"]),
           f("08", "Plumber supplementary report: no leakage predates 14 March 2026", ["Q-START"])]
    return state.model_copy(update={"documents": [*state.documents, doc], "facts": [*state.facts, *new]})


# ------------------------------------------------------------------------------------------------ the wrapper
def test_the_agent_declares_its_id_and_writes_only_the_fields_it_owns():
    ag, _ = agent()
    assert (ag.agent_id, set(ag.owns)) == ("gap", {"open_questions", "contradictions", "missing_evidence", "counterfactuals"})
    patch = ag.run(fixture_state())
    assert set(patch.updates) == set(ag.owns)
    apply_patch(fixture_state(), patch)                                      # an owned-fields-only patch is accepted
    with pytest.raises(OwnershipViolation):
        apply_patch(fixture_state(), StatePatch(agent_id="gap", updates={"facts": []}))


def test_the_first_run_is_a_full_assessment_that_reproduces_the_labelled_gaps_of_s01():
    ag, log = agent()
    state = step(ag, fixture_state())
    by = {q.question_id: q.status for q in state.open_questions}
    assert by["Q-INFORCE"] == "established" and by["Q-SUBLIMIT"] == "established" and by["Q-SUDDEN"] == "requires_validation"
    assert ag.last_run.cold and {c.question_id for c in state.contradictions} == {"Q-SUDDEN"} and log.proposals == 1
    assert {m.minimum_evidence_request.addressee_role for m in state.missing_evidence} >= {"plumber", "claimant", "surveyor"}
    assert [s.scenario_id for s in state.counterfactuals] == ["A", "B", "C"] and ag.last_run.problems == []
    assert "FULL assessment" in agent()[0].run(fixture_state()).rationale


# ------------------------------------------------------------------------------------------------ the heart of T4.7
def test_a_supplementary_plumber_report_changes_only_the_questions_linked_to_its_facts():
    ag, log = agent()
    s1 = step(ag, fixture_state())
    base_log = (list(log.links), list(log.criteria), list(log.conflicts), list(log.requests), list(log.scenarios), log.proposals)
    s2 = step(ag, with_supplementary_report(s1))
    run = ag.last_run
    assert not run.cold and run.affected == {"Q-SUDDEN", "Q-START"}
    # unaffected questions keep IDENTICAL results
    before, after = {q.question_id: q for q in s1.open_questions}, {q.question_id: q for q in s2.open_questions}
    for qid in ("Q-INFORCE", "Q-EXTENT", "Q-AMOUNT", "Q-SUBLIMIT"):
        assert after[qid] == before[qid], qid
    assert (before["Q-SUDDEN"].status, after["Q-SUDDEN"].status) == ("requires_validation", "established") and {"F-S01-07"} <= set(after["Q-SUDDEN"].linked_fact_ids)
    assert (before["Q-START"].status, after["Q-START"].status) == ("missing", "established") and s2.contradictions == []   # the conflict is resolved by the report
    assert run.diff.questions_unchanged and {c.question_id for c in run.diff.status_changes} <= {"Q-SUDDEN", "Q-START"}
    # and only what the new facts touch was sent anywhere
    assert log.links[len(base_log[0]):] == ["F-S01-07", "F-S01-08"]                       # only the new facts were linked
    assert set(log.criteria[len(base_log[1]):]) <= {"Q-SUDDEN", "Q-START"}
    assert set(log.conflicts[len(base_log[2]):]) <= {"Q-SUDDEN", "Q-START"}
    assert log.proposals == base_log[5] + 1                                               # the model's claim-specific questions are re-read once for the new facts


def test_the_requests_of_addressees_whose_needs_did_not_change_are_carried_over_untouched():
    ag, log = agent()
    s1 = step(ag, fixture_state())
    seen = len(log.requests)
    s2 = step(ag, with_supplementary_report(s1))
    old = {m.minimum_evidence_request.addressee_role: m for m in s1.missing_evidence}
    new = {m.minimum_evidence_request.addressee_role: m for m in s2.missing_evidence}
    assert "surveyor" in old and new["surveyor"] == old["surveyor"] and "surveyor" in ag.last_run.kept_requests
    assert "surveyor" not in log.requests[seen:]                                          # not even re-written by the model
    assert set(ag.last_run.regenerated_requests) <= {"plumber", "claimant"}


def test_scenarios_are_recomputed_only_for_questions_that_were_affected():
    ag, log = agent()
    s1 = step(ag, fixture_state())
    seen = len(log.scenarios)
    step(ag, with_supplementary_report(s1))
    assert log.scenarios[seen:] in ([], ["Q-SUDDEN"]) and ag.last_run.recomputed_scenarios == ["Q-SUDDEN"]


def test_a_rerun_with_nothing_new_changes_nothing_and_asks_no_model():
    ag, log = agent()
    s1 = step(ag, fixture_state())
    counts = (len(log.links), len(log.criteria), len(log.conflicts), len(log.requests), len(log.scenarios), log.proposals)
    s2 = step(ag, s1)
    assert ag.last_run.affected == set() and ag.last_run.diff.empty and "NOTHING NEW" in ag.run(s1).rationale
    for f in ("open_questions", "contradictions", "missing_evidence", "counterfactuals"):
        assert getattr(s2, f) == getattr(s1, f)
    assert counts == (len(log.links), len(log.criteria), len(log.conflicts), len(log.requests), len(log.scenarios), log.proposals)


def test_the_memory_can_be_saved_and_restored_so_a_restarted_agent_stays_incremental():
    ag, _ = agent()
    s1 = step(ag, fixture_state())
    saved = ag.memory.model_dump_json()
    fresh, log2 = agent()
    fresh.memory = GapMemory.model_validate_json(saved)
    step(fresh, with_supplementary_report(s1))
    assert not fresh.last_run.cold and fresh.last_run.affected == {"Q-SUDDEN", "Q-START"} and log2.links == ["F-S01-07", "F-S01-08"]


def test_without_memory_or_for_another_claim_the_run_is_a_full_one():
    ag, _ = agent()
    step(ag, fixture_state("S-01"))
    other = fixture_state("S-02")
    step(ag, other)
    assert ag.last_run.cold and ag.last_run.affected == {q.question_id for q in apply_patch(other, ag.run(other)).open_questions}


# ------------------------------------------------------------------------------------------------ other things that touch a question
def test_rejecting_or_removing_a_fact_re_runs_the_questions_it_was_linked_to():
    ag, log = agent()
    s1 = step(ag, fixture_state())
    rejected = s1.model_copy(update={"facts": [f.model_copy(update={"verification_status": "rejected"}) if f.fact_id == "F-S01-03" else f for f in s1.facts]})
    step(ag, rejected)
    assert {"Q-SUDDEN", "Q-START"} <= ag.last_run.affected and "Q-EXTENT" not in ag.last_run.affected
    removed = rejected.model_copy(update={"facts": [f for f in rejected.facts if f.fact_id != "F-S01-04"]})
    step(ag, removed)
    assert {"Q-SUDDEN", "Q-EXTENT"} <= ag.last_run.affected


def test_a_changed_policy_re_runs_the_policy_questions_and_not_the_evidence_ones():
    ag, _ = agent()
    s1 = step(ag, fixture_state("S-01"))
    s5 = PolicySnapshot(**json.loads((ROOT / "data/fixtures/S-05/policy.json").read_text(encoding="utf-8"))["policy_snapshot"])
    step(ag, s1.model_copy(update={"policy_snapshot": s5}))
    assert {"Q-INFORCE", "Q-SUBLIMIT"} <= ag.last_run.affected and "Q-AMOUNT" not in ag.last_run.affected and "Q-EXTENT" not in ag.last_run.affected


def test_a_policy_that_is_only_refetched_is_not_a_change():
    s = fixture_state()
    refetched = s.model_copy(update={"policy_snapshot": s.policy_snapshot.model_copy(update={"retrieved_at": None})})
    assert not snapshot_changed(s.model_copy(update={"policy_snapshot": s.policy_snapshot.model_copy(update={"retrieved_at": None})}), refetched)


def test_a_lapsed_policy_is_read_as_answered_not_as_missing():
    s3 = fixture_state("S-03")
    ag, _ = agent()
    state = step(ag, s3)
    assert {q.question_id: q.status for q in state.open_questions} == {"Q-INFORCE": "established"}
    no_answer = s3.model_copy(update={"claim_header": s3.claim_header.model_copy(update={"policy_ref": None})})
    ag2, _ = agent()
    assert {q.question_id: q.status for q in step(ag2, no_answer).open_questions} == {"Q-INFORCE": "missing"}


# ------------------------------------------------------------------------------------------------ the pure helpers
def test_affected_questions_uses_tags_links_and_what_a_fact_was_linked_to_before():
    s = fixture_state()
    new = [Fact(fact_id="F-X", claim_id=s.claim_id, proposition="p", source_document_id="D", page=1, evidence_text="p", evidence_type="neutral",
                related_question_ids=["Q-EXTENT"], extraction_confidence=0.9)]
    assert affected_questions(s, new) == {"Q-EXTENT"}
    untagged = [new[0].model_copy(update={"related_question_ids": []})]
    assert affected_questions(s, untagged) == set()                                       # nothing is known to touch: only a link can say so
    from claimtower.evidence.linker import Link
    assert affected_questions(s, untagged, links=[Link("F-X", "Q-AMOUNT", "neutral", "x", "model")]) == {"Q-AMOUNT"}
    old = next(f for f in s.facts if f.fact_id == "F-S01-04")
    assert {"Q-SUDDEN", "Q-EXTENT"} <= affected_questions(s, [old])                       # an edited fact re-runs the questions it was linked to
    assert affected_questions(s, [new[0].model_copy(update={"related_question_ids": ["Q-NOT-A-QUESTION"]})]) == set()


def test_new_or_changed_facts_finds_additions_edits_and_removals():
    s = fixture_state()
    edited = [f.model_copy(update={"evidence_text": "changed"}) if f.fact_id == "F-S01-02" else f for f in s.facts if f.fact_id != "F-S01-06"]
    ids = {f.fact_id for f in new_or_changed_facts(s, s.model_copy(update={"facts": edited}))}
    assert ids == {"F-S01-02", "F-S01-06"}


def test_diff_gap_reports_status_contradiction_request_and_scenario_changes():
    ag, _ = agent()
    s1 = step(ag, fixture_state())
    s2 = step(ag, with_supplementary_report(s1))
    d = diff_gap(s1, s2)
    assert not d.empty and "Q-SUDDEN requires_validation -> established" in d.describe() and d.contradictions_removed and d.scenarios_changed == ["Q-SUDDEN"] and d.questions_unchanged
    assert diff_gap(s1, s1).empty and diff_gap(s1, s1).describe() == "no change"


def test_the_factory_builds_a_rule_only_agent_without_a_model():
    ag = make_gap_agent(PACK, None)
    patch = ag.run(fixture_state())
    assert ag.last_run.cold and {q.question_id for q in patch.updates["open_questions"]} >= {"Q-INFORCE", "Q-SUDDEN"}
    assert "REVIEW" in patch.rationale                                                     # without a model the fact questions abstain: a person decides


# ------------------------------------------------------------------------------------------------ contradictions are sticky
def late_plumber_note(state, text="Plumber supplementary report: the joint was re-tested on site and holds pressure"):
    f = Fact(fact_id="F-S01-09", claim_id=state.claim_id, proposition=text, source_document_id="DOC-S01-2", page=1, evidence_text=text, evidence_type="neutral",
             related_question_ids=["Q-SUDDEN"], extraction_confidence=0.9)
    return state.model_copy(update={"facts": [*state.facts, f]})


def test_an_earlier_contradiction_stays_when_a_later_model_call_does_not_repeat_it():
    ag, _ = agent(forget=True)
    s1 = step(ag, fixture_state())
    assert [c.question_id for c in s1.contradictions] == ["Q-SUDDEN"]
    patch = ag.run(late_plumber_note(s1))                                             # affects Q-SUDDEN; the judge reports nothing this time
    s2 = apply_patch(late_plumber_note(s1), patch)
    assert s2.contradictions == s1.contradictions and ag.last_run.retained_contradictions == [s1.contradictions[0].contradiction_id]
    assert next(q for q in s2.open_questions if q.question_id == "Q-SUDDEN").status == "requires_validation"           # still open: nothing resolved it
    assert "kept (nothing resolved them)" in patch.rationale


def test_an_earlier_contradiction_is_removed_only_by_a_retraction_that_cites_a_new_fact():
    ag, _ = agent(forget=True)
    s1 = step(ag, fixture_state())
    patch = ag.run(with_supplementary_report(s1))
    assert apply_patch(with_supplementary_report(s1), patch).contradictions == []
    assert len(ag.last_run.retracted_contradictions) == 1 and "explains the corrosion away" in ag.last_run.retracted_contradictions[0]
    assert "resolved by new evidence" in patch.rationale


def test_a_contradiction_whose_cited_fact_was_edited_is_not_carried_over_blindly():
    ag, log = agent(forget=True)
    s1 = step(ag, fixture_state())
    cited = s1.contradictions[0].fact_ids_b[0]
    edited = s1.model_copy(update={"facts": [f.model_copy(update={"evidence_text": f.evidence_text + " (revised)", "proposition": f.proposition + " (revised)"})
                                             if f.fact_id == cited else f for f in s1.facts]})
    s2 = step(ag, edited)
    assert ag.last_run.retained_contradictions == [] and "Q-SUDDEN" in ag.last_run.affected
    assert s2.contradictions == s1.contradictions                                    # judged afresh on the edited fact (the scripted model finds the same conflict again)
