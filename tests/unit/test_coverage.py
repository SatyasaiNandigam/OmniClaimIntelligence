"""T5.1 - coverage issue matrix: rows from the pack, status from a decision table, a checked model reading, citations from the snapshot, never a verdict."""
import json
from pathlib import Path

import pytest

from claimtower.agents.coverage import CoverageAgent, make_coverage_agent, not_in_force
from claimtower.core.config import CoverageConfig, IssueRow, load_settings
from claimtower.core.schemas import ClaimHeader, ClaimState, OpenQuestion, PolicySnapshot
from claimtower.core.state import OwnershipViolation, StatePatch, apply_patch
from claimtower.coverage.issue_matrix import IssueMatrixBuilder, Reading, ReadingOutcome, control_state
from claimtower.coverage.llm_reading import LLMReadingJudge, ReadingAnswer, make_builder

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()
PACK = SETTINGS.pack
CFG = PACK.coverage
SCENARIOS = ("S-01", "S-02", "S-03", "S-04", "S-05")


def fixture(sid) -> ClaimState:
    return ClaimState(**json.loads((ROOT / f"data/fixtures/{sid}/claim_state.json").read_text(encoding="utf-8")))


class Judge:
    """A scripted reader: the same reading for every row it is asked about (or per issue id); records what it was shown."""

    def __init__(self, value="favourable", per_issue=None, refused=None, unavailable=False):
        self.value, self.per_issue, self.refused, self.unavailable, self.seen = value, per_issue or {}, refused, unavailable, []

    def read(self, ctx):
        self.seen.append(ctx)
        if self.unavailable:
            return ReadingOutcome(unavailable=True, detail="down")
        v = self.per_issue.get(ctx.spec.id, self.value)
        if v == "unclear":
            return ReadingOutcome(Reading("unclear"), refused=self.refused)
        return ReadingOutcome(Reading(v, (ctx.facts[0].fact_id,), (ctx.clauses[0].clause_id,), "quote", "the facts on file indicate it", 0.9))


def build(state, judge=None, **kw):
    return IssueMatrixBuilder(CFG, judge, 1).build(state.policy_snapshot, not_in_force(state), state.facts, state.open_questions, state.contradictions, **kw)


def statuses(result):
    return {i.issue_id: i.status for i in result.issues}


# ------------------------------------------------------------------------------------------------ the labelled matrices
@pytest.mark.parametrize("sid", SCENARIOS)
def test_with_a_favourable_reading_the_matrix_has_the_labelled_statuses_and_every_row_is_cited(sid):
    state = fixture(sid)
    r = build(state, Judge("favourable"))
    assert statuses(r) == {i.issue_id: i.status for i in state.coverage_issues}, sid
    assert r.problems == [] and r.skipped == ([] if sid != "S-03" else ["ISS-COVER-WATER", "ISS-EXC-GRADUAL", "ISS-SUBLIMIT"])
    for i in r.issues:
        if i.issue_id != "ISS-INFORCE" or sid != "S-03":
            assert i.clause_citations and all(c.text and c.document for c in i.clause_citations), (sid, i.issue_id)


def test_without_a_model_the_settled_rows_are_identified_and_nothing_is_concluded():
    r = build(fixture("S-02"))
    assert statuses(r) == {"ISS-INFORCE": "established", "ISS-COVER-WATER": "identified", "ISS-EXC-GRADUAL": "identified", "ISS-SUBLIMIT": "identified"}
    assert {d.row_id for i, d in r.details.items() if i in ("ISS-COVER-WATER", "ISS-EXC-GRADUAL")} == {"R15-not-judged"}


def test_the_running_example_is_unresolved_and_the_exclusion_is_potentially_relevant_whatever_a_model_would_say():
    j = Judge("favourable")
    r = build(fixture("S-01"), j)
    assert statuses(r) == {"ISS-INFORCE": "established", "ISS-COVER-WATER": "unresolved", "ISS-EXC-GRADUAL": "potentially_relevant", "ISS-SUBLIMIT": "identified"}
    assert j.seen == []                                                                  # contested rows are decided by the table: the model is not asked
    assert r.get("ISS-COVER-WATER").fact_ids == ["F-S01-02", "F-S01-03", "F-S01-04"]    # the facts of the question, both sides of the conflict
    assert "conflicts" in r.get("ISS-COVER-WATER").note


# ------------------------------------------------------------------------------------------------ the reading: an adverse one never establishes
def test_an_adverse_reading_is_never_established():
    r = build(fixture("S-02"), Judge("adverse"))
    assert statuses(r)["ISS-COVER-WATER"] == "unresolved" and statuses(r)["ISS-EXC-GRADUAL"] == "potentially_relevant"
    assert "may not satisfy" in r.get("ISS-COVER-WATER").note and "may engage" in r.get("ISS-EXC-GRADUAL").note and r.problems == []


def test_an_unclear_reading_abstains_and_a_refused_one_is_recorded():
    r = build(fixture("S-02"), Judge("unclear", refused="its quote is not verbatim text of a cited fact"))
    assert statuses(r)["ISS-COVER-WATER"] == "abstain" and statuses(r)["ISS-EXC-GRADUAL"] == "abstain"
    assert [w for _, w in r.refused] == ["its quote is not verbatim text of a cited fact"] * 2


def test_a_model_that_is_down_leaves_the_rows_identified_with_a_warning():
    r = build(fixture("S-02"), Judge(unavailable=True))
    assert statuses(r)["ISS-COVER-WATER"] == "identified" and any("unavailable" in w and "down" in w for w in r.warnings)


def test_the_rows_that_were_read_cite_the_facts_the_reading_rests_on_and_show_its_hedged_reason():
    r = build(fixture("S-02"), Judge("favourable"))
    row = r.get("ISS-COVER-WATER")
    assert row.fact_ids == ["F-S02-02"] and row.note.endswith("Reading: the facts on file indicate it")
    assert r.details["ISS-COVER-WATER"].reading.value == "favourable" and r.details["ISS-COVER-WATER"].row_id == "R11-favourable"


def test_a_rationale_that_states_a_coverage_conclusion_is_not_shown():
    class Bold(Judge):
        def read(self, ctx):
            return ReadingOutcome(Reading("favourable", (ctx.facts[0].fact_id,), (ctx.clauses[0].clause_id,), "q", "the claim is covered in full", 0.9))

    r = build(fixture("S-02"), Bold())
    assert "covered" not in r.get("ISS-COVER-WATER").note and r.problems == []


# ------------------------------------------------------------------------------------------------ citations, missing wording, not in force
def test_a_missing_endorsement_makes_the_sublimit_row_abstain_and_cites_the_cover_it_would_modify():
    r = build(fixture("S-05"))
    row = r.get("ISS-SUBLIMIT")
    assert row.status == "abstain" and [c.clause_id for c in row.clause_citations] == ["CL-COV-WATER"] and "not available" in row.note
    assert r.details["ISS-SUBLIMIT"].row_id == "R03-wording-missing"


def test_a_row_with_no_clause_to_cite_abstains():
    state = fixture("S-02")
    snap = state.policy_snapshot.model_copy(update={"clauses": [c for c in state.policy_snapshot.clauses if c.clause_type != "exclusion"]})
    r = IssueMatrixBuilder(CFG, Judge(), 1).build(snap, False, state.facts, state.open_questions, state.contradictions)
    assert r.get("ISS-EXC-GRADUAL").status == "abstain" and r.get("ISS-EXC-GRADUAL").clause_citations == [] and "could be cited" in r.get("ISS-EXC-GRADUAL").note
    assert r.problems == []


def test_a_lapsed_policy_gets_the_one_row_that_says_so_and_nothing_else():
    state = fixture("S-03")
    r = build(state)
    assert [(i.issue_id, i.status) for i in r.issues] == [("ISS-INFORCE", "identified")] and "no policy in force" in r.issues[0].note
    assert not_in_force(state) and not not_in_force(fixture("S-01")) and not not_in_force(ClaimState(claim_id="x"))


def test_citations_carry_the_verbatim_clause_and_the_endorsements_that_modify_it():
    r = build(fixture("S-01"))
    c = next(c for c in r.get("ISS-COVER-WATER").clause_citations if c.clause_id == "CL-COV-WATER")
    assert c.text.startswith("We cover loss of or damage to the Buildings") and c.page and c.document == "BASE-HOME-1"
    assert isinstance(c.modified_by, list)


def test_a_row_exists_only_when_its_questions_are_in_the_register():
    state = fixture("S-01")
    r = IssueMatrixBuilder(CFG, None, 1).build(state.policy_snapshot, False, state.facts, [q for q in state.open_questions if q.question_id != "Q-START"], [])
    assert "ISS-EXC-GRADUAL" in r.skipped and "ISS-EXC-GRADUAL" not in statuses(r)


# ------------------------------------------------------------------------------------------------ control state and the table
@pytest.mark.parametrize("statuses_,want", [(["established"], "settled"), (["established", "established"], "settled"), (["requires_validation", "missing"], "contested"),
                                            (["missing", "partially_established"], "missing"), (["partially_established", "established"], "partial"),
                                            (["unverified"], "open"), (["possible", "established"], "open"), ([], "open")])
def test_the_controlling_questions_collapse_to_one_state(statuses_, want):
    assert control_state(statuses_) == want


@pytest.mark.parametrize("q_status,kind_status", [("missing", {"ISS-COVER-WATER": "incomplete"}), ("partially_established", {"ISS-COVER-WATER": "incomplete"}),
                                                  ("unverified", {"ISS-COVER-WATER": "unresolved"}), ("possible", {"ISS-COVER-WATER": "unresolved"})])
def test_unsettled_cover_questions_give_incomplete_or_unresolved(q_status, kind_status):
    state = fixture("S-02")
    qs = [q.model_copy(update={"status": q_status}) if q.question_id == "Q-SUDDEN" else q for q in state.open_questions]
    r = IssueMatrixBuilder(CFG, Judge(), 1).build(state.policy_snapshot, False, state.facts, qs, [])
    assert statuses(r)["ISS-COVER-WATER"] == kind_status["ISS-COVER-WATER"]


def test_the_pack_cannot_say_the_claim_is_covered():
    b = IssueMatrixBuilder(CFG, None, 1)
    for bad in ("The claim is covered.", "This loss is not covered", "the exclusion applies", "We should deny the claim", "payment is payable", "fully covered"):
        assert b.lint(bad), bad
    for good in ("The clause is identified and cited.", "The evidence conflicts; a person decides.", "The cover condition may not be satisfied."):
        assert b.lint(good) == [], good
    assert all(b.lint(row.note) == [] for row in CFG.table)                                # every table note is itself clean


def test_the_self_check_catches_an_established_row_on_an_adverse_reading():
    r = build(fixture("S-02"), Judge("favourable"))
    from claimtower.coverage.issue_matrix import RowDetail
    r.details["ISS-COVER-WATER"] = RowDetail("ISS-COVER-WATER", "R11-favourable", "settled", Reading("adverse"), True)
    assert any("established on an adverse reading" in p for p in IssueMatrixBuilder(CFG, None, 1).self_check(r))


# ------------------------------------------------------------------------------------------------ the model is asked once per unchanged row
def test_an_unchanged_row_is_not_read_again_and_a_changed_fact_is():
    j = Judge("favourable")
    b = IssueMatrixBuilder(CFG, j, 1)
    state = fixture("S-02")
    args = (state.policy_snapshot, False, state.facts, state.open_questions, state.contradictions)
    first = b.build(*args)
    asked = len(j.seen)
    second = b.build(*args)
    assert len(j.seen) == asked and b.cache_hits == asked and statuses(first) == statuses(second)
    edited = [f.model_copy(update={"proposition": f.proposition + " (revised)"}) if f.fact_id == "F-S02-02" else f for f in state.facts]
    b.build(state.policy_snapshot, False, edited, state.open_questions, state.contradictions)
    assert len(j.seen) > asked


# ------------------------------------------------------------------------------------------------ the checks on the model's answer
class FakeLLM:
    def __init__(self, answer=None, fail=None, cause="yes"):
        self.answer, self.fail, self.cause, self.prompts = answer, fail, cause, []

    def structured(self, model_cls, prompt, **kw):
        self.prompts.append(prompt.text)
        if self.fail:
            raise self.fail
        if model_cls.__name__ == "AmbiguityAnswer":
            return model_cls(ambiguous=False)
        if model_cls.__name__ == "CauseCheck":
            return model_cls(states_cause=self.cause)
        return self.answer


def ctx_for(sid="S-02", issue="ISS-COVER-WATER"):
    from claimtower.coverage.issue_matrix import IssueContext
    state = fixture(sid)
    spec = next(i for i in CFG.issues if i.id == issue)
    b = IssueMatrixBuilder(CFG, None, 1)
    clauses = b._select(spec.clauses, state.policy_snapshot.clauses)
    qs = [q for q in state.open_questions if q.question_id in spec.questions]
    facts = [f for f in state.facts if f.fact_id in {i for q in qs for i in q.linked_fact_ids}]
    return IssueContext(spec, clauses, facts, qs, [])


def good_answer(ctx, **over):
    base = dict(reading="supports", clause_ids=[ctx.clauses[0].clause_id], fact_ids=[ctx.facts[0].fact_id], evidence_quote=ctx.facts[0].proposition[:20],
                rationale="The plumber's statement indicates a sudden failure.", confidence=0.95)
    base.update(over)
    return ReadingAnswer(**base)


def judge_for(answer=None, fail=None):
    return LLMReadingJudge(FakeLLM(answer, fail), PACK)


def test_a_grounded_reading_is_accepted_with_capped_confidence():
    ctx = ctx_for()
    out = judge_for().check(good_answer(ctx), ctx)
    assert out.refused is None and out.reading.value == "favourable" and out.reading.confidence == CFG.reading.confidence_cap and out.reading.rationale
    assert judge_for().check(good_answer(ctx, reading="against"), ctx).reading.value == "adverse"


@pytest.mark.parametrize("over,why", [
    (dict(reading="maybe"), "allowed reading"), (dict(clause_ids=[]), "cites no clause"), (dict(fact_ids=[]), "cites no clause or fact"),
    (dict(clause_ids=["CL-NOPE"]), "not offered"), (dict(fact_ids=["F-NOPE"]), "not offered"),
    (dict(evidence_quote="something never said"), "verbatim"), (dict(evidence_quote=""), "verbatim"), (dict(confidence=0.3), "below"),
])
def test_an_ungrounded_reading_is_refused_and_becomes_unclear(over, why):
    ctx = ctx_for()
    out = judge_for().check(good_answer(ctx, **over), ctx)
    assert why in (out.refused or "") and out.reading.value == "unclear"


def test_unclear_is_an_honest_answer_and_a_conclusion_in_the_rationale_is_dropped_not_the_reading():
    ctx = ctx_for()
    assert judge_for().check(ReadingAnswer(reading="unclear", confidence=0.4), ctx).reading.value == "unclear"
    out = judge_for().check(good_answer(ctx, rationale="The exclusion is not engaged and the claim is covered."), ctx)
    assert out.reading.value == "favourable" and out.reading.rationale == ""


def test_the_prompt_shows_the_issue_the_clauses_the_facts_and_the_state_of_the_questions():
    ctx = ctx_for()
    llm = FakeLLM(good_answer(ctx))
    out = LLMReadingJudge(llm, PACK).read(ctx)
    assert len(llm.prompts) == 2                                  # a cover row has no exception clause: the ambiguity question is not asked
    p = llm.prompts[0]
    for needle in ["Escape of water: sudden and accidental", "CL-COV-WATER", "CL-DEF-SUDDEN", "F-S02-02", "Q-SUDDEN (established)", "the condition appears satisfied", "hedged"]:
        assert needle in p, needle
    assert out.reading.value == "favourable"
    down = LLMReadingJudge(FakeLLM(fail=RuntimeError("down")), PACK).read(ctx)
    assert down.unavailable and "down" in down.detail


def test_the_factory_uses_the_model_only_when_the_pack_enables_it_and_a_client_is_given():
    assert make_builder(PACK, None).judge is None and make_builder(PACK, FakeLLM()).judge is not None


# ------------------------------------------------------------------------------------------------ config
def test_the_pack_config_is_validated():
    base = CFG.model_dump()
    with pytest.raises(ValueError, match="no conditions"):
        CoverageConfig(**{**base, "table": base["table"][:-1]})
    with pytest.raises(ValueError, match="unknown condition"):
        IssueRow(id="x", when={"bogus": True}, status="identified")
    with pytest.raises(ValueError, match="one or more of"):
        IssueRow(id="x", when={"control": ["half"]}, status="identified")
    with pytest.raises(ValueError, match="duplicate coverage issue ids"):
        CoverageConfig(**{**base, "issues": [base["issues"][0], base["issues"][0]]})
    with pytest.raises(ValueError, match="pattern"):
        CoverageConfig(**{**base, "forbidden_patterns": ["("]})
    assert CFG.reading.enabled and [i.id for i in CFG.issues] == ["ISS-INFORCE", "ISS-COVER-WATER", "ISS-EXC-GRADUAL", "ISS-SUBLIMIT"]


def test_every_issue_names_questions_that_exist_in_the_pack():
    known = {q.id for q in PACK.question_templates.questions}
    assert all(set(i.questions) <= known for i in CFG.issues)


# ------------------------------------------------------------------------------------------------ the agent
def test_the_agent_declares_its_id_writes_only_coverage_issues_and_explains_itself():
    agent = CoverageAgent(IssueMatrixBuilder(CFG, Judge("favourable"), 1))
    assert (agent.agent_id, set(agent.owns)) == ("coverage", {"coverage_issues"})
    state = fixture("S-02").model_copy(update={"coverage_issues": []})
    patch = agent.run(state)
    assert isinstance(patch, StatePatch) and set(patch.updates) == {"coverage_issues"}
    assert "4 issue(s)" in patch.rationale and "REVIEW: read by a model" in patch.rationale and "ISS-COVER-WATER" in patch.rationale
    after = apply_patch(state, patch)
    assert [i.status for i in after.coverage_issues] == ["established"] * 3 + ["identified"]
    with pytest.raises(OwnershipViolation):
        apply_patch(state, StatePatch(agent_id="coverage", updates={"facts": []}))


def test_the_agent_without_a_model_says_nothing_was_read_and_is_idempotent():
    agent = make_coverage_agent(PACK, None, 1)
    state = fixture("S-02")
    first, second = agent.run(state), agent.run(state)
    assert first.updates == second.updates and "REVIEW" not in first.rationale and "identified" in first.rationale


def test_a_claim_with_no_snapshot_yet_gets_no_rows_that_need_clauses():
    state = ClaimState(claim_id="C", claim_header=ClaimHeader(claim_id="C"), open_questions=[OpenQuestion(question_id="Q-INFORCE", text="t", category="coverage_condition",
                                                                                                        status="established")])
    r = build(state)
    assert [(i.issue_id, i.status) for i in r.issues] == [("ISS-INFORCE", "abstain")] and r.problems == []
    _ = PolicySnapshot


def test_a_favourable_reading_needs_the_independent_check_that_its_quote_states_the_cause():
    ctx = ctx_for()
    ok = LLMReadingJudge(FakeLLM(good_answer(ctx), cause="yes"), PACK).read(ctx)
    assert ok.reading.value == "favourable" and ok.refused is None
    for cause in ("no", "partly"):
        out = LLMReadingJudge(FakeLLM(good_answer(ctx), cause=cause), PACK).read(ctx)
        assert out.reading.value == "unclear" and "how or when" in out.refused
    adverse = LLMReadingJudge(FakeLLM(good_answer(ctx, reading="against"), cause="no"), PACK).read(ctx)
    assert adverse.reading.value == "adverse"                                  # the check guards only the favourable direction


# ------------------------------------------------------------------------------------------------ T5.2: abstention reasons and the ambiguity check
def amb(*readings, ambiguous=True, confidence=0.85):
    from claimtower.coverage.llm_reading import AmbiguityAnswer, AmbiguousReading
    return AmbiguityAnswer(ambiguous=ambiguous, readings=[AmbiguousReading(interpretation=i, clause_quote=q, points=p) for i, q, p in readings], confidence=confidence)


def exc_ctx():
    from claimtower.coverage.issue_matrix import IssueContext
    state = fixture("S-02")
    spec = next(i for i in CFG.issues if i.id == "ISS-EXC-GRADUAL")
    clauses = IssueMatrixBuilder(CFG, None, 1)._select(spec.clauses, state.policy_snapshot.clauses)
    return IssueContext(spec, clauses, state.facts[:2], [], []), clauses


def test_two_readings_resting_on_different_words_and_pointing_opposite_ways_are_an_ambiguity():
    ctx, clauses = exc_ctx()
    q1, q2 = clauses[0].text[:30], (clauses[1].text[:30] if len(clauses) > 1 else clauses[0].text[-30:])
    out = judge_for().check_ambiguity(amb(("the exclusion governs", q1, "adverse"), ("the exception governs", q2, "favourable")), ctx)
    assert out.reading.value == "ambiguous" and len(out.reading.alternatives) == 2 and out.reading.alternatives[0].startswith("the exclusion governs")


@pytest.mark.parametrize("answer", [
    lambda q1, q2: amb(("a", q1, "adverse"), ("b", q1, "favourable")),                    # the very same words read two ways: a re-reading of the facts
    lambda q1, q2: amb(("a", q1, "adverse"), ("b", q2, "adverse")),                       # both point the same way: nothing to choose between
    lambda q1, q2: amb(("a", q1, "adverse"), ("b", "words that are not in any clause", "favourable")),
    lambda q1, q2: amb(("a", q1, "adverse")),                                              # one reading is not an ambiguity
    lambda q1, q2: amb(("a", q1, "adverse"), ("b", q2, "favourable"), confidence=0.3),
    lambda q1, q2: amb(("a", q1, "adverse"), ("b", q2, "favourable"), ambiguous=False),
])
def test_anything_less_than_a_grounded_opposed_pair_is_not_an_ambiguity(answer):
    ctx, clauses = exc_ctx()
    q1, q2 = clauses[0].text[:30], (clauses[1].text[:30] if len(clauses) > 1 else clauses[0].text[-30:])
    assert judge_for().check_ambiguity(answer(q1, q2), ctx) is None


def test_an_interpretation_that_states_a_conclusion_is_not_shown():
    ctx, clauses = exc_ctx()
    q1, q2 = clauses[0].text[:30], clauses[0].text[-30:]
    out = judge_for().check_ambiguity(amb(("the claim is not covered", q1, "adverse"), ("the exception governs", q2, "favourable")), ctx)
    assert out.reading.value == "ambiguous" and "not covered" not in out.reading.alternatives[0]


def test_an_ambiguous_reading_abstains_with_its_reason_and_shows_both_readings():
    class Amb(Judge):
        def read(self, ctx):
            return ReadingOutcome(Reading("ambiguous", (), (ctx.clauses[0].clause_id,), "", "", 0.8, ("first way (\"x\")", "second way (\"y\")")))

    r = build(fixture("S-02"), Amb())
    row = r.get("ISS-EXC-GRADUAL")
    assert row.status == "abstain" and "Reading A: first way" in row.note and "Reading B: second way" in row.note and row.clause_citations
    assert r.details["ISS-EXC-GRADUAL"].row_id == "R13b-ambiguous" and r.abstentions()["ISS-EXC-GRADUAL"] == "AMBIGUOUS_CLAUSE" and r.problems == []


def test_every_row_that_does_not_conclude_has_a_reason_and_a_concluded_row_has_none():
    assert set(build(fixture("S-05")).abstentions().values()) == {"WORDING_MISSING"}
    assert build(fixture("S-01")).abstentions() == {"ISS-COVER-WATER": "CONTROL_CONTESTED", "ISS-EXC-GRADUAL": "CONTROL_CONTESTED"}
    r = build(fixture("S-02"), Judge("favourable"))
    assert r.abstentions() == {}
    assert build(fixture("S-02"), Judge("adverse")).abstentions() == {"ISS-COVER-WATER": "READING_ADVERSE", "ISS-EXC-GRADUAL": "READING_ADVERSE"}
    assert {x.reason for x in CFG.table if x.status == "abstain"} == {"NO_CITATION", "WORDING_MISSING", "AMBIGUOUS_CLAUSE", "UNCLEAR_READING"}


def test_no_note_of_any_scenario_states_a_coverage_conclusion_even_with_every_reading():
    b = IssueMatrixBuilder(CFG, None, 1)
    for sid in SCENARIOS:
        for reading in ("favourable", "adverse", "unclear"):
            for i in build(fixture(sid), Judge(reading)).issues:
                assert b.lint(i.note or "") == [], (sid, reading, i.issue_id)


def test_the_ambiguity_question_is_asked_only_where_an_exclusion_has_an_exception_clause():
    j = judge_for()
    ctx, _ = exc_ctx()
    assert not j.has_exception(ctx) and not j.has_exception(ctx_for())                  # BASE-HOME-1: exclusions without an exception; a cover row
    from claimtower.coverage.issue_matrix import IssueContext
    from evals.metrics.coverage import load_cases
    h2 = next(c for c in load_cases() if c.id == "A-1").state
    spec = next(i for i in CFG.issues if i.id == "ISS-EXC-GRADUAL")
    clauses = IssueMatrixBuilder(CFG, None, 1)._select(spec.clauses, h2.policy_snapshot.clauses)
    assert {c.clause_id for c in clauses} == {"CL-H2-EXC-GRADUAL", "CL-H2-EXC-GRADUAL-EXC"} and j.has_exception(IssueContext(spec, clauses, [], [], []))


def test_a_clear_exclusion_is_read_without_the_ambiguity_question():
    ctx, _ = exc_ctx()
    fake = FakeLLM(good_answer(ctx))
    LLMReadingJudge(fake, PACK).read(ctx)
    assert len(fake.prompts) == 2 and "Policy clauses" in fake.prompts[0] and "WORDING" not in fake.prompts[0]          # reading + cause check only
