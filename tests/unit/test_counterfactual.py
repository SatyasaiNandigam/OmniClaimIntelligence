"""T4.6 - counterfactual scenarios: A / B / C per unresolved material question, never a predicted decision."""
import json
from pathlib import Path

import pytest

from claimtower.core.config import CounterfactualConfig, load_settings
from claimtower.core.schemas import ClaimHeader, CounterfactualScenario, CoverageIssue, MinimumEvidenceRequest, PolicySnapshot
from claimtower.core.schemas.evidence import MissingEvidence
from claimtower.evidence.counterfactual import CounterfactualEngine, ScenarioDraft, WriterOutcome
from claimtower.evidence.llm_counterfactual import LLMScenarioWriter, RewriteAnswer, ScenarioModel, make_engine
from claimtower.evidence.questions import QuestionRegister
from claimtower.evidence.sufficiency import CriterionAssessment, QuestionAssessment, SufficiencyResult

ROOT = Path(__file__).resolve().parents[2]
PACK = load_settings().pack
TEMPLATES = PACK.question_templates
CFG = TEMPLATES.counterfactuals
CRIT = {q.id: q.establishing_criteria for q in TEMPLATES.questions}
SNAP = PolicySnapshot(**json.loads((ROOT / "data/fixtures/S-01/policy.json").read_text(encoding="utf-8"))["policy_snapshot"])
QUESTIONS = QuestionRegister(TEMPLATES).build(ClaimHeader(claim_id="C"), SNAP).questions
ISSUES = [CoverageIssue(**i) for i in json.loads((ROOT / "data/fixtures/S-01/coverage.json").read_text(encoding="utf-8"))["coverage_issues"]]
GT = json.loads((ROOT / "data/ground_truth/S-01.json").read_text(encoding="utf-8"))["counterfactuals"]


def assess(qid, status, gap=None, abstained=False):
    crit = tuple(CriterionAssessment(i, c, "unmet", "model", "x") for i, c in enumerate(CRIT[qid]))
    return QuestionAssessment(qid, status, gap, "R", crit, (), (), abstained, "x")


SUFF = SufficiencyResult([assess("Q-SUDDEN", "requires_validation", "contradictory"), assess("Q-START", "missing", "missing"), assess("Q-INFORCE", "established")])
REQUESTS = [MissingEvidence(question_id="Q-SUDDEN", gap_type="contradictory", resolves_question_ids=["Q-SUDDEN", "Q-START"],
                            minimum_evidence_request=MinimumEvidenceRequest(addressee_role="plumber", items=["x"])),
            MissingEvidence(question_id="Q-START", gap_type="missing", resolves_question_ids=["Q-START"],
                            minimum_evidence_request=MinimumEvidenceRequest(addressee_role="claimant", items=["x"]))]


def run(engine=None, suff=SUFF, issues=ISSUES, requests=REQUESTS):
    return (engine or CounterfactualEngine(TEMPLATES)).generate(QUESTIONS, suff, issues, SNAP.clauses, requests)


class FakeWriter:
    def __init__(self, drafts=None, unavailable=False):
        self.drafts, self.unavailable, self.contexts = drafts or [], unavailable, []

    def write(self, ctx):
        self.contexts.append(ctx)
        return WriterOutcome(list(self.drafts), unavailable=self.unavailable, detail="down" if self.unavailable else "")


# ------------------------------------------------------------------------------------------------ the scenarios
def test_s01_gets_scenarios_a_b_c_for_the_cause_question_and_the_issues_the_labelled_table_names():
    r = run()
    assert [(s.scenario_id, s.question_id) for s in r.scenarios] == [("A", "Q-SUDDEN"), ("B", "Q-SUDDEN"), ("C", "Q-SUDDEN")] and r.problems == []
    for s, g in zip(r.scenarios, GT):
        assert (s.scenario_id, s.question_id, s.affected_issue_ids) == (g["scenario_id"], g["question_id"], g["affected_issue_ids"])
    assert [s.assumed_finding for s in r.scenarios] == ["New evidence supports sudden failure", "New evidence supports prolonged leakage", "New evidence is inconclusive"]


def test_each_scenario_says_what_the_finding_would_change_and_the_next_step():
    a, b, c = run().scenarios
    assert "Escape of water: sudden and accidental would move toward established" in a.implication_text and "Gradual leakage exclusion would be re-assessed as not engaged" in a.implication_text
    assert "would not be established" in b.implication_text and "would become potentially applicable" in b.implication_text and "senior human review" in b.implication_text
    assert "would remain unresolved" in c.implication_text
    assert "the plumber" in a.next_step and "re-run coverage and settlement" in a.next_step          # who is outstanding comes from the requests
    assert "senior adjuster" in b.next_step and "no automatic decision" in b.next_step and "independent expert" in c.next_step


def test_only_open_issues_that_share_a_clause_and_a_kind_the_question_re_assesses_are_affected():
    ids = [i.issue_id for i in ISSUES]
    assert ids == ["ISS-INFORCE", "ISS-COVER-WATER", "ISS-EXC-GRADUAL", "ISS-SUBLIMIT"]
    assert run().scenarios[0].affected_issue_ids == ["ISS-COVER-WATER", "ISS-EXC-GRADUAL"]                  # not the established one, not the sublimit (amount)
    with_amount = TEMPLATES.model_copy(update={"counterfactuals": CFG.model_copy(update={"framings": [
        CFG.framings[0].model_copy(update={"issue_kinds": ["cover", "exclusion", "amount"]})]})})
    assert run(CounterfactualEngine(with_amount)).scenarios[0].affected_issue_ids == ["ISS-COVER-WATER", "ISS-EXC-GRADUAL", "ISS-SUBLIMIT"]


def test_nothing_is_produced_for_an_established_or_unframed_question_but_an_abstained_one_still_gets_scenarios():
    assert run(suff=SufficiencyResult([assess("Q-SUDDEN", "established")])).scenarios == []
    assert run(suff=SufficiencyResult([assess("Q-START", "missing", "missing")])).scenarios == []                 # Q-START is not framed
    assert len(run(suff=SufficiencyResult([assess("Q-SUDDEN", "unverified", None, abstained=True)])).scenarios) == 3


def test_a_question_no_open_issue_shares_a_clause_with_is_skipped_with_a_warning():
    r = run(issues=[i for i in ISSUES if i.issue_id in ("ISS-INFORCE", "ISS-SUBLIMIT")])
    assert r.scenarios == [] and r.skipped == ["Q-SUDDEN"] and any("nothing for a scenario" in w for w in r.warnings)


def test_without_requests_the_next_step_still_reads_and_the_contract_holds():
    s = run(requests=[]).scenarios[0]
    assert "the outstanding requests" in s.next_step and isinstance(s, CounterfactualScenario)


# ------------------------------------------------------------------------------------------------ the linter
@pytest.mark.parametrize("text", [
    "The claim will be approved.", "This would be paid in full.", "The claim is not covered.", "We should deny the claim.", "The loss is excluded and payment is denied.",
    "Payment would be approved once the report arrives.", "The claimant is entitled to the full amount.", "It is likely to be rejected.", "We must reject the claim.",
    "Approve the claim and pay out.", "The insurer is liable for the damage.",
])
def test_the_linter_catches_wording_that_predicts_or_recommends_a_decision(text):
    assert CounterfactualEngine(TEMPLATES).lint(text), text


@pytest.mark.parametrize("text", [
    "The gradual-leakage exclusion would become potentially applicable.", "The cover condition would not be established; this needs senior human review.",
    "Cause and start date would remain unresolved.", "Refer to a senior adjuster with the cited clauses; no automatic decision.", "Consider an independent expert inspection.",
])
def test_the_linter_passes_neutral_wording(text):
    assert CounterfactualEngine(TEMPLATES).lint(text) == []


def test_an_implication_must_be_conditional():
    e = CounterfactualEngine(TEMPLATES)
    assert any("not conditional" in p for p in e.lint_implication("The exclusion applies."))
    assert e.lint_implication("The exclusion could apply.") == []


def test_every_rule_built_scenario_of_the_labelled_claim_is_clean_and_the_contract_still_refuses_outcomes():
    for s in run().scenarios:
        e = CounterfactualEngine(TEMPLATES)
        assert e.lint_implication(s.implication_text) == [] and e.lint(s.next_step) == []
    with pytest.raises(ValueError, match="outcome"):
        CounterfactualScenario(scenario_id="A", implication_text="The claim will be approved.", next_step="x", assumed_finding="y")


# ------------------------------------------------------------------------------------------------ the model rewrite
GOOD = {
    "A": ScenarioDraft("A", "If the plumber confirms a sudden failure of the joint, Escape of water: sudden and accidental would move toward established and Gradual leakage exclusion "
                            "would be re-assessed as not engaged.", "Collect the plumber's statement, then re-run coverage and settlement for human review."),
    "B": ScenarioDraft("B", "If the corrosion is shown to be long-standing, Escape of water: sudden and accidental would not be established and Gradual leakage exclusion would become "
                            "potentially applicable; this needs senior human review.", "Refer to a senior adjuster with the cited clauses and the plumber's report."),
}


def rewrite(**over):
    d = GOOD["A"]
    return ScenarioDraft(over.get("scenario_id", "A"), over.get("implication", d.implication_text), over.get("next", d.next_step))


def test_a_clean_claim_specific_rewrite_replaces_the_rule_text_scenario_by_scenario():
    r = run(CounterfactualEngine(TEMPLATES, FakeWriter(list(GOOD.values()))))
    assert r.refused == [] and r.problems == []
    assert r.source == {("Q-SUDDEN", "A"): "model", ("Q-SUDDEN", "B"): "model", ("Q-SUDDEN", "C"): "rule"}
    assert r.scenarios[0].implication_text.startswith("If the plumber confirms")


@pytest.mark.parametrize("over,why", [
    (dict(implication="If checked, the claim would be approved and Escape of water: sudden and accidental would move toward established."), "predicts"),
    (dict(implication="Escape of water: sudden and accidental moves toward established and Gradual leakage exclusion is re-assessed as not engaged."), "not conditional"),
    (dict(implication="x" * 500), "longer"),
    (dict(next="x" * 400), "longer"),
    (dict(next="Then the claim should be paid."), "predicts"),
    (dict(implication="Escape of water: sudden and accidental would move toward established, Gradual leakage exclusion would be re-assessed as not engaged (see ISS-INVENTED)."), "ids"),
    (dict(implication="The cover condition would move toward established."), "does not mention"),
    (dict(implication="Escape of water: sudden and accidental would not be established; Gradual leakage exclusion would become potentially applicable."), "keep the effect"),
    (dict(scenario_id="D"), "not a scenario"),
])
def test_a_bad_rewrite_is_refused_and_the_rule_text_is_kept(over, why):
    r = run(CounterfactualEngine(TEMPLATES, FakeWriter([rewrite(**over)])))
    assert why in r.refused[0].reason and r.problems == []
    assert all(v == "rule" for v in r.source.values())
    assert r.scenarios[0].implication_text.startswith("Escape of water: sudden and accidental would move toward established")


def test_model_outage_keeps_the_rule_scenarios_and_warns():
    r = run(CounterfactualEngine(TEMPLATES, FakeWriter(unavailable=True)))
    assert len(r.scenarios) == 3 and any("unavailable" in w for w in r.warnings) and all(v == "rule" for v in r.source.values())


def test_the_writer_sees_the_issues_effects_clauses_and_facts_and_is_off_when_switched_off():
    w = FakeWriter()
    run(CounterfactualEngine(TEMPLATES, w))
    ctx = w.contexts[0]
    assert [i.issue_id for i in ctx.issues] == ["ISS-COVER-WATER", "ISS-EXC-GRADUAL"] and set(ctx.base) == {"A", "B", "C"}
    assert ctx.effects["ISS-EXC-GRADUAL"]["B"] == "would become potentially applicable" and "CL-EXC-GRADUAL" in ctx.clauses
    off = TEMPLATES.model_copy(update={"counterfactuals": CFG.model_copy(update={"enabled": False})})
    w2 = FakeWriter(list(GOOD.values()))
    run(CounterfactualEngine(off, w2))
    assert w2.contexts == [] and make_engine(PACK, None).writer is None and make_engine(PACK, FakeLLM()).writer is not None


class FakeLLM:
    def __init__(self, answer=None, fail=None):
        self.answer, self.fail, self.prompts = answer or RewriteAnswer(), fail, []

    def structured(self, model_cls, prompt, **kw):
        self.prompts.append(prompt.text)
        if self.fail:
            raise self.fail
        return self.answer


def test_the_prompt_carries_the_scenarios_effects_wording_and_limits():
    llm = FakeLLM(RewriteAnswer(scenarios=[ScenarioModel(scenario_id="a", implication_text=GOOD["A"].implication_text, next_step=GOOD["A"].next_step)]))
    r = run(CounterfactualEngine(TEMPLATES, LLMScenarioWriter(llm, PACK)))
    p = llm.prompts[0]
    for needle in ["Q-SUDDEN", "New evidence supports prolonged leakage", "would become potentially applicable", "ISS-EXC-GRADUAL", "gradual leakage, seepage", "at most 420", "never swap A and B"]:
        assert needle in p, needle
    assert r.source[("Q-SUDDEN", "A")] == "model"                                  # the scenario id was normalised ('a' -> 'A')
    assert run(CounterfactualEngine(TEMPLATES, LLMScenarioWriter(FakeLLM(fail=RuntimeError("down")), PACK))).warnings


# ------------------------------------------------------------------------------------------------ config
def test_counterfactual_config_is_validated():
    base = CFG.model_dump()
    with pytest.raises(ValueError, match="effects"):
        CounterfactualConfig(**{**base, "effects": {**base["effects"], "cover": {"A": "x"}}})
    with pytest.raises(ValueError, match="missing issue kind"):
        CounterfactualConfig(**{**base, "effects": {k: v for k, v in base["effects"].items() if k != "amount"}})
    with pytest.raises(ValueError, match="next_steps"):
        CounterfactualConfig(**{**base, "next_steps": {"A": "x"}})
    with pytest.raises(ValueError, match="pattern"):
        CounterfactualConfig(**{**base, "forbidden_patterns": ["("]})
    assert CFG.enabled and [f.question for f in CFG.framings] == ["Q-SUDDEN"]


def test_an_issue_title_that_reads_like_a_decision_is_replaced_by_its_id_so_the_scenarios_stay_clean():
    bad = [i.model_copy(update={"issue": "Claim likely to be denied: gradual leakage exclusion"}) if i.issue_id == "ISS-EXC-GRADUAL" else i for i in ISSUES]
    r = run(issues=bad)
    assert r.problems == [] and "ISS-EXC-GRADUAL would be re-assessed as not engaged" in r.scenarios[0].implication_text
    assert "denied" not in " ".join(s.implication_text for s in r.scenarios)
