"""T4.6 - the recorded model rewrites replay offline; the structural and wording guarantees hold on every labelled case."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from claimtower.core.config import load_settings  # noqa: E402
from claimtower.core.llm import Cassette, LLMClient  # noqa: E402
from claimtower.evidence.counterfactual import CounterfactualEngine  # noqa: E402
from claimtower.evidence.llm_counterfactual import make_engine  # noqa: E402
from claimtower.evidence.questions import QuestionRegister  # noqa: E402
from evals.metrics.counterfactuals import evaluate, load_cases  # noqa: E402

SETTINGS = load_settings(ROOT)
CASSETTE = ROOT / "data" / "cassettes" / "counterfactual_eval.json"


@pytest.fixture(scope="module")
def cases():
    return load_cases(QuestionRegister(SETTINGS.pack.question_templates))


@pytest.fixture(scope="module")
def llm():
    c = LLMClient.from_settings(SETTINGS)
    c.mode, c.cassette = "replay", Cassette(CASSETTE)
    return c


def summaries(engine, cases):
    return {k: t.summary() for k, t in evaluate(engine, cases).items()}


def test_both_splits_have_cases_with_and_without_scenarios(cases):
    for split in ("dev", "fresh"):
        in_split = [c for c in cases if c.split == split]
        assert any(c.expect for c in in_split) and any(not c.expect for c in in_split)


def test_rule_built_scenarios_are_complete_right_and_clean_on_every_case(cases):
    for split, s in summaries(CounterfactualEngine(SETTINGS.pack.question_templates), cases).items():
        assert (s["complete_a_b_c"], s["affected_issues_exact"], s["scenarios_where_none_expected"], s["lint_failures"], s["self_check_problems"]) == (1.0, 1.0, 0, 0, 0), split
        assert s["model_text_share"] == 0.0


def test_model_rewritten_scenarios_keep_every_guarantee_and_are_mostly_the_models_text(cases, llm):
    """Recorded: dev 4 cases, unseen 7 cases (one with an issue title that reads like a decision): every rewrite was accepted, none refused."""
    for split, s in summaries(make_engine(SETTINGS.pack, llm), cases).items():
        assert (s["complete_a_b_c"], s["affected_issues_exact"], s["scenarios_where_none_expected"], s["lint_failures"], s["self_check_problems"]) == (1.0, 1.0, 0, 0, 0), split
        assert s["model_text_share"] >= 0.9 and s["refused_rewrites"] == {}, split


def test_no_scenario_text_of_any_case_contains_decision_wording(cases, llm):
    eng = make_engine(SETTINGS.pack, llm)
    for c in cases:
        for s in eng.generate(c.questions, c.sufficiency, c.issues, c.clauses, c.requests, c.facts, c.links).scenarios:
            assert eng.lint_implication(s.implication_text) == [] and eng.lint(s.next_step) == [], (c.id, s.scenario_id)
            assert "denied" not in s.implication_text.lower() and "approved" not in s.implication_text.lower()
