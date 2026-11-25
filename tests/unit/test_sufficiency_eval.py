"""T4.4 - the recorded model answers replay offline and reproduce the reported sufficiency numbers (a prompt, table or check change shows up here)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from claimtower.core.config import load_settings  # noqa: E402
from claimtower.core.llm import Cassette, LLMClient  # noqa: E402
from claimtower.evidence.llm_sufficiency import make_classifier  # noqa: E402
from claimtower.evidence.questions import QuestionRegister  # noqa: E402
from claimtower.evidence.sufficiency import SufficiencyClassifier  # noqa: E402
from evals.metrics.sufficiency import evaluate, load_cases  # noqa: E402

SETTINGS = load_settings(ROOT)
TEMPLATES = SETTINGS.pack.question_templates
MODES = {q.id: q.criteria_mode for q in TEMPLATES.questions}
CASSETTE = ROOT / "data" / "cassettes" / "sufficiency_eval.json"


@pytest.fixture(scope="module")
def cases():
    return load_cases(QuestionRegister(TEMPLATES))


@pytest.fixture(scope="module")
def llm():
    c = LLMClient.from_settings(SETTINGS)
    c.mode, c.cassette = "replay", Cassette(CASSETTE)
    return c


def summaries(clf, cases):
    return {k: t.summary() for k, t in evaluate(clf, cases, MODES).items()}


def test_start_date_criteria_are_alternatives_and_the_others_are_all_required():
    assert MODES["Q-START"] == "any" and all(v == "all" for k, v in MODES.items() if k != "Q-START")


def test_rules_alone_never_claim_established_without_a_model_for_fact_questions(cases):
    r = summaries(SufficiencyClassifier(TEMPLATES.sufficiency), cases)
    assert r["dev"]["over_claims"] == 0 and r["fresh"]["over_claims"] == 0
    assert r["dev"]["abstained"] > 0 and r["fresh"]["abstained"] > 0                    # without a model the fact questions abstain, they are not guessed


def test_dev_results_replay_offline(cases, llm):
    d = summaries(make_classifier(SETTINGS.pack, llm), cases)["dev"]
    assert d["over_claims"] == 0 and d["abstained"] == 0 and round(d["status_accuracy"], 2) == 0.76 and round(d["gap_accuracy"], 2) == 0.79


def test_unseen_results_are_the_ones_reported(cases, llm):
    """Recorded on cases written before any result was seen, after the criteria were relaxed (Q-P10: 'dated' and 'itemised' dropped): 13 of
    15 questions get the labelled status, 14 of 15 the labelled gap type, none is called established when it is not. Before the change
    it was 11 and 12 of 15."""
    f = summaries(make_classifier(SETTINGS.pack, llm), cases)["fresh"]
    assert (f["questions_scored"], f["over_claims"], f["abstained"], f["refused_judgements"]) == (15, 0, 0, 0)
    assert round(f["status_accuracy"], 2) == 0.87 and round(f["gap_accuracy"], 2) == 0.93


def test_every_status_traces_to_a_decision_table_row_and_its_criteria(cases, llm):
    clf = make_classifier(SETTINGS.pack, llm)
    rows = {r.id for r in TEMPLATES.sufficiency.table}
    for c in cases[:5]:
        for a in clf.classify(c.questions, c.facts, c.links, c.contradictions, c.snapshot, c.documents, MODES).assessments:
            assert a.row_id in rows and len(a.criteria) == len(next(q for q in c.questions if q.question_id == a.question_id).establishing_criteria)
