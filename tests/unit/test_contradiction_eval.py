"""T4.3 - the recorded model answers replay offline and reproduce the reported contradiction numbers (a prompt, rule or check change shows up here)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from claimtower.core.config import load_settings  # noqa: E402
from claimtower.core.llm import Cassette, LLMClient  # noqa: E402
from claimtower.evidence.contradiction import ContradictionDetector  # noqa: E402
from claimtower.evidence.llm_contradiction import make_detector  # noqa: E402
from claimtower.evidence.questions import QuestionRegister  # noqa: E402
from evals.metrics.contradictions import evaluate, load_cases  # noqa: E402

SETTINGS = load_settings(ROOT)
CASSETTE = ROOT / "data" / "cassettes" / "contradiction_eval.json"


@pytest.fixture(scope="module")
def cases():
    return load_cases(QuestionRegister(SETTINGS.pack.question_templates))


@pytest.fixture(scope="module")
def llm():
    c = LLMClient.from_settings(SETTINGS)
    c.mode, c.cassette = "replay", Cassette(CASSETTE)
    return c


def summaries(detector, cases):
    return {k: t.summary() for k, t in evaluate(detector, cases).items()}


def test_both_splits_have_positives_and_negatives(cases):
    for split in ("dev", "fresh"):
        in_split = [c for c in cases if c.split == split]
        assert any(c.labels for c in in_split) and any(not c.labels for c in in_split)


def test_rules_alone_find_every_date_and_amount_conflict_but_no_narrative_one(cases):
    r = summaries(ContradictionDetector(SETTINGS.pack.question_templates.contradictions.model_copy(update={"enabled": False})), cases)
    for split in ("dev", "fresh"):
        by = r[split]["by_kind"]
        assert by["amount"]["recall"] == 1.0 and by["cause"]["recall"] == 0.0 and by["extent"]["recall"] == 0.0
        assert r[split]["precision"] == 1.0                                       # a rule never raises a false alarm on these cases


def test_dev_results_replay_offline(cases, llm):
    d = summaries(make_detector(SETTINGS.pack, llm), cases)["dev"]
    assert (d["labelled"], d["predicted"], d["precision"], d["recall"]) == (5, 5, 1.0, 1.0)


def test_unseen_results_are_the_ones_reported(cases, llm):
    """Recorded once on cases written before any result was seen: all 8 labelled contradictions found, no false alarm in 11 cases
    (consistent claims, a hedge, facts about different points, close amounts)."""
    f = summaries(make_detector(SETTINGS.pack, llm), cases)["fresh"]
    assert (f["labelled"], f["predicted"], f["precision"], f["recall"]) == (8, 8, 1.0, 1.0)
    assert f["under_cited"] == 0 and f["refused"] == 0
    assert {k: v["recall"] for k, v in f["by_kind"].items()} == {"date": 1.0, "amount": 1.0, "cause": 1.0, "extent": 1.0}


def test_no_contradiction_is_ever_reported_without_two_cited_facts(cases, llm):
    for t in evaluate(make_detector(SETTINGS.pack, llm), cases).values():
        assert t.under_cited == 0
