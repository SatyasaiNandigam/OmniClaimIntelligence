"""T4.1 - the recorded model answers replay offline and reproduce the baseline numbers (a prompt or check change shows up here)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from claimtower.core.config import load_settings  # noqa: E402
from claimtower.core.llm import Cassette, LLMClient  # noqa: E402
from claimtower.evidence.llm_questions import make_register  # noqa: E402
from evals.metrics.questions import evaluate, load_cases  # noqa: E402

CASSETTE = ROOT / "data" / "cassettes" / "question_eval.json"


@pytest.fixture(scope="module")
def results():
    settings = load_settings(ROOT)
    llm = LLMClient.from_settings(settings)
    llm.mode, llm.cassette = "replay", Cassette(CASSETTE)
    return {k: t.summary() for k, t in evaluate(make_register(settings.pack, llm), load_cases()).items()}


def test_labelled_cases_cover_both_splits_and_both_kinds_of_label():
    cases = load_cases()
    assert {c.split for c in cases} == {"dev", "fresh"}
    assert any(c.none_needed for c in cases) and any(c.needed for c in cases)
    assert all(bool(c.needed) != c.none_needed for c in cases)


def test_dev_results_replay_offline(results):
    d = results["dev"]
    assert (d["needed_groups"], d["hits"], d["spurious_cases"]) == (3, 3, 0)


def test_unseen_results_are_the_ones_reported(results):
    """Recorded on cases written before the dev results were seen: 2 of 4 needed questions, 1 spurious in 3 'nothing needed' claims (3 of 4 and 2 of 3 in the
    first recording: the model's answers vary a little between recordings, so the sample is a guide, not a measurement)."""
    f = results["fresh"]
    assert (f["needed_groups"], f["hits"], f["none_needed_cases"], f["spurious_cases"]) == (4, 2, 3, 1)
    assert f["unavailable"] == 0


def test_no_accepted_question_is_ungrounded(results):
    assert all(r["unavailable"] == 0 for r in results.values())
