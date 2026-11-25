"""T4.5 - the recorded model answers replay offline; the structural guarantees of the request generator hold on every labelled case."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from claimtower.core.config import load_settings  # noqa: E402
from claimtower.core.llm import Cassette, LLMClient  # noqa: E402
from claimtower.evidence.llm_requests import make_generator  # noqa: E402
from claimtower.evidence.min_evidence import MinimumEvidenceGenerator  # noqa: E402
from claimtower.evidence.questions import QuestionRegister  # noqa: E402
from evals.metrics.requests import evaluate, load_cases  # noqa: E402

SETTINGS = load_settings(ROOT)
CASSETTE = ROOT / "data" / "cassettes" / "request_eval.json"


@pytest.fixture(scope="module")
def cases():
    return load_cases(QuestionRegister(SETTINGS.pack.question_templates))


@pytest.fixture(scope="module")
def llm():
    c = LLMClient.from_settings(SETTINGS)
    c.mode, c.cassette = "replay", Cassette(CASSETTE)
    return c


def summaries(gen, cases):
    return {k: t.summary() for k, t in evaluate(gen, cases).items()}


def test_all_three_splits_are_labelled_with_requests_and_with_nothing_to_ask(cases):
    assert {c.split for c in cases} == {"dev", "fresh", "fresh2"}
    for split in ("dev", "fresh", "fresh2"):
        assert any(c.expect for c in cases if c.split == split) and any(not c.expect for c in cases if c.split == split)


def test_rule_written_requests_are_complete_and_grouped_on_every_case(cases):
    r = summaries(MinimumEvidenceGenerator(SETTINGS.pack.question_templates), cases)
    for split, s in r.items():
        assert (s["addressee_precision"], s["addressee_recall"], s["question_recall"]) == (1.0, 1.0, 1.0), split
        assert s["self_check_problems"] == 0 and s["duplicates_of_received"] == 0 and s["internal_needs_sent_externally"] == 0 and s["extra_question_pairs"] == 0
        assert s["rule_written_item_share"] == 1.0


def test_model_worded_requests_keep_every_guarantee_and_mostly_write_the_items_themselves(cases, llm):
    r = summaries(make_generator(SETTINGS.pack, llm), cases)
    for split, s in r.items():
        assert (s["addressee_precision"], s["addressee_recall"], s["question_recall"]) == (1.0, 1.0, 1.0), split
        assert s["self_check_problems"] == 0 and s["duplicates_of_received"] == 0 and s["internal_needs_sent_externally"] == 0 and s["extra_question_pairs"] == 0
    assert r["fresh2"]["rule_written_item_share"] <= 0.2 and r["fresh"]["rule_written_item_share"] <= 0.3 and r["dev"]["rule_written_item_share"] <= 0.4


def test_an_abstained_question_is_never_asked_externally(cases, llm):
    assert all(s["abstained_wrongly_asked"] == 0 for s in summaries(make_generator(SETTINGS.pack, llm), cases).values())
