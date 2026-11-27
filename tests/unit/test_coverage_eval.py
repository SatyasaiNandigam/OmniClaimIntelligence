"""T5.1 - the recorded model readings replay offline (no live call); the numbers pinned here are the baseline in evals/baselines/coverage.json."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from claimtower.core.config import load_settings  # noqa: E402
from claimtower.core.llm import Cassette, LLMClient  # noqa: E402
from claimtower.coverage.issue_matrix import IssueMatrixBuilder  # noqa: E402
from claimtower.coverage.llm_reading import make_builder  # noqa: E402
from evals.metrics.coverage import evaluate, load_cases  # noqa: E402

SETTINGS = load_settings(ROOT)


@pytest.fixture(scope="module")
def cases():
    return load_cases()


@pytest.fixture(scope="module")
def llm():
    c = LLMClient.from_settings(SETTINGS)
    c.mode, c.cassette = "replay", Cassette(ROOT / "data" / "cassettes" / "coverage_eval.json")
    return c


def test_both_splits_have_supports_against_and_unclear_labels(cases):
    for split in ("dev", "fresh"):
        got = {v for c in cases if c.split == split for v in c.labels.values()}
        assert got == {"supports", "against", "unclear", "ambiguous"}, split


def test_without_a_model_no_row_is_ever_established_from_facts_alone(cases):
    for split, t in evaluate(IssueMatrixBuilder(SETTINGS.pack.coverage, None, 1), cases).items():
        s = t.summary()
        assert s["status_accuracy"] == 0.0 and s["established_against_the_label"] == 0 and s["lint_failures"] == 0 and s["self_check_problems"] == 0, split


def test_the_recorded_readings_replay_to_the_baseline_and_keep_the_guarantees(cases, llm):
    """Recorded (T5.2 v2: the ambiguity question is asked only where an exclusion has an exception clause): dev 35/46 (seen cases, tuned); fresh H-* 9/11,
    run once, then frozen. No row is established against its label; every ambiguity label was found (4/4 dev, 2/2 fresh). False ambiguity: 4 dev, 2 fresh
    (clear BASE-HOME-2 cases still flagged: the exception clause is there). Fresh also exposed a pack gap: BASE-HOME-3's exclusion is not matched (NO_CITATION)."""
    res = {k: t.summary() for k, t in evaluate(make_builder(SETTINGS.pack, llm, 1), cases).items()}
    assert (res["dev"]["rows"], round(res["dev"]["status_accuracy"] * 46)) == (46, 35)
    assert (res["fresh"]["rows"], round(res["fresh"]["status_accuracy"] * 11)) == (11, 9)
    for split, s in res.items():
        assert s["established_against_the_label"] == 0 and s["lint_failures"] == 0 and s["self_check_problems"] == 0, split
        assert s["ambiguity_found"] == s["ambiguity_expected"], split
        assert set(s["abstentions_by_reason"]) <= {"", "AMBIGUOUS_CLAUSE", "READING_ADVERSE", "UNCLEAR_READING", "NO_CITATION"}, split
