"""T5.5 - the recorded model readings of the amounts replay offline (no live call); the numbers pinned here are the baseline in evals/baselines/settlement_inputs.json."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from claimtower.agents.settlement import SettlementAgent, make_settlement_agent  # noqa: E402
from claimtower.core.config import load_settings  # noqa: E402
from claimtower.core.llm import Cassette, LLMClient  # noqa: E402
from claimtower.settlement.engine import SettlementCalculator  # noqa: E402
from claimtower.settlement.rules_loader import rules_by_version  # noqa: E402
from evals.metrics.settlement_inputs import evaluate, load_cases  # noqa: E402

SETTINGS = load_settings(ROOT)
CALC = SettlementCalculator(rules_by_version(SETTINGS.pack.calculation))


@pytest.fixture(scope="module")
def cases():
    return load_cases()


@pytest.fixture(scope="module")
def llm():
    c = LLMClient.from_settings(SETTINGS)
    c.mode, c.cassette = "replay", Cassette(ROOT / "data" / "cassettes" / "settlement_inputs_eval.json")
    return c


def test_both_splits_have_cases_with_and_without_amounts(cases):
    for split in ("dev", "fresh"):
        in_split = [c for c in cases if c.split == split]
        assert any(c.assessed for c in in_split) and any(c.assessed is None for c in in_split) and any(c.conflict for c in in_split)


def test_the_rules_alone_never_produce_a_wrong_number(cases):
    res = {k: t.summary() for k, t in evaluate(SettlementAgent(CALC, "sample-1.0"), cases).items()}
    assert all(s["wrong_numbers"] == 0 for s in res.values())
    assert res["dev"]["assessed_correct"] == 1.0 and res["fresh"]["missing_when_labelled"] == 2                     # fresh F-1: wording the rules do not know


def test_the_recorded_model_readings_replay_to_the_baseline(cases, llm):
    """Recorded: dev 6/6 (rules already complete); fresh 7/7 with the model finding F-1 and F-7 values the rules could not (flagged for review); no wrong number."""
    res = {k: t.summary() for k, t in evaluate(make_settlement_agent(SETTINGS.pack, CALC, "sample-1.0", None, None, llm), cases).items()}
    for split, s in res.items():
        assert s["wrong_numbers"] == 0 and s["missing_when_labelled"] == 0, split
        assert s["assessed_correct"] == 1.0 and s["non_covered_correct"] == 1.0, split
    assert res["fresh"]["model_only_values"] == 2
