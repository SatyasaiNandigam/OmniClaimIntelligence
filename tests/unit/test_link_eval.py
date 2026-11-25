"""T4.2 - the recorded model answers replay offline and reproduce the reported linker numbers (a prompt or check change shows up here)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from claimtower.core.config import load_settings  # noqa: E402
from claimtower.core.llm import Cassette, LLMClient  # noqa: E402
from claimtower.evidence.linker import FactQuestionLinker  # noqa: E402
from claimtower.evidence.llm_linking import make_linker  # noqa: E402
from claimtower.evidence.questions import QuestionRegister  # noqa: E402
from evals.metrics.links import evaluate, load_items  # noqa: E402

SETTINGS = load_settings(ROOT)
CASSETTE = ROOT / "data" / "cassettes" / "link_eval.json"


@pytest.fixture(scope="module")
def items():
    return load_items(QuestionRegister(SETTINGS.pack.question_templates))


@pytest.fixture(scope="module")
def llm():
    c = LLMClient.from_settings(SETTINGS)
    c.mode, c.cassette = "replay", Cassette(CASSETTE)
    return c


def test_both_splits_are_labelled(items):
    assert {i.split for i in items} == {"dev", "fresh"} and all(i.labels for i in items if i.split == "fresh")


def test_tags_alone_never_judge_a_direction(items):
    r = {k: t.summary() for k, t in evaluate(FactQuestionLinker(SETTINGS.pack.question_templates.fact_links), items).items()}
    assert r["dev"]["link_recall"] == 1.0 and r["fresh"]["link_recall"] < 0.75     # the tags miss links on unseen cases
    assert r["dev"]["direction_accuracy"] < 0.5 and r["fresh"]["direction_accuracy"] < 0.5


def test_dev_results_replay_offline(items, llm):
    d = {k: t.summary() for k, t in evaluate(make_linker(SETTINGS.pack, llm), items).items()}["dev"]
    assert d["link_precision"] == 1.0 and d["link_recall"] == 1.0 and d["direction_accuracy"] >= 0.85


def test_unseen_results_are_the_ones_reported(items, llm):
    """Recorded on cases written before any result was seen (after the criteria wording was relaxed, which changes the prompt and so the
    recording): recall 94% (tags alone 69%), direction 82% (tags alone 48%), one contradicting link read as supporting (none in the first
    recording, 88%). Precision stays 81% because the model never dropped a wrong tag."""
    f = {k: t.summary() for k, t in evaluate(make_linker(SETTINGS.pack, llm), items).items()}["fresh"]
    assert (f["labelled_links"], f["predicted_links"], f["correct_links"]) == (36, 42, 34)
    assert f["critical_errors"] == 1 and f["dropped_links"] == 0 and f["refused_judgements"] == 0
    assert round(f["direction_accuracy"], 2) == 0.82


def test_without_the_tags_the_model_still_finds_most_links(items, llm):
    f = {k: t.summary() for k, t in evaluate(make_linker(SETTINGS.pack, llm), items, tags="none").items()}["fresh"]
    assert f["link_recall"] >= 0.85
