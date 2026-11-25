"""T4.7 - the real pipeline on late evidence for S-01, replayed offline from recorded model answers: only the questions the new document touches are re-run."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from claimtower.agents.gap import make_gap_agent  # noqa: E402
from claimtower.core.config import load_settings  # noqa: E402
from claimtower.core.llm import Cassette, LLMClient  # noqa: E402
from claimtower.core.schemas import ClaimState  # noqa: E402
from evals.runners.eval_gap_agent import SCENARIOS, add_evidence, step  # noqa: E402

SETTINGS = load_settings(ROOT)


@pytest.fixture(scope="module")
def runs():
    llm = LLMClient.from_settings(SETTINGS)
    llm.mode, llm.cassette = "replay", Cassette(ROOT / "data" / "cassettes" / "gap_agent.json")
    base = ClaimState(**json.loads((ROOT / "data/fixtures/S-01/claim_state.json").read_text(encoding="utf-8")))
    out = {}
    for sc in SCENARIOS:
        agent = make_gap_agent(SETTINGS.pack, llm)
        s1 = step(agent, base)
        n0 = len(llm.calls)
        s2 = step(agent, add_evidence(s1, sc))
        out[sc[0]] = (sc, s1, s2, agent.last_run, len(llm.calls) - n0)
    return out


def test_every_question_the_document_touches_is_re_run_and_no_other_is(runs):
    for sid, (sc, s1, s2, run, _) in runs.items():
        assert not run.cold and sc[3] <= run.affected, sid                             # at least the labelled questions (the model may find another link)
        before, after = {q.question_id: q for q in s1.open_questions}, {q.question_id: q for q in s2.open_questions}
        assert all(before[q] == after[q] for q in before if q not in run.affected), sid


def test_the_two_plumber_reports_re_run_exactly_the_cause_and_start_questions(runs):
    for sid in ("L1-plumber-explains-the-corrosion", "L2-plumber-confirms-long-term-leak"):
        assert runs[sid][3].affected == {"Q-SUDDEN", "Q-START"}


def test_unaffected_contradictions_and_requests_are_carried_over_unchanged(runs):
    for sid, (_, s1, s2, run, _) in runs.items():
        assert all(c in s2.contradictions for c in s1.contradictions if c.question_id not in run.affected), sid
        old = {m.minimum_evidence_request.addressee_role: m for m in s1.missing_evidence}
        assert all(old[r] in s2.missing_evidence for r in run.kept_requests), sid
        assert run.problems == [], sid


def test_a_conflict_is_only_resolved_with_cited_evidence_and_stays_open_otherwise(runs):
    _, s1, s2, run, _ = runs["L2-plumber-confirms-long-term-leak"]
    assert {c.contradiction_id for c in s1.contradictions} <= {c.contradiction_id for c in s2.contradictions}              # the cause conflict is still open
    assert next(q for q in s2.open_questions if q.question_id == "Q-SUDDEN").status == "requires_validation"
    _, s1, s2, run, _ = runs["L1-plumber-explains-the-corrosion"]
    assert run.retracted_contradictions and "C-" in run.retracted_contradictions[0]                                         # resolved, with the explanation recorded


def test_the_incremental_run_asks_the_model_less_than_a_full_one(runs):
    assert all(calls <= 12 for *_, calls in runs.values())
