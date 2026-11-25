"""The generic comparison of two ClaimState versions (used by GET /claims/{id}/diff)."""
import json
from pathlib import Path

from claimtower.core.schemas import ClaimState
from claimtower.core.state_diff import diff_states

ROOT = Path(__file__).resolve().parents[2]


def fixture(sid="S-01") -> ClaimState:
    return ClaimState(**json.loads((ROOT / f"data/fixtures/{sid}/claim_state.json").read_text(encoding="utf-8")))


def test_identical_states_have_no_diff():
    s = fixture()
    d = diff_states(s, s.model_copy())
    assert d.empty and d.status_changes == [] and d.describe() == "no change"


def test_lists_are_compared_by_identifier_and_statuses_are_called_out():
    a = fixture()
    b = a.model_copy(update={
        "facts": [f for f in a.facts if f.fact_id != "F-S01-06"],
        "open_questions": [q.model_copy(update={"status": "established"}) if q.question_id == "Q-START" else q for q in a.open_questions],
        "contradictions": [], "resolution_status": "approved", "state_version": a.state_version + 1})
    d = diff_states(a, b)
    by = {c.field: c for c in d.changes}
    assert by["facts"].removed == ["F-S01-06"] and by["open_questions"].changed == ["Q-START"]
    assert by["contradictions"].removed == ["C-S01-01"]
    assert by["resolution_status"].before == "awaiting_evidence" and by["resolution_status"].after == "approved"
    assert [(s.question_id, s.before, s.after) for s in d.status_changes] == [("Q-START", "missing", "established")]
    assert (d.from_version, d.to_version) == (a.state_version, a.state_version + 1)
    assert "Q-START missing -> established" in d.describe()


def test_a_refetched_policy_snapshot_is_not_a_change_but_a_new_version_is():
    a = fixture()
    same = a.model_copy(update={"policy_snapshot": a.policy_snapshot.model_copy(update={"retrieved_at": None})})
    assert diff_states(a, same).empty
    newer = a.model_copy(update={"policy_snapshot": a.policy_snapshot.model_copy(update={"version": "2026-02"})})
    assert [c.field for c in diff_states(a, newer).changes] == ["policy_snapshot"]
