import pytest
from pydantic import ValidationError

from claimtower.core.schemas import ClaimState, OpenQuestion
from claimtower.core.state import (
    OWNERSHIP, InMemoryVersionStore, OwnershipViolation, StatePatch, VersionNotFound,
    apply_patch, commit_patch,
)


def q(i):
    return OpenQuestion(question_id=i, text=i, category="cause")


def test_ownership_map_covers_every_state_field_exactly_once():
    owned = [f for fields in OWNERSHIP.values() for f in fields]
    assert len(owned) == len(set(owned)), "a field is owned by two agents"
    shared = {"claim_id", "state_version", "correlation_id"}
    assert set(owned) == set(ClaimState.model_fields) - shared


def test_foreign_field_raises():
    with pytest.raises(OwnershipViolation):
        apply_patch(ClaimState(claim_id="c"), StatePatch(agent_id="gap", updates={"settlement": None}))


@pytest.mark.parametrize("field", ["state_version", "claim_id", "not_a_field"])
def test_protected_and_unknown_fields_raise(field):
    with pytest.raises(OwnershipViolation):
        apply_patch(ClaimState(claim_id="c"), StatePatch(agent_id="supervisor", updates={field: 9}))


def test_unknown_agent_raises():
    with pytest.raises(OwnershipViolation):
        apply_patch(ClaimState(claim_id="c"), StatePatch(agent_id="rogue", updates={}))


def test_patch_bumps_version_and_does_not_mutate_input():
    s0 = ClaimState(claim_id="c")
    s1 = apply_patch(s0, StatePatch(agent_id="gap", updates={"open_questions": [q("Q1")]}))
    assert (s0.state_version, s1.state_version) == (0, 1)
    assert s0.open_questions == [] and len(s1.open_questions) == 1


def test_invalid_patch_data_is_rejected():
    with pytest.raises(ValidationError):
        apply_patch(ClaimState(claim_id="c"), StatePatch(agent_id="supervisor", updates={"resolution_status": "bogus"}))


def test_history_replay_and_diff():
    store, s = InMemoryVersionStore(), ClaimState(claim_id="c")
    s = commit_patch(s, StatePatch(agent_id="gap", updates={"open_questions": [q("Q1")]}, rationale="init"), store)
    s = commit_patch(s, StatePatch(agent_id="gap", updates={"open_questions": [q("Q1"), q("Q2")]}), store)
    s = commit_patch(s, StatePatch(agent_id="supervisor", updates={"resolution_status": "assessing"}), store)

    assert store.versions("c") == [0, 1, 2, 3]
    assert [x.question_id for x in store.replay("c", 1).open_questions] == ["Q1"]
    assert [x.question_id for x in store.replay("c", 2).open_questions] == ["Q1", "Q2"]
    assert store.replay("c", 3) == s
    assert store.replay("c", 0).resolution_status == "registered"
    assert set(store.diff("c", 1, 3)) == {"open_questions", "resolution_status"}
    assert [e.state_version_after for e in store.audit_events("c")] == [1, 2, 3]
    with pytest.raises(VersionNotFound):
        store.replay("c", 7)


def test_failed_patch_leaves_no_history():
    store, s = InMemoryVersionStore(), ClaimState(claim_id="c")
    with pytest.raises(OwnershipViolation):
        commit_patch(s, StatePatch(agent_id="gap", updates={"settlement": None}), store)
    with pytest.raises(VersionNotFound):
        store.versions("c")
