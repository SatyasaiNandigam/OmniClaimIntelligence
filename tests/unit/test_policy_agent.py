"""T3.5 Policy agent: the Agent contract (writes only `policy_snapshot`), the fixture scenarios, and honest behaviour when it cannot run."""
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from claimtower.agents.base import Agent
from claimtower.agents.policy import PolicyAgent, titles_from_facts
from claimtower.core.config import load_settings
from claimtower.core.schemas import ClaimHeader, ClaimState, Fact
from claimtower.core.state import OWNERSHIP, OwnershipViolation, StatePatch, apply_patch
from claimtower.policy.decompose import PolicyDecomposer
from claimtower.policy.relations import RelationExtractor
from claimtower.policy.snapshot import PolicyDocumentMissing, SnapshotBuilder
from evals.metrics.policy_snapshot import MockPas, corpus_library

ROOT = Path(__file__).resolve().parents[2]
STRUCTURE = load_settings().pack.policy_structure
DECOMPOSER, EXTRACTOR = PolicyDecomposer(STRUCTURE), RelationExtractor(STRUCTURE)
LIBRARY, TO_LABEL = corpus_library(DECOMPOSER)
NOW = datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)
SCENARIOS = ("S-01", "S-02", "S-03", "S-04", "S-05")


def agent(library=LIBRARY, clock=lambda: NOW):
    return PolicyAgent(SnapshotBuilder(MockPas(), library, EXTRACTOR, clock=clock))


def fixture_state(sid, **over):
    d = json.loads((ROOT / f"data/fixtures/{sid}/claim_state.json").read_text(encoding="utf-8"))
    d["policy_snapshot"] = None
    return ClaimState.model_validate({**d, **over})


def label_set(snapshot):
    return {TO_LABEL.get(c.clause_id, c.clause_id) for c in snapshot.clauses}


# ------------------------------------------------------------------ the Agent contract
def test_the_agent_declares_its_id_and_owns_exactly_what_the_ownership_map_says():
    assert PolicyAgent.agent_id == "policy" and PolicyAgent.owns == OWNERSHIP["policy"] == frozenset({"policy_snapshot"})
    a: Agent = agent()
    assert callable(a.run)


@pytest.mark.parametrize("sid", SCENARIOS)
def test_a_run_writes_only_policy_snapshot_and_changes_nothing_else(sid):
    before = fixture_state(sid)
    snapshot_before = before.model_dump()
    patch = agent().run(before)
    assert isinstance(patch, StatePatch) and patch.agent_id == "policy" and set(patch.updates) == {"policy_snapshot"}
    assert before.model_dump() == snapshot_before                                   # the input state was not mutated
    after = apply_patch(before, patch)                                              # ownership enforced by the core
    assert after.state_version == before.state_version + 1
    assert {k: v for k, v in after.model_dump().items() if k not in ("policy_snapshot", "state_version")} == \
           {k: v for k, v in before.model_dump().items() if k not in ("policy_snapshot", "state_version")}


def test_writing_any_other_field_would_be_rejected_by_the_core():
    with pytest.raises(OwnershipViolation):
        apply_patch(fixture_state("S-01"), StatePatch(agent_id="policy", updates={"settlement": None}))


# ------------------------------------------------------------------ the five scenarios against the fixtures
@pytest.mark.parametrize("sid", ("S-01", "S-02", "S-04"))
def test_the_snapshot_matches_the_fixture_clause_set_endorsements_and_pas_reference(sid):
    fixture = json.loads((ROOT / f"data/fixtures/{sid}/policy.json").read_text(encoding="utf-8"))["policy_snapshot"]
    got = apply_patch(fixture_state(sid), agent().run(fixture_state(sid))).policy_snapshot
    catalogue = {c["clause_id"] for c in fixture["clauses"]}
    assert label_set(got) & catalogue == catalogue
    assert [e.endorsement_id for e in got.endorsements] == [e["endorsement_id"] for e in fixture["endorsements"]]
    assert (got.policy_id, got.pas_status, got.source_system_response_ref, got.mock) == (
        fixture["policy_id"], fixture["pas_status"], fixture["source_system_response_ref"], True)


def test_s04_cites_the_endorsement_in_force_and_says_the_later_one_is_not_yet_effective():
    got = agent().run(fixture_state("S-04")).updates["policy_snapshot"]
    assert {"CL-SUB-WATER-E1"} <= label_set(got) and "CL-SUB-WATER-E2" not in label_set(got)
    assert any("END-02 is listed but not effective until 2026-02-01" in r for r in got.precedence_rules)


def test_s03_a_lapsed_policy_leaves_no_snapshot_and_the_rationale_carries_the_pas_reason():
    fixture = json.loads((ROOT / "data/fixtures/S-03/get_snapshot_result.json").read_text(encoding="utf-8"))["not_in_force"]
    patch = agent().run(fixture_state("S-03"))
    assert patch.updates == {"policy_snapshot": None} and patch.rationale == f"NOT_IN_FORCE: {fixture['reason']}"
    assert json.loads((ROOT / "data/fixtures/S-03/policy.json").read_text(encoding="utf-8")) == {"policy_snapshot": None}


def test_s05_names_the_missing_endorsement_and_abstains_on_the_clause_it_affects_using_the_schedule_fact():
    a = agent(LIBRARY.without("END-01"))
    state = fixture_state("S-05")
    assert titles_from_facts(state.facts) == {"END-01": "Water Damage Sublimit Endorsement"}
    patch = a.run(state)
    snap = patch.updates["policy_snapshot"]
    assert snap.endorsements == [] and "CL-SUB-WATER-E1" not in label_set(snap)
    assert "listed but wording missing: END-01" in patch.rationale
    assert [TO_LABEL[c] for c in a.last_outcome.effective.abstain_clause_ids] == ["CL-COV-WATER"]


def test_when_the_missing_wording_is_uploaded_the_next_run_gives_the_sublimit_clause():
    state = apply_patch(fixture_state("S-05"), agent(LIBRARY.without("END-01")).run(fixture_state("S-05")))
    assert "CL-SUB-WATER-E1" not in label_set(state.policy_snapshot)
    patch = agent(LIBRARY).run(state)                                              # the library now has END-01
    assert "CL-SUB-WATER-E1" in label_set(patch.updates["policy_snapshot"]) and "unchanged" not in patch.rationale


# ------------------------------------------------------------------ cannot run / errors
@pytest.mark.parametrize("header, missing", [
    (None, "policy number and loss date"),
    (ClaimHeader(claim_id="C", policy_ref="POL-1001"), "loss date"),
    (ClaimHeader(claim_id="C", loss_date=date(2026, 3, 14)), "policy number"),
])
def test_without_a_policy_number_or_loss_date_the_agent_clears_the_snapshot_and_says_why(header, missing):
    stale = apply_patch(fixture_state("S-01"), agent().run(fixture_state("S-01"))).policy_snapshot
    state = fixture_state("S-01", claim_header=header, policy_snapshot=stale)
    patch = agent().run(state)
    assert patch.updates == {"policy_snapshot": None} and patch.rationale.startswith("NOT_RUN") and missing in patch.rationale


def test_an_inexact_policy_number_is_not_in_force_not_a_fuzzy_match():
    state = fixture_state("S-01", claim_header=fixture_state("S-01").claim_header.model_copy(update={"policy_ref": "pol-1001"}))
    patch = agent().run(state)
    assert patch.updates == {"policy_snapshot": None} and patch.rationale.startswith("NOT_IN_FORCE") and "not_found" in patch.rationale


def test_a_missing_base_wording_is_raised_not_swallowed():
    with pytest.raises(PolicyDocumentMissing):
        agent(LIBRARY.without("BASE-HOME-1")).run(fixture_state("S-01"))


def test_a_policy_system_failure_is_raised_not_swallowed():
    class Down:
        def get_policy_version(self, *a):
            raise RuntimeError("PAS timeout")

    with pytest.raises(RuntimeError, match="PAS timeout"):
        PolicyAgent(SnapshotBuilder(Down(), LIBRARY, EXTRACTOR)).run(fixture_state("S-01"))


# ------------------------------------------------------------------ idempotence
def test_rerunning_with_nothing_new_keeps_the_existing_snapshot_and_its_timestamp():
    ticks = iter(NOW + timedelta(minutes=i) for i in range(10))
    a = agent(clock=lambda: next(ticks))
    state = apply_patch(fixture_state("S-04"), a.run(fixture_state("S-04")))
    again = a.run(state)
    assert again.updates["policy_snapshot"] is state.policy_snapshot and again.rationale.startswith("unchanged: ")
    assert state.policy_snapshot.retrieved_at == NOW


def test_a_changed_loss_date_gives_a_new_snapshot():
    state = apply_patch(fixture_state("S-04"), agent().run(fixture_state("S-04")))
    moved = state.model_copy(update={"claim_header": state.claim_header.model_copy(update={"loss_date": date(2026, 3, 1)})})
    new = agent().run(moved).updates["policy_snapshot"]
    assert new != state.policy_snapshot and "CL-SUB-WATER-E2" in label_set(new) and "CL-SUB-WATER-E1" not in label_set(new)


def test_the_agent_is_deterministic():
    assert agent().run(fixture_state("S-04")) == agent().run(fixture_state("S-04"))


# ------------------------------------------------------------------ titles from the schedule
def fact(text):
    return Fact(fact_id="F-1", claim_id="C", proposition="p", source_document_id="D", evidence_text=text, evidence_type="neutral",
                extraction_confidence=0.9)


@pytest.mark.parametrize("text, expected", [
    ("Endorsements attached: END-01 Water Damage Sublimit Endorsement", {"END-01": "Water Damage Sublimit Endorsement"}),
    ("Endorsements attached: END-01 Water Damage Sublimit, END-02 Revised Sublimit Endorsement",
     {"END-01": "Water Damage Sublimit", "END-02": "Revised Sublimit Endorsement"}),
    ("Endorsements: END-03 - Escape of Water Deductible Endorsement; END-04 Unoccupied Property Condition Endorsement",
     {"END-03": "Escape of Water Deductible Endorsement", "END-04": "Unoccupied Property Condition Endorsement"}),
    ("Endorsements attached: END-01", {}),
    ("Date of loss: 14 March 2026", {}),
])
def test_titles_are_read_from_the_schedule_text(text, expected):
    assert titles_from_facts([fact(text)]) == expected


def test_the_policy_agent_imports_nothing_it_should_not():
    import re

    src = (ROOT / "src/claimtower/agents/policy.py").read_text(encoding="utf-8")
    for imp in re.findall(r"^(?:from|import) (claimtower[\w.]*)", src, re.MULTILINE):
        assert imp.startswith(("claimtower.core", "claimtower.policy")), imp
