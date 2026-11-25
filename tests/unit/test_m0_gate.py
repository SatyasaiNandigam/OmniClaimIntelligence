"""M0 gate (BUILD_PLAN section 8): contracts frozen, protocols exported, 5 scenarios + ground truth ready, fixtures available."""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from claimtower.core.schemas import ClaimState
from claimtower.core.state import OWNERSHIP, StatePatch, apply_patch
from evals.ground_truth import load_all_ground_truth

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "docs/contracts"
FIXTURES = ROOT / "data/fixtures"
IDS = ["S-01", "S-02", "S-03", "S-04", "S-05"]
PIPELINE = ["intake", "policy", "evidence", "gap", "coverage", "settlement", "action", "audit", "supervisor"]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(*args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=ROOT)


# ------------------------------------------------------------------ contracts
def test_contracts_match_the_freeze():
    r = run("scripts/freeze_contracts.py", "--check")
    assert r.returncode == 0, r.stdout + r.stderr


def test_protocols_and_ownership_are_exported():
    protocols = json.loads((CONTRACTS / "protocols.json").read_text(encoding="utf-8"))
    assert set(protocols["PASClient"]["methods"]) == {"get_policy_version", "get_status"}
    assert set(protocols["ClaimsClient"]["methods"]) == {"create_claim", "update_status", "create_task", "get_claim"}
    assert set(protocols["ReconClient"]["methods"]) == {"submit_payment_instruction", "get_match_status", "list_breaks"}
    assert set(protocols["Agent"]["attributes"]) == {"agent_id", "owns"} and "run" in protocols["Agent"]["methods"]
    assert {"extract_facts"} == set(protocols["FactExtractor"]["methods"])
    assert {"get_snapshot", "retrieve_clauses"} == set(protocols["PolicyService"]["methods"])
    assert {"compute_settlement"} == set(protocols["SettlementEngine"]["methods"]) and {"route"} == set(protocols["Router"]["methods"])
    own = json.loads((CONTRACTS / "ownership.json").read_text(encoding="utf-8"))
    assert set(own["agents"]) == set(OWNERSHIP) and own["system_fields"] == ["claim_id", "correlation_id", "state_version"]
    for name in ("ClaimState", "StatePatch", "PaymentInstruction", "ReconAck", "SettlementInputs"):
        assert (CONTRACTS / f"{name}.schema.json").is_file(), name


def test_writes_carry_idempotency_keys_and_responses_are_labelled_mock():
    for name in ("ClaimCreateRequest", "ClaimStatusUpdate", "TaskRequest", "PaymentInstruction", "ReconAck"):
        schema = json.loads((CONTRACTS / f"{name}.schema.json").read_text(encoding="utf-8"))
        assert "idempotency_key" in schema["properties"], name
    for name in ("PolicyVersionResponse", "ClaimAck", "ReconAck", "MatchStatus"):
        schema = json.loads((CONTRACTS / f"{name}.schema.json").read_text(encoding="utf-8"))
        assert schema["properties"]["mock"]["default"] is True, name


def test_freeze_blocks_unapproved_change_and_approval_is_recorded(tmp_path, monkeypatch):
    fc = script("freeze_contracts")
    shutil.copytree(CONTRACTS, tmp_path / "contracts")
    monkeypatch.setattr(fc, "DIR", tmp_path / "contracts")
    monkeypatch.setattr(fc, "LOCK", tmp_path / "contracts/CONTRACT_LOCK.json")
    monkeypatch.setattr(fc, "CHANGELOG", tmp_path / "contracts/CHANGELOG.md")
    monkeypatch.setattr(fc.subprocess, "run", lambda *a, **k: None)         # never regenerate the real files from a test
    assert fc.check() == 0

    victim = tmp_path / "contracts/Fact.schema.json"
    victim.write_text(victim.read_text(encoding="utf-8").replace('"Fact"', '"FactRenamed"'), encoding="utf-8")
    assert fc.check() == 1                                                  # silent change is blocked

    assert fc.approve("rename for test", "tester", "minor") == 0            # deliberate approval
    assert fc.check() == 0
    lock = json.loads((tmp_path / "contracts/CONTRACT_LOCK.json").read_text(encoding="utf-8"))
    assert lock["contract_version"] == "1.1.0" and lock["approved_by"] == "tester"
    log = (tmp_path / "contracts/CHANGELOG.md").read_text(encoding="utf-8")
    assert "v1.1.0" in log and "rename for test" in log and "Fact.schema.json" in log and "v1.0.0" in log


# ------------------------------------------------------------------ scenarios + fixtures
def test_five_scenarios_have_ground_truth_and_fixtures():
    assert sorted(load_all_ground_truth()) == IDS
    for sid in IDS:
        assert (FIXTURES / sid / "claim_state.json").is_file(), sid
        assert {p.stem for p in (FIXTURES / sid).glob("*.json")} >= {"claim_state", *PIPELINE}, sid


def test_fixtures_are_up_to_date_with_ground_truth():
    r = run("scripts/build_fixtures.py", "--check")
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("sid", IDS)
def test_fixtures_are_valid_and_respect_agent_ownership(sid):
    expected = ClaimState.model_validate_json((FIXTURES / sid / "claim_state.json").read_text(encoding="utf-8"))
    state = ClaimState(claim_id=expected.claim_id, correlation_id=expected.correlation_id)
    for agent in PIPELINE:
        updates = json.loads((FIXTURES / sid / f"{agent}.json").read_text(encoding="utf-8"))
        assert set(updates) == OWNERSHIP[agent], f"{sid}/{agent}.json must hold exactly the fields {agent} owns"
        state = apply_patch(state, StatePatch(agent_id=agent, updates=updates))     # ownership + validation
    assert state == expected, "applying the agent slices in order must reproduce claim_state.json"
    assert state.state_version == len(PIPELINE)


def test_fixtures_encode_the_headline_behaviours():
    def state(sid):
        return ClaimState.model_validate_json((FIXTURES / sid / "claim_state.json").read_text(encoding="utf-8"))

    s1, s3, s4, s5 = state("S-01"), state("S-03"), state("S-04"), state("S-05")
    assert {m.gap_type for m in s1.missing_evidence} == {"contradictory", "missing", "insufficient"} and s1.routing.tier == "complex"
    assert s1.settlement.status == "cannot_calculate" and s1.resolution_status == "awaiting_evidence"
    assert s3.policy_snapshot is None and s3.settlement is None
    assert json.loads((FIXTURES / "S-03/get_snapshot_result.json").read_text(encoding="utf-8"))["not_in_force"]["policy_id"] == "POL-1003"
    assert str(s4.settlement.payable) == "100000" and [e.endorsement_id for e in s4.policy_snapshot.endorsements] == ["END-01"]
    assert s5.settlement.missing_inputs == ["water_damage_sublimit"]
    assert all(s.human_approval_required for s in map(state, IDS))


def test_adr_is_accepted():
    assert "Status:** Accepted" in (ROOT / "docs/adr/0001-framework.md").read_text(encoding="utf-8")
