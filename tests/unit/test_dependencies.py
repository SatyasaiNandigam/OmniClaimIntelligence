"""T6.2 - the dependency map: a re-assessment re-runs only what new evidence can change, says why, and says what it skipped."""
import json
from pathlib import Path

import pytest

from claimtower.composition import build_registry
from claimtower.core.config import load_settings
from claimtower.core.schemas import ClaimState, Fact
from claimtower.orchestration.dependencies import NODE_INPUTS, changed_fields, downstream, input_digest
from claimtower.orchestration.graph import ClaimRunner
from claimtower.orchestration.registry import PIPELINE

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()
META = {"claim_id", "state_version", "correlation_id", "audit_ref"}


def fixture(sid) -> ClaimState:
    return ClaimState(**json.loads((ROOT / f"data/fixtures/{sid}/claim_state.json").read_text(encoding="utf-8")))


def dump(patch):
    return json.dumps({k: v for k, v in patch.updates.items()}, default=lambda o: o.model_dump(mode="json") if hasattr(o, "model_dump") else str(o), sort_keys=True)


# ---------------------------------------------------------------------------------------------------------------- the map itself
def test_every_node_of_the_pipeline_has_an_entry_and_only_intake_and_evidence_are_never_skipped():
    assert set(NODE_INPUTS) == set(PIPELINE)
    assert {n for n, f in NODE_INPUTS.items() if f is None} == {"intake", "evidence"}
    assert all(f <= set(ClaimState.model_fields) for f in NODE_INPUTS.values() if f is not None)


def test_the_digest_changes_exactly_when_a_declared_input_changes():
    s = fixture("S-02")
    d = input_digest("coverage", s)
    assert set(d) == NODE_INPUTS["coverage"] and input_digest("intake", s) is None
    assert changed_fields(d, input_digest("coverage", s.model_copy(update={"audit_ref": "x"}))) == []                       # not an input of coverage
    other = s.model_copy(update={"facts": [*s.facts, Fact(fact_id="F-NEW", claim_id=s.claim_id, proposition="p", source_document_id="D", page=1, evidence_text="p",
                                                          evidence_type="neutral", extraction_confidence=0.9)]})
    assert changed_fields(d, input_digest("coverage", other)) == ["facts"] and changed_fields(None, d) == sorted(d)
    assert input_digest("settlement", s, {"human_overrides": {"a": "1"}}) != input_digest("settlement", s)
    assert downstream("gap", PIPELINE) == ["coverage", "settlement", "action", "audit", "routing"]


@pytest.mark.parametrize("node", ["gap", "coverage", "settlement", "action", "routing"])
@pytest.mark.parametrize("sid", ["S-01", "S-02"])
def test_the_declared_inputs_are_enough_clearing_any_other_field_never_changes_what_the_node_returns(node, sid):
    """The proof of the declarations: a node whose declared inputs are unchanged returns the same thing, so skipping it is safe."""
    declared = NODE_INPUTS[node]
    state = fixture(sid)
    blank = ClaimState(claim_id=state.claim_id)

    def run(s):
        return dump(build_registry(SETTINGS)[node].agent.run(s))

    base = run(state)
    for field in ClaimState.model_fields:
        if field in declared or field in META:
            continue
        assert run(state.model_copy(update={field: getattr(blank, field)})) == base, f"{node} depends on '{field}': add it to NODE_INPUTS['{node}']"


# ---------------------------------------------------------------------------------------------------------------- the run log
def mock_runner():
    return ClaimRunner(build_registry(SETTINGS, force_mock=("intake", "gap", "coverage", "settlement", "action", "audit", "routing")))


def test_a_reassessment_with_nothing_new_skips_every_node_that_reads_nothing_new_and_says_so():
    r = mock_runner()
    r.start(ClaimState(claim_id="CLM-0002"))
    r.decide("CLM-0002", {"decision": "request_more_evidence", "reviewer_id": "rev-1", "reviewer_role": "senior_adjuster", "reason": "r"})
    st = r.reassess("CLM-0002", from_node="gap")
    skipped = r.events.of_kind("CLM-0002", "node_skipped")
    assert [e.node for e in skipped] == ["coverage", "settlement", "action", "audit", "routing"] == st.skipped_nodes
    assert all(e.message.startswith("skipped (unchanged)") and e.data["inputs"] is not None for e in skipped)
    gap = [e for e in r.events.of_kind("CLM-0002", "agent_finished") if e.node == "gap"][-1]
    assert gap.data["rerun_reason"] == ["coverage_issues"]                                       # gap also reads the coverage issues, which are new since its first run
    assert [e.data["rerun_reason"] for e in r.events.of_kind("CLM-0002", "agent_finished") if e.node == "gap"][0] is None


def test_the_status_reports_what_was_skipped_in_the_latest_assessment_only():
    r = mock_runner()
    r.start(ClaimState(claim_id="CLM-0002"))
    assert r.status("CLM-0002").skipped_nodes == []
    r.decide("CLM-0002", {"decision": "request_more_evidence", "reviewer_id": "rev-1", "reviewer_role": "senior_adjuster", "reason": "r"})
    r.reassess("CLM-0002", from_node="gap")
    assert r.status("CLM-0002").skipped_nodes == ["coverage", "settlement", "action", "audit", "routing"]
    r.decide("CLM-0002", {"decision": "request_more_evidence", "reviewer_id": "rev-1", "reviewer_role": "senior_adjuster", "reason": "again"})
    r.reassess("CLM-0002", from_node="audit")
    assert r.status("CLM-0002").skipped_nodes == ["routing"]                                     # the node asked for always runs; the one after it had nothing new


def test_a_node_that_failed_last_time_is_run_again_not_skipped():
    r = mock_runner()
    sup = r.supervisor
    r.start(ClaimState(claim_id="CLM-0002"))
    assert sup._last_digest("CLM-0002", "coverage") is not None
    r.events.add("CLM-0002", "agent_failed", "coverage", "boom", reason="boom")
    assert sup._last_digest("CLM-0002", "coverage") is None                                       # a failure after the last success: the next assessment runs it again


def test_skipping_can_be_switched_off():
    r = ClaimRunner(build_registry(SETTINGS, force_mock=("intake", "gap", "coverage", "settlement", "action", "audit", "routing")))
    r.supervisor.skip_unchanged = False
    r.start(ClaimState(claim_id="CLM-0002"))
    r.decide("CLM-0002", {"decision": "request_more_evidence", "reviewer_id": "rev-1", "reviewer_role": "senior_adjuster", "reason": "r"})
    r.reassess("CLM-0002", from_node="gap")
    assert r.events.of_kind("CLM-0002", "node_skipped") == [] and [e.node for e in r.events.of_kind("CLM-0002", "agent_finished")][-5:] == ["coverage", "settlement", "action", "audit", "routing"]


def test_the_node_a_person_asks_to_re_run_always_runs_even_when_its_inputs_look_unchanged():
    r = mock_runner()
    r.start(ClaimState(claim_id="CLM-0002"))
    r.decide("CLM-0002", {"decision": "request_more_evidence", "reviewer_id": "rev-1", "reviewer_role": "senior_adjuster", "reason": "r"})
    r.reassess("CLM-0002", from_node="coverage")
    assert "coverage" not in r.status("CLM-0002").skipped_nodes and [e.node for e in r.events.of_kind("CLM-0002", "agent_finished")][-1] == "coverage"
