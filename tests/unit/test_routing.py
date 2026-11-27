"""T5.6 - the tiering engine: the rules file decides, the features come from the claim state, extraction confidence is not an input, and the five labelled scenarios agree."""
import copy
import json
from pathlib import Path

import pytest

from claimtower.agents.routing import RoutingAgent
from claimtower.composition import build_registry
from claimtower.core.config import RoutingRules, load_settings
from claimtower.core.schemas import ClaimState, Contradiction, CoverageIssue, MissingEvidence, MinimumEvidenceRequest
from claimtower.core.state import OwnershipViolation, StatePatch, apply_patch
from claimtower.routing.features import derive_features
from claimtower.routing.tiering import DEFAULT_CODE, TieringEngine, holds
from evals.ground_truth import load_all_ground_truth

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()
ENGINE = TieringEngine(SETTINGS.routing)
GT = load_all_ground_truth()
AMOUNT_ISSUES = [i.id for i in SETTINGS.pack.coverage.issues if i.kind == "amount"]


def fixture(sid) -> ClaimState:
    return ClaimState(**json.loads((ROOT / f"data/fixtures/{sid}/claim_state.json").read_text(encoding="utf-8")))


def agent() -> RoutingAgent:
    return RoutingAgent(ENGINE, amount_issue_ids=AMOUNT_ISSUES)


# ---------------------------------------------------------------------------------------------------------------- the engine
@pytest.mark.parametrize("sid", sorted(GT))
def test_the_engine_reproduces_the_expert_tier_reason_codes_and_approver_from_the_labelled_features(sid):
    want = GT[sid].routing
    got = ENGINE.route(GT[sid].routing_features)
    assert (got.tier, got.reason_codes, got.required_approver_role, got.rules_version) == (want.tier, want.reason_codes, want.required_approver_role, want.rules_version)


@pytest.mark.parametrize("sid", sorted(GT))
def test_features_derived_from_the_claim_state_give_the_same_routing_as_the_expert(sid):
    """The tier agreement on the labelled scenarios: derived from the fixture state, not from the labelled features."""
    p = agent().run(fixture(sid))
    assert p.updates["routing"].model_dump() == GT[sid].routing.model_dump(), sid


def base() -> dict:
    return dict(GT["S-02"].routing_features)


def test_mandatory_referral_overrides_everything_else():
    f = base() | {"policy_status": "not_in_force"}
    d = ENGINE.route(f)
    assert d.tier == "complex" and d.reason_codes == ["POLICY_NOT_IN_FORCE"] and d.required_approver_role == "senior_adjuster"
    assert ENGINE.route(base() | {"specialist_required": True}).reason_codes == ["SPECIALIST_REQUIRED"]


def test_each_escalation_trigger_makes_a_claim_complex_with_its_own_code():
    assert ENGINE.route(base() | {"contradiction_severity": "high"}).reason_codes == ["CONTRADICTION_HIGH"]
    assert ENGINE.route(base() | {"evidence_completeness": "major_gap"}).reason_codes == ["EVIDENCE_GAP_MATERIAL"]
    assert ENGINE.route(base() | {"coverage_certainty": "unresolved"}).reason_codes == ["COVERAGE_UNRESOLVED"]
    assert ENGINE.route(base() | {"exposure_inr": 500001}).reason_codes == ["EXPOSURE_HIGH"]


def test_the_medium_band_and_the_default():
    assert ENGINE.route(base() | {"exposure_inr": 100001}).tier == "medium"
    assert ENGINE.route(base() | {"exposure_inr": 100000}).tier == "simple"
    odd = base() | {"coverage_certainty": "not_assessed"}                        # not complex, not simple, no medium trigger: the default
    d = ENGINE.route(odd)
    assert d.tier == "complex" and d.reason_codes == [DEFAULT_CODE]


def test_an_unknown_feature_value_never_satisfies_a_condition_so_an_unknown_exposure_cannot_be_simple():
    d = ENGINE.route(base() | {"exposure_inr": None})
    assert d.tier != "simple" and d.reason_codes == [DEFAULT_CODE]
    assert holds(SETTINGS.routing.tiers.simple.all_of[0], {}) is False


def test_changing_the_yaml_changes_the_routing():
    raw = copy.deepcopy(SETTINGS.routing.model_dump(by_alias=True))
    for c in raw["tiers"]["simple"]["all_of"]:
        if c["feature"] == "exposure_inr":
            c["lte"] = 50000
    stricter = TieringEngine(RoutingRules.model_validate(raw))
    assert ENGINE.route(base()).tier == "simple" and stricter.route(base()).tier == "complex"        # 98,000 is no longer within the front-line ceiling
    raw["approver_roles"]["complex"] = "supervisor"
    assert TieringEngine(RoutingRules.model_validate(raw)).route(base() | {"exposure_inr": 600000}).required_approver_role == "supervisor"


def test_extraction_confidence_is_not_a_routing_feature_and_changes_nothing():
    raw = copy.deepcopy(SETTINGS.routing.model_dump(by_alias=True))
    raw["features"]["extraction_confidence"] = "number"
    raw["tiers"]["medium"]["any_of"].append({"feature": "extraction_confidence", "lt": 0.5, "code": "LOW_CONFIDENCE"})
    with pytest.raises(ValueError, match="extraction confidence"):
        RoutingRules.model_validate(raw)
    s = fixture("S-02")
    low = s.model_copy(update={"facts": [f.model_copy(update={"extraction_confidence": 0.1}) for f in s.facts]})
    assert agent().run(low).updates["routing"] == agent().run(s).updates["routing"]


# ---------------------------------------------------------------------------------------------------------------- the features
def test_policy_features():
    s = fixture("S-03")
    assert derive_features(s)[0]["policy_status"] == "not_in_force" and derive_features(s)[0]["coverage_certainty"] == "not_assessed"
    no_ref = s.model_copy(update={"claim_header": s.claim_header.model_copy(update={"policy_ref": None})})
    assert derive_features(no_ref)[0]["policy_match"] == "mismatch" and ENGINE.route(derive_features(no_ref)[0]).reason_codes[0] in ("POLICY_NOT_IN_FORCE", "POLICY_MISMATCH")
    assert derive_features(fixture("S-05"))[0]["policy_certainty"] == "partial" and derive_features(fixture("S-02"))[0]["policy_certainty"] == "exact"


def test_evidence_completeness_separates_a_material_gap_from_a_minor_one():
    assert derive_features(fixture("S-01"))[0]["evidence_completeness"] == "major_gap"            # the cause is contested
    assert derive_features(fixture("S-05"))[0]["evidence_completeness"] == "minor_gap"            # only the sublimit wording
    assert derive_features(fixture("S-02"))[0]["evidence_completeness"] == "complete"
    s = fixture("S-02")
    qs = [q.model_copy(update={"status": "unverified"}) if q.question_id == "Q-SUDDEN" else q for q in s.open_questions]
    assert derive_features(s.model_copy(update={"open_questions": qs}))[0]["evidence_completeness"] == "major_gap"        # not yet judged is not complete


def test_contradiction_severity_is_the_highest_recorded():
    s = fixture("S-02")
    mk = lambda i, sev: Contradiction(contradiction_id=f"C{i}", question_id="Q-SUDDEN", fact_ids_a=["a"], fact_ids_b=["b"], severity=sev, explanation="x")      # noqa: E731
    assert derive_features(s.model_copy(update={"contradictions": [mk(1, "low"), mk(2, "medium")]}))[0]["contradiction_severity"] == "medium"
    assert derive_features(s.model_copy(update={"contradictions": [mk(1, "low"), mk(2, "high")]}))[0]["contradiction_severity"] == "high"
    assert derive_features(s)[0]["contradiction_severity"] == "none"


def test_coverage_certainty_is_resolved_only_when_every_non_amount_issue_is_established():
    s = fixture("S-02")
    assert derive_features(s, amount_issue_ids=AMOUNT_ISSUES)[0]["coverage_certainty"] == "resolved"
    weaker = [i.model_copy(update={"status": "identified"}) if i.issue_id == "ISS-COVER-WATER" else i for i in s.coverage_issues]
    assert derive_features(s.model_copy(update={"coverage_issues": weaker}), amount_issue_ids=AMOUNT_ISSUES)[0]["coverage_certainty"] == "unresolved"
    only_amount = [CoverageIssue(issue_id="ISS-SUBLIMIT", issue="x", status="abstain", clause_citations=[])]
    assert derive_features(s.model_copy(update={"coverage_issues": only_amount}), amount_issue_ids=AMOUNT_ISSUES)[0]["coverage_certainty"] == "not_assessed"


def test_exposure_is_the_assessed_loss_then_the_largest_amount_fact_then_unknown():
    assert derive_features(fixture("S-02"))[0]["exposure_inr"] == 98000.0
    s = fixture("S-01")                                                                           # no assessed loss: the contractor estimate
    assert derive_features(s)[0]["exposure_inr"] == 180000.0
    none = s.model_copy(update={"facts": [], "settlement": None})
    f, assumed = derive_features(none)
    assert f["exposure_inr"] is None and any("exposure_inr" in a for a in assumed)


def test_the_assumed_features_are_listed_so_nobody_reads_them_as_checked():
    _, assumed = derive_features(fixture("S-02"))
    assert any("duplicate_status" in a for a in assumed) and any("specialist_required" in a for a in assumed)


# ---------------------------------------------------------------------------------------------------------------- the agent
def test_the_agent_writes_only_the_supervisors_fields_and_sets_the_status():
    a = agent()
    assert (a.agent_id, set(a.owns)) == ("supervisor", {"routing", "human_approval_required", "resolution_status"})
    p = a.run(fixture("S-02"))
    assert set(p.updates) == {"routing", "human_approval_required", "resolution_status"} and p.updates["resolution_status"] == "awaiting_human" and p.updates["human_approval_required"]
    assert "tier simple" in p.rationale and "assumed:" in p.rationale
    assert a.run(fixture("S-01")).updates["resolution_status"] == "awaiting_evidence"
    with pytest.raises(OwnershipViolation):
        apply_patch(fixture("S-02"), StatePatch(agent_id="supervisor", updates={"facts": []}))


def test_an_open_evidence_request_means_the_claim_waits_for_evidence_even_when_nothing_else_is_missing():
    s = fixture("S-02").model_copy(update={"missing_evidence": [MissingEvidence(question_id="Q-EXTENT", gap_type="missing", resolves_question_ids=["Q-EXTENT"],
                                                                                 minimum_evidence_request=MinimumEvidenceRequest(addressee_role="claimant", items=["x"]))]})
    p = agent().run(s)
    assert p.updates["resolution_status"] == "awaiting_evidence" and p.updates["routing"].tier == "medium"


def test_the_agent_is_pure_and_is_wired_into_the_default_registry():
    s = fixture("S-04")
    assert agent().run(s).updates == agent().run(s).updates
    reg = build_registry(SETTINGS)
    assert "routing" in reg.implemented and reg["routing"].agent.agent_id == "supervisor"
