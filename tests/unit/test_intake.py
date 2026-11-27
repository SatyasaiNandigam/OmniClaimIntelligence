"""T6.7 - the Intake agent: the registration test group (complete, missing field, invalid policy ref, duplicate submission, not effective)."""
import json
from datetime import date
from pathlib import Path

import pytest

from claimtower.actions.planner import ActionPlanner, ActionTemplates
from claimtower.agents.intake import IntakeAgent
from claimtower.composition import SYNTHETIC, build_registry, ingest_claim_folder
from claimtower.core.config import load_settings
from claimtower.core.schemas import ClaimHeader, ClaimState
from claimtower.core.state import OwnershipViolation, StatePatch, apply_patch
from claimtower.ingestion.classifier import HybridClassifier, RuleClassifier
from claimtower.ingestion.extractors import HybridExtractor, RuleExtractor
from claimtower.ingestion.store import DocumentStore
from claimtower.integrations.base import AdapterError
from claimtower.integrations.dispatcher import Dispatcher
from claimtower.integrations.mock.claims_mock import MockClaimsAdapter
from claimtower.integrations.mock.faults import FaultInjector, FaultPlan
from claimtower.integrations.mock.pas_mock import MockPasAdapter
from claimtower.integrations.retry import RetryPolicy
from claimtower.routing.features import derive_features
from claimtower.routing.tiering import TieringEngine

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()
PACK = SETTINGS.pack
FIXED = __import__("datetime").datetime(2026, 6, 1, 9, 0, tzinfo=__import__("datetime").timezone.utc)


def make_agent(store=None, claims=None, **kw) -> tuple[IntakeAgent, MockClaimsAdapter]:
    claims = claims or MockClaimsAdapter()
    cls = HybridClassifier(RuleClassifier(PACK.document_types))
    ext = HybridExtractor(RuleExtractor(PACK.document_types))
    d = Dispatcher(policy=RetryPolicy(max_attempts=2), sleeper=lambda s: None)
    return IntakeAgent(PACK.intake_checklist, MockPasAdapter(), claims, d, store=store, classifier=cls if store else None, extractor=ext if store else None, clock=lambda: FIXED, **kw), claims


def store_for(tmp_path, claim="CLM-0002") -> DocumentStore:
    s = DocumentStore(tmp_path / "store")
    ingest_claim_folder(s, SYNTHETIC / "claims" / claim)
    return s


def items(patch):
    return {i.item_id: i for i in patch.updates["intake_checklist"]}


# ---------------------------------------------------------------------------------------------------------------- complete
def test_a_complete_registration_reads_the_notice_of_loss_registers_the_claim_and_finds_nothing_missing(tmp_path):
    a, claims = make_agent(store_for(tmp_path))
    p = a.run(ClaimState(claim_id="CLM-0002"))
    h = p.updates["claim_header"]
    assert (h.claimant_name, h.policy_ref, h.loss_date, h.loss_type) == ("SYNTHETIC Claimant B", "POL-1001", date(2026, 5, 2), "water_damage")
    assert "pipe" in h.loss_description.lower() and h.registered_at == FIXED
    assert all(i.received for i in p.updates["intake_checklist"].__iter__()) and set(items(p)) == {"field:claimant_name", "field:policy_ref", "field:loss_date", "field:loss_description",
                                                                                                 "doc:fnol", "doc:estimate", "policy:exact_match", "policy:in_force"}
    assert claims.claim_count == 1 and claims.get_claim("CLM-0002").claim_header.policy_ref == "POL-1001"
    assert "registered in the claims system" in p.rationale and "STILL NEEDED" not in p.rationale and "loss type assumed" in p.rationale


@pytest.mark.parametrize("claim,sid", [("CLM-0001", "S-01"), ("CLM-0002", "S-02"), ("CLM-0004", "S-04"), ("CLM-0005", "S-05")])
def test_the_header_read_from_the_synthetic_notice_matches_the_labelled_claim(tmp_path, claim, sid):
    h = make_agent(store_for(tmp_path, claim))[0].run(ClaimState(claim_id=claim)).updates["claim_header"]
    fx = json.loads((ROOT / f"data/fixtures/{sid}/claim_state.json").read_text(encoding="utf-8"))["claim_header"]
    assert (h.claimant_name, h.policy_ref, str(h.loss_date)) == (fx["claimant_name"], fx["policy_ref"], fx["loss_date"])


# ---------------------------------------------------------------------------------------------------------------- missing field
def test_a_missing_loss_date_is_tracked_and_asked_for_but_does_not_block_registration():
    a, claims = make_agent()
    p = a.run(ClaimState(claim_id="CLM-X1", claim_header=ClaimHeader(claim_id="CLM-X1", claimant_name="A B", policy_ref="POL-1001", loss_description="a pipe burst")))
    i = items(p)
    assert not i["field:loss_date"].received and i["field:policy_ref"].received and "policy:exact_match" not in i      # no date: no policy lookup either
    assert claims.claim_count == 1 and "STILL NEEDED" in p.rationale and "Date of loss" in p.rationale


def test_everything_still_needed_goes_to_the_claimant_in_one_consolidated_request():
    a, _ = make_agent()
    s = ClaimState(claim_id="CLM-X1", claim_header=ClaimHeader(claim_id="CLM-X1", policy_ref="POL-1001"))
    s = apply_patch(s, a.run(s))
    plan = ActionPlanner(SETTINGS.autonomy, ActionTemplates.load(PACK)).plan(s)
    reqs = [x for x in plan if x.kind == "draft" and x.draft_kind == "information_request"]
    assert len(reqs) == 1 and reqs[0].party == "claimant" and reqs[0].action_type == "draft_consolidated_request"
    assert {"Name of the claimant", "Date of loss", "A description of what happened"} <= set(reqs[0].items)


# ---------------------------------------------------------------------------------------------------------------- invalid policy ref
def test_a_policy_number_the_policy_system_does_not_know_goes_to_an_operator_not_a_guess():
    a, claims = make_agent()
    h = ClaimHeader(claim_id="CLM-X2", claimant_name="A B", policy_ref="POL-9999", loss_date=date(2026, 5, 2), loss_description="x")
    p = a.run(ClaimState(claim_id="CLM-X2", claim_header=h))
    assert not items(p)["policy:exact_match"].received and "policy:in_force" not in items(p) and "an operator takes it from here" in p.rationale and claims.claim_count == 1
    s = apply_patch(ClaimState(claim_id="CLM-X2", claim_header=h), p)
    feats, _ = derive_features(s)
    assert feats["policy_match"] == "mismatch" and TieringEngine(SETTINGS.routing).route(feats).reason_codes[0] in ("POLICY_NOT_IN_FORCE", "POLICY_MISMATCH")


# ---------------------------------------------------------------------------------------------------------------- not effective
def test_a_policy_that_is_not_in_force_on_the_loss_date_is_flagged_and_still_registered():
    a, claims = make_agent()
    h = ClaimHeader(claim_id="CLM-X3", claimant_name="A B", policy_ref="POL-1003", loss_date=date(2026, 2, 13), loss_description="x")
    p = a.run(ClaimState(claim_id="CLM-X3", claim_header=h))
    i = items(p)
    assert i["policy:exact_match"].received and not i["policy:in_force"].received and "lapsed" in i["policy:in_force"].description and claims.claim_count == 1


# ---------------------------------------------------------------------------------------------------------------- duplicate submission
def test_submitting_the_same_claim_twice_registers_it_once_and_changes_nothing(tmp_path):
    a, claims = make_agent(store_for(tmp_path))
    s = ClaimState(claim_id="CLM-0002")
    first = a.run(s)
    state = apply_patch(s, first)
    second = a.run(state)
    assert claims.claim_count == 1 and "already registered (replayed)" in second.rationale and claims.calls == [("create_claim", "CLM-0002:create_claim")]       # the dispatcher did not call again
    assert second.updates["claim_header"] == first.updates["claim_header"] and second.updates["intake_checklist"] == first.updates["intake_checklist"]
    assert second.updates["claim_header"].registered_at == FIXED


def test_a_second_agent_with_a_fresh_dispatcher_still_cannot_create_a_second_claim(tmp_path):
    claims = MockClaimsAdapter()
    make_agent(store_for(tmp_path), claims)[0].run(ClaimState(claim_id="CLM-0002"))
    again = make_agent(store_for(tmp_path / "b"), claims)[0].run(ClaimState(claim_id="CLM-0002"))
    assert claims.claim_count == 1 and "already registered" in again.rationale                       # the claims system's own idempotency


# ---------------------------------------------------------------------------------------------------------------- the header: supplied vs notice
def test_a_supplied_value_that_differs_from_the_notice_is_kept_and_reported_for_review(tmp_path):
    a, _ = make_agent(store_for(tmp_path))
    h = ClaimHeader(claim_id="CLM-0002", policy_ref="POL-1002")
    p = a.run(ClaimState(claim_id="CLM-0002", claim_header=h))
    assert p.updates["claim_header"].policy_ref == "POL-1002" and "REVIEW, supplied values differ" in p.rationale and "policy_ref: supplied 'POL-1002'" in p.rationale
    assert p.updates["claim_header"].claimant_name == "SYNTHETIC Claimant B"                          # the missing fields were still filled in


# ---------------------------------------------------------------------------------------------------------------- failure
def test_a_claim_that_cannot_be_registered_stops_the_flow_with_the_reason():
    claims = FaultInjector(MockClaimsAdapter(), FaultPlan({"create_claim": ["timeout"] * 5}))
    a, _ = make_agent(claims=claims)
    with pytest.raises(AdapterError, match="could not be registered .* dead-letter queue"):
        a.run(ClaimState(claim_id="CLM-X4", claim_header=ClaimHeader(claim_id="CLM-X4", policy_ref="POL-1001", loss_date=date(2026, 5, 2))))
    assert len(a.dispatcher.dead_letters.open("CLM-X4")) == 1


def test_a_graph_run_parks_an_unregistered_claim_for_a_person():
    from claimtower.orchestration.graph import ClaimRunner
    claims = FaultInjector(MockClaimsAdapter(), FaultPlan({"create_claim": ["timeout"] * 9}))
    from claimtower.integrations import Adapters
    reg = build_registry(SETTINGS, adapters=Adapters(MockPasAdapter(), claims, None), dispatcher=Dispatcher(policy=RetryPolicy(max_attempts=2), sleeper=lambda s: None))
    st = ClaimRunner(reg).start(ClaimState(claim_id="CLM-X5", claim_header=ClaimHeader(claim_id="CLM-X5", policy_ref="POL-1001", loss_date=date(2026, 5, 2))))
    assert st.phase == "halted" and "could not be registered" in st.halted_reason and st.paused_at == "intake"


# ---------------------------------------------------------------------------------------------------------------- ownership and wiring
def test_the_agent_owns_the_intake_fields_only_and_leaves_duplicates_to_t6_8():
    a, _ = make_agent()
    assert (a.agent_id, set(a.owns)) == ("intake", {"claim_header", "intake_checklist", "duplicate_candidates"})
    p = a.run(ClaimState(claim_id="CLM-X6"))
    assert p.updates["duplicate_candidates"] == [] and set(p.updates) == {"claim_header", "intake_checklist", "duplicate_candidates"}
    with pytest.raises(OwnershipViolation):
        apply_patch(ClaimState(claim_id="C"), StatePatch(agent_id="intake", updates={"facts": []}))
    assert "intake" in build_registry(SETTINGS).implemented
