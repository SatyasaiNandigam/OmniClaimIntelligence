"""T6.4 - the action planner, drafts and tasks: the labelled next steps for every scenario, one consolidated request per addressee, no duplicates on a re-run, never sent."""
import json
from pathlib import Path

import pytest

from claimtower.actions.drafts import DraftWriter, draft_id
from claimtower.actions.planner import ActionPlanner, ActionTemplates, execution_mode
from claimtower.actions.tasks import task_request
from claimtower.agents.action import ActionAgent, make_action_agent
from claimtower.composition import build_registry
from claimtower.core.config import load_settings
from claimtower.core.schemas import ClaimState, MinimumEvidenceRequest, MissingEvidence, ProposedAction
from claimtower.core.state import OwnershipViolation, StatePatch, apply_patch

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()
TEMPLATES = ActionTemplates.load(SETTINGS.pack)
PATTERNS = SETTINGS.pack.coverage.forbidden_patterns


def fixture(sid) -> ClaimState:
    """The labelled state with the action slice removed: the agent plans from the gaps, not from the answer key."""
    s = ClaimState(**json.loads((ROOT / f"data/fixtures/{sid}/claim_state.json").read_text(encoding="utf-8")))
    return s.model_copy(update={"proposed_actions": [], "drafts": []})


def labelled(sid):
    raw = json.loads((ROOT / f"data/fixtures/{sid}/action.json").read_text(encoding="utf-8"))["proposed_actions"]
    return sorted((a["action_type"], a["responsible_party"], a["execution_mode"], a["idempotency_key"]) for a in raw)


def agent(rewriter=None) -> ActionAgent:
    return ActionAgent(ActionPlanner(SETTINGS.autonomy, TEMPLATES), DraftWriter(TEMPLATES, PATTERNS, rewriter))


def req(role, items, qids, qid=None) -> MissingEvidence:
    return MissingEvidence(question_id=qid or qids[0], gap_type="missing", resolves_question_ids=qids,
                           minimum_evidence_request=MinimumEvidenceRequest(addressee_role=role, items=items))


def keys(patch):
    return [(a.action_type, a.responsible_party, a.execution_mode, a.idempotency_key) for a in patch.updates["proposed_actions"]]


# ---------------------------------------------------------------------------------------------------------------- the labelled scenarios
@pytest.mark.parametrize("sid", ["S-01", "S-02", "S-03", "S-04", "S-05"])
def test_every_scenario_gets_the_labelled_actions_with_their_modes_and_keys(sid):
    assert sorted(keys(agent().run(fixture(sid)))) == labelled(sid)


def test_one_consolidated_request_per_addressee_with_every_item_once():
    p = agent().run(fixture("S-01"))
    reqs = [d for d in p.updates["drafts"] if d.kind == "information_request"]
    assert sorted(d.addressee_role for d in reqs) == ["claimant", "plumber"]                          # one each, the surveyor gets a task instead
    plumber = next(d for d in reqs if d.addressee_role == "plumber")
    assert plumber.draft_id == draft_id("CLM-0001:draft_consolidated_request:plumber") and plumber.status == "draft"
    for item in ("Statement of the failure mechanism of the joint", "Close-up photographs of the failed joint"):
        assert plumber.body.count(item) == 1
    assert "one message" in plumber.body


def test_two_requests_to_the_same_role_are_merged_into_one_message_without_repeating_an_item():
    s = fixture("S-02").model_copy(update={"missing_evidence": [req("claimant", ["Photos of the pipe", "Invoice"], ["Q-A"]), req("claimant", ["Invoice", "Date of repair"], ["Q-B"])]})
    p = agent().run(s)
    d = [x for x in p.updates["drafts"] if x.kind == "information_request"]
    assert len(d) == 1 and d[0].body.count("Invoice") == 1 and "Date of repair" in d[0].body
    assert ("draft_consolidated_request", "claimant") in [(a[0], a[1]) for a in keys(p)]


def test_a_task_role_gets_a_controlled_task_not_a_draft_and_the_endorsement_id_names_it():
    p = agent().run(fixture("S-05"))
    task = next(a for a in p.updates["proposed_actions"] if a.action_type == "create_task")
    assert task.execution_mode == "controlled_api" and task.responsible_party == "request_endorsement_END-01_from_policy_administration"
    assert task.payload_ref == f"task:{task.idempotency_key}" and not [d for d in p.updates["drafts"] if d.kind == "information_request"]


def test_the_settlement_is_held_only_while_a_policy_side_input_is_missing_and_presented_when_calculated():
    held = [a.action_type for a in agent().run(fixture("S-05")).updates["proposed_actions"]]
    assert "hold_settlement_until_endorsement_received" in held and "present_settlement_for_human_approval" not in held
    ok = [a.action_type for a in agent().run(fixture("S-02")).updates["proposed_actions"]]
    assert "present_settlement_for_human_approval" in ok and "hold_settlement_until_endorsement_received" not in ok
    assert "hold_settlement_until_endorsement_received" not in [a.action_type for a in agent().run(fixture("S-01")).updates["proposed_actions"]]


def test_a_lapsed_policy_is_registered_routed_and_acknowledged_with_nothing_decided():
    p = agent().run(fixture("S-03"))
    assert [a.action_type for a in p.updates["proposed_actions"]] == ["register_claim", "route_to_operator", "draft_acknowledgement"]
    body = p.updates["drafts"][0].body
    assert p.updates["drafts"][0].kind == "acknowledgement" and "Nothing has been decided" in body and DraftWriter(TEMPLATES, PATTERNS).lint(body) == []


# ---------------------------------------------------------------------------------------------------------------- idempotency
def test_running_again_creates_no_duplicate_actions_or_drafts():
    a, s = agent(), fixture("S-01")
    first = a.run(s)
    state = apply_patch(s, first)
    second = a.run(state)
    assert [x.model_dump() for x in second.updates["proposed_actions"]] == [x.model_dump() for x in first.updates["proposed_actions"]]
    assert [d.model_dump() for d in second.updates["drafts"]] == [d.model_dump() for d in first.updates["drafts"]]
    assert len({x.idempotency_key for x in second.updates["proposed_actions"]}) == len(second.updates["proposed_actions"])


def test_an_action_a_person_moved_on_keeps_its_status_and_a_new_gap_adds_only_the_new_action():
    a, s = agent(), fixture("S-01")
    state = apply_patch(s, a.run(s))
    moved = [x.model_copy(update={"status": "executed"}) if x.action_type == "create_task" else x for x in state.proposed_actions]
    approved = [d.model_copy(update={"status": "approved"}) if d.addressee_role == "plumber" else d for d in state.drafts]
    state = state.model_copy(update={"proposed_actions": moved, "drafts": approved, "missing_evidence": [*state.missing_evidence, req("contractor", ["Quotation"], ["Q-X"])]})
    p = a.run(state)
    acts = {x.idempotency_key: x for x in p.updates["proposed_actions"]}
    assert acts["CLM-0001:create_task:surveyor_followup"].status == "executed" and "CLM-0001:draft_request:contractor" in acts
    assert next(d for d in p.updates["drafts"] if d.addressee_role == "plumber").status == "approved"
    assert len(p.updates["proposed_actions"]) == len(state.proposed_actions) + 1 and "1 new" in p.rationale


def test_an_action_whose_need_has_gone_is_cancelled_with_its_unsent_draft_never_deleted():
    a, s = agent(), fixture("S-01")
    state = apply_patch(s, a.run(s))
    resolved = state.model_copy(update={"missing_evidence": [m for m in state.missing_evidence if m.minimum_evidence_request.addressee_role != "claimant"]})
    p = a.run(resolved)
    acts = {x.idempotency_key: x for x in p.updates["proposed_actions"]}
    assert acts["CLM-0001:draft_request:claimant"].status == "cancelled" and "cancelled" in p.rationale
    assert "claimant" not in [d.addressee_role for d in p.updates["drafts"] if d.kind == "information_request"]


# ---------------------------------------------------------------------------------------------------------------- the model rewording is checked
def test_a_reworded_draft_is_used_only_if_it_keeps_every_item_adds_no_amount_and_states_no_conclusion():
    s = fixture("S-01")
    items = next(m for m in s.missing_evidence if m.minimum_evidence_request.addressee_role == "plumber").minimum_evidence_request.items

    def good(text, kind, its):
        return "Hello, we need these for the claim:\n" + "\n".join(f"* {i}" for i in its) + "\nThank you."

    d = next(x for x in agent(good).run(s).updates["drafts"] if x.addressee_role == "plumber")
    assert d.body.startswith("Hello") and all(i in d.body for i in items)
    for bad in (lambda t, k, i: "Please send the photographs.",                                                  # items dropped
                lambda t, k, i: t + " The claim is covered up to INR 99,999.",                                  # a new amount (and a conclusion)
                lambda t, k, i: t + " The claim is covered.",                                                    # a conclusion
                lambda t, k, i: t + " x" * 500):                                                                  # runaway length
        ag = agent(bad)
        p = ag.run(s)
        assert next(x for x in p.updates["drafts"] if x.addressee_role == "plumber").body == agent().run(s).updates["drafts"][0].body and "rewording dropped" in p.rationale
    assert agent(lambda t, k, i: None).run(s).updates == agent().run(s).updates                                  # a model that fails changes nothing


def test_the_status_message_never_states_an_outcome_or_an_amount():
    w = DraftWriter(TEMPLATES, PATTERNS)
    for sid in ("S-01", "S-02", "S-04", "S-05"):
        d = next(x for x in agent().run(fixture(sid)).updates["drafts"] if x.kind == "status_message")
        assert w.lint(d.body) == [] and "INR" not in d.body and "reviews every claim" in d.body
    assert "asked you for further information" in next(x for x in agent().run(fixture("S-01")).updates["drafts"] if x.kind == "status_message").body
    assert "not need anything more from you" in next(x for x in agent().run(fixture("S-02")).updates["drafts"] if x.kind == "status_message").body


# ---------------------------------------------------------------------------------------------------------------- modes, tasks, ownership
def test_the_autonomy_matrix_decides_the_mode_and_a_human_only_operation_is_never_planned():
    assert execution_mode(SETTINGS.autonomy, "draft_information_request") == "draft" and execution_mode(SETTINGS.autonomy, "create_investigation_task") == "controlled_api"
    with pytest.raises(ValueError, match="authorised person only"):
        execution_mode(SETTINGS.autonomy, "release_payment")


def test_a_task_request_carries_exactly_what_is_needed_and_the_actions_key():
    a, s = agent(), fixture("S-05")
    act = next(x for x in a.run(s).updates["proposed_actions"] if x.action_type == "create_task")
    planned = next(p for p in a.last_plan if p.key == act.idempotency_key)
    r = task_request(act, planned, TEMPLATES, s)
    assert r.claim_id == "CLM-0005" and r.assignee_role == "policy_administration" and r.idempotency_key == act.idempotency_key
    assert "END-01" in r.description and r.task_type == act.responsible_party


def test_the_agent_owns_actions_and_drafts_only_and_is_wired_into_the_registry():
    a = make_action_agent(SETTINGS.pack, SETTINGS.autonomy)
    assert (a.agent_id, set(a.owns)) == ("action", {"proposed_actions", "drafts"})
    with pytest.raises(OwnershipViolation):
        apply_patch(fixture("S-02"), StatePatch(agent_id="action", updates={"facts": []}))
    reg = build_registry(SETTINGS)
    assert "action" in reg.implemented and isinstance(reg["action"].agent, ActionAgent)
    assert ProposedAction  # the frozen contract is used unchanged
