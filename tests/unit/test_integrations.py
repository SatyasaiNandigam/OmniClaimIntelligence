"""T6.5 - the PAS, Claims and Recon adapters (mocks): the production contract (idempotent writes, every response labelled mock), failure injection, mock -> live as a flag."""
from datetime import date
from decimal import Decimal

import pytest

from claimtower.composition import build_registry
from claimtower.core.config import load_settings
from claimtower.core.schemas import ClaimCreateRequest, ClaimHeader, ClaimStatusUpdate, PaymentInstruction, TaskRequest
from claimtower.integrations import build_adapters, register_adapters
from claimtower.integrations.base import AdapterError, AdapterNotConfigured, AdapterTimeout, ClaimsClient, PASClient, ReconClient
from claimtower.integrations.mock.claims_mock import MockClaimsAdapter
from claimtower.integrations.mock.faults import FaultInjector, FaultPlan
from claimtower.integrations.mock.pas_mock import MockPasAdapter
from claimtower.integrations.mock.recon_mock import MockReconAdapter

SETTINGS = load_settings()
HEADER = ClaimHeader(claim_id="CLM-T1", policy_ref="POL-1001", loss_date=date(2026, 5, 2))


def create_req(key="CLM-T1:create", header=HEADER):
    return ClaimCreateRequest(claim_header=header, idempotency_key=key, correlation_id="corr-1")


def task_req(key="CLM-T1:create_task:surveyor_followup"):
    return TaskRequest(claim_id="CLM-T1", task_type="surveyor_followup", assignee_role="surveyor", description="d", idempotency_key=key)


def pay(key="CLM-T1:pay:1", amount="80000", approved_by="rev-9", approval_ref="EVT-1"):
    return PaymentInstruction(claim_id="CLM-T1", amount=Decimal(amount), payee_ref="PAYEE-1", approved_by=approved_by, approval_ref=approval_ref, idempotency_key=key)


def methods(obj, proto):
    return all(callable(getattr(obj, n, None)) for n in proto.__protocol_attrs__)


# ---------------------------------------------------------------------------------------------------------------- the contract
def test_each_mock_implements_its_protocol_and_says_it_is_a_mock():
    pas, claims, recon = MockPasAdapter(), MockClaimsAdapter(), MockReconAdapter()
    assert methods(pas, PASClient) and methods(claims, ClaimsClient) and methods(recon, ReconClient)
    assert pas.mock and claims.mock and recon.mock


def test_the_pas_answers_status_and_version_consistently_and_labels_every_answer():
    pas = MockPasAdapter()
    v, s = pas.get_policy_version("POL-1001", date(2026, 5, 2)), pas.get_status("POL-1001", date(2026, 5, 2))
    assert (v.status, s.status) == ("in_force", "in_force") and v.mock and s.mock and s.source_system == "PAS-MOCK" and s.policy_no == "POL-1001"
    assert pas.get_status("POL-1003", date(2026, 2, 13)).status == "lapsed" and pas.get_status("POL-NOPE", date(2026, 5, 2)).status == "not_found"
    assert pas.get_schedule("POL-1001", date(2026, 5, 2))["standard_deductible"] == "10000"


def test_the_claims_system_creates_one_claim_however_often_the_same_call_arrives():
    c = MockClaimsAdapter()
    acks = [c.create_claim(create_req()) for _ in range(5)]
    assert [a.created for a in acks] == [True, False, False, False, False] and c.claim_count == 1 and len({a.claim_id for a in acks}) == 1
    assert all(a.mock and a.source_system == "CLAIMS-MOCK" and a.correlation_id == "corr-1" for a in acks)
    again = c.create_claim(create_req("another-key"))
    assert again.created is False and c.claim_count == 1                                          # the same claim under another key: still no second claim
    assert len(c.calls) == 6 and c.get_claim("CLM-T1").claim_header.policy_ref == "POL-1001"


def test_a_status_update_and_a_task_are_idempotent_and_need_an_existing_claim():
    c = MockClaimsAdapter()
    with pytest.raises(AdapterError, match="unknown claim"):
        c.create_task(task_req())
    c.create_claim(create_req())
    upd = ClaimStatusUpdate(claim_id="CLM-T1", status="awaiting_human", idempotency_key="CLM-T1:status:1")
    assert [c.update_status(upd).created for _ in range(3)] == [True, False, False] and c.get_claim("CLM-T1").status == "awaiting_human"
    tasks = [c.create_task(task_req()) for _ in range(5)]
    assert c.task_count == 1 and [t.created for t in tasks] == [True, False, False, False, False] and len({t.task_id for t in tasks}) == 1
    c.create_task(task_req("a different task"))
    assert c.task_count == 2
    with pytest.raises(AdapterError):
        c.update_status(ClaimStatusUpdate(claim_id="CLM-NOPE", status="closed", idempotency_key="k"))
    with pytest.raises(AdapterError):
        c.get_claim("CLM-NOPE")


def test_a_payment_instruction_replayed_five_times_is_one_record():
    r = MockReconAdapter()
    acks = [r.submit_payment_instruction(pay(), "CLM-T1:pay:1") for _ in range(5)]
    assert [a.status for a in acks] == ["accepted"] + ["duplicate_ignored"] * 4 and r.instruction_count == 1 and len({a.recon_id for a in acks}) == 1
    assert all(a.mock and a.source_system == "RECON-MOCK" for a in acks) and r.list_breaks() == []
    assert r.get_match_status(acks[0].recon_id).state == "pending"


def test_a_payment_without_a_recorded_human_approval_is_rejected_and_nothing_is_recorded():
    r = MockReconAdapter()
    for kw in ({"approved_by": ""}, {"approval_ref": "  "}):
        assert r.submit_payment_instruction(pay(key=f"k{len(kw)}{list(kw)[0]}", **kw), f"k{len(kw)}{list(kw)[0]}").status == "rejected"
    assert r.instruction_count == 0
    with pytest.raises(AdapterError, match="does not match"):
        r.submit_payment_instruction(pay(), "some other key")


def test_the_same_payment_under_another_key_is_flagged_for_a_person_not_merged_silently():
    r = MockReconAdapter()
    first = r.submit_payment_instruction(pay(), "CLM-T1:pay:1")
    second = r.submit_payment_instruction(pay(key="CLM-T1:pay:2"), "CLM-T1:pay:2")
    assert second.status == "accepted" and second.recon_id != first.recon_id and r.instruction_count == 2
    b = r.list_breaks("CLM-T1")
    assert len(b) == 1 and "possible duplicate payment" in b[0].reason and r.get_match_status(second.recon_id).state == "unmatched" and r.list_breaks("CLM-OTHER") == []
    with pytest.raises(AdapterError):
        r.get_match_status("RECON-NOPE")


# ---------------------------------------------------------------------------------------------------------------- failure injection
def test_a_timeout_before_the_call_creates_nothing_and_the_retry_creates_it_once():
    c = MockClaimsAdapter()
    f = FaultInjector(c, FaultPlan({"create_claim": ["timeout"]}))
    with pytest.raises(AdapterTimeout):
        f.create_claim(create_req())
    assert c.claim_count == 0
    assert f.create_claim(create_req()).created is True and c.claim_count == 1 and f.injected == [("create_claim", "timeout")]


def test_a_timeout_after_the_call_took_effect_does_not_double_it_on_retry_with_the_same_key():
    c = MockClaimsAdapter()
    f = FaultInjector(c, FaultPlan({"create_task": ["timeout_after"]}))
    f.create_claim(create_req())
    with pytest.raises(AdapterTimeout):
        f.create_task(task_req())
    assert c.task_count == 1                                                                       # it DID happen; the caller could not know
    assert f.create_task(task_req()).created is False and c.task_count == 1


def test_a_duplicate_message_is_absorbed_by_the_idempotent_system():
    r = MockReconAdapter()
    f = FaultInjector(r, FaultPlan({"submit_payment_instruction": ["duplicate"]}))
    ack = f.submit_payment_instruction(pay(), "CLM-T1:pay:1")
    assert ack.status == "duplicate_ignored" and r.instruction_count == 1 and r.calls == ["CLM-T1:pay:1"] * 2


def test_a_stale_answer_is_the_previous_one_and_slow_calls_use_the_injected_sleeper():
    pas = MockPasAdapter()
    slept = []
    f = FaultInjector(pas, FaultPlan({"get_policy_version": ["ok", "stale", "slow"]}, latency_s=2.5), sleeper=slept.append)
    first = f.get_policy_version("POL-1001", date(2026, 5, 2))
    stale = f.get_policy_version("POL-1003", date(2026, 2, 13))
    assert stale is first and stale.policy_no == "POL-1001"                                         # the second question got the first answer
    assert f.get_policy_version("POL-1003", date(2026, 2, 13)).policy_no == "POL-1003" and slept == [2.5]
    assert f.get_status("POL-1001", date(2026, 5, 2)).status == "in_force" and f.mock


def test_an_unknown_fault_kind_is_refused():
    with pytest.raises(ValueError, match="unknown fault kind"):
        FaultPlan({"create_claim": ["explode"]})


# ---------------------------------------------------------------------------------------------------------------- mock -> live is a flag
def test_the_flags_pick_the_mock_adapters_by_default():
    register_adapters()
    register_adapters()                                                                            # safe twice
    a = build_adapters(SETTINGS)
    assert isinstance(a.pas, MockPasAdapter) and isinstance(a.claims, MockClaimsAdapter) and isinstance(a.recon, MockReconAdapter)
    assert a.claims is not build_adapters(SETTINGS).claims                                          # a fresh adapter per call: no shared state between runs


def live_settings(**flags):
    flags = SETTINGS.app.flags.model_copy(update=flags)
    return SETTINGS.model_copy(update={"app": SETTINGS.app.model_copy(update={"flags": flags})})


@pytest.mark.parametrize("flag,kind", [("use_mock_pas", "PAS"), ("use_mock_claims", "CLAIMS"), ("use_mock_recon", "RECON")])
def test_turning_a_flag_off_selects_the_live_adapter_which_fails_loudly_instead_of_falling_back(flag, kind):
    with pytest.raises(AdapterNotConfigured, match=f"live {kind} adapter is not implemented"):
        build_adapters(live_settings(**{flag: False}))


def test_the_registry_uses_the_adapter_the_flags_select_for_the_pas():
    with pytest.raises(AdapterNotConfigured):
        build_registry(live_settings(use_mock_pas=False))
