"""T6.6 - idempotency, bounded retry with backoff, dead-letter, and the commit step built on them: a retry never creates a second claim, task or payment."""
from datetime import date
from decimal import Decimal

import pytest

from claimtower.composition import build_commit_service, build_registry
from claimtower.core.config import load_settings
from claimtower.core.schemas import ClaimHeader, ClaimState, PaymentInstruction
from claimtower.integrations import Adapters
from claimtower.integrations.base import AdapterError, AdapterNotConfigured, AdapterTimeout
from claimtower.integrations.deadletter import DeadLetterQueue
from claimtower.integrations.dispatcher import Dispatcher
from claimtower.integrations.idempotency import IdempotencyStore
from claimtower.integrations.mock.claims_mock import MockClaimsAdapter
from claimtower.integrations.mock.faults import FaultInjector, FaultPlan
from claimtower.integrations.mock.pas_mock import MockPasAdapter
from claimtower.integrations.mock.recon_mock import MockReconAdapter
from claimtower.integrations.retry import RetriesExhausted, RetryPolicy, call_with_retry, is_retryable
from claimtower.orchestration.graph import ClaimRunner
from claimtower.orchestration.supervisor import HumanDecision

SETTINGS = load_settings()


def instruction(key="CLM-T:pay:1"):
    return PaymentInstruction(claim_id="CLM-T", amount=Decimal(80000), payee_ref="P", approved_by="rev-9", approval_ref="EVT-1", idempotency_key=key)


def dispatcher(sleeps=None, **kw):
    return Dispatcher(IdempotencyStore(), DeadLetterQueue(), RetryPolicy(**kw) if kw else RetryPolicy(), sleeper=(sleeps.append if sleeps is not None else (lambda s: None)))


# ---------------------------------------------------------------------------------------------------------------- idempotency store
def test_a_key_is_bound_to_one_operation_and_remembers_its_result():
    s = IdempotencyStore()
    e = s.begin("k", "create_task")
    assert e.new and e.state == "started"
    s.attempt("k")
    s.complete("k", "answer")
    again = s.begin("k", "create_task")
    assert not again.new and again.state == "done" and again.result == "answer" and again.attempts == 1
    with pytest.raises(ValueError, match="was used for 'create_task'"):
        s.begin("k", "update_status")
    s.reopen("k")
    assert s.get("k").state == "done"                                                            # reopen only affects a failed key: a done one stays done
    s.begin("j", "op")
    s.fail("j", "boom")
    s.reopen("j")
    assert s.get("j").state == "started" and s.keys() == ["k", "j"]


# ---------------------------------------------------------------------------------------------------------------- retry
def test_backoff_doubles_up_to_the_cap_and_the_policy_is_validated():
    p = RetryPolicy(max_attempts=6, base_delay_s=0.5, factor=2.0, max_delay_s=3.0)
    assert [p.delay(n) for n in range(1, 6)] == [0.5, 1.0, 2.0, 3.0, 3.0]
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)


def test_a_timeout_is_retried_with_backoff_until_it_succeeds():
    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise AdapterTimeout("slow")
        return "ok"

    result, attempts = call_with_retry(flaky, RetryPolicy(max_attempts=4, base_delay_s=1.0), sleeper=sleeps.append)
    assert (result, attempts, sleeps) == ("ok", 3, [1.0, 2.0])


def test_running_out_of_attempts_raises_and_a_non_retryable_error_is_not_retried():
    sleeps = []
    with pytest.raises(RetriesExhausted, match="gave up after 3"):
        call_with_retry(lambda: (_ for _ in ()).throw(AdapterTimeout("down")), RetryPolicy(max_attempts=3, base_delay_s=0.1), sleeper=sleeps.append)
    assert len(sleeps) == 2                                                                        # no sleep after the last attempt
    n = []
    with pytest.raises(AdapterNotConfigured):
        call_with_retry(lambda: n.append(1) or (_ for _ in ()).throw(AdapterNotConfigured("no")), RetryPolicy(max_attempts=5), sleeper=lambda s: None)
    assert n == [1] and not is_retryable(AdapterError("refused")) and not is_retryable(ValueError("bug")) and is_retryable(AdapterTimeout("t"))


# ---------------------------------------------------------------------------------------------------------------- dispatcher + dead letter
def test_replaying_the_same_payment_instruction_five_times_yields_exactly_one_recon_record_and_one_call():
    recon, d = MockReconAdapter(), dispatcher()
    outs = [d.execute("submit_payment_instruction", "CLM-T:pay:1", "CLM-T", lambda: recon.submit_payment_instruction(instruction(), "CLM-T:pay:1")) for _ in range(5)]
    assert [o.status for o in outs] == ["done"] + ["replayed"] * 4 and recon.instruction_count == 1 and len(recon.calls) == 1        # the system was asked once
    assert len({o.result.recon_id for o in outs}) == 1


def test_a_timeout_after_the_payment_was_booked_does_not_book_it_twice_when_retried():
    recon = MockReconAdapter()
    flaky = FaultInjector(recon, FaultPlan({"submit_payment_instruction": ["timeout_after", "timeout", "ok"]}))
    sleeps = []
    out = dispatcher(sleeps).execute("submit_payment_instruction", "CLM-T:pay:1", "CLM-T", lambda: flaky.submit_payment_instruction(instruction(), "CLM-T:pay:1"))
    assert out.status == "done" and out.attempts == 3 and recon.instruction_count == 1 and out.result.status == "duplicate_ignored" and len(sleeps) == 2


def test_a_system_that_stays_down_lands_in_the_dead_letter_queue_after_the_bounded_attempts():
    claims = MockClaimsAdapter()
    down = FaultInjector(claims, FaultPlan({"create_claim": ["timeout"] * 10}))
    d = dispatcher(max_attempts=3)
    out = d.execute("create_claim", "CLM-T:create_claim", "CLM-T", lambda: down.create_claim(None), {"x": 1})
    assert out.status == "dead_lettered" and out.attempts == 3 and claims.claim_count == 0 and not out.ok
    item = d.dead_letters.open("CLM-T")[0]
    assert item.reason == "retries_exhausted" and item.attempts == 3 and item.key == "CLM-T:create_claim" and "gave up after 3" in item.error
    assert d.store.get("CLM-T:create_claim").state == "failed"


def test_a_poison_message_goes_to_the_dead_letter_queue_at_once_without_retries():
    sleeps, n = [], []
    d = dispatcher(sleeps)

    def poison():
        n.append(1)
        raise AdapterError("the claims system refused the message: malformed")

    out = d.execute("create_task", "CLM-T:t", "CLM-T", poison, payload="the message")
    assert out.status == "dead_lettered" and n == [1] and sleeps == [] and d.dead_letters.open()[0].reason == "non_retryable"
    assert d.dead_letters.open()[0].payload == "the message"
    again = d.execute("create_task", "CLM-T:t", "CLM-T", poison, payload="the message")          # the same key failing again: still ONE open entry
    assert again.status == "dead_lettered" and len(d.dead_letters.open()) == 1 and d.dead_letters.open()[0].attempts == 2


def test_after_a_person_repairs_the_cause_the_dead_letter_is_requeued_and_closed_once():
    claims = MockClaimsAdapter()
    d = dispatcher(max_attempts=2)
    header = ClaimHeader(claim_id="CLM-T")
    from claimtower.core.schemas import ClaimCreateRequest
    req = ClaimCreateRequest(claim_header=header, idempotency_key="CLM-T:create_claim")
    down = FaultInjector(claims, FaultPlan({"create_claim": ["timeout", "timeout"]}))
    out = d.execute("create_claim", "CLM-T:create_claim", "CLM-T", lambda: down.create_claim(req))
    assert out.status == "dead_lettered"
    entry = d.dead_letters.open()[0]
    fixed = d.requeue(entry.entry_id, lambda: down.create_claim(req), by="ops-3")
    assert fixed.status == "done" and claims.claim_count == 1 and d.dead_letters.open() == []
    assert d.dead_letters.get(entry.entry_id).status == "repaired" and d.dead_letters.get(entry.entry_id).resolved_by == "ops-3"
    with pytest.raises(ValueError, match="already repaired"):
        d.requeue(entry.entry_id, lambda: None, by="ops-3")
    with pytest.raises(ValueError, match="named person"):
        DeadLetterQueue().resolve("x", "discarded", by=" ")


# ---------------------------------------------------------------------------------------------------------------- the commit step
def make_adapters(claims=None, recon=None):
    return Adapters(MockPasAdapter(), claims or MockClaimsAdapter(), recon or MockReconAdapter())


def approved_state(sid="S-05", role_ok=True):
    import json
    from pathlib import Path
    s = ClaimState(**json.loads((Path(__file__).resolve().parents[2] / f"data/fixtures/{sid}/claim_state.json").read_text(encoding="utf-8")))
    return s.model_copy(update={"correlation_id": f"corr-{s.claim_id}", "resolution_status": "approved"})


def test_the_commit_creates_the_claim_the_status_and_each_controlled_task_once_and_leaves_drafts_alone():
    claims, recon = MockClaimsAdapter(), MockReconAdapter()
    svc = build_commit_service(SETTINGS, make_adapters(claims, recon), dispatcher())
    state = approved_state("S-05")
    r = svc.commit(state, reviewer_id="rev-9", reviewer_role="senior_adjuster", approval_ref="EVT-7")
    assert claims.claim_count == 1 and claims.get_claim("CLM-0005").status == "approved"
    assert claims.task_count == 2                                                                   # the endorsement task and the hold; the status message is a draft
    by_type = {a.action_type: a.status for a in r.actions}
    assert by_type["create_task"] == "executed" and by_type["hold_settlement_until_endorsement_received"] == "executed" and by_type["draft_status_message"] == "proposed"
    assert r.mock and "not instructed" in r.payment
    again = svc.commit(state, reviewer_id="rev-9", reviewer_role="senior_adjuster", approval_ref="EVT-7")      # the whole commit repeated
    assert claims.claim_count == 1 and claims.task_count == 2 and all(o.status == "replayed" for o in again.outcomes)


def test_a_payment_is_instructed_only_for_a_calculated_payable_and_a_reviewer_who_may_release_it():
    recon, claims = MockReconAdapter(), MockClaimsAdapter()
    svc = build_commit_service(SETTINGS, make_adapters(claims, recon), dispatcher())
    state = approved_state("S-02")
    no = svc.commit(state, reviewer_id="rev-1", reviewer_role="senior_adjuster", approval_ref="EVT-1")
    assert "needs a reviewer with the role supervisor or finance" in no.payment and recon.instruction_count == 0
    yes = svc.commit(state, reviewer_id="rev-2", reviewer_role="supervisor", approval_ref="EVT-2")
    assert "accepted" in yes.payment and recon.instruction_count == 1
    again = svc.commit(state, reviewer_id="rev-2", reviewer_role="supervisor", approval_ref="EVT-2")
    assert recon.instruction_count == 1 and recon.calls == ["CLM-0002:pay:1"]
    assert "replayed" in [o.status for o in again.outcomes]
    assert "no calculated payable" in svc.commit(approved_state("S-05"), reviewer_id="r", reviewer_role="supervisor", approval_ref="E").payment


def test_a_task_that_cannot_be_delivered_is_dead_lettered_and_marked_failed_without_stopping_the_rest():
    claims = MockClaimsAdapter()
    down = FaultInjector(claims, FaultPlan({"create_task": ["timeout"] * 20}))
    d = dispatcher(max_attempts=2)
    svc = build_commit_service(SETTINGS, make_adapters(down, MockReconAdapter()), d)
    r = svc.commit(approved_state("S-05"), reviewer_id="rev-9", reviewer_role="senior_adjuster", approval_ref="EVT-7")
    assert claims.claim_count == 1 and claims.get_claim("CLM-0005").status == "approved"            # the claim and its status went through
    assert len(r.dead_lettered) == 2 and len(d.dead_letters.open("CLM-0005")) == 2
    assert {a.status for a in r.actions if a.execution_mode == "controlled_api"} == {"failed"}
    assert "2 in the dead-letter queue" in r.summary()


def test_the_graph_commits_through_the_adapters_after_an_approval_and_records_it():
    claims, recon = MockClaimsAdapter(), MockReconAdapter()
    reg = build_registry(SETTINGS, force_mock=("gap", "coverage", "settlement", "action", "audit", "routing"))
    runner = ClaimRunner(reg, commit_service=build_commit_service(SETTINGS, make_adapters(claims, recon), dispatcher()))
    runner.start(ClaimState(claim_id="CLM-0002"))
    done = runner.decide("CLM-0002", HumanDecision(decision="approve", reviewer_id="rev-2", reviewer_role="supervisor"))
    ev = runner.events.of_kind("CLM-0002", "commit")[0]
    assert done.phase == "completed" and ev.mock and "approval committed through the adapters" in ev.message and "YET TO IMPLEMENT" not in ev.message
    assert claims.get_claim("CLM-0002").status == "approved" and recon.instruction_count == 1 and ev.data["dead_lettered"] == []
    assert "accepted" in ev.data["payment"] and ev.state_version == runner.state("CLM-0002").state_version
    assert recon.list_breaks() == [] and date(2026, 1, 1)


def test_without_a_commit_service_the_commit_stays_the_labelled_mock():
    reg = build_registry(SETTINGS, force_mock=("gap", "coverage", "settlement", "action", "audit", "routing"))
    runner = ClaimRunner(reg)
    runner.start(ClaimState(claim_id="CLM-0002"))
    runner.decide("CLM-0002", HumanDecision(decision="approve", reviewer_id="rev-2", reviewer_role="supervisor"))
    assert "YET TO IMPLEMENT" in runner.events.of_kind("CLM-0002", "commit")[0].message
