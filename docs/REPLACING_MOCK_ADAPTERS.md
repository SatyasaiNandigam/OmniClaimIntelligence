# Replacing a mock adapter with the real system

The prototype talks to three external systems through adapters: the **PAS** (policy administration, read only), the **Claims** system and **Recon** (payment reconciliation). Today all three are in-process mocks. There are no mock servers. This note says what a real adapter must do and where to plug it in.

## What exists today

| System | Protocol (`integrations/base.py`) | Mock | Used by |
|---|---|---|---|
| PAS | `get_policy_version`, `get_status` | `integrations/mock/pas_mock.py` (reads `data/synthetic/pas_mock_db.json`; also `get_schedule`, see below) | Policy agent, Settlement agent |
| Claims | `create_claim`, `update_status`, `create_task`, `get_claim` | `integrations/mock/claims_mock.py` (in memory; also `MockPaymentsLedger`) | the commit step after an approval (`orchestration/commit.py`) |
| Recon | `submit_payment_instruction`, `get_match_status`, `list_breaks` | `integrations/mock/recon_mock.py` (in memory) | the commit step, only for a calculated payable and an approver who may release payment |

`build_adapters(settings)` (`integrations/__init__.py`) returns the three adapters. Which one you get comes from the flags in `configs/app.yaml` (`use_mock_pas`, `use_mock_claims`, `use_mock_recon`), which the environment variables `USE_MOCK_PAS`, `USE_MOCK_CLAIMS` and `USE_MOCK_RECON` override. A flag that is off selects the **live** adapter. The live adapters are stubs that raise `AdapterNotConfigured`, so a half-configured system never falls back to a mock without saying so.

## The rules every adapter follows (they are tested on the mocks)

1. **Label every response.** `mock=True` for a mock, `mock=False` for a real system, and set `source_system`. The UI and the audit trail show it.
2. **Writes are idempotent on `idempotency_key`.** Replaying the same key returns the first answer with `created=False` (Claims) or `status="duplicate_ignored"` (Recon) and creates nothing. Five replays of one payment instruction must leave exactly one record.
3. **A failure of the system is an exception, a business answer is a response.** "No policy found" is a `not_found` response. "The system did not answer" raises `AdapterTimeout` (subclass of `AdapterError`). A timeout does not tell the caller whether the call took effect, so the caller retries with the same key.
4. **The PAS is read only.** Recon is called only after a recorded human decision: an instruction without `approved_by` and `approval_ref` is rejected.

## Steps to replace one

1. Write a class with the methods of the protocol in `integrations/base.py`. Use the request and response models in `core/schemas/integration.py` unchanged. Those are frozen contracts, and `scripts/freeze_contracts.py --check` guards them.
2. Put it in `integrations/live/` (for example `live/pas.py`), read its endpoint and credentials from the environment, never from code or the repository.
3. In `integrations/__init__.py` register it in place of the stub: `"pas:live": LivePasAdapter` instead of `not_configured("pas")`.
4. Turn the flag off for that system only (`USE_MOCK_PAS=false`). The other two stay mocks.
5. Run the same contract tests against it. `tests/unit/test_integrations.py` is the checklist: idempotent writes, labelled responses, timeouts as `AdapterTimeout`. Point them at the real adapter (or its sandbox) and add its own cases.
6. Check the Settlement agent's inputs. Today it asks the PAS adapter for `get_schedule(policy_no, as_of)` (building limit and standard deductible) and the payments ledger for `paid_on_claim` and `paid_against_policy`. Those two are not in the frozen protocols, because the schedule and payment history have no contract yet. When the real PAS or Claims system provides them, add them to the protocol and the schema, and re-freeze the contracts on purpose.

## Testing failure handling without a real system

`integrations/mock/faults.py` wraps any adapter and misbehaves as planned, so retry and dead-letter logic (T6.6) can be tested:

```python
plan = FaultPlan({"create_task": ["timeout_after", "ok"], "get_policy_version": ["stale"]})
claims = FaultInjector(MockClaimsAdapter(), plan)
```

The kinds are `timeout` (before the call), `timeout_after` (the call took effect), `duplicate` (the message arrives twice), `stale` (the previous answer again), `slow` (waits `latency_s`) and `ok`. Faults are used one per call, in order.

## Retry, idempotency and the dead-letter queue (T6.6)

The graph never calls an adapter directly. The commit step goes through a `Dispatcher` (`integrations/dispatcher.py`), so a real adapter gets this for free:

- **Idempotency store** (`integrations/idempotency.py`): a key whose call completed is answered from the store and the system is not called again. A key is bound to one operation.
- **Bounded retry** (`integrations/retry.py`): `AdapterTimeout` is retried with exponential backoff (default 4 attempts, 0.5 s doubling, capped at 10 s) using the **same key**. Anything else is not retried. So a real adapter must raise `AdapterTimeout` for a timeout and `AdapterError` for a refusal, and must be idempotent on the key, because a timeout does not say whether the call took effect.
- **Dead-letter queue** (`integrations/deadletter.py`): a call that runs out of attempts, or fails in a way retrying cannot fix (a poison message, an unconfigured live adapter), is parked with its payload and error for a person. `Dispatcher.requeue` tries it again after the cause is repaired and closes the entry.

In the prototype the store and the queue are in memory. Replace them with tables (a unique constraint on the key) when the claims system is real.

## Not done yet

- Mock **servers** (a process that answers HTTP) are not in the prototype plan. If you want one, wrap a mock adapter in a small FastAPI app and write a live adapter that calls it. The protocols do not change.
- The dead-letter queue has no API endpoint or UI yet; its content is in the `commit` event of the audit log (`dead_lettered`).
