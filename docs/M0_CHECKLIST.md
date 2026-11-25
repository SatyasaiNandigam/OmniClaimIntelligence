# M0 checkpoint — contracts frozen

**Definition (BUILD_PLAN §8):** `ClaimState` + sub-models, agent protocol and adapter protocols exported as JSON Schema; 5 scenarios + ground truth ready.
**Status (2026-11-25):** technically met and machine-checked; **two human sign-offs outstanding** (below). Verify everything with `python -m pytest` — the gate is `tests/unit/test_m0_gate.py`.

## Gate criteria

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `ClaimState` + all sub-models are real, strict Pydantic contracts | ✅ | `src/claimtower/core/schemas/`, `tests/unit/test_schemas.py` |
| 2 | Agent protocol exported (`Agent`, `StatePatch`, ownership map) | ✅ | `StatePatch.schema.json`, `ownership.json`, `protocols.json` |
| 3 | Adapter protocols exported (`PASClient`, `ClaimsClient`, `ReconClient`) with typed request/response models, `mock` flag and idempotency keys | ✅ | `integrations/base.py`, `core/schemas/integration.py`, `protocols.json` |
| 4 | Section-11 interface signatures defined (`FactExtractor`, `ImageAnalyser`, `PolicyService`, `SettlementEngine`, `Router`, `AuditLog`) | ✅ | `core/interfaces.py`, `protocols.json` |
| 5 | JSON Schemas exported in CI and fail on drift | ✅ | `scripts/export_schemas.py --check` (also covers protocols + ownership) |
| 6 | Post-M0 changes need explicit approval | ✅ | `scripts/freeze_contracts.py`, `docs/contracts/CONTRACT_LOCK.json` (v1.0.0), `CHANGELOG.md`; the test proves an unapproved change fails and an approved one is logged |
| 7 | 5 scenarios + ground truth | ✅ | `data/scenarios/scenarios.yaml`, `data/ground_truth/S-01…S-05.json` |
| 8 | Fixture rule: every scenario has a full expected `ClaimState` and one slice per agent | ✅ | `data/fixtures/` (51 files), rebuilt by `scripts/build_fixtures.py`; slices replay to the full state through `apply_patch` |
| 9 | Policy corpus + stable clause IDs | ✅ | `data/synthetic/policies/` (T1.1) |
| 10 | Business rules are SAMPLE and validated | ✅ | `configs/`, `docs/assumptions.md` (T1.5) |
| 11 | Framework decision proven | ✅ | `docs/adr/0001-framework.md`, `tests/integration/test_spike.py` (T0.1) |
| 12 | Lint + import boundary (`settlement/` has no LLM imports) | ✅ | `ruff`, `import-linter` (verified failing on a probe violation) |

## Outstanding — need a human

| Item | Why it is open | Suggested action |
|---|---|---|
| **Peer review of the contracts** (T0.2 done-when: one reviewer per workstream) | Working solo; nobody else has reviewed them. | Have at least one other person read `docs/contracts/README.md` and the schemas before real agents depend on them. If they request changes, use the approval procedure — the freeze is v1.0.0, so nothing is irreversible. |
| **Second labeller for ground truth** (T1.2 done-when: ≥ 90% agreement on 5 scenarios) | One person (and the assistant) wrote all labels. | Have a claims-savvy reviewer independently label S-01…S-05, then diff against `data/ground_truth/`. |
| Freeze approver is recorded as "project owner (solo tech lead)" | Placeholder identity. | Re-run `--approve "<reason>" --by "<your name>"` only if you want your name on the lock (bumps to v1.1.0). |
| `make` and `pre-commit` hooks not executed | No `make` on this machine; repo is not yet a git repository. | After `git init`: `pre-commit install && pre-commit run --all-files`. |

## What M0 unlocks (BUILD_PLAN §8 / TASKS.md "After M0")

Capabilities can now be built in parallel against `data/fixtures/` and the frozen signatures. Suggested next: T5.3/T5.4 settlement tests, then T1.3/T1.4 (document generator + mock PAS), T2.x ingestion, T3.x policy, T4.x evidence ★.

## Known contract notes carried forward

- `SettlementInputs.unavailable_inputs` exists so a sublimit that applies but cannot be determined (S-05) makes the engine return `cannot_calculate`; the engine honours it since T5.3; the Settlement agent must populate it (T5.5).
- `settlement_v1.yaml` keeps `water_damage_sublimit` `optional` (skip when truly absent); `unavailable_inputs` covers the applies-but-unknown case; see `docs/assumptions.md` Q-S4.
- Endorsement effective dates marked `schedule` are resolved from the PAS response (`PolicyVersionResponse.endorsements`) — to be built in T1.4/T3.3.
