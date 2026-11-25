# Fixtures (SYNTHETIC)

Generated from the ground truth by `python scripts/build_fixtures.py` (checked by `tests/unit/test_m0_gate.py`). Do not edit by hand.

Per scenario `S-xx/`:

| File | Content |
|---|---|
| `claim_state.json` | Complete EXPECTED final `ClaimState` (valid against the frozen contract) |
| `intake.json` `policy.json` `evidence.json` `gap.json` `coverage.json` `settlement.json` `action.json` `audit.json` `supervisor.json` | One slice per agent holding exactly the fields it owns (`core.state.OWNERSHIP`). Applying them in this order with `apply_patch` reproduces `claim_state.json` |
| `get_snapshot_result.json` | S-03 only: the `NotInForce` answer a `PolicyService` returns |

**Fixture rule (BUILD_PLAN section 11):** develop against these; never block on someone else's code. A stub agent returns its slice
(`json.load` the file and wrap it in a `StatePatch`). Real agents are measured against the same files by the T8.1 eval.
