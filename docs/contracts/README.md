# Contracts

Source: `src/claimtower/core/schemas/` (Pydantic v2). JSON Schemas here are **generated** — run `make schemas`; `python scripts/export_schemas.py --check` (also a pytest) fails on drift. All models reject unknown fields. Money is `Decimal` (serialised as a string). Schema changes after M0 need Lead approval.

Example: [examples/valid_claim_state_s01.json](examples/valid_claim_state_s01.json) (SYNTHETIC; the S-01 running example).

## `ClaimState` field owners (BUILD_PLAN §5)

| Field(s) | Owner (writes) | Module |
|---|---|---|
| `claim_header`, `intake_checklist`, `duplicate_candidates` | Intake agent | `intake.py` |
| `policy_snapshot` | Policy agent | `policy.py` |
| `documents`, `facts`, `image_findings` | Evidence agent | `intake.py`, `evidence.py` |
| `open_questions`, `contradictions`, `missing_evidence`, `counterfactuals` | Gap Analysis agent ★ | `evidence.py` |
| `coverage_issues` | Coverage agent | `decision.py` |
| `settlement` | Settlement agent | `decision.py` |
| `proposed_actions`, `drafts` | Action Planning agent | `decision.py` |
| `qa_report`, `audit_ref` | Quality & Audit agent | `decision.py` |
| `routing`, `human_approval_required`, `resolution_status` | Supervisor (deterministic) | `claim_state.py` |
| `claim_id`, `state_version`, `correlation_id` | `core/state.py` only | `claim_state.py` |

## Built-in invariants

- `Fact.evidence_text` non-empty; confidence in 0..1 (verbatim-substring check is added in T2.4).
- `Contradiction` needs ≥1 fact on each side.
- `CounterfactualScenario` rejects text predicting approval/denial.
- `RoutingDecision` needs ≥1 reason code.
- `PolicySnapshot.mock` defaults to `true` (guardrail 10).

## Frozen at M0 (v1.0.0)

The whole contract surface is locked by `CONTRACT_LOCK.json` (sha256 of every file below). Locked files:

| File(s) | What |
|---|---|
| `*.schema.json` | `ClaimState` + sub-models, integration request/response models (PAS / Claims / Recon), `SettlementInputs`, `StatePatch` (the agent protocol payload) |
| `ownership.json` | Which agent may write which `ClaimState` field |
| `protocols.json` | Method signatures of `Agent`, `PASClient`, `ClaimsClient`, `ReconClient` and the section-11 interfaces (`FactExtractor`, `ImageAnalyser`, `PolicyService`, `SettlementEngine`, `Router`, `AuditLog`) |

Python sources: `core/schemas/`, `core/interfaces.py`, `integrations/base.py`, `agents/base.py`, `core/state.py`.

### Changing a frozen contract
1. Edit the code, then `python scripts/export_schemas.py` (regenerates the files above).
2. `python scripts/freeze_contracts.py --check` now fails (and so does CI / pytest) - the change cannot slip in silently.
3. The tech lead accepts it on purpose: `python scripts/freeze_contracts.py --approve "<reason>" --by "<name>" [--bump major|minor|patch]`.
   This rewrites the lock, bumps `contract_version` and appends to [CHANGELOG.md](CHANGELOG.md). Commit the code, the regenerated files, the lock and the
   changelog together, and re-run `python scripts/build_fixtures.py` if fixtures are affected.
