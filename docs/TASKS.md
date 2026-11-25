# Next Tasks (solo order)

Source of truth: [BUILD_PLAN.md](BUILD_PLAN.md) §9 (task cards) and §10.3 (dependencies). Priority: P1 must · P2 should · P3 stretch. Size: S ½ day · M 1 day · L 2+ days.

## Status

| Done | Notes |
|---|---|
| Folder scaffold (BUILD_PLAN §7) | `src/claimtower/*`, `configs/`, `packs/`, `data/`, `tests/`, `evals/`, `scripts/` |
| Dependencies pinned | `pyproject.toml` |
| Skeletons | `ClaimState` + `Fact`, `apply_patch` (`core/state.py`), agent protocol, settlement engine + sample YAML |
| Tests | 2 passing (settlement example, ownership violation) |

CODEOWNERS / 4-person split is deferred while working solo.

## Next tasks — Phase 0 → M0 (contracts frozen)

| # | Task | Pri/Size | Purpose | Deliverable | Done when |
|---|---|---|---|---|---|
| 1 | **T0.1 (finish)** Framework spike | P1 / M | Prove LangGraph supports typed state, checkpoint/resume, human interrupt, partial re-run | `tests/integration/test_spike.py` (2 dummy nodes + interrupt + resume), `docs/adr/0001-framework.md` finalised, `pre-commit` config | Spike passes: interrupt → resume → partial re-run |
| 2 | **T0.2 (finish)** Core schemas | P1 / L | The contract everything builds on | Replace `_Stub` models in `core/schemas/` with real ones (BUILD_PLAN §6); `scripts/export_schemas.py` → JSON Schema in `docs/contracts/` | All sub-models real; export runs; schema-drift check possible |
| 3 | **T0.3 (finish)** State patching & versioning | P1 / M | Agents write only owned fields; every patch bumps version | Harden `core/state.py`: version history store, replay, ownership map per agent | Foreign write raises `OwnershipViolation`; any version replayable |
| 4 | **T0.5 (finish)** Config & pack loader | P1 / S | Keep business rules out of code | `core/config.py`, `core/registry.py`; YAML validation at startup | Editing `routing_rules.yaml` changes behaviour; bad YAML fails clearly |
| 5 | **T0.4 (finish)** LLM wrapper | P1 / M | One place to swap providers, log model/prompt versions | `core/llm.py`: `chat()`, `structured()`, `vision()`; record/replay cassettes; prompt loader | Same call works on 2 providers via config; replay is deterministic |
| 6 | **T1.2 (finish)** First 5 scenarios + ground truth | P1 / L | Drive development and evaluation | `data/scenarios/scenarios.yaml` (S-01–S-05), `data/ground_truth/*.json` | 5 scenarios labelled (expected facts, gaps, issues, tier, calc) |
| 7 | **T1.1 (finish)** Synthetic policy corpus | P1 / L | Tricky wording to test policy/coverage agents | Policy + endorsements in `data/synthetic/policies/` + `INDEX.md` (stable clause IDs, all marked SYNTHETIC) | Wording plausible; clause IDs stable |
| 8 | **T1.5 (finish)** SAMPLE business rules | P1 / S | Plausible values without inventing insurer policy | Review `routing_rules.yaml`, `settlement_v1.yaml`, `docs/assumptions.md`; questions for a domain expert | Every number has `# SAMPLE` |
| 9 | **M0 checkpoint (finish)** | — | Freeze contracts | Schemas + protocols exported; 5 scenarios ready | Schema changes after this are deliberate |

## After M0 — suggested solo order

| Step | Tasks | Why this order |
|---|---|---|
| 1 | T5.3 / T5.4 settlement tests & edge cases (finish) | Engine already exists; lock behaviour early |
| 2 | T1.3, T1.4 doc generator + mock PAS data (finish) | Needed as input for ingestion/policy |
| 3 | T2.1–T2.4 ingestion → facts with provenance (finish) | Feeds the hero engine |
| 4 | T3.1–T3.5 policy decomposition → snapshot → retrieval (finish)| Needed for open questions |
| 4a | **Reminder, right after T3.5 (optional) (finish):** LLM tie-breaker for clauses the decomposer leaves in `unclassified` (T3.1) | Same pattern as the classifier tie-breaker: allowed types only, verbatim quote, confidence cap, validated on a fresh unseen policy document before it is trusted. Do it earlier only if a demo scenario needs an unclassified clause |
| 5 | **T4.1–T4.7 evidence sufficiency ★ (finish)** | Hero feature; build on fixtures if upstream isn't ready |
| 6 | T5.1, T5.2, T5.5, T5.6 coverage, abstention, settlement agent, routing | Consume WS4 output |
| 7 | T6.5–T6.7 mock adapters, idempotency, intake | Integration frame |
| 8 | T6.1–T6.4 graph, incremental reassessment, human gate, actions → **M2/M3** | Wire real agents together |
| 9 | T7.1–T7.5 API + UI | Reviewer-facing |
| 10 | T8.1–T8.7 eval, audit, guards, demo → **M4/M5** | Scorecard and rehearsal |

## Rules of thumb

- Build against fixtures in `data/fixtures/`; never block on unfinished upstream code.
- If time is short, cut in this order: T6.8 duplicates → T2.5/T2.6 images → T7.6 → T6.9 → T8.6 → T5.2 depth.
- `settlement/` never imports LLM code; no float money; mark invented numbers `# SAMPLE`.
