# Next Tasks (solo order)

Source of truth: [BUILD_PLAN.md](BUILD_PLAN.md) §9 (task cards) and §10.3 (dependencies). Priority: P1 must · P2 should · P3 stretch. Size: S ½ day · M 1 day · L 2+ days.

**Legend:** ✅ done · 🟡 partly done (what is left is in the Notes) · ⬜ to do · ⏸ deferred (P2/P3). "(finish)" in a task name means it was completed (your convention, kept).
Status as of 2026-11-26.

## Status

| Done | Notes |
|---|---|
| Folder scaffold (BUILD_PLAN §7) | `src/claimtower/*`, `configs/`, `packs/`, `data/`, `tests/`, `evals/`, `scripts/` |
| Dependencies pinned | `pyproject.toml` (FastAPI, Streamlit, python-multipart added to the install) |
| Contracts frozen | `ClaimState` and all sub-models, ownership map, protocols: freeze v1.0.0 (`scripts/freeze_contracts.py --check`) |
| Settlement engine | deterministic engine + YAML rules + edge-case suite (T5.3, T5.4) |
| Ingestion → facts | loaders, OCR, classifier, extractors, facts with verbatim provenance (T2.1–T2.4) |
| Policy | decomposition, relations/precedence, effective set, snapshot, hybrid retrieval, Policy agent, model clause typing (T3.1–T3.5, 4a) |
| Evidence sufficiency ★ | question register, fact links, contradictions, sufficiency, minimum-evidence requests, counterfactuals, Gap agent + incremental reassessment (T4.1–T4.7) |
| Claim graph | LangGraph graph with the human gate; **real agents: Evidence, Policy, Gap, Coverage, Settlement, Action, Routing; no mocks left** (`docs/orchestration.md`) |
| API + UI | FastAPI (11 endpoints, OpenAPI contract in `docs/contracts/openapi.json`) and a simple Streamlit UI (`docs/api.md`) |
| Samples, docs, CI | `samples/claim_sample_1`, `TEAM_HEADSUP.md`, `API_FLOW.md`, `AZURE_OPENAI_SETUP.md`, `NEW_INSURANCE_CHALLENGES.md`, GitHub Actions (ruff, import-linter, pytest) |
| Tests | 1163 passing, 1 skipped (no test calls a live model: replay cassettes and fakes only) |

CODEOWNERS / 4-person split is deferred while working solo.

## Phase 0 → M0 (contracts frozen): all done

| # | Task | Pri/Size | Status | Purpose | Deliverable | Done when |
|---|---|---|---|---|---|---|
| 1 | **T0.1 (finish)** Framework spike | P1 / M | ✅ | Prove LangGraph supports typed state, checkpoint/resume, human interrupt, partial re-run | `tests/integration/test_spike.py` (2 dummy nodes + interrupt + resume), `docs/adr/0001-framework.md` finalised, `pre-commit` config | Spike passes: interrupt → resume → partial re-run |
| 2 | **T0.2 (finish)** Core schemas | P1 / L | ✅ | The contract everything builds on | Replace `_Stub` models in `core/schemas/` with real ones (BUILD_PLAN §6); `scripts/export_schemas.py` → JSON Schema in `docs/contracts/` | All sub-models real; export runs; schema-drift check possible |
| 3 | **T0.3 (finish)** State patching & versioning | P1 / M | ✅ | Agents write only owned fields; every patch bumps version | Harden `core/state.py`: version history store, replay, ownership map per agent | Foreign write raises `OwnershipViolation`; any version replayable |
| 4 | **T0.5 (finish)** Config & pack loader | P1 / S | ✅ | Keep business rules out of code | `core/config.py`, `core/registry.py`; YAML validation at startup | Editing `routing_rules.yaml` changes behaviour; bad YAML fails clearly |
| 5 | **T0.4 (finish)** LLM wrapper | P1 / M | ✅ | One place to swap providers, log model/prompt versions | `core/llm.py`: `chat()`, `structured()`, `vision()`, `embed()`; record/replay cassettes; prompt loader | Same call works on 2 providers via config (Azure path untested: `AZURE_OPENAI_SETUP.md`); replay is deterministic |
| 6 | **T1.2 (finish)** First 5 scenarios + ground truth | P1 / L | ✅ | Drive development and evaluation | `data/scenarios/scenarios.yaml` (S-01–S-05), `data/ground_truth/*.json` | 5 scenarios labelled (expected facts, gaps, issues, tier, calc) |
| 7 | **T1.1 (finish)** Synthetic policy corpus | P1 / L | ✅ | Tricky wording to test policy/coverage agents | Policy + endorsements in `data/synthetic/policies/` + `INDEX.md` (stable clause IDs, all marked SYNTHETIC) | Wording plausible; clause IDs stable |
| 8 | **T1.5 (finish)** SAMPLE business rules | P1 / S | ✅ | Plausible values without inventing insurer policy | Review `routing_rules.yaml`, `settlement_v1.yaml`, `docs/assumptions.md`; questions for a domain expert | Every number has `# SAMPLE` |
| 9 | **M0 checkpoint (finish)** | — | ✅ | Freeze contracts | Schemas + protocols exported; 5 scenarios ready | Schema changes after this are deliberate |

## After M0: what is done, what is left, and the order to do it

### Done (steps 1–5)

| Step | Tasks | Status |
|---|---|---|
| 1 | T5.3 / T5.4 settlement engine, tests & edge cases (finish) | ✅ |
| 2 | T1.3, T1.4 doc generator + mock PAS data (finish) | ✅ |
| 3 | T2.1–T2.4 ingestion → facts with provenance (finish) | ✅ |
| 4 | T3.1–T3.5 policy decomposition → snapshot → retrieval (finish) | ✅ |
| 4a | LLM tie-breaker for unclassified policy clauses (finish) | ✅ (on; wrong-type risk logged as Q-P5) |
| 5 | **T4.1–T4.7 evidence sufficiency ★ (finish)** | ✅ |

### Built since, outside the original order

| Task | Status | What exists / what is left |
|---|---|---|
| T6.1 Graph and supervisor | ✅ | LangGraph graph, node wrapper (ownership, version, events), routing, failure parking; agents not built yet run as mocks |
| T7.1 FastAPI service | ✅ | all T7.1 endpoints + documents/status/agents; 30 contract tests |
| T7.2 Streamlit shell | ✅ | simple UI over the API only (create claim, upload, decide, tabs for facts, gaps, policy, history) |
| T6.3 Human approval gate | ✅ | approve / edit / reject / request more evidence by a named reviewer; role order from `rbac.yaml`, reject needs the `deny_claim` role; edits change settlement inputs with a recorded reason; no path to a payment without a recorded authorised decision (tested) |
| T6.2 Incremental reassessment | ✅ | `orchestration/dependencies.py` declares what each node reads (proved by a test); unchanged nodes are skipped and logged, changed ones say why; `skipped_nodes` in the status |
| T6.5 Integration protocols and mock adapters | ✅ | PAS (+ `get_status`), Claims and Recon mock adapters (in-process, idempotent, every response `mock=True`), failure injection (`mock/faults.py`), mock to live as a flag (`build_adapters`; the live adapters are stubs that fail loudly). How to replace one: `docs/REPLACING_MOCK_ADAPTERS.md`. No mock servers (not in the plan) |
| T7.3 Document-AI panel | ✅ | Document reader tab: the page picture with the place a fact was read marked, the source excerpt beside it, type and reason, OCR flag, extracted fields with page and confidence, low confidence marked, a duplicate badge that says "not done yet" (T6.8), what the agents did. The facts table opens a fact in the reader. No accusation wording anywhere (tested) |
| T7.4 Gap board and counterfactual view | ✅ | Gap board: conflicting, missing and insufficient evidence in three columns (both sides of a conflict next to each other with document, page and quote; who is asked and for what), not-yet-judged and settled questions below; counterfactual view: scenarios A/B/C per open question, the issues each would re-assess, and a scenario-by-issue matrix. Tested on the running example (S-01) |

### Left to do, in the order to work on it

| Order | Step | Tasks | Status | Why this order |
|---|---|---|---|---|
| 1 | 6 | **T5.1** coverage issue matrix | ✅ | Done: pack decision table + checked model reading (`coverage/`, `agents/coverage.py`). Fresh-set eval: 7/12, 2 exclusion rows wrongly `established` on facts that say nothing: fix before relying on the model reading (`reading.enabled`) |
| 2 | 6 | **T5.2** abstention policy | ✅ | Done: reason codes in the pack table, model ambiguity check (BASE-HOME-2 "resulting damage"), abstentions counted by reason in the eval. Ambiguity asked only where an exclusion has an exception clause (rule); fresh 9/11, ambiguity 2/2. Open: clear cases under BASE-HOME-2 still flagged (2 fresh); BASE-HOME-3 exclusion not matched by the pack selectors |
| 3 | 6 | **T5.5** settlement agent (extract inputs → engine → explain) | ✅ | Done: rules + checked model reading of amounts, schedule/payments from the mock PAS and claims systems, verified explanation, reserve forecast kept apart from the payable. Fresh amounts eval 7/7, no wrong number |
| 4 | 6 | **T5.6** routing / tiering engine | ✅ | Done: `routing/tiering.py` (rules evaluator), `routing/features.py` (features from the state), `agents/routing.py`. 5/5 labelled tiers agree. Without a model every real claim routes complex (nothing is established yet); duplicates and specialist flags are assumed until T6.8 / vision |
| 5 | 7 | **T6.7** intake agent (FNOL, checklist, PAS lookup) | ✅ | Done: header from the notice of loss, one consolidated checklist, exact policy lookup, registration through the claims adapter; the five registration cases pass. Duplicate candidates stay with T6.8 |
| 6 | 7 | **T6.5** ✅, **T6.6** ✅ idempotency store, bounded retry with backoff, dead-letter queue, and the real commit step | ✅ | The integration frame the actions and the commit need |
| 7 | 8 | **T6.4** action planner, drafts, tasks (and the real commit step) | ✅ | Planner, drafts, tasks and the commit step through the adapters (claim, status, controlled tasks, payment only for an authorised approver) |
| 8 | 8 | **T6.2** dependency map, **T6.3** gate extras | ✅ | Harden the human-approval flow |
| 9 | 9 | **T7.5** issue matrix, calculation trace, approval UI; then **T7.3, T7.4** panels | ✅ | Show what now exists (coverage, settlement, routing) |
| 10 | 10 | **T8.4** autonomy matrix + RBAC guards, **T8.5** Quality & Audit agent | ✅ | **T8.4** `audit/guards.py` enforces the autonomy matrix (default deny, no agent for human-only operations, payments only through a guarded Recon, reject through the guard, blocked actions at commit), denials logged as `action_denied`; governance group tested. **T8.5** quality checks block the release to the gate |
| 11 | 10 | **T8.1** evaluation harness (one scorecard), **T8.2** scenario replay, **T8.3** audit trail + dossier export | ✅ | **T8.1** scorecard (`make eval`): 13 end-to-end cases scored against the labelled scenarios, per-scenario failure list, component baselines, committed baseline with a regression check. **T8.2** 13 scenario replays in CI, two with recorded model answers; `scripts/replay_scenario.py`. **T8.3** audit trail and decision pack done earlier. See `docs/EVALUATION.md` |
| 12 | 10 | **T8.7** demo script and rehearsal → **M4/M5** | ⬜ | Last |

### Decided follow-ups (not tasks in the plan)

| Item | Decision | When |
|---|---|---|
| Evidence model fallback (document intelligence) | ✅ Switched on in the Evidence agent when a model is given (classifier tie-breaker, field filler, scan OCR); measuring on real-format files still skipped | Done |
| Q-SUBLIMIT for a policy with no sublimit endorsement | Keep current behaviour (reported `missing`, request to policy administration); POL-1013 is the sample policy that avoids it | No action |
| Adjuster review of requests and scenarios | Adjuster rates `evals/review/request_review.csv` and `counterfactual_review.csv` | Whenever an adjuster is available |
| Open domain questions | Q-R, Q-S, Q-P, Q-D, Q-O, Q-A in `docs/assumptions.md` | Before any value leaves SAMPLE |

### Deferred (P2/P3): cut in this order if time is short
T6.8 duplicates → T2.5 image quality / T2.6 visible-damage observations → T7.6 state-version timeline and metrics → T6.9 event bus → T8.6 benefits and metrics → T5.2 depth.

### Known prototype limits (not scheduled)
No authentication; claims, checkpoints and events in memory (lost on restart); one claim processed at a time; synchronous requests; Azure OpenAI path untested; one pack only. See `NEW_INSURANCE_CHALLENGES.md`.

## Rules of thumb

- Build against fixtures in `data/fixtures/`; never block on unfinished upstream code.
- If time is short, cut in the order given under "Deferred".
- `settlement/` never imports LLM code; no float money; mark invented numbers `# SAMPLE`.
- No test calls a live model: use replay cassettes or fakes. Mocks in the graph say `YET TO IMPLEMENT`; replacing one is a line in `claimtower/composition.py`.
