# Navigation & Team Ownership

## The one idea
`ClaimState` (in `core/schemas`) is the contract. Each **agent** writes only the fields it **owns**.
Each **capability folder** holds the logic; agents are thin wrappers. Mock vs live and LOB differences
live in `integrations/` and `packs/`, never in core code.

## 4-person ownership (BUILD_PLAN section 10.2)
| Who | Folders | Tasks |
|---|---|---|
| **A** Lead | `core/` `orchestration/` `integrations/` `actions/` `duplicates/` `configs/` | WS0, WS6, T8.4 |
| **B** Docs + Evidence (hero) | `ingestion/` `vision/` `evidence/` + packs (questions, doc types) | WS2, WS4 |
| **C** Policy + Settlement | `policy/` `coverage/` `settlement/` `routing/` + packs (calculation) | WS3, WS5 |
| **D** Data + UI + Eval | `data/` `api/` `ui/` `audit/` `evals/` `docs/` `scripts/` | WS1, WS7, WS8 |

## Where do I find...?
| Looking for | Go to |
|---|---|
| A data model / contract | `src/claimtower/core/schemas/` |
| Which agent writes which field | `src/claimtower/agents/<name>.py` (`owns`) |
| Business rules (routing, calc, autonomy) | `configs/`, `packs/*/calculation/` - never hard-coded |
| Hero logic | `src/claimtower/evidence/` |
| Mock PAS / Claims / Recon | `src/claimtower/integrations/mock/` |
| Test scenarios, fixtures | `data/scenarios/`, `data/fixtures/` |
| A task card | BUILD_PLAN.md section 9 (task id is in each folder's `__init__.py`) |

## Working rules
1. Build against **fixtures** in `data/fixtures/`; never wait for a teammate's code.
2. Touch only your folders; need a change elsewhere -> PR to the owner (CODEOWNERS).
3. Schema/signature changes after M0 need Lead approval.
4. `settlement/` never imports LLM code. No float money. No invented thresholds (mark `# SAMPLE`).
5. Branches: `<initial>/<task-id>-short-name`, e.g. `b/T4.3-contradictions`. Small PRs, 1 reviewer.
6. Mocks and synthetic data are always labelled.

## Flow (see BUILD_PLAN section 4)
`intake -> policy -> evidence -> gap* -> coverage -> settlement -> action -> audit -> routing -> HUMAN GATE -> Claims/Recon`
