# The API (T7.1)

FastAPI service in `src/claimtower/api/`; the only thing the UI and any integration talk to. The contract is `docs/contracts/openapi.json` (regenerate with
`python scripts/export_openapi.py`; a test fails if it drifts) and the live interactive docs are at `/docs`.

```
python scripts/run_api.py                  # http://127.0.0.1:8000/docs    (--llm: the Gap agent also uses the model; --store DIR: keep uploads)
uvicorn claimtower.api.main:build_default_app --factory
pytest tests/api                            # 30 contract tests
```

What runs behind it: the claim graph (`docs/orchestration.md`) with the real Evidence, Policy and Gap agents and mocks ("YET TO IMPLEMENT") for the rest, on SYNTHETIC data.
Every response carries `X-Mock-Agents: intake,coverage,...`; `GET /health` and `GET /agents` say which agent is which.

| Endpoint | What it does | Errors |
|---|---|---|
| `GET /health` | service, mock and implemented agents, claim count | |
| `GET /agents` | the roster: status (`implemented` / `YET TO IMPLEMENT (mock)`), the task that will implement each | |
| `POST /claims` | create a claim and run it to the human gate. Body (all optional): `claim_id`, `policy_ref`, `loss_date`, `loss_type`, `loss_description`, `claimant_name`, `channel`, `use_synthetic_documents` (demo: load `data/synthetic/claims/<id>`). Returns the claim view | 409 duplicate id, 404 no synthetic folder, 422 bad body |
| `GET /claims` | one line per claim (phase, tier, open questions by status) | |
| `GET /claims/{id}` | the claim view: `status` (phase, gate, nodes run, mock nodes) and the full `state` at the current version | 404 |
| `GET /claims/{id}/status` | the status only (cheap to poll) | 404 |
| `GET /claims/{id}/versions` | every committed version and the node that produced it | 404 |
| `GET /claims/{id}/diff?from=&to=` | what changed between two versions: added / removed / changed items per field, and every question whose status moved | 404 unknown claim or version, 422 |
| `GET /claims/{id}/documents` | the claim's documents (type, pages, source) | 404 |
| `POST /claims/{id}/documents` | multipart upload (`file`, optional `uploaded_by`, `source_channel`; `?reassess=false` to only store). Re-assesses the claim from the evidence node: only the new document is processed, and a recommendation waiting at the gate is superseded | 404, 413 (over 20 MB), 422 empty |
| `GET /claims/{id}/documents/{doc}/file` | the original file | 404 |
| `POST /claims/{id}/decision` | the human gate: `{decision: approve\|reject\|request_more_evidence, reviewer_id, reviewer_role, reason}` | 409 not waiting for a decision, 422 refused (the reason in `detail.error`; the gate stays open) |
| `GET /claims/{id}/audit` | the event log: every agent (with `mock`), routing, human decisions and refusals | 404 |

## The UI

A simple Streamlit UI sits on top of exactly these endpoints (`src/claimtower/ui/client.py` is its only way to the system):
```
python scripts/run_api.py                   # terminal 1
python scripts/run_ui.py                    # terminal 2: http://localhost:8501      (python scripts/run_ui.py --start-api starts both)
```
Sidebar: the API address, the agents (real or mock), the claims, and a form that creates a claim and runs it to the gate. For the open claim: **Overview & decision** (tier,
open questions, the human decision form), **Documents** (upload, download), **Facts** (each with its verbatim evidence), **Questions & gaps** (questions, contradictions side by side,
evidence requests, scenarios), **Policy**, **History** (versions, diff, audit log). A yellow banner names the mock agents whenever their output is on screen.

## The claim view and the phases

`status.phase`: `awaiting_human_decision` (paused at the gate: `status.gate` holds what the reviewer sees), `completed` (decided), `halted` (an agent failed: `halted_reason`,
`paused_at`; the claim is parked `awaiting_human`), `not_started`. `resolution_status` is the claim's business status (`awaiting_evidence`, `awaiting_human`, `approved`, `closed`, ...).

## Rules the API keeps

- The system only **recommends**. `approve` pays and denies nothing: the commit step is a mock. A reviewer's role must reach the claim's `required_approver_role`
  (adjuster < senior_adjuster < supervisor, SAMPLE); reject and request_more_evidence need a reason; a refused decision changes nothing.
- A mock agent's output is a labelled fixture, not computed; it is always flagged (`mock: true` in the audit log, `mock_nodes_run`, the header).

## Prototype limits (do not build on them)

- **No authentication:** the reviewer's id and role are asserted by the caller. Real identity and RBAC (`configs/rbac.yaml`) come with T6.3 / T7.
- **In memory:** claims, checkpoints and the audit log are lost on restart (uploaded documents survive only with `--store`). One claim is processed at a time.
- **Synchronous:** a request returns when the graph reaches the gate. Without the model that is about a second; with `--llm` it can take a minute or more.
- **No idempotency keys** on `POST /claims` yet (T6.6 / T6.8): a repeated call with the same `claim_id` is a 409, not a replay.
