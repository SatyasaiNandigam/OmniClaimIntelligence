# API FLOW: a claim from first notice to a decision, end to end

Status as of 2026-11-26, after all P1 tasks. Prototype, SYNTHETIC data only. Every agent is real; the three outside systems (policy system, claims system, payment reconciliation) are in-process mocks, and nothing is sent anywhere.

More detail: `docs/api.md` (every endpoint), `docs/orchestration.md` (the graph), `TEAM_HEADSUP.md` (what uses a model), `docs/EVALUATION.md` (the scorecard).

## 0. Start it

```
python scripts/run_api.py                       # http://127.0.0.1:8000/docs   (interactive: "Try it out" has a file picker)
python scripts/run_api.py --llm                 # the Evidence, Gap, Coverage, Settlement and Action agents also use the model (needs LLM_API_KEY in .env)
python scripts/run_api.py --store var/store     # keep uploaded documents AND the audit trail between runs
python scripts/run_ui.py                        # the Streamlit UI on http://localhost:8501 (talks to this API only); --start-api starts both
```

Without `--llm` the API makes no model calls at all. Without `--store` the documents and the audit trail live in a temporary folder; claims, checkpoints and the event log are in memory either way and are lost on restart.

## 1. What happens to a claim

```
create claim ─► intake ─► policy ─► evidence ─► gap ─► coverage ─► settlement ─► action ─► audit ─► routing ─► HUMAN GATE ─► commit
                                                                                              (quality checks)         approve only
```

| Step | What it does | Uses the model (`--llm`) |
|---|---|---|
| **intake** | Reads the first notice of loss, fills any header field you did not supply, builds ONE checklist of what is still needed, looks the policy up exactly (policy number and loss date), registers the claim in the claims system | only through the document classifier / field fallback |
| **policy** | The policy in force on the loss date: version, endorsements, clauses. No policy in force: no snapshot, and the claim goes to an operator | no |
| **evidence** | Types every document, extracts fields and facts with the page and the verbatim quote, transcribes scans | tie-breaker, field filler, scan OCR |
| **gap** | The open questions, which clause each bears on, contradictions, what evidence is missing and who to ask, scenarios A/B/C | yes |
| **coverage** | One row per issue with its clause citations and a status that says how well it is resolved. Never a verdict | yes (reads the facts against the clause, checks for ambiguity) |
| **settlement** | Reads the amounts, calculates with the deterministic engine, explains it. Assessed loss, payable and reserve forecast are kept apart | reads amounts, rewords the explanation (the engine, not the model, calculates) |
| **action** | Next steps: one consolidated request per addressee, tasks, a status message. All drafts; nothing is sent | rewords drafts (checked) |
| **audit** | Quality checks before the gate: citations, the settlement total, outcome wording, nothing executed early. A failure parks the claim with the reasons | no |
| **routing** | Tier (simple, medium, complex), reason codes, the role that must decide. From the routing rules, never from model confidence | no |
| **gate** | Pauses until a named person decides | |
| **commit** | After an approval only: claim status, tasks, and (supervisor only) the payment instruction, through the mock adapters | no |

## 2. The flow, step by step

### Step 1: create the claim

Two ways to start a claim: create it and then upload (this step and the next), or create it WITH its documents in one request (`POST /claims/with-documents`, below). Later evidence always goes to `POST /claims/{id}/documents`.

```
curl -X POST http://127.0.0.1:8000/claims -H "content-type: application/json" \
  -d '{"claim_id":"CLM-1001","policy_ref":"POL-1013","loss_date":"2026-05-22","loss_type":"water_damage","loss_description":"Dishwasher inlet hose burst"}'
```

- Everything in the body is optional, but give **`policy_ref` and `loss_date`**: the policy must exist in the mock policy system (`data/synthetic/pas_mock_db.json`, for example `POL-1001`, `POL-1013`) and the loss date decides which version and which endorsements apply. Without them there is no policy snapshot.
- Use a new id. `CLM-0001` to `CLM-0005` are the labelled demo claims: add `"use_synthetic_documents": true` to load their documents, and the intake agent then reads the header from the notice of loss.
- The claim runs to the gate at once. With no documents yet it has no facts, so it stops at the gate marked complex with the settlement `cannot_calculate`.
- The claim is registered in the claims system here (idempotent: the same claim is never created twice).

### Shortcut: create the claim with its documents in one request

```
curl -X POST http://127.0.0.1:8000/claims/with-documents -F "claim_id=CLM-1001" -F "policy_ref=POL-1013" -F "loss_date=2026-05-22" -F "uploaded_by=claimant" \n  -F "files=@samples/claim_sample_1/fnol.pdf" -F "files=@samples/claim_sample_1/plumber_report.pdf" -F "files=@samples/claim_sample_1/surveyor_report.pdf"
```

- Every field is optional; what is missing is read from the claim form by the intake agent, and what you supply is kept.
- All files are stored first, then the claim is assessed ONCE. The response has `claim` (the claim view), `stored` and `skipped` (a file that is empty, too large or unreadable is skipped with its reason; the claim is still created if at least one file was stored, and refused with 422 if none was).
- After that, more evidence goes to `POST /claims/{id}/documents` (step 2), which re-assesses the claim.

### Step 2: upload the documents

```
curl -X POST "http://127.0.0.1:8000/claims/CLM-1001/documents?reassess=false" -F "file=@samples/claim_sample_1/fnol.pdf" -F "uploaded_by=claimant"
curl -X POST "http://127.0.0.1:8000/claims/CLM-1001/documents?reassess=false" -F "file=@samples/claim_sample_1/plumber_report.pdf" -F "uploaded_by=plumber"
...surveyor_report.pdf, estimate.pdf, invoice.pdf the same way...
curl -X POST "http://127.0.0.1:8000/claims/CLM-1001/documents" -F "file=@samples/claim_sample_1/claimant_email.pdf" -F "uploaded_by=claimant"      # the last one re-assesses
```

- Use `?reassess=false` on all but the last so the claim is re-assessed once, not once per file. (Each upload is checked for size, 20 MB limit, and an empty file is refused.)
- A re-assessment starts at the **intake** node: the notice of loss is read, the checklist is refreshed, and only the new documents are processed. A node whose inputs did not change is **skipped**; `GET /claims/{id}/status` lists them in `skipped_nodes`, and the audit log says "skipped (unchanged)".
- If the claim was waiting at the gate, the new evidence **supersedes** that recommendation: nobody had decided yet, and the gate is presented again after the re-assessment.
- Until the claim is assessed a new file shows `detected_type: "unknown"`; afterwards `GET /claims/{id}/documents` shows the real types. A document the rules cannot type is kept as `unknown` and yields no facts (there is no endpoint yet to type one by hand).

### Step 3: look at what the system found

| To see | Call |
|---|---|
| Everything: the status and the full state | `GET /claims/CLM-1001` |
| Just where it stands (cheap to poll): phase, gate, nodes run, nodes skipped | `GET /claims/CLM-1001/status` |
| Documents with their detected type | `GET /claims/CLM-1001/documents` |
| What the system read in one document: the type and why, the pages and whether the text is OCR, the extracted fields with page, region and confidence | `GET /claims/CLM-1001/documents/{doc}/details` |
| The page as a picture; with `?fact=<id>` the place that fact was read is marked | `GET /claims/CLM-1001/documents/{doc}/page/1?fact=F-...` |
| The original file | `GET /claims/CLM-1001/documents/{doc}/file` |
| Every version and which node made it; what changed between two | `GET /claims/CLM-1001/versions`, `GET /claims/CLM-1001/diff?from=3&to=18` |
| The run log: every agent, skipped node, human decision and blocked attempt | `GET /claims/CLM-1001/audit` |

In `state` you will find: `claim_header`, `intake_checklist`, `documents`, `facts` (each with page, verbatim `evidence_text`, `extraction_confidence`), `policy_snapshot`, `open_questions`, `contradictions`, `missing_evidence` (who to ask and for what), `counterfactuals`, `coverage_issues` (with `clause_citations`), `settlement`, `proposed_actions`, `drafts`, `qa_report`, `routing`, `resolution_status`.

### Step 4: read the gate

When `status.phase` is `awaiting_human_decision`, `status.gate` is what the reviewer sees: `tier`, `reason_codes`, `required_approver_role`, `open_questions`, `contradictions`, `evidence_requests`, `coverage_issues`, `settlement`, `overrides` (edits already in force), `low_confidence_facts` (readings to check against the document), `allowed_decisions`, and `error` when the last decision was refused.

Without `--llm` the facts stay unverified, so the claim is routed **complex** and needs a `senior_adjuster`. With the model, the tier follows the evidence. On the sample claim without the model, after the uploads: settlement `calculated`, assessed loss 64,500 (the surveyor's assessment, not the claimant's 72,000 estimate), payable 54,500, tier complex, quality checks passed.

### Step 5: decide

```
curl -X POST http://127.0.0.1:8000/claims/CLM-1001/decision -H "content-type: application/json" \
  -d '{"decision":"approve","reviewer_id":"rev-9","reviewer_role":"supervisor"}'
```

| Decision | Needs | What happens |
|---|---|---|
| `approve` | the claim's required role or higher; no reason needed | the claim goes to the commit step (step 6) |
| `edit` | the required role, a `reason`, and `edits` such as `{"non_covered_items_total":"28000"}` | the settlement is calculated again with your input (you change an INPUT, never the payable); the gate opens again; the edit is recorded with who and why. Inputs: `assessed_covered_loss`, `non_covered_items_total`, `applicable_deductible`, `water_damage_sublimit`, `remaining_policy_limit`, `prior_payments_total` |
| `request_more_evidence` | the required role and a `reason` | the claim waits for evidence; upload documents (step 2) and it comes back to the gate |
| `reject` | `senior_adjuster` or higher (the autonomy matrix: deny claim) and a `reason` | the claim is closed; nothing is committed |

- Roles are ordered `adjuster` < `senior_adjuster` < `supervisor` (SAMPLE, `configs/rbac.yaml`). A role below the claim's `required_approver_role` gets **422** with the reason in `detail.error`, and **the gate stays open**: a refused decision changes nothing. The reviewer's id and role are asserted by the caller (no authentication yet).
- An approval records a decision event with the reviewer; a rejected attempt (for example an adjuster trying to reject) is also logged as a blocked action.

### Step 6: what an approval does

The commit step runs through the mock adapters, idempotently (a repeat or a retry never creates a second claim, task or payment):

1. the claim status becomes `approved` in the claims system (the claim was registered at intake);
2. each proposed **controlled** action becomes a task in the claims system (for example the endorsement request, the surveyor follow-up). **Drafts are never sent**: they wait for a person;
3. a **payment instruction** goes to the reconciliation system **only** when the settlement is calculated **and** the approver's role may release payments (`supervisor`). A senior adjuster's approval pays nothing, and the audit log says why.

Look at it in `GET /claims/CLM-1001/audit` (the `commit` event lists the calls and the payment result). Example from the sample claim approved by a supervisor: `approval committed through the adapters (mock systems): 2 call(s) made, 1 already done earlier, 0 in the dead-letter queue; payment: instruction RECON-0001 accepted` (the "already done" call is the registration from intake, replayed).

### Step 7: the record of the decision

| To get | Call |
|---|---|
| The persistent audit trail of the claim, hash-chained, and whether its chain is intact | `GET /claims/CLM-1001/trail` |
| The **decision pack**: who decided what, on which evidence and wording, with the calculation, the overrides and the model and prompt versions; built from the trail alone | `GET /claims/CLM-1001/decision-pack` (Markdown) or `?format=pdf` |

The trail holds the full output of every agent and every human decision, so the claim state at any version can be rebuilt from it alone. It survives a restart only if the server was started with `--store`.

### Adding a policy wording (before or between claims)

The policy **wording** (the clauses) is separate from the policy **system** (the mock PAS, which says which policy exists, which wording and endorsements apply on which dates, and the schedule values). Wordings are added through the API; the PAS entry is still `data/synthetic/pas_mock_db.json` (add it in `scripts/seed_pas_mock.py` and re-run it; the real system replaces it, see `docs/REPLACING_MOCK_ADAPTERS.md`).

```
curl -X POST "http://127.0.0.1:8000/policies/documents" -F "file=@From the broker (2).pdf" -F "uploaded_by=ops-1"
curl http://127.0.0.1:8000/policies/documents                       # what the library holds
curl -X POST http://127.0.0.1:8000/policies/reload                  # read the policy folders again (for files you dropped in by hand)
curl -X POST "http://127.0.0.1:8000/claims/CLM-1001/reassess"        # pick the new wording up on a claim already assessed (default node: policy)
```

- **The file name does not matter.** With `--llm` the **model reads the document's identity**: the identifier the document gives itself (a wording or form reference, an endorsement number, or a policy number when the document is the policy), its version, title and kind. The id it names must appear in the document's text (it is checked, never invented), and the response shows the words it relied on (`id_evidence`) and the policy number and period it saw (`details`, for information). Without `--llm`, or when the model cannot decide, fixed patterns for lines such as `Policy wording BASE-HOME-1 | Version 2025-01` are tried (`packs/property_water_damage/policy_documents.yaml`). Slashes and spaces in an identifier become `-` (`HSP/2026/B` is filed as `HSP-2026-B`). That id must match the `base_wording_id` or endorsement id the PAS names. A document with no id inside is refused (422); send it with `-F document_id=...` if you know what it is.
- **Same content again** is `unchanged` (200). **Different content under an existing id** is refused (409) unless you add `?replace=true`: a changed wording changes every claim assessed afterwards, so that must be deliberate.
- **Refused (422)** also when the file is not a PDF, has no text layer (a scan), or has no numbered clauses (it must look like `2.2 Escape of water` or `A1 Sudden` under headings). A clause the rules cannot type is listed in the result and left for a person, never guessed.
- Uploaded wordings are kept in `<store>/policies` (set `CLAIMTOWER_POLICIES` to move it), so they survive a restart only when the server runs with `--store`.
- **Claims already assessed are not changed.** Re-assess one with `POST /claims/{id}/reassess` (from the policy node by default; nodes after it whose inputs did not change are skipped). A claim whose sublimit wording was missing then gets its payable.

### Late evidence, a second round

If more evidence arrives, upload it (step 2). Only what depends on it re-runs; the gate is presented again. The same applies after `request_more_evidence`. If the quality checks fail on a re-assessment, the claim is parked instead of reaching the gate (see below).

## 3. Phases and statuses

`status.phase`: `awaiting_human_decision` (paused at the gate), `completed` (decided), `halted` (an agent failed or the quality checks failed: `halted_reason` says why, `paused_at` where; the claim is parked for a person), `not_started`.

`state.resolution_status` is the business status: `assessing`, `awaiting_evidence` (a request is open or a material question is not settled), `awaiting_human`, `approved`, `closed`.

A claim is **halted** when: the policy system is unreachable, a policy wording is missing, the claim cannot be registered, an agent raises an error, or the **quality checks fail** (a citation that does not match the policy text, a settlement explanation whose total differs from the engine's, wording that states an outcome, something executed before an approval). It never reaches the gate with a defective assessment.

## 4. Endpoint cheat sheet

| Endpoint | Purpose |
|---|---|
| `GET /health`, `GET /agents` | which agents are real or mock (all real now), the low-confidence threshold |
| `POST /claims` | create and run to the gate |
| `POST /claims/with-documents` | create a claim and send all its documents in one multipart request; assessed once (a bad file is skipped and listed) |
| `GET /claims`, `GET /claims/{id}`, `GET /claims/{id}/status` | list, full view, status only |
| `GET /claims/{id}/versions`, `GET /claims/{id}/diff?from=&to=` | version history and what changed |
| `GET /claims/{id}/documents`, `POST /claims/{id}/documents` | list, upload and re-assess |
| `GET /claims/{id}/documents/{doc}/file`, `/details`, `/page/{n}?fact=` | the original, what was read, the page with the fact marked |
| `POST /claims/{id}/decision` | approve, edit, request more evidence, reject |
| `POST /claims/{id}/reassess?from_node=` | run a claim again from a node (default `policy`) |
| `GET/POST /policies/documents`, `POST /policies/reload` | the policy wordings: list, add (any file name, the id is read from the document), re-read the folders |
| `GET /claims/{id}/audit` | the run log |
| `GET /claims/{id}/trail`, `GET /claims/{id}/decision-pack` | the hash-chained trail, the decision pack |

Errors: **404** unknown claim, document, page, fact or version; **409** duplicate claim id, or the claim is not waiting for a decision; **413** file over 20 MB; **422** invalid body or a refused decision (the reason is in `detail.error`). Every response carries `X-Mock-Agents`, empty when every agent is real.

## 5. The same flow in the UI

`python scripts/run_ui.py`, then: create a claim in the sidebar; **Documents** (upload), **Document reader** (a fact marked on its page with the source excerpt), **Facts**, **Questions & gaps** (conflicting, missing and insufficient evidence side by side, and the scenarios), **Assessment & decision** (coverage issues with their clauses, the calculation trace, and the decision form: edit needs a reason and an input to change), **Policy**, **History** (versions, diff, every human decision and override with its reason, the audit log).

## 6. Try it with the sample claim, start to finish

`samples/claim_sample_1/` has six new SYNTHETIC documents (policy `POL-1013`, loss date 2026-05-22; its README has the files and the expected result).

```
python scripts/run_api.py --store var/store
# 1. create CLM-1001 (step 1), 2. upload the six files (step 2), 3. GET /claims/CLM-1001
# 4. POST /decision as an adjuster      -> 422 "needs 'senior_adjuster' or higher", the gate stays open
# 5. POST /decision with an edit        -> {"decision":"edit","reviewer_id":"rev-3","reviewer_role":"senior_adjuster","reason":"a second upgrade was confirmed","edits":{"non_covered_items_total":"28000"}}
#                                          payable 54,500 becomes 26,500; the gate opens again
# 6. POST /decision as a supervisor     -> approved; the claim is committed and one payment instruction is accepted
# 7. GET /claims/CLM-1001/trail         -> chain_ok true;   GET /claims/CLM-1001/decision-pack?format=pdf
```

## 7. What to watch for

- **Header at creation:** give `policy_ref` and `loss_date` when you create the claim; the claimant name and description are filled from the notice of loss once it is uploaded.
- **Document types:** Evidence classifies by rules, so documents must look like the pack's types (first notice of loss, plumber report, surveyor report, estimate, invoice, e-mail, photo, policy schedule). Scans are read only with `--llm` (vision OCR).
- **Rules only vs the model:** without `--llm` fact questions stay `unverified` and every real claim routes complex (nothing is established without a judgement). The scorecard (`make eval`) shows the difference on the labelled scenarios.
- **Outside systems are mocks:** the policy system, claims system and payment reconciliation are in-process mocks, every response is labelled `mock`, and nothing leaves the machine. How to replace one: `docs/REPLACING_MOCK_ADAPTERS.md`.
- **Limits:** no authentication (reviewer identity and role are asserted); claims, checkpoints and the event log are in memory; one claim is processed at a time; a request returns when the claim reaches the gate (about a second without the model, up to a minute or more with it); the duplicate check is not built yet (the Document reader says "not done yet").
