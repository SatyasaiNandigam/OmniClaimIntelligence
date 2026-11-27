# TEAM HEADS-UP: what uses an LLM today, and how to test the API

Status as of 2026-11-25. Prototype, SYNTHETIC data only. Three agents are implemented (Evidence, Policy, Gap); the other six are mocks ("YET TO IMPLEMENT") that return labelled fixtures.
More detail: `docs/orchestration.md` (the graph), `docs/api.md` (the endpoints), `docs/assumptions.md` (open questions).

## 1. Which implemented agents use an LLM

| Agent | LLM capability built | Used when you run the API today |
|---|---|---|
| **Gap** | Yes, in 6 places: claim-specific questions (plus a duplicate check), link direction, narrative contradictions and retractions, criterion judgement, request wording, scenario rewrite | Only with `--llm` or `CLAIMTOWER_LLM=1`. That is about 10-15 calls per claim. Without it the Gap agent is rules only, and fact questions abstain (`unverified`). |
| **Intake** | No: it reads the notice of loss with the Evidence agent's classifier and extractor, so the model fallback applies there when `--llm` is on | With `--llm` only through those fallbacks. |
| **Evidence** | Yes: a document-type tie-breaker, a field filler for required fields, and vision OCR for scans | Only with `--llm`. The model is asked only where the rules abstain or find nothing: an unclear document type, a required field the rules missed, a page with no text layer. |
| **Coverage** | Yes: reads the facts against the clauses (cause and exclusion rows), checks whether a clause can be read two ways, and double-checks favourable readings | Only with `--llm`. Without it the settled cover and exclusion rows are only `identified`. |
| **Settlement** | Yes: reads the assessed loss and non-covered amounts from the facts and rewords the explanation; the engine, not the model, computes | Only with `--llm`. The amounts come from rules without it. |
| **Action** | Yes: rewords the drafted messages; the planner and the tasks are rules | Only with `--llm`. A reworded draft is kept only if every requested item is still in it, no amount or conclusion was added. Nothing is ever sent. |
| **Audit** | No, by design: a safety net must not depend on the thing it checks | Never. |
| **Routing** | No, by design: the tier follows the routing rules, never model confidence | Never. |
| **Policy** | Yes, at ingestion: typing of clauses the rules can't type, and embedding retrieval. The agent's own snapshot logic never uses a model. | **No.** The default app loads the policy PDFs with rules only, and retrieval isn't in the graph flow. |

With no flag, **the API makes zero LLM calls**. With `--llm`, the Evidence, Gap, Coverage, Settlement and Action agents call the model (gpt-5-mini, from `LLM_API_KEY` in `.env`). Routing and the mock agents never use one.

The policy clause typing at ingestion is still rules only in the API; that is a small change in `composition.py` and `main.py`.

## 2. API flow to test

Start the server and open the interactive docs. You can run every step below with "Try it out":

```
python scripts/run_api.py            # add --llm for the model
http://127.0.0.1:8000/docs
```

1. **Create a claim** with `POST /claims` and body `{"claim_id":"CLM-0001","use_synthetic_documents":true}`. It runs to the human gate and returns `status.gate` (tier, reason codes, open questions) plus the full `state`.
   - CLM-0001 is complex (cause conflict).
   - CLM-0002 is simple.
   - CLM-0003 is a lapsed policy: coverage records the one "no policy in force" row and settlement is skipped.
   - CLM-0004 is medium.
   - CLM-0005 has an endorsement issue.
   - Other ids work, but the mocks have no fixture for them and leave their fields empty.
2. **Inspect it** with:
   - `GET /claims/CLM-0001`;
   - `/versions`;
   - `/diff?from=3&to=6`;
   - `/audit`, where each event is flagged `mock` or real;
   - `/documents`.
3. **Add late evidence** with `POST /claims/CLM-0001/documents`, a multipart upload (try `data/synthetic/claims/CLM-0001/plumber_report.pdf`). It re-runs from the intake node (unchanged nodes are skipped), which supersedes the open gate, so look at `/audit` and `/diff`.
4. **Decide** with `POST /claims/CLM-0001/decision` and body `{"decision":"approve","reviewer_id":"rev-1","reviewer_role":"senior_adjuster"}`.
   - `adjuster` on a complex claim returns 422 and the gate stays open. Without `--llm` the real routing sends every claim to complex (nothing is established yet), so use `senior_adjuster`.
   - `reject`, `edit` and `request_more_evidence` need a `reason`. `edit` also needs `edits`, for example `{"non_covered_items_total":"28000"}`: the settlement is calculated again and the gate opens again. `reject` needs `senior_adjuster` or higher.
   - After `request_more_evidence`, upload a document (step 3) and the claim returns to the gate.
5. **Check the end state** with `GET /claims`, `/health` and `/agents`. Every response also carries the `X-Mock-Agents` header.

A request returns when the claim reaches the gate: about a second without the model, longer with `--llm`. There is no authentication yet. Claims and events are in memory and are lost when you restart the server.
