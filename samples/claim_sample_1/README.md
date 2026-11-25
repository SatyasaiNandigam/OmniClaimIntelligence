# Claim sample 1: dishwasher hose burst (SYNTHETIC)

Six new documents for trying the API flow with your own uploads. **Nothing here has been seen by the system before**: new claimant, policy, dates, amounts and wording
(`python scripts/generate_sample_claim.py --check` proves that no file matches a document already in `data/synthetic`). All data is SYNTHETIC and every amount is SAMPLE.

| File | Type the system should detect | What it says |
|---|---|---|
| `fnol.pdf` | fnol | SYNTHETIC Claimant K, policy `POL-1013`, loss on 22 May 2026: the dishwasher inlet hose burst; estimated loss INR 72,000 |
| `plumber_report.pdf` | plumber_report | inspected 23 May; the hose split at the crimped end, a fresh tear, no earlier leakage or corrosion; hose replaced |
| `surveyor_report.pdf` | surveyor_report | site visit 26 May; moisture only in the utility room floor and lower hallway wall; **assessed repair cost INR 64,500** |
| `estimate.pdf` | estimate | utility room INR 41,000 + hallway INR 25,000 = **INR 66,000** |
| `invoice.pdf` | invoice | INV-5521 dated 3 June, INR 65,000 including GST |
| `claimant_email.pdf` | email | the claimant confirms the repair and that the damage was only in two rooms |

It is a **clean, consistent claim** with one deliberate wrinkle: the claimant's estimate (INR 72,000) is 10% above the surveyor's assessment (INR 64,500). That is a *low*-severity
amount difference: the system notes it but it does not block the question.

## Try it

```
python scripts/run_api.py                  # add --llm to let the Gap agent use the model; then open http://127.0.0.1:8000/docs
```
1. Create the claim (a new id; the policy and the loss date matter):
   ```
   curl -X POST http://127.0.0.1:8000/claims -H "content-type: application/json" -d '{"claim_id":"CLM-1001","policy_ref":"POL-1013","loss_date":"2026-05-22","loss_type":"water_damage","loss_description":"Dishwasher inlet hose burst"}'
   ```
2. Upload the six files (`reassess=false` on all but the last, so the claim is re-assessed once):
   ```
   curl -X POST "http://127.0.0.1:8000/claims/CLM-1001/documents?reassess=false" -F "file=@samples/claim_sample_1/fnol.pdf" -F "uploaded_by=claimant"
   ... plumber_report.pdf, surveyor_report.pdf, estimate.pdf, invoice.pdf the same way ...
   curl -X POST "http://127.0.0.1:8000/claims/CLM-1001/documents" -F "file=@samples/claim_sample_1/claimant_email.pdf" -F "uploaded_by=claimant"
   ```
3. Look: `GET /claims/CLM-1001`, `/documents`, `/audit`, `/diff?from=3&to=18`. Then decide: `POST /claims/CLM-1001/decision` (a `simple` claim: any role from `adjuster` up).

Notes: the upload response shows `detected_type: "unknown"` for a file until the claim is assessed; after the assessment `GET /claims/CLM-1001/documents` shows the real types.
Do not reuse `CLM-0001` to `CLM-0005`: those ids trigger the labelled fixtures of the demo scenarios.

## What you should see

| | rules only (default) | with `--llm` (Gap agent uses the model) |
|---|---|---|
| Documents | all six typed correctly | same |
| Facts | 33, each with the page and the verbatim text | same |
| Policy | `POL-1013` in force on 22 May 2026, END-01 applied, 13 clauses | same |
| Open questions | Q-INFORCE and Q-SUBLIMIT `established`; Q-SUDDEN, Q-START, Q-EXTENT, Q-AMOUNT `unverified` (no model: they abstain, they are not guessed) | all six standard questions `established` |
| Contradictions | one **low** amount difference (72,000 against 64,500) | same |
| Extra | none | the model may add claim-specific questions (typically whether the dishwasher hose counts as a fixed part of the plumbing) and a request to the claimant; that is the known tendency to over-ask, see Q-P6 in `docs/assumptions.md` |
| Gate | paused, waiting for a decision | same |

The coverage, settlement, action, audit and routing agents are still mocks, so for a new id the gate shows no tier and no settlement: those parts are not computed.

## Regenerate

`python scripts/generate_sample_claim.py` rewrites these files (deterministic); `--check` fails if they differ or are no longer new. `manifest.json` lists each file with its hashes and expected type.
