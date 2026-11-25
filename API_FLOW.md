# API FLOW: starting a claim by uploading your own synthetic documents

Status as of 2026-11-25. Prototype, SYNTHETIC data only. Start the server first: `python scripts/run_api.py` (add `--llm` for the model), then open `http://127.0.0.1:8000/docs`.
More detail: `docs/api.md` (every endpoint), `TEAM_HEADSUP.md` (what uses an LLM today).

Ready-made files to try this with: `samples/claim_sample_1/` (six new SYNTHETIC documents, policy `POL-1013`, loss date 2026-05-22; its README gives the exact commands and the expected result).

Create the claim first, then upload the files. A claim can't be created from files in one call.

1. **Create the claim** with `POST /claims`. Use a new id, because `CLM-0001` to `CLM-0005` trigger the labelled fixtures. Give it the policy and loss date:
   ```
   curl -X POST http://127.0.0.1:8000/claims -H "content-type: application/json" \
     -d '{"claim_id":"CLM-1001","policy_ref":"POL-1001","loss_date":"2026-03-14","loss_type":"water_damage","loss_description":"Kitchen pipe burst"}'
   ```
   It runs to the human gate with no documents yet (no facts).

2. **Upload each file** with `POST /claims/CLM-1001/documents` as multipart. Add `?reassess=false` on all but the last so the claim re-runs once, not once per file:
   ```
   curl -X POST "http://127.0.0.1:8000/claims/CLM-1001/documents?reassess=false" -F "file=@fnol.pdf" -F "uploaded_by=claimant"
   curl -X POST "http://127.0.0.1:8000/claims/CLM-1001/documents?reassess=false" -F "file=@plumber_report.pdf" -F "uploaded_by=plumber"
   curl -X POST "http://127.0.0.1:8000/claims/CLM-1001/documents" -F "file=@estimate.pdf"      # the last one re-assesses
   ```
   The re-run starts at the evidence node, so intake and policy are not re-run. You can also use `/docs` ("Try it out" has a file picker).

3. **Check and decide:**
   - `GET /claims/CLM-1001` shows facts, open questions and the gate.
   - `/documents` shows each document's detected type.
   - `/audit` shows the event log.
   - `POST /claims/CLM-1001/decision` decides.

## What to watch for

- **Policy number:** the `policy_ref` must exist in the mock PAS database (`data/synthetic/pas_mock_db.json`, for example `POL-1001`), and `loss_date` must be set. Without them the Policy agent doesn't run and there is no snapshot.
- **Document types:** Evidence classifies by rules, so your documents need to look like the pack's types (fnol, plumber_report, surveyor_report, estimate, invoice, email, photo, policy schedule). An unrecognised document is stored as type `unknown` and yields no facts. There is no endpoint yet to re-type a document by hand.
- **Mock fields:** for a new id the coverage, settlement and routing mocks have no fixture, so those fields stay empty. The gate shows no tier, and an `adjuster` can decide. The real agents' outputs (facts, policy snapshot, open questions, requests) are the part that reflects your files.
- **Model:** add `--llm` when starting the server if you want the Gap agent to judge the facts. Without it, fact questions stay `unverified`.
