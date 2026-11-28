# TODO (after the end-to-end tests of the real claims)

## Done
- [x] Create a claim with its documents in one call: `POST /claims/with-documents`.
- [x] UI: Claims, Claim workspace, Human decision, Audit & decision pack (trail, pack download), Policies pages.
- [x] Many policy documents in one request: `POST /policies/documents/batch` (parallel reading, one result per file).
- [x] Register a policy in the mock PAS from a schedule: `POST /pas/schedules/read` (proposal, checked against the text) then `POST /pas/policies` (confirm).
- [x] Schedule items given in words ("as per wording clause 5.1") kept as text; settlement reads the standard deductible from the wording clause when the schedule gives no sum.
- [x] Re-assess from the UI without new documents (Documents tab; Overview button for a halted claim).
- [x] A question about timing or cause can be settled on the evidence as a whole (`purpose` in question_templates.yaml).

## Next
- [ ] Verify the `purpose` step live on S-02 (expect Q-START established, no information requests, simple tier); then run the other test-pack scenarios.
- [ ] Scenario scorecard from the test pack's `expected/*.json` (tier, payable, question statuses, requests, contradictions, forbidden words), several runs each.
- [ ] Keep uploaded policy documents between restarts (start the API with `CLAIMTOWER_STORE`; consider a persistent default).
- [ ] Refuse or flag a schedule uploaded to `POST /policies/documents` (filed as a wording today).
- [ ] Ignore the literal placeholder `string` in the upload form fields (Swagger default).

## Decide / optional
- [ ] Invoice-versus-estimate amount note: drop it for clean claims? (kept by decision 2026-11-26)
- [ ] Should `unverified` also hold the payable (assumption Q-S8)?
- [ ] Stop rule for information requests; adjuster "settled, with a reason" action.
- [ ] Check an endorsement's policy number / insured against the policy that lists it.
- [ ] Delete a policy from the mock PAS (route and button).
- [ ] Cross-check a schedule uploaded as a claim document against the PAS values.

## Known backend issues (not fixed)
- Plumber evidence request does not match the corrosion-severity conflict and does not mention inspecting the preserved pipe.
- The corrosion conflict is filed under both Q-SUDDEN and Q-START.
- "Cannot be determined" is flagged as a conflict.
- Audit node skipped on re-assessment.
- Intermittent false extent contradiction; wrapped-row estimate drop; PDF underscore rendering; contradiction detection varies between runs.
