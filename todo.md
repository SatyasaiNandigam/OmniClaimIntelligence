# TODO (after the current end-to-end test of the real claim)

- [x] Create a claim with its documents in one call: `POST /claims/with-documents` (done).
- [ ] Refuse or flag an uploaded `schedule` in `POST /policies/documents` (it is filed as a wording today).
- [ ] Ignore the literal placeholder `string` in the upload form fields (Swagger default).
- [ ] A Policies page, decision-pack and audit-trail downloads in the UI.
- [ ] Endpoint to register a policy in the mock policy system (today: `data/local/pas_policies.json`).
