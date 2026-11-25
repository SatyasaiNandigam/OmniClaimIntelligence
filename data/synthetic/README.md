# Synthetic data (ALL SYNTHETIC)

Nothing here is real: names, policies, amounts, vendors and images are invented. Every document and photo says so on its face, and every file below is
**generated and reproducible** — `make data` rebuilds everything (`make data-check` verifies nothing drifted), byte for byte.

| Path | What | Generator / source | Task |
|---|---|---|---|
| `policies/` | 3 base wordings + 5 endorsements (PDF + DOCX), `clause_index.json`, `INDEX.md` | `scripts/generate_policy_corpus.py` from `policies/corpus.yaml` | T1.1 |
| `claims/CLM-000x/` | The scenario documents as PDFs, a few photos, `manifest.json` (sha256 of every file) | `scripts/generate_synthetic_docs.py` from `data/scenarios/scenarios.yaml` | T1.3 |
| `claims/CLM-0002/surveyor_report_scanned.pdf` | Image-only, rotated, noisy copy of a report: **needs OCR** (`needs_ocr: true`, expected text in the manifest) | same | T1.3 |
| `photos/quality_set/` | Labelled good / blurry / low-res / glare / dark / unrelated photos with expected quality verdicts | same | T1.3 (for T2.5) |
| `templates/` | Invoice and e-mail layouts (no scenario uses these types yet) | same | T1.3 |
| `classification_set/items.yaml` | 77 labelled text/e-mail documents (dev / holdout / fresh / fresh2 / adversarial) for the document-type classifier eval | hand-written; see `evals/baselines/README.md` | T2.2 |
| `extraction_set/` | `corpus_labels.yaml` (expected fields/tables for the 19 rendered documents) + `items.yaml` (70 text documents: dev / holdout / fresh..fresh4) | hand-written; see `evals/baselines/README.md` | T2.3 |
| `pas_mock_db.json` | Mock PAS: policies, versions, endorsements, status history, 18 labelled lookups | `scripts/seed_pas_mock.py` | T1.4 |
| `prior_claims.json` | Earlier claims + payments (duplicate-FNOL, payment idempotency, remaining limit) | same | T1.4 |
| `invoice_registry.json` | Known invoices (duplicate-invoice and near-match cases) | same | T1.4 |

## Rules
- **Do not edit generated files by hand.** Change the source (scenario YAML, `corpus.yaml`, or the generator/seed script) and regenerate.
- **Scenario text is the contract.** The PDFs render `documents[].pages` line for line; only decoration (letterheads, stamps, boxes, footers) is added, so ground-truth
  `evidence_text` remains extractable. Match provenance on **whitespace-normalised** page text (a long line may wrap in the PDF).
- Extra files (scanned variant, photos, templates) are marked `in_scenario: false` in the manifests: they are **not** part of the scenario ground truth.
- Photos are drawn by the generator (no third-party imagery), watermarked "SYNTHETIC ILLUSTRATION - CC0", licence CC0-1.0.
- PDFs are made reproducible by pinning the random trailer `/ID`; images by fixed seeds and a pinned Pillow.

## Mock PAS semantics (`claimtower.integrations.mock.data.PasDatabase.lookup`)
Exact policy-number match only (no fuzzy matching). Dates are inclusive. Status comes from `status_history`, the version from `versions` (for a lapsed or cancelled
policy the latest earlier version is returned, informationally). Endorsements are returned as listed on the schedule **with their effective dates**; deciding which
are effective on the loss date is the snapshot builder's job (T3.3). Structural cases: lapse boundary (POL-1010), cancelled-then-reinstated (POL-1011), two versions
straddling a date (POL-1012), plus S-01..S-05's policies.
