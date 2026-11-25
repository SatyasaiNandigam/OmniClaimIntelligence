# ADR-0002: Azure Document Intelligence vs. our ingestion and extractors

- **Status:** Proposed (not decided). Related: BUILD_PLAN §3 (stack swap table) and §17 open decision #3 (local OCR vs Azure Document Intelligence).
- **Question:** Can Azure Document Intelligence (DI) replace the ingestion and extraction we built in T2.1–T2.3, or is it a different thing?
- **Short answer:** It can replace *part* of it, not all of it.

## What DI can replace

| Our component | Overlap with DI |
|---|---|
| **T2.1 loaders and OCR** (`ingestion/loaders.py`, `ocr.py`) | Reading PDFs and scans, including OCR for pages we currently flag `needs_ocr` (e.g. the scanned surveyor report). DI returns page text, word and line positions, tables and key-value pairs. |
| **Part of T2.3 extraction** (`extractors/lines.py`, `table.py`) | Its layout and table models overlap with our line merging and table-row logic. Its prebuilt *invoice* model overlaps with the invoice fields. |
| **Swap point** | BUILD_PLAN §3 already lists DI as the production swap for the loaders. It would sit behind the same contract: pages, text, geometry, confidence. |

## What DI does not replace

| Our component | Why it stays |
|---|---|
| **Arithmetic self-check, "never silent" flags, range/ambiguity handling, verbatim provenance** | These are our guardrails. They must run on top of whatever text DI returns, not be delegated to it. |
| **Classification and domain fields** | Prebuilt models do not know the claim types (FNOL, plumber report, surveyor report, policy schedule). Custom models would need labelled training data, and a custom model still cannot say whether a value is trustworthy. |
| **Store, tamper-evident originals, human overrides, evals, audit trail** | Outside what DI does. |

## Recommended path (if adopted)

1. Treat DI as an upgrade to **T2.1** behind the existing interface: DI is the *reader* (OCR, layout, tables), our extractor and checks remain the *judge*.
2. Compare it against our eval sets (`data/synthetic/extraction_set/`, `evals/baselines/`). Those sets are the fair test of whether it beats the current approach. Note that our unseen-round numbers (extraction 87.0% / 84.2% / 64.3% / 82.3%) are the baseline to beat, not the tuned 100%.
3. Keep the rule-based path available as the offline / replay fallback.

## Must be decided first

DI is a cloud service: claim documents would leave the local environment. A **data-residency and privacy decision** is required before any real use (BUILD_PLAN open decision #3). The prototype only ever uses synthetic data.

## Consequences

- No code change is needed now; this records the direction so T2.4 and later work keep the reader behind a stable interface.
- Revisit when the Azure environment and the privacy decision are settled.
