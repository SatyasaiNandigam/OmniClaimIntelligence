# Eval baselines

`classification.json` is the history of the document-type classifier eval (T2.2); `extraction.json` is the field/table extraction eval (T2.3) and `facts.json` the facts-with-provenance eval (T2.4); both have sections below. Each entry records the date, a hash of `document_types.yaml`
(the exact signals and thresholds) and the result per split. **A number is only meaningful with the signals that produced it.**

Run: `python evals/runners/eval_classifier.py --details` · record: `... --write-baseline "<label>"`.
`tests/unit/test_classifier.py` fails if `document_types.yaml` changed without a new baseline, or if any split regressed against the latest entry.

## Splits (`data/synthetic/classification_set/items.yaml` + the rendered corpus)
| Split | What | Gated? |
|---|---|---|
| corpus | the 50 rendered T1.1/T1.3 files (PDF, DOCX, photos, the scan before/after OCR) | yes: 100% |
| dev | written alongside the signals | yes: 100% |
| holdout | written in different words than the signals | yes: no wrong answers |
| fresh | written after the v2 design; first measured against v2, then used to fix v3 (**no longer unseen**) | yes: no wrong answers |
| fresh2 | written after v3 was fixed: **the unseen check** | yes: accuracy >= 90%, no wrong answers |
| adversarial | keyword-rich look-alikes (privacy policy, advert, ...) | **tracked, not gated** |

## History (what changed and why)
1. **v1** initial signals. holdout: 96.5% but 1 *wrong* answer (a gym's terms classified as policy wording on vocabulary alone).
2. **v2** vocabulary alone never decides a type (a heading or header block is required). fresh (unseen): 66.7%, 2 wrong (an article titled "Estimate your reading time").
3. **v3** a heading alone is not enough either: it needs >= 1 supporting phrase. fresh2 (unseen): 12/12. adversarial: 4/4 wrong.

## Proposed targets  (PROPOSED by the assistant - **need your confirmation**, you are the lead)
- Never a confident guess on the gated splits: **0 wrong answers, answered precision 100%, abstain recall 100%** (met).
- corpus and dev 100%, fresh2 >= 90%, pooled accuracy over dev+holdout+fresh+fresh2 >= 85% (met: 87.7% pooled; the shortfall is *safe* abstentions).
- Cost of the safety stance: ~12% of ordinary documents with informal or unusual titles go to a human (they are never misclassified).

## Known limit (read before relying on `confidence`)
Rules cannot tell a real document from a keyword-rich look-alike. All 4 adversarial documents were classified wrongly at confidence 0.99: **confidence measures how much
rule evidence was found, not that the document is genuine.** Mitigations that exist: the evidence snippet is shown to the reviewer; a human can override
(`DocumentStore.set_type`); the optional LLM tie-breaker only ever answers where rules abstained. Not built: an LLM *veto* on rule answers (needs a configured model).

---

# Field and table extraction (T2.3): `extraction.json`

Run: `python evals/runners/eval_extractor.py --details` - record: `... --write-baseline "<label>"`. The entry hash covers the extractor parameters and every
type's field/table spec in `document_types.yaml`. `tests/unit/test_extractors.py` fails if they changed without a new baseline, if any split's
field score regressed, or if any split has more *silent wrong values* than the latest entry. Data: `data/synthetic/extraction_set/`.

**Metrics.** `field score` = share of labelled fields whose NORMALISED value (ISO date, plain decimal, canonical id) matches. `silent wrong` = a wrong
value that was NOT flagged low-confidence: the dangerous failure, target 0. Also: missed, spurious (extracted but labelled absent), flag recall
(ambiguous / unparseable values must come out flagged), table score (status, items, stated total, tax, flags vs label), and that `missing_required` matches.

| Split | What |
|---|---|
| corpus | 19 rendered PDFs (T1.3) with hand-written labels: 60 fields, 5 tables (geometry, regions, stamps, letterheads) |
| dev | 8 text documents written with the extractor in mind |
| holdout | 26 text documents in other formats (ambiguous / impossible dates, Rs. / rupee / lakh amounts, Indian grouping, mismatching tables, ...) |
| fresh, fresh2, fresh3, fresh4 | written AFTER the previous round's fixes; **each was unseen the first time it was measured** |

## The honest history: what each UNSEEN round showed
| Round (first run on unseen data) | Fields right | Silent wrong | What it exposed |
|---|---|---|---|
| v1 on corpus / dev / holdout | 100% / 100% / 100% | 0 | but 2 real defects: a bare number (`Invoice number: 5521`) counted as a line item; an ambiguous date not flagged (0.71 > 0.70) |
| v2 on **fresh** | **87.0%** | **2** | a date range and an amount range read as one value; `Total (incl. GST)` not recognised |
| v3 on **fresh2** | **84.2%** | **3** | more range spellings (`from 3 May to 5 May`, `INR 45,000 - 60,000`, `45,000-60,000`). *(observed and fixed; not recorded as its own entry)* |
| v3 on **fresh3** | **64.3%** | **2** | `INR 40k` read as 40 (a 1000x error); three more total-row labels |
| v4 on **fresh4** | **82.3%** | **0** | only safe misses: `3rd of June`, `60,000 INR`, `Date of loss (approx)` |

The `FINAL` entry shows 100% everywhere, but every split in it has been tuned on: **do not read it as generalisation.** The unseen rows above are the real
evidence: each new round of formats still found a new gap in a long tail, though the *last* round found no silent wrong value.

## Proposed targets  (PROPOSED by the assistant - **need your confirmation**)
- **0 silent wrong values** on every split (met, including the final unseen round); every required field that is absent is reported; nothing extracted from a field labelled absent.
- Corpus (rendered documents) 100% fields and tables (met).
- Expect **~80-85% field score on documents in formats the labels have not seen** (the unseen rounds), with the shortfall as *missed or flagged* fields, not wrong ones.

## Known limits (read before relying on extraction)
- Evidence is rule-based on `Label: value` lines. Labels, spellings and layouts not listed in `document_types.yaml` are **missed or flagged, not guessed**; each unseen round added a few.
- Geometry rules (wrapped lines, same-row pieces) are validated on documents rendered by this project's own templates. Real PDFs may lay text out differently.
- Free-text fields (findings, observations) are verbatim passages. Unlabelled body passages are deliberately low-confidence (0.65 < 0.70) and flagged.
- The optional LLM filler covers only REQUIRED fields the rules could not find, and only with a verbatim quote; it was tested with a fake model, never a real one.
- `confidence` is extraction confidence (is this text the field's value?), never coverage probability, and, as in classification, not a guarantee that the document is genuine.

---

# Facts with provenance (T2.4): `facts.json`

Run: `python evals/runners/eval_facts.py --details` - record: `... --write-baseline "<label>"`. Rules only, no model. `tests/unit/test_provenance.py` fails
if citation correctness or ground-truth coverage regress against the latest entry, or if the extractor config changed without a new entry.

**Citation correctness (the done-when).** Every fact is re-checked against the ORIGINAL document, independently of the derived text the extractor used:
the cited page exists, `evidence_text` is verbatim text of that page (whitespace-normalised), and for PDFs the stored region actually contains the evidence.
Measured over 414 facts: 73 from the five scenarios run through the real Evidence agent + 341 from the 70 labelled text documents. Result: **100% verbatim,
100% region, 0 refused**. A fact that cannot cite its evidence is refused before it is emitted, so this is a guarantee by construction that is also *checked*.

**Ground-truth coverage.** 31/31 hand-labelled facts are reproduced (same document and page, evidence containing the labelled text). 40/40 question links match.

| Entry | Question links | What changed |
|---|---|---|
| v1 first run | 39/40 (97.5%) | missing: `F-S04-01 -> Q-SUBLIMIT` (the loss date also selects the date-effective endorsement) |
| v2 | 40/40 | `fnol.loss_date` template now tags `Q-INFORCE` and `Q-SUBLIMIT` |

The tag change is a **domain judgement, not a measured improvement**: a fact "bears on" a question; whether it is *relevant* or *supports/contradicts* is decided later by the
linker (T4.2). Tagging every loss date with `Q-SUBLIMIT` is slightly more inclusive than the ground truth, which tags it in S-04 only.

**Limits.** Facts are field- and sentence-level and rule-built: propositions are templates in `fact_templates.yaml`, not paraphrases. Contradictory alternative
values (e.g. two different loss dates) stay in the extraction result (`alternatives`) but are not yet separate facts with their own evidence lines: T4.3 will need that.
`evidence_type` is always `neutral` here. Region is the box of the whole logical line, not of the sentence.

