"""ingestion

OWNER   : B (Documents + Evidence hero)
TASKS   : T2.1-T2.4
PURPOSE : PDF/image/DOCX/e-mail -> pages -> classified docs -> fields -> Facts with provenance.

Modules (status)
  loaders.py     T2.1 DONE  bytes -> pages (PDF, image, DOCX, .eml, text); format from content; scans flagged `needs_ocr`
  ocr.py         T2.1 DONE  OcrEngine protocol + VisionLLMOcr (vision-model fallback; deterministic with cassettes)
  store.py       T2.1 DONE  immutable, content-addressed originals + metadata + page text; tamper-evident reads
  classifier.py  T2.2 DONE  document type + verbatim evidence snippet + confidence; abstains to 'unknown' (human); optional LLM tie-breaker
  extractors/    T2.3 DONE  typed fields (date/money/id) + tables with an arithmetic self-check; page, region, verbatim line, confidence
  provenance.py  T2.4 DONE  extraction -> Facts (verbatim evidence, page, region, question tags); independent citation check; FactExtractor

Provenance basis: match evidence against `loaders.normalise(page.text)` (whitespace collapsed), because a long line can wrap in a PDF.

Rules: depend only on claimtower.core (contracts). Never import another
capability's internals - use the signatures in BUILD_PLAN.md section 11.
"""
