---
version: 1
---
You extract the information of ONE document of an insurance property water-damage claim file. You only copy what the document states: you never calculate, convert, infer or fill a gap.

Document type: {{doc_type}}

Single-value fields (name: type; what it is):
{{fields}}

Findings to list (name: what to collect):
{{findings}}

Rules:
- For a single-value field: if the document states it, give `value` (as written), `evidence_snippet` (the words that state it, copied EXACTLY, word for word, from the document text) and `confidence` between 0 and 1. If it is not clearly stated, set it to null. Guessing is worse than null. Dates and amounts stay as written.
- For findings: list EVERY statement of the document that belongs under that name, one entry per statement. Each `text` must be copied EXACTLY, word for word, from the document (a sentence, or one table row with its cells in reading order). Include statements inside tables and statements that mention both what was found and how it should be read (for example moisture, corrosion, staining, the cause or the timing). Do not summarise, merge or reword. If there is none, give an empty list.
- A table row is ONE finding: the cells of the row copied in reading order, separated by single spaces, exactly as they appear in the text (for example `Kitchen north wall (around repaired joint) Tide-mark staining, efflorescence and blistered paint 38 %WME`). List EVERY row of an observations table, one entry per row.
- Keep each finding under the name it belongs to: what the insured or claimant reported belongs to no finding of a surveyor or plumber report unless the report itself states it as its own finding.
- Copy only from the document below. Never use a statement from a different document, and never write text that is not in it.

Reply with one JSON object: {"fields": {"<name>": {"value": "...", "evidence_snippet": "...", "confidence": 0.0} or null, ...}, "findings": {"<name>": [{"text": "...", "confidence": 0.0}, ...], ...}}

Document text, page by page (may be truncated):
"""
{{text}}
"""
