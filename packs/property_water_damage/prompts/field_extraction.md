---
version: 1
---
You extract fields from one document of an insurance property water-damage claim file.

Document type: {{doc_type}}

Fields to find (name: what it is):
{{fields}}

Rules:
- Answer ONLY for fields that are clearly stated in the document. If a field is not stated, set it to null. Guessing is worse than null.
- For each field give `value` (as written in the document), `evidence_snippet` (copied EXACTLY, word for word, from the document text: the words that state the value) and `confidence` between 0 and 1.
- Never calculate, convert or infer a value. Dates and amounts stay as written.

Reply with one JSON object: {"fields": {"<field name>": {"value": "...", "evidence_snippet": "...", "confidence": 0.0} or null, ...}}

Document text (may be truncated):
"""
{{text}}
"""
