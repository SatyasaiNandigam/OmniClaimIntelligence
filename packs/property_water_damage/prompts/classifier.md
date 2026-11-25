---
version: 2
---
You classify one document from an insurance property water-damage claim file. A wrong type is far worse than "unknown": a person reviews every "unknown", but nobody reviews a confident wrong answer.

Allowed document types: {{types}}

Answer with a type ONLY if ALL of these are true:
1. The document itself declares what it is: a title, heading or header block (or a plain statement such as "this endorsement forms part of the policy") that names the type or its unmistakable purpose. Vocabulary that merely resembles a type (words like scope, labour, total, dear sir) is not enough.
2. The document is ONE document of that type, not a letter, note, message or list that only mentions or forwards such documents.
3. The text is clean and readable. If it is garbled, broken into fragments or looks like bad OCR, answer "unknown".
4. It is a real claim-file document, not an advert, article, how-to, form template or unrelated text.

Otherwise answer "unknown". A generic cover note, complaint, chaser, "please find enclosed" message, or a document that mixes several types is "unknown".

Other rules:
- Never answer "other".
- evidence_snippet MUST be copied exactly (verbatim, same words) from the document text below, and must be the heading or statement from point 1. Leave it empty when the answer is "unknown".
- confidence is your own estimate between 0 and 1 that the type is right. Use 0.9 or more only when point 1 is clearly met; use 0.5 or less when you are unsure.

Document text (may be truncated):
"""
{{text}}
"""
