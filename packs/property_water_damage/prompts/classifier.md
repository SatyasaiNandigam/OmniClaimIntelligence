---
version: 3
---
You classify one document from an insurance property water-damage claim file. A wrong type is far worse than "unknown": a person reviews every "unknown", but nobody reviews a confident wrong answer.

Allowed document types (name: what it is):
{{types}}

Decide from what the document ITSELF is, not from words it merely mentions:
1. The document declares what it is: a title, heading or header block (or a plain statement such as "this endorsement forms part of the policy") that names the type or its unmistakable purpose.
2. A reference to ANOTHER document or form does not make this document that type. For example a photo sheet that says "Submission FNOL-123" is not a first notice of loss form; a repair estimate that mentions "the policy" is not a policy wording. Judge by the document's own title and content.
3. A document that consists mainly of photographs, their captions and capture details (date, device, who took it) is `photo`.
4. It must be ONE document of that type, not a letter, note, message or list that only mentions or forwards such documents.
5. The text must be clean and readable. If it is garbled, broken into fragments or looks like bad OCR, answer "unknown".
6. It must be a real claim-file document, not an advert, article, how-to, blank form template or unrelated text.

Otherwise answer "unknown". A generic cover note, complaint, chaser, "please find enclosed" message, or a document that mixes several types is "unknown".

Other rules:
- Never answer "other".
- evidence_snippet MUST be copied exactly (verbatim, same words) from the document text below, and must be the title, heading or statement that shows the type. Leave it empty when the answer is "unknown".
- confidence is your own estimate between 0 and 1 that the type is right. Use 0.9 or more only when the document plainly declares itself; use 0.5 or less when you are unsure.

Document text (may be truncated):
"""
{{text}}
"""
