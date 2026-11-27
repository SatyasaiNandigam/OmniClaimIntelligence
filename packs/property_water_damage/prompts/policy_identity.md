---
version: 4
---
You read the first pages of an insurance policy document and say which document it is, so that it can be filed under its own identifier. You do not interpret cover.

Text of the first pages:
{{text}}

Find:
- `document_id`: the identifier the document gives ITSELF: a wording or form reference, a policy or certificate number, an endorsement number. Give ONLY the identifier itself, exactly as written (letters, digits, dashes, slashes): for the line "Document ref HSP/2026/B (edition 4)" the identifier is `HSP/2026/B`. Exception: when the only identifier is a bare number ("Endorsement No. 7"), give it with its label (`Endorsement No. 7`), because "7" alone identifies nothing; and when the document has both a form reference and a bare number, prefer the form reference. Otherwise do NOT include the label before it ("Document ref", "Policy wording", "Endorsement", "No."), the version or edition, or the title. The policy number is NOT the identifier of an endorsement or a wording: an endorsement attached to policy MM-55-0098213 is identified by its own number or form reference (for example "Endorsement No. 7" or "Form EW-07"), and the policy number goes in `policy_number`. Use the policy number as `document_id` only when the document IS the policy itself and has no wording, form or endorsement reference of its own. If several identifiers appear, choose the one that identifies THIS document (for example "Policy wording HP-2025-A" or "Endorsement END-07"), not one it merely refers to. If the document states no identifier at all, leave it empty.
- `evidence_quote`: the shortest words copied EXACTLY (verbatim) from the text above that state that identifier, for example `Document ref HSP/2026/B (edition 4)`: one line at most, never the whole page.
- `version`: the version, edition or revision it states, if any (for example "2025-01", "Edition 3"). Empty if none.
- `title`: the title of the document as written.
- `kind`: `policy_wording` for a policy wording, conditions or full policy document; `endorsement` for an endorsement; `schedule` for a schedule or certificate only; `other` otherwise.
- `policy_number`: the policy number the document is issued under, if it states one. Empty if none.
- `period`: the period of insurance exactly as written, if stated. Empty if none.
- `confidence`: your estimate between 0 and 1 that `document_id` is the right identifier. Use 0.9 or more only when the document plainly names itself; 0.5 or less when unsure.

Rules: never invent an identifier or build one from other words; if you are not sure, leave `document_id` empty and say why in `note`.
