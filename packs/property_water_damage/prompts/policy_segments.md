---
version: 2
---
You read ONE page of an insurance policy document and mark where its clauses begin, so the page can be cut into clauses. You do not interpret cover and you do not rewrite anything.

Document title: {{title}}
Page {{page}} lines, one per row, in order:
{{lines}}

Find:
- `clauses`: every line that is the HEADING of a clause, provision, condition, exclusion, definition, limit or endorsement item. A heading is a short line that names what the text under it is about (for example "2.2 Escape of water", "Exclusions: wear and tear", "Definition of Buildings", "A1 Sudden damage", "Excess"). Copy each `heading` EXACTLY as the line is written above (the whole line, same characters). Not a heading: a sentence of the body text, a page number, a footer or banner, the document title, a table row of figures.
  - `number`: the numbering the line starts with, exactly as written ("2.2", "A1", "Clause 4", "(c)"), or empty when the heading carries no number.
- `sections`: lines that are headings of a whole GROUP of clauses (for example "SECTION 2 - WHAT IS COVERED", "General Exclusions"), copied exactly. Empty if none. A line is either a clause heading or a section heading, never both.

How to mark:
- A clause is one complete provision: a rule, definition, cover, exclusion, limit, deductible, condition or amendment, written as a heading with text under it, as a single heading-plus-sentence line, or as a paragraph with no heading at all.
- When a provision has no heading (for example the paragraphs of an endorsement: "Clause 5.2 of the wording is amended as follows: the maximum We will pay ... is INR 300,000 ..."), mark ONLY the FIRST line of the paragraph. A paragraph that wraps over several lines is ONE clause: the lines that continue the SAME sentence (a line starting with a lower-case word or a bracket, or finishing a sentence the line above left open) are NEVER marked; they stay part of the clause above. But when the line above ENDS a sentence (it finishes with a full stop or a closing bracket and full stop) and this line begins a new sentence with a capital letter, this line starts a NEW provision and IS marked.
- A table or block of document details (labels and values such as Policy number, Insured, Endorsement number, Effective date, Issued on, signatory, address, page banner, "It is hereby agreed and declared that:") is NOT clause text: mark none of its lines. The title of the document and any subtitle under it (for example "Escape of Water Aggregate Sublimit Endorsement") are not clauses either.

Rules: only lines from the list above, copied exactly; keep the order of the page; never invent a line; when a line could be body text rather than a heading, leave it out (a missed heading is reviewed by a person, a wrong one cuts a clause in the wrong place). A page with no headings gives empty lists.
