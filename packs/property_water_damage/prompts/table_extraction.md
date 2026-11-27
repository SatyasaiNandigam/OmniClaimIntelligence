---
version: 1
---
You read the text of ONE document of an insurance property water-damage claim file (a {{doc_type}}) and list the priced lines of its table and its stated total. You never calculate, convert or infer: you only copy.

Each line of the document is shown on its own row below. A priced line is a row that names an item of work, material or service and states ITS OWN amount (the last amount on the row, the line total, not the unit rate). Header rows, notes, validity or payment terms, and the total row itself are not items.

Find:
- `items`: every priced line, in order. For each: `line` (the row copied EXACTLY, word for word, from below), `description` (the item as written on that row, without the number, quantity, unit and rates) and `amount` (the line's own total, digits as written, for example 21,000).
- `total`: the total the document itself states for all lines ("Total estimate", "Total payable", "Grand total"): `line` (the row, copied exactly) and `amount`. Null if the document states no total.

If a row of the table continues on the next row (a description wrapped over two rows), copy the row that carries the amount. If you are unsure whether a row is a priced line, leave it out.

Reply with one JSON object: {"items": [{"line": "...", "description": "...", "amount": "..."}], "total": {"line": "...", "amount": "..."} or null}

Document rows:
"""
{{text}}
"""
