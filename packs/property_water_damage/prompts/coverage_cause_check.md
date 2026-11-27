---
version: 2
---
You check one quotation from a property water-damage claim file. You do not decide anything about cover.

Quotation: "{{quote}}"

Question: does this quotation, by itself, state HOW or WHEN the water escaped (the event, such as a pipe joint failing, a hose splitting, a frost split, or how long it had been leaking)?

Answer:
- yes: it names the event or its timing or duration.
- partly: it hints at the cause but does not name it.
- no: it only reports when water was noticed or arrived (for example "overnight", "this morning") without saying what failed or how long it had been escaping, or it only reports damage, costs, dates of invoices or inspections, that water was present, or something unrelated to the cause.

Give `states_cause` as yes, partly or no, and `what_it_states` as a few words (what the quotation says about the cause, or empty).
