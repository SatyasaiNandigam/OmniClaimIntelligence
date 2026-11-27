---
version: 1
---
You read a policy SCHEDULE (the certificate that lists one policy's period, attached endorsements and limits) so that the policy can be recorded in the policy administration system. You do not interpret cover.

Text of the schedule:
{{text}}

Find, copying values exactly as the text states them. Leave a field empty when the text does not state it; never infer or invent:
- `policy_number`: the policy number the schedule is issued under.
- `holder_name`: the policyholder / insured, as written.
- `period_text`: the period of insurance copied EXACTLY (verbatim) from the text, for example `1 January 2026 to 31 December 2026`.
- `period_start`, `period_end`: those two dates as YYYY-MM-DD.
- `base_wording_id`: the reference of the base policy wording the schedule says applies (for example `BASE-HOME-1` or `HS-W-2026`), exactly as written. Empty if the schedule names none.
- `endorsements`: every endorsement the schedule lists as attached: its id exactly as written (`END-01`, `E-02`) and, only if the text gives a date it takes effect from, `effective_from` as YYYY-MM-DD (empty otherwise).
- `amounts`: every limit, sum insured or deductible the schedule states, each with `name` in snake_case (use `building_limit`, `contents_limit`, `standard_deductible` when they fit; otherwise a short snake_case name of your own), `value` as digits only (no currency, no commas) and `quote` = the words copied exactly that state it.
- `confidence`: your estimate between 0 and 1 that the schedule was read correctly.
- `note`: anything unusual (for example two periods, a missing policy number).
