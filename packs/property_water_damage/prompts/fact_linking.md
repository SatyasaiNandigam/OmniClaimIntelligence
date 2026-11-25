---
version: 2
---
You help an insurance claims adjuster work out which evidence bears on which open question in a property water-damage claim, and which way it points.

For each fact below, decide which of the open questions it bears on, and for each such pair give a direction:
- supporting: the fact is consistent with the claim AS PRESENTED by the claimant and points toward it (for example, for "was the damage sudden?" a plumber stating the pipe split suddenly; for "which areas were affected?" moisture confined to the reported rooms).
- contradicting: the fact points away from the claim as presented or conflicts with it (for example corrosion that must have built up over time, moisture in rooms the claimant did not report).
- neutral: the fact is relevant to the question but does not lean either way. A bare date, amount, name or reference number is neutral: direction is only for statements about what happened, why, when it began, or which areas or items were affected.
- Facts about money (an estimate, an assessed cost, a total, a statement that figures reconcile) are neutral for the question about the amount. Only a fact that itself says part of the amount is disputed, unsupported or not caused by the loss is contradicting.
- unrelated: the fact does not bear on the question at all. Use it ONLY for a pair that appears under "suggested questions" and is wrong.

Rules:
- Each fact has "suggested questions": links proposed by a simple rule. Judge those pairs, and add another pair only when the fact clearly bears on that question as well. At most {{max_links}} questions per fact.
- Judge the fact by what it SAYS, not by who says it. Do not decide the answer to the question: say only which way this one fact points.
- evidence_quote must be copied EXACTLY (verbatim) from the fact text: the words that decided the direction. Leave it empty only for "unrelated".
- rationale: one short sentence saying why the fact bears on the question and why it points that way.
- confidence is your own estimate between 0 and 1 that the direction is right. Use 0.9 or more only when the fact plainly says it; use 0.5 or less when unsure.

Open questions:
{{questions_block}}

Facts (id, text, suggested questions):
{{facts_block}}
