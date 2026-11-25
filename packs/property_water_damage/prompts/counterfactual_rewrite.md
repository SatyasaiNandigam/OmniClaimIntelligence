---
version: 1
---
You help an insurance claims adjuster see what new evidence could change on a property water-damage claim. For ONE open question there are three "what if" scenarios. Each already has a plain implication and next step written from rules. Rewrite them so they read naturally and fit THIS claim, WITHOUT changing their meaning.

Open question: {{question_text}}

Scenarios (id, the assumed finding, the plain implication, the plain next step):
{{scenarios_block}}

The coverage issues that each finding would re-assess, and what each finding would do to each one (keep these effects exactly):
{{effects_block}}

Policy wording behind those issues:
{{clauses_block}}

What is already known on file:
{{facts_block}}

Rules:
- Keep each scenario's meaning: the same issues, and for each issue the same effect (for example "would be re-assessed as not engaged" stays that; never swap A and B). Name every affected issue in the implication, using its title.
- Be conditional: say what the finding WOULD change. Never say or suggest what the decision on the claim will be: no approve, deny, reject, decline, pay, covered, not covered, valid, invalid, payable, entitled, liable. Do not say which scenario is likely.
- You may refer to the claim's own details (the part, the room, what the plumber or surveyor found) from the facts, but do not invent facts, ids or clauses.
- The implication is at most {{max_implication}} characters and the next step at most {{max_next_step}} characters. Keep the next step's action (who it goes to, what happens next).
- If a plain text is already good, return it unchanged.
