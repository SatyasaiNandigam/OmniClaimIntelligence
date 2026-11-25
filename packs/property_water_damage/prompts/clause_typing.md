---
version: 2
---
You classify ONE clause of an insurance policy document. A wrong type can change what an insurer pays, so a wrong type is much worse than "unknown": a person reviews every "unknown", but nobody reviews a confident wrong answer.

Allowed types, with what each one means:
- definition: says what a word or phrase means ("X means ...").
- coverage: says what the insurer WILL pay for or repair (an insured event or peril).
- exclusion: a rule of the COVER that removes cover for a kind of loss, cause or situation (what the insurer will not pay for or repair). It is about the loss, not about how the policyholder behaves.
- limit: the most the insurer will pay in total, for the whole policy or a whole class of claims (sum insured, building limit).
- sublimit: a lower maximum for one specific kind of loss (for example water damage), inside the overall limit.
- deductible: the part of each claim the policyholder bears first (excess, deductible).
- condition: something the policyholder must do, or that must be true, for cover to apply to a claim (notify a loss, take care, maintain, occupy, pay the premium, the period of insurance). It is a requirement ON the policyholder or the claim.

Answer "unknown" for everything else. In particular these are NEVER one of the seven types, even when they mention claims, payment, "must" or "we will not":
- what happens if the policyholder is dishonest or makes a fraudulent claim, and any other penalty or remedy for the policyholder's behaviour;
- the insurer's or the policyholder's rights and powers over the policy itself: cancelling, ending, renewing, refusing to renew, changing the terms, transferring the policy, refunds;
- governing law, disputes, complaints, data protection, language, currency, how headings or the document are read;
- anything you cannot place with confidence.
Ask yourself: does this clause decide WHETHER or HOW MUCH the insurer pays for a loss, or what the policyholder must do for a claim to be paid? If not, answer "unknown".

Rules:
- Judge the clause by what it SAYS, not by how it is titled or where it sits.
- evidence_snippet MUST be copied exactly (verbatim) from the clause heading or the clause text below, and must be the words that decided your answer. Leave it empty when the answer is "unknown".
- confidence is your own estimate between 0 and 1 that the type is right. Use 0.9 or more only when the clause plainly is that type; use 0.5 or less when you are unsure.

Document: {{doc_title}}
Section: {{section}}
Clause heading: {{heading}}
Clause text:
"""
{{text}}
"""
