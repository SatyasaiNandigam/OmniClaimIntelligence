# Scenarios & ground truth (SYNTHETIC)

| File | What |
|---|---|
| `scenarios.yaml` | Inputs: clause catalogue, per scenario the claim header, expected PAS state, document **page text** and time-ordered events |
| `../ground_truth/S-0x.json` | Labels per scenario: facts, questions, contradictions, missing evidence + minimum request, counterfactuals, issue matrix, policy snapshot, routing tier, settlement, expected actions |
| `evals/ground_truth.py` | Typed loader (reuses the T0.2 contracts); the T8.1 eval runner imports it |
| `tests/unit/test_ground_truth.py` | Validates labels (run `pytest tests/unit/test_ground_truth.py`) |

Scenarios: S-01 running example (contradiction + gaps) · S-02 clean happy path · S-03 lapsed policy · S-04 two endorsements / as-of date · S-05 endorsement missing from upload.

## Rules the tests enforce
- Every fact's `evidence_text` is a verbatim substring of the cited page in `scenarios.yaml`.
- Every question, clause, fact and issue reference resolves; every issue cites ≥1 clause from `clause_catalogue`.
- Missing-evidence requests map to a question's `establishing_criteria`.
- Calculated settlements match the deterministic engine with `settlement_v1.yaml`.
- No fraud / approve / deny outcome language.

## Contracts for later tasks
- **T1.1** must write policy wording using the `clause_catalogue` IDs (stable). Endorsement effective dates per policy are in each scenario's `pas_expectation`.
- **T1.3 (done)** renders `documents[].pages` to PDFs in `data/synthetic/claims/`; keep the text unchanged or facts stop matching.
- **T1.4 (done)** mock PAS in `data/synthetic/pas_mock_db.json` agrees with every `pas_expectation` (tested). See `data/synthetic/README.md`.

## Adding a scenario
Add it to `scenarios.yaml`, add `data/ground_truth/S-xx.json`, add its id to `IDS` in the test. All amounts are SAMPLE.

## Known follow-ups surfaced while labelling
1. ~~S-01 routed to `medium`~~ - fixed in T1.5: `routing_rules.yaml` (sample-2) now has escalation triggers; `tests/unit/test_routing_rules_sample.py` proves all five tiers and reason codes.
2. ~~S-05 optional sublimit~~ - fixed at engine level in T5.3: `SettlementInputs.unavailable_inputs` makes the engine return `cannot_calculate`; the Settlement agent (T5.5) must populate it.
3. Labels were written by one person; the plan's two-labeller ≥90% agreement check (T1.2 done-when) still needs a second reviewer.
