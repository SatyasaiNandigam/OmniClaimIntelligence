# Evaluation: the scorecard and the scenario replays

Baselines first: no target numbers are set until a first run exists (BUILD_PLAN section 13). Everything here runs offline and deterministically. Nothing calls a live model; the model's answers are replayed from recorded cassettes.

## The scorecard (`make eval`)

`python evals/runners/run_eval.py` runs every end-to-end case, scores it against the labelled scenarios and writes `evals/reports/scorecard.md` (git-ignored) and `scorecard.json`. It has four parts:

1. **Behaviour checks:** pass or fail per case (reaches the gate, quality checks pass, the state can be rebuilt from the audit trail, nothing is paid before a decision, ...).
2. **Against the labelled scenarios:** document type, fact recall, question status, contradiction recall, evidence-request addressees, coverage status, abstention, settlement exact match, routing tier and reason codes, actions, citation correctness. Shown for rules alone (S-01 to S-05), for rules on the two scenarios the model was recorded on, and for rules plus the model's recorded answers on those two.
3. **Where the system and the labels differ:** one line per scenario and metric that missed its label.
4. **Component baselines:** the latest committed baseline of every component eval (`evals/baselines/*.json`) with its headline numbers.

`python evals/runners/run_eval.py --write-baseline "label"` commits the current numbers as the scorecard baseline (`evals/baselines/scorecard.json`); `--check` exits 1 when a case or metric is worse than that baseline, and `tests/unit/test_scorecard.py` enforces the same in CI.

## Scenario replays (`tests/e2e`, `scripts/replay_scenario.py`)

`evals/e2e.py` holds 13 cases: the five labelled scenarios; late evidence (S-01: only what depends on it re-runs); approval pays only for a supervisor; an edit recalculates and the edited amount is what is paid; the endorsement wording arriving later (S-05); the never-seen sample claim; the model being unreachable; and two scenarios (S-01, S-02) with the model's recorded answers. `pytest` replays all of them in CI.

```
python scripts/replay_scenario.py --list
python scripts/replay_scenario.py S-01-with-model          # checks, metrics, run log
python scripts/replay_scenario.py S-02-approve --pack      # and the decision pack built from the audit trail
python scripts/replay_scenario.py S-01-with-model --record # re-record the model's answers (needs LLM_API_KEY; sends only synthetic data)
```

Re-record a cassette whenever a prompt or the pack configuration the model reads changes (a changed request is a miss in replay mode), then re-run `make eval` and review the difference before committing a new baseline.

## What the first scorecard says

Rules alone settle the money correctly (5 of 5) and read every document and fact, but they judge nothing: question status agreement is 40% and routing tier agreement 40%, because without a model nothing is established and every claim goes to a person. With the model's recorded answers on S-01 and S-02, question status agreement is 83%, coverage status 100%, contradiction and evidence-request recall 100%. S-02 still routes to `medium` instead of `simple`, because the model asked for one extra piece of evidence on a clean claim: an over-request to look at before any target is set.

Limits: five labelled scenarios (the plan has fifteen to eighteen), two with a model recording; the human-rated measures (minimal-request quality, adjuster review) are in `evals/review/` and are still unrated; duplicate precision and recall wait for T6.8.
