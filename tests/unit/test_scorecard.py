"""T8.1 - the scorecard: it is built from the end-to-end runs, committed as a baseline, and a metric that falls below the baseline fails the build."""
import json

import pytest

from evals.e2e import CASES, MODEL_CASES, has_cassette
from evals.scorecard import SCORECARD_BASELINE, aggregate, build, components, headline, regressions, run_cases, to_markdown


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    return run_cases(tmp_path_factory.mktemp("card"))


@pytest.fixture(scope="module")
def card(results):
    return build(results)


def test_the_scorecard_covers_every_runnable_case_and_both_modes(results, card):
    assert set(results) == {n for n in CASES if n not in MODEL_CASES or has_cassette(n)} and len(results) >= 10
    assert set(card["end_to_end"]) == {"rules", "rules + model (replayed)", "rules on the same scenarios as the model runs"}
    assert card["end_to_end"]["rules"]["fact_recall"] == [31, 31] and card["end_to_end"]["rules"]["settlement_exact_match"] == [5, 5]


def test_the_markdown_scorecard_lists_the_checks_the_metrics_the_differences_and_the_component_baselines(results, card):
    md = to_markdown(card, results)
    for needle in ["# Scorecard", "## 1. Behaviour checks", "13 of 13 cases pass", "## 2. Against the labelled scenarios", "question_status_agreement", "rules + model (replayed) (2 scenarios)",
                   "## 3. Where the system and the labels differ", "S-02 (rules): routing_tier_agreement 0/1", "## 4. Component baselines", "| coverage |", "Incremental efficiency: "]:
        assert needle in md, needle


def test_the_per_scenario_failure_list_names_every_metric_that_missed_its_label(card):
    assert any(f.startswith("S-01 (rules): contradiction_recall 0/1") for f in card["failures"])
    assert not any("check failed" in f for f in card["failures"])                                   # every behaviour check holds


def test_the_committed_baseline_exists_and_the_current_numbers_do_not_regress_against_it(card):
    base = json.loads(SCORECARD_BASELINE.read_text(encoding="utf-8"))["entries"][-1]
    assert base["label"].startswith("T8.1") and len(base["cases"]) >= 10
    assert regressions(card, base) == []


def test_a_case_that_stops_passing_or_a_metric_that_falls_is_reported_as_a_regression(card):
    base = json.loads(json.dumps(card))
    worse = json.loads(json.dumps(card))
    worse["cases"]["S-02"]["passed"] = False
    worse["end_to_end"]["rules"]["fact_recall"] = [20, 31]
    out = regressions(worse, base)
    assert any("S-02 used to pass" in x for x in out) and any("fact_recall fell from 31/31 to 20/31" in x for x in out)
    assert regressions(base, worse) == []                                                          # an improvement is not a regression


def test_the_aggregate_adds_numerators_and_denominators_over_the_labelled_scenarios(results):
    agg = aggregate(results)
    s = sum(r.metrics["document_type_accuracy"][1] for n, r in results.items() if n in ("S-01", "S-02", "S-03", "S-04", "S-05"))
    assert agg["rules"]["document_type_accuracy"][1] == s == 18


def test_the_component_table_reads_the_committed_baselines_and_tolerates_an_odd_layout():
    rows = {c["component"]: c for c in components()}
    assert {"classification", "coverage", "settlement_inputs", "facts"} <= set(rows) and "scorecard" not in rows
    assert ("wrong numbers, fresh", "0") in rows["settlement_inputs"]["headline"]
    assert headline("classification", {"unexpected": 1}) == [] and headline("nonexistent", {}) == []
