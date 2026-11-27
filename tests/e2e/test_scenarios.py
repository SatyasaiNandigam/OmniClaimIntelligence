"""T8.2 - every scenario case replayed end to end in CI: the real graph, offline, deterministic (the model's answers come from recorded cassettes, nothing is live)."""
import pytest

from evals.e2e import CASES, MODEL_CASES, Context, has_cassette

RUNNABLE = [n for n in CASES if n not in MODEL_CASES or has_cassette(n)]


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    return Context.create(tmp_path_factory.mktemp("e2e"))


@pytest.fixture(scope="module")
def runs(ctx):
    return {n: CASES[n](ctx) for n in RUNNABLE}


def test_at_least_ten_scenarios_are_replayed_and_the_model_cases_have_recordings():
    assert len(RUNNABLE) >= 10 and set(MODEL_CASES) <= set(RUNNABLE)


@pytest.mark.parametrize("name", RUNNABLE)
def test_every_check_of_the_case_holds(runs, name):
    r = runs[name]
    assert r.checks, name
    failed = [k for k, v in r.checks.items() if not v]
    assert not failed, f"{name}: {failed}"


def test_a_replay_is_deterministic_the_same_scores_every_time(ctx, runs):
    for name in ("S-02", "S-05", "S-01-with-model", "S-02-with-model"):
        again = CASES[name](ctx)
        assert again.metrics == runs[name].metrics, name
        assert again.state.model_dump(mode="json")["coverage_issues"] == runs[name].state.model_dump(mode="json")["coverage_issues"], name


def test_the_recorded_model_answers_lift_the_judgement_metrics_that_rules_alone_cannot_make(runs):
    for sid in ("S-01", "S-02"):
        rules, model = runs[sid].metrics, runs[f"{sid}-with-model"].metrics
        for k in ("question_status_agreement", "coverage_status_agreement"):
            assert model[k][0] >= rules[k][0] and model[k][1] == rules[k][1], (sid, k)
    assert runs["S-02-with-model"].metrics["question_status_agreement"] == (6, 6) and runs["S-01-with-model"].metrics["request_addressee_recall"] == (3, 3)


def test_nothing_in_any_case_instructs_a_payment_without_a_recorded_authorised_decision(runs):
    for name, r in runs.items():
        if r.adapters is not None and name not in ("S-02-approve", "S-02-edit-then-approve"):
            assert r.adapters.recon.instruction_count == 0, name
    assert runs["S-02-approve"].adapters.recon.instruction_count == 1
