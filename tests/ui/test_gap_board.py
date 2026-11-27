"""T7.4 - the gap board and the counterfactual view: the three kinds of problem shown distinctly, and the scenario table, on the running example (S-01)."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from claimtower.api.app import create_app
from claimtower.composition import SYNTHETIC, build_registry
from claimtower.core.config import load_settings
from claimtower.ingestion.store import DocumentStore
from claimtower.orchestration.graph import ClaimRunner
from claimtower.core.schemas import ClaimState
from claimtower.ui.client import ApiClient
from claimtower.ui.components import counterfactual_view as cv
from claimtower.ui.components import gap_board as gb

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()
APP = str(ROOT / "src" / "claimtower" / "ui" / "app.py")
ALL = ("intake", "gap", "coverage", "settlement", "action", "audit", "routing")


def s01() -> dict:
    return json.loads((ROOT / "data/fixtures/S-01/claim_state.json").read_text(encoding="utf-8"))


@pytest.fixture
def api(tmp_path) -> ApiClient:
    store = DocumentStore(tmp_path / "store")
    runner = ClaimRunner(build_registry(SETTINGS, force_mock=ALL), settings=SETTINGS)      # the labelled running example: contradiction, missing and insufficient evidence, scenarios A/B/C
    c = ApiClient(http=TestClient(create_app(runner, store, synthetic_claims=SYNTHETIC / "claims")))
    c.create_claim(claim_id="CLM-0001")
    return c


def page(api) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["_client"] = api
    at.session_state["selected_claim"] = "CLM-0001"
    return at.run()


def texts(at) -> str:
    return " ".join([*[m.value for m in at.markdown], *[c.value for c in at.caption], *[i.value for i in at.info], *[h.value for h in at.subheader]])


# ---------------------------------------------------------------------------------------------------------------- grouping
def test_the_running_example_has_one_question_of_each_kind_of_problem():
    g = gb.group(s01())
    assert {k: [c["question"] for c in v] for k, v in g["kinds"].items()} == {"contradictory": ["Q-SUDDEN"], "missing": ["Q-START"], "insufficient": ["Q-EXTENT"]}
    assert [q["question_id"] for q in g["unjudged"]] == ["Q-AMOUNT"] and sorted(q["question_id"] for q in g["settled"]) == ["Q-INFORCE", "Q-SUBLIMIT"]


def test_a_conflict_shows_both_sides_with_the_document_page_and_verbatim_words():
    card = gb.group(s01())["kinds"]["contradictory"][0]
    (x,) = card["contradictions"]
    assert x["kind"] == "cause" and x["severity"] == "high" and len(x["a"]) == 1 and len(x["b"]) == 2
    assert all(it["quote"] and "page" in it["where"] and it["text"] for it in x["a"] + x["b"])


def test_every_request_names_who_is_asked_and_what_for():
    g = gb.group(s01())["kinds"]
    assert [r["to"] for r in g["contradictory"][0]["requests"]] == ["plumber"] and g["missing"][0]["requests"][0]["to"] == "claimant" and g["insufficient"][0]["requests"][0]["to"] == "surveyor"
    assert all(r["items"] for c in sum(g.values(), []) for r in c["requests"])


def test_two_requests_about_one_question_form_one_card():
    s = s01()
    s["missing_evidence"].append({**s["missing_evidence"][0], "minimum_evidence_request": {**s["missing_evidence"][0]["minimum_evidence_request"], "addressee_role": "claimant", "items": ["x"]}})
    card = gb.group(s)["kinds"]["contradictory"]
    assert len(card) == 1 and [r["to"] for r in card[0]["requests"]] == ["plumber", "claimant"]


# ---------------------------------------------------------------------------------------------------------------- scenarios
def test_the_scenario_rows_and_matrix_say_which_issues_each_scenario_touches():
    s = s01()
    r = cv.rows(s)
    assert [x["scenario"] for x in r] == ["A", "B", "C"] and all("ISS-COVER-WATER" in x["issues to re-assess"] for x in r)
    m = cv.matrix(s)
    assert list(m.columns) == ["issue", "today", "Q-SUDDEN A", "Q-SUDDEN B", "Q-SUDDEN C"] and len(m) == 2 and set(m["Q-SUDDEN A"]) == {"re-assess"}
    assert cv.matrix({"coverage_issues": [], "counterfactuals": []}).empty


# ---------------------------------------------------------------------------------------------------------------- the page
def test_the_tab_counts_each_kind_shows_the_cards_with_both_sides_and_the_scenario_table(api):
    at = page(api)
    assert not at.exception
    m = {x.label: x.value for x in at.metric}
    assert m["🟣 Conflicting evidence"] == "1" and m["🔴 Missing evidence"] == "1" and m["🟡 Insufficient evidence"] == "1"
    t = texts(at)
    for needle in ["**Side A**", "**Side B**", "**Ask plumber**", "**Ask claimant**", "**Ask surveyor**", "Scenario A", "Scenario B", "Scenario C", "What would resolve this claim?", "Which scenario touches which issue"]:
        assert needle in t, needle
    assert "### ⚪ Not yet judged" not in t and "### ✅ Settled" not in t                              # the default view is what needs attention


def test_the_other_views_show_what_is_not_yet_judged_and_what_is_settled(api):
    at = page(api)
    next(r for r in at.radio if r.key == "gaps_view_CLM-0001").set_value("All")
    at.run()
    t = texts(at)
    assert not at.exception and "### ⚪ Not yet judged" in t and "### ✅ Settled" in t


def test_the_scenarios_never_predict_a_decision_and_the_view_says_so(api):
    t = texts(page(api)).lower()
    assert "never predict whether the claim is approved, declined or paid" in t
    for banned in ("will be approved", "will be denied", "will be rejected", "will be paid", "fraud"):
        assert banned not in t.replace("never predict whether the claim is approved, declined or paid", "")


def info_of(state: dict) -> str:
    code = "import json; from claimtower.ui.components import counterfactual_view as cv; cv.render({'state': json.loads(%r)})" % json.dumps(state)
    return " ".join(i.value for i in AppTest.from_string(code, default_timeout=30).run().info)


def test_with_no_scenarios_the_view_explains_why_not_in_two_different_ways():
    empty = ClaimState(claim_id="CLM-0001").model_dump(mode="json")
    assert "No open question needs a scenario." in info_of(empty)
    s = s01()
    s["counterfactuals"] = []
    assert "re-assess it" in info_of(s)


def test_a_settled_question_is_a_card_with_what_establishes_it_and_the_facts_that_prove_it(api):
    at = page(api)
    next(r for r in at.radio if r.key == "gaps_view_CLM-0001").set_value("Settled")
    at.run()
    assert not at.exception
    t = texts(at)
    assert "**What establishes it**" in t and "Rests on" in t and "Q-INFORCE" in t and "Q-SUBLIMIT" in t
    assert any(e.label.startswith("Evidence behind this question") for e in at.expander)                 # the verbatim facts are one click away
