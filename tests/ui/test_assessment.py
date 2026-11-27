"""T7.5 - the issue matrix, the calculation trace and the approval form: clause citations to check, three figures kept apart, a step-by-step trace, and a decision whose reason is mandatory and visible in the audit view."""
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
from claimtower.orchestration.overrides import OverrideStore
from claimtower.ui.client import ApiClient
from claimtower.ui.components import approval, calc_trace, issue_matrix

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()
APP = str(ROOT / "src" / "claimtower" / "ui" / "app.py")
MOCKS = ("intake", "gap", "coverage", "action", "audit", "routing")          # the labelled fixtures; the REAL settlement agent calculates
REASON = "Reason (needed for edit, reject and request more evidence)"


@pytest.fixture
def api(tmp_path) -> ApiClient:
    ov = OverrideStore()
    runner = ClaimRunner(build_registry(SETTINGS, force_mock=MOCKS, overrides=ov), overrides=ov, settings=SETTINGS)
    store = DocumentStore(tmp_path / "store")
    c = ApiClient(http=TestClient(create_app(runner, store, synthetic_claims=SYNTHETIC / "claims")))
    for cid in ("CLM-0002", "CLM-0005"):
        c.create_claim(claim_id=cid)
    return c


def page(api, claim="CLM-0002", decide=False) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["_client"] = api
    at.session_state["selected_claim"] = claim
    at.run()
    if decide:                                                                                 # the form is on the Human decision page: reach it as a user does
        next(b for b in at.button if b.label == "Go to the decision").click()
        at.run()
    return at


def texts(at) -> str:
    return " ".join([*[m.value for m in at.markdown], *[c.value for c in at.caption], *[i.value for i in at.info], *[x.value for x in at.success], *[w.value for w in at.warning], *[e.value for e in at.error],
                     *[h.value for h in at.subheader], *[f"{m.label} {m.value}" for m in at.metric]])


def widget(at, kind, label):
    return next(w for w in getattr(at, kind) if w.label == label)


def submit(at):
    next(b for b in at.button if b.label == "Submit decision").click()
    return at.run()


# ---------------------------------------------------------------------------------------------------------------- the issue matrix
def test_each_issue_row_names_its_clauses_and_the_endorsements_that_modify_them():
    state = json.loads((ROOT / "data/fixtures/S-01/claim_state.json").read_text(encoding="utf-8"))
    r = issue_matrix.rows(state)
    assert [x["status"] for x in r] == ["established", "unresolved", "potentially relevant", "identified"] and all(x["clauses"] != "none cited" for x in r)
    assert "modified by CL-SUB-WATER-E1" in r[1]["clauses"] and issue_matrix.STATUS["abstain"][1] == "the system declined to conclude"


def test_the_page_shows_the_verbatim_clause_text_and_says_a_status_is_not_a_verdict(api):
    t = texts(page(api))
    assert "It is never a verdict" in t and "Coverage issues" in t and "CL-COV-WATER" in t and "We cover loss of or damage to the Buildings" in t
    assert "fraud" not in t.lower()


# ---------------------------------------------------------------------------------------------------------------- the calculation trace
def test_the_three_figures_are_kept_apart_and_the_trace_shows_every_step(api):
    at = page(api)
    m = {x.label: x.value for x in at.metric}
    assert m["Assessed loss"] == "INR 98,000" and m["Payable"] == "INR 80,000" and m["Reserve forecast"] == "none"
    rows = calc_trace.rows(api.claim("CLM-0002")["state"]["settlement"])
    assert [r["step"] for r in rows][:3] == ["Assessed covered loss", "Less non-covered items", "Less deductible"] and rows[1]["change"] == "-8,000" and rows[2]["change"] == "-10,000"
    assert rows[0]["before"] == "INR 0" and rows[-1]["step"] == "Rounding" and rows[-1]["after"] == "INR 80,000" and "In words" in texts(at)


def test_a_calculation_that_cannot_be_made_shows_no_payable_names_what_is_missing_and_a_forecast_only(api):
    at = page(api, "CLM-0005")
    m = {x.label: x.value for x in at.metric}
    assert m["Payable"] == "not calculated" and m["Reserve forecast"] == "INR 100,000" and m["Assessed loss"] == "INR 110,000"
    t = texts(at)
    assert "cannot be calculated yet. Missing: water damage sublimit" in t and "never a payment" in t


def test_money_is_shown_in_whole_rupees_and_decimals_only_when_there_are_any():
    assert calc_trace.money("98000.00") == "INR 98,000" and calc_trace.money("1234.50") == "INR 1,234.50" and calc_trace.money(None) == "not available"


# ---------------------------------------------------------------------------------------------------------------- the approval form
def test_a_reason_is_mandatory_for_edit_reject_and_request_more_evidence_but_not_for_approve():
    assert approval.problem("approve", "", {}) is None
    for d in ("edit", "reject", "request_more_evidence"):
        assert "needs a reason" in approval.problem(d, "  ", {"x": "1"})
    assert "at least one settlement input" in approval.problem("edit", "why", {}) and approval.problem("edit", "why", {"x": "1"}) is None


def test_an_edit_without_a_reason_is_stopped_in_the_form_and_nothing_is_decided(api):
    at = page(api, decide=True)
    widget(at, "radio", "Decision").set_value("edit").run()
    at.text_input(key="dec_edit_non_covered_items_total_CLM-0002").set_value("28000")
    at = submit(at)
    assert "needs a reason" in texts(at) and api.claim("CLM-0002")["status"]["phase"] == "awaiting_human_decision" and api.audit("CLM-0002")["human_decisions"] == []


def test_a_reject_without_a_reason_is_stopped_in_the_form(api):
    at = page(api, decide=True)
    widget(at, "radio", "Decision").set_value("reject")
    assert "needs a reason" in texts(submit(at)) and api.claim("CLM-0002")["status"]["phase"] == "awaiting_human_decision"


def test_an_edit_recalculates_the_engine_and_the_reason_and_the_changed_input_appear_in_the_audit_view(api):
    at = page(api, decide=True)
    widget(at, "radio", "Decision").set_value("edit").run()
    at.text_input(key="dec_edit_non_covered_items_total_CLM-0002").set_value("28,000")
    widget(at, "text_input", REASON).set_value("the surveyor confirmed a second upgrade")
    at = submit(at)
    assert not at.exception
    state = api.claim("CLM-0002")
    assert state["state"]["settlement"]["payable"] == "60000" and state["status"]["phase"] == "awaiting_human_decision"
    t = texts(at)
    assert "Edits already in force: Non-covered items total = 28000" in t
    assert "Edited by a reviewer at the gate" in texts(page(api))                                     # and the workspace's calculation says so
    d = api.audit("CLM-0002")["human_decisions"]
    assert d[-1]["data"]["decision"] == "edit" and d[-1]["data"]["reason"] == "the surveyor confirmed a second upgrade" and d[-1]["data"]["edits"] == {"non_covered_items_total": "28000"}
    assert "Human decisions and overrides" not in texts(page(api))                                  # moved to the Audit & decision pack page


def test_the_api_refuses_a_role_that_is_too_low_and_the_form_shows_why(api):
    at = page(api, decide=True)
    widget(at, "selectbox", "Reviewer role").set_value("adjuster")
    widget(at, "radio", "Decision").set_value("reject").run()
    widget(at, "text_input", REASON).set_value("outside the period")
    t = texts(submit(at))
    assert "deny_claim" in t and api.claim("CLM-0002")["status"]["phase"] == "awaiting_human_decision"


def test_an_approval_needs_no_reason_and_closes_the_gate(api):
    at = page(api, decide=True)
    widget(at, "selectbox", "Reviewer role").set_value("supervisor")
    at = submit(at)
    assert not at.exception and api.claim("CLM-0002")["state"]["resolution_status"] == "approved"
    assert "Decided" in texts(at) and "Decisions recorded" in texts(at)


from claimtower.ui.components.actions import label  # noqa: E402


def test_the_proposed_next_steps_are_shown_on_the_overview_and_the_decision_page(api):
    n = len([a for a in api.claim("CLM-0002")["state"]["proposed_actions"] if a["status"] != "cancelled"])
    for at in (page(api), page(api, decide=True)):
        assert not at.exception and "Proposed next steps" in [h.value for h in at.subheader]
    at = page(api)
    shown = " ".join(m.value for m in at.markdown) + " ".join(e.label for e in at.expander)
    assert n == 0 or all(label(a) in shown for a in api.claim("CLM-0002")["state"]["proposed_actions"] if a["status"] != "cancelled")
