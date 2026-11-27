"""The claim workspace: the banner's headline figure, and the Overview's list of what needs attention (both are plain functions of the claim state)."""
from pathlib import Path

from claimtower.ui.components.header import settlement_figure
from claimtower.ui.components.overview import attention

STATUS = {"phase": "awaiting_human_decision", "paused_at": None, "halted_reason": None, "mock_nodes_run": []}


def settlement(**kw):
    return {"currency": "INR", "status": "calculated", "payable": "229260", "forecast_low": None, "forecast_high": None, "missing_inputs": [], **kw}


def test_the_headline_is_the_payable_only_when_it_is_calculated_else_the_forecast_range_else_nothing():
    assert settlement_figure({"settlement": settlement()})[:2] == ("Payable", "INR 229,260")
    held = settlement(status="cannot_calculate", payable=None, forecast_low="204260", forecast_high="425000", missing_inputs=["assessed_covered_loss"])
    label, value, _ = settlement_figure({"settlement": held})
    assert label == "Forecast range" and value == "INR 204,260 to INR 425,000"
    assert settlement_figure({"settlement": settlement(status="cannot_calculate", payable=None, missing_inputs=["applicable_deductible"])})[1:] == ("not calculated", "applicable deductible")
    assert settlement_figure({})[1] == "–"


def test_the_attention_list_orders_the_serious_things_first_and_ignores_minor_differences():
    state = {"contradictions": [{"severity": "high", "question_id": "Q-SUDDEN"}, {"severity": "low", "question_id": "Q-AMOUNT"}],
             "missing_evidence": [{"gap_type": "contradictory", "resolves_question_ids": ["Q-SUDDEN"], "minimum_evidence_request": {"addressee_role": "plumber", "items": ["Say when you attended."]}}],
             "settlement": settlement(status="cannot_calculate", payable=None, missing_inputs=["assessed_covered_loss"]), "qa_report": None}
    items = attention(state, {**STATUS, "phase": "halted", "paused_at": "evidence", "halted_reason": "store down"})
    assert [level for level, _ in items] == ["error", "warning", "info", "warning"]
    text = " ".join(t for _, t in items)
    assert "1 conflict(s)" in text and "Q-AMOUNT" not in text and "Ask plumber" in text and "held" in text


def test_a_clean_claim_has_nothing_to_attend_to():
    assert attention({"contradictions": [], "missing_evidence": [], "settlement": settlement(), "qa_report": {"passed": True, "failures": []}}, STATUS) == []


# ---------------------------------------------------------------------------------------------------------------- the Human decision page
def test_the_decision_page_without_a_claim_points_to_the_claims_page():
    from streamlit.testing.v1 import AppTest

    from claimtower.ui.client import ApiClient
    code = ("import streamlit as st\nfrom claimtower.ui.views import decision\nfrom claimtower.ui.client import ApiClient\n"
            "decision.render(ApiClient('http://127.0.0.1:9'), lambda: None)")
    at = AppTest.from_string(code, default_timeout=30).run()
    assert not at.exception and "No claim is open" in " ".join(i.value for i in at.info) and [b.label for b in at.button] == ["Go to Claims"]
    assert ApiClient


def test_the_decision_page_shows_the_briefing_then_the_form_and_after_a_decision_the_outcome(tmp_path):
    from fastapi.testclient import TestClient
    from streamlit.testing.v1 import AppTest

    from claimtower.api.app import create_app
    from claimtower.composition import SYNTHETIC, build_registry, policy_library_from_pdfs
    from claimtower.core.config import load_settings
    from claimtower.ingestion.store import DocumentStore
    from claimtower.orchestration.graph import ClaimRunner
    from claimtower.ui.client import ApiClient

    settings = load_settings()
    store = DocumentStore(tmp_path / "store")
    api = ApiClient(http=TestClient(create_app(ClaimRunner(build_registry(settings, document_store=store, policy_library=policy_library_from_pdfs(settings))), store,
                                                synthetic_claims=SYNTHETIC / "claims")))
    api.create_claim(claim_id="CLM-0001", use_synthetic_documents=True)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "src" / "claimtower" / "ui" / "app.py"), default_timeout=120)
    at.session_state["_client"], at.session_state["selected_claim"] = api, "CLM-0001"
    at.run()
    next(b for b in at.button if b.label == "Go to the decision").click()
    at.run()
    texts = " ".join([*[m.value for m in at.markdown], *[h.value for h in at.subheader]])
    assert not at.exception and "Proposed next steps" in texts and "Coverage issues" in [t.label for t in at.tabs] and "Human decision" in texts
    assert not any(t.label in ("Overview", "Documents", "Facts") for t in at.tabs)                       # the page is for deciding: no workspace tabs here
    next(s for s in at.selectbox if s.label == "Reviewer role").set_value("senior_adjuster")
    next(b for b in at.button if b.label == "Submit decision").click()
    at.run()
    assert not at.exception and any("Decided" in x.value for x in at.success) and "Decisions recorded" in " ".join(h.value for h in at.subheader)


def test_proposed_actions_pair_with_their_drafts_and_hide_cancelled_ones():
    from claimtower.ui.components.actions import label, rows
    act = lambda i, t, p, st="proposed", ref=None: {"action_id": i, "action_type": t, "responsible_party": p, "execution_mode": "draft", "status": st, "idempotency_key": i, "payload_ref": ref}  # noqa: E731
    state = {"proposed_actions": [act("A1", "request_information", "plumber", ref="D1"), act("A2", "create_task", "adjuster"), act("A3", "request_information", "owner", "cancelled", "D2")],
             "drafts": [{"draft_id": "D1", "subject": "s", "body": "b", "addressee_role": "plumber", "status": "draft", "kind": "information_request"}]}
    got = rows(state)
    assert [r["action"]["action_id"] for r in got] == ["A1", "A2"] and got[0]["draft"]["draft_id"] == "D1" and got[1]["draft"] is None
    assert label(got[1]["action"]) == "create task · adjuster"
