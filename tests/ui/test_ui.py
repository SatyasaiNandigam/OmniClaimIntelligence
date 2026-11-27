"""The Streamlit UI (T7.2 first cut): the client speaks only the API, and the page works through it. No browser: Streamlit's AppTest drives the script in-process."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from claimtower.api.app import create_app
from claimtower.composition import SYNTHETIC, build_registry, policy_library_from_pdfs
from claimtower.core.config import load_settings
from claimtower.ingestion.store import DocumentStore
from claimtower.orchestration.graph import ClaimRunner
from claimtower.ui.client import ApiClient, ApiError

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()
APP = str(ROOT / "src" / "claimtower" / "ui" / "app.py")
PDF = (SYNTHETIC / "claims" / "CLM-0001" / "plumber_report.pdf").read_bytes()


@pytest.fixture(scope="module")
def library():
    return policy_library_from_pdfs(SETTINGS)


@pytest.fixture
def api(tmp_path, library) -> ApiClient:
    store = DocumentStore(tmp_path / "store")
    runner = ClaimRunner(build_registry(SETTINGS, document_store=store, policy_library=library))
    return ApiClient(http=TestClient(create_app(runner, store, synthetic_claims=SYNTHETIC / "claims")))


def page(api: ApiClient | None, claim: str | None = None, **state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    if api is not None:
        at.session_state["_client"] = api
    if claim:
        at.session_state["selected_claim"] = claim
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def texts(at: AppTest) -> str:
    return " ".join([*[e.value for e in at.error], *[w.value for w in at.warning], *[i.value for i in at.info], *[m.value for m in at.markdown], *[c.value for c in at.caption]])


# ------------------------------------------------------------------------------------------------ the client
def test_the_client_covers_every_endpoint_the_ui_needs(api):
    assert api.health()["mock_agents"] == [] and {a["node"] for a in api.agents()} >= {"gap", "coverage"}
    v = api.create_claim(claim_id="CLM-0001", use_synthetic_documents=True, policy_ref="")
    assert v["status"]["phase"] == "awaiting_human_decision" and [c["claim_id"] for c in api.claims()] == ["CLM-0001"]
    assert api.claim("CLM-0001")["state_version"] == 10 and len(api.versions("CLM-0001")) == 10
    assert api.diff("CLM-0001", 3, 6)["changes"] and api.audit("CLM-0001")["events"]
    docs = api.documents("CLM-0001")
    assert docs and api.document_file("CLM-0001", docs[0]["document_id"])[:4] == b"%PDF"
    up = api.upload("CLM-0001", "late.pdf", PDF, uploaded_by="plumber", reassess=False)
    assert up["document"]["filename"] == "late.pdf" and up["reassessment"] is None
    assert api.decide("CLM-0001", "approve", "rev-1", "senior_adjuster")["status"]["phase"] == "completed"


def test_the_client_turns_api_refusals_and_a_missing_server_into_readable_errors(api):
    api.create_claim(claim_id="CLM-0001", use_synthetic_documents=True)
    with pytest.raises(ApiError) as e:
        api.decide("CLM-0001", "approve", "rev-1", "adjuster")
    assert e.value.status == 422 and "senior_adjuster" in e.value.message
    with pytest.raises(ApiError) as e:
        api.claim("CLM-9999")
    assert e.value.status == 404
    with pytest.raises(ApiError) as e:
        api.create_claim(claim_id="x")
    assert e.value.status == 422 and "claim_id" in e.value.message
    with pytest.raises(ApiError) as e:
        ApiClient("http://127.0.0.1:9", timeout=2).health()
    assert e.value.status == 0 and "cannot reach the API" in e.value.message


# ------------------------------------------------------------------------------------------------ the page
def test_the_page_says_so_when_the_api_is_not_running():
    at = page(ApiClient("http://127.0.0.1:9", timeout=2))
    assert not at.exception and "Cannot reach the API" in texts(at) and "run_api.py" in texts(at)


def test_with_no_claim_the_page_explains_how_to_start(api):
    at = page(api)
    assert not at.exception and "Create a claim in the sidebar" in texts(at) and [b.label for b in at.button] == ["Refresh", "Create and run to the gate"]


def test_an_open_claim_shows_the_tabs_the_tier_and_the_open_questions(api):
    api.create_claim(claim_id="CLM-0001", use_synthetic_documents=True)
    at = page(api, "CLM-0001")
    assert not at.exception and not at.error
    assert [t.label for t in at.tabs] == ["Overview & decision", "Documents", "Document reader", "Facts", "Questions & gaps", "Assessment & decision", "Policy", "History"]
    m = {x.label: x.value for x in at.metric}
    assert m["Tier"] == "complex" and m["Version"] == "10" and m["Required approver"] == "senior adjuster"
    assert "YET TO IMPLEMENT" not in texts(at) and "Q-SUDDEN" in texts(at)                                # every agent is real now
    assert len(at.dataframe) >= 5                                                                    # documents, facts, questions, policy clauses, history, audit


def test_the_decision_form_refuses_a_role_that_is_too_low_and_accepts_an_authorised_one(api):
    api.create_claim(claim_id="CLM-0001", use_synthetic_documents=True)
    at = page(api, "CLM-0001")
    next(s for s in at.selectbox if s.label == "Reviewer role").set_value("adjuster")
    next(b for b in at.button if b.label == "Submit decision").click()
    at.run()
    assert not at.exception and "may not decide this claim" in texts(at)
    assert api.claim("CLM-0001")["status"]["phase"] == "awaiting_human_decision"                    # nothing was decided
    next(s for s in at.selectbox if s.label == "Reviewer role").set_value("senior_adjuster")
    next(b for b in at.button if b.label == "Submit decision").click()
    at.run()
    assert not at.exception and api.claim("CLM-0001")["state"]["resolution_status"] == "approved"
    assert "decided" in texts(at) or any("decided" in s.value for s in at.subheader)


def test_a_reject_without_a_reason_is_refused_by_the_api_and_shown(api):
    api.create_claim(claim_id="CLM-0002", use_synthetic_documents=True)
    at = page(api, "CLM-0002")
    next(r for r in at.radio if r.label == "Decision").set_value("reject")
    next(b for b in at.button if b.label == "Submit decision").click()
    at.run()
    assert "needs a reason" in texts(at) and api.claim("CLM-0002")["status"]["phase"] == "awaiting_human_decision"


def test_creating_a_claim_from_the_sidebar_runs_it_and_opens_it(api):
    at = page(api)
    next(t for t in at.text_input if t.label == "Claim id (optional)").set_value("CLM-0003")
    next(c for c in at.checkbox if c.label.startswith("Load the demo documents")).set_value(True)
    next(b for b in at.button if b.label == "Create and run to the gate").click()
    at.run()
    assert not at.exception and at.session_state["selected_claim"] == "CLM-0003"
    assert api.claim("CLM-0003")["status"]["gate"]["reason_codes"] == ["POLICY_NOT_IN_FORCE"] and [t.label for t in at.tabs][0] == "Overview & decision"


def test_creating_a_claim_with_a_bad_id_shows_the_api_message(api):
    at = page(api)
    next(t for t in at.text_input if t.label == "Claim id (optional)").set_value("x")
    next(b for b in at.button if b.label == "Create and run to the gate").click()
    at.run()
    assert not at.exception and "claim_id" in texts(at)


def test_a_halted_claim_shows_which_agent_failed(api):
    runner = api._http.app.state.runner

    class Down:
        agent_id, owns = "evidence", frozenset({"documents", "facts", "image_findings"})

        def run(self, state):
            raise RuntimeError("the document store is down")

    from claimtower.orchestration.registry import AgentRegistry
    slots = {s.node: (s.agent, s.status) for s in runner.registry}
    slots["evidence"] = (Down(), "implemented")
    runner.registry = AgentRegistry(slots)
    runner.graph = __import__("claimtower.orchestration.graph", fromlist=["build_graph"]).build_graph(runner.registry, runner.supervisor, runner.graph.checkpointer)
    api.create_claim(claim_id="CLM-0001")
    at = page(api, "CLM-0001")
    assert not at.exception and "An agent failed at **evidence**" in texts(at) and "document store is down" in texts(at)


# ------------------------------------------------------------------------------------------------ typing in the boxes
def test_no_box_is_inside_a_form_so_enter_can_never_create_a_claim_or_submit_a_decision():
    assert "st.form" not in Path(APP).read_text(encoding="utf-8")


def test_the_press_enter_hint_under_the_text_boxes_is_hidden(api):
    at = page(api)
    assert any('data-testid="InputInstructions"' in m.value and "display: none" in m.value for m in at.markdown)


def test_the_create_boxes_are_cleared_after_a_claim_is_created_and_the_decision_boxes_are_per_claim(api):
    at = page(api)
    next(t for t in at.text_input if t.label == "Claim id (optional)").set_value("CLM-0002")
    next(c for c in at.checkbox if c.label.startswith("Load the demo documents")).set_value(True)
    next(b for b in at.button if b.label == "Create and run to the gate").click()
    at.run()
    assert not at.exception and at.session_state["selected_claim"] == "CLM-0002"
    assert next(t for t in at.text_input if t.label == "Claim id (optional)").value == "" and not next(c for c in at.checkbox if c.label.startswith("Load the demo")).value
    next(t for t in at.text_input if t.label.startswith("Reason")).set_value("typed for CLM-0002")
    api.create_claim(claim_id="CLM-0004", use_synthetic_documents=True)
    at.session_state["selected_claim"] = "CLM-0004"
    at.run()
    assert next(t for t in at.text_input if t.label.startswith("Reason")).value == ""                  # a draft reason does not follow you to another claim


def test_the_mock_warning_appears_when_an_agent_is_still_a_mock(tmp_path, library):
    store = DocumentStore(tmp_path / "store")
    runner = ClaimRunner(build_registry(SETTINGS, document_store=store, policy_library=library, force_mock=("action",)))
    mocked = ApiClient(http=TestClient(create_app(runner, store, synthetic_claims=SYNTHETIC / "claims")))
    mocked.create_claim(claim_id="CLM-0001", use_synthetic_documents=True)
    assert "YET TO IMPLEMENT" in texts(page(mocked, "CLM-0001"))
