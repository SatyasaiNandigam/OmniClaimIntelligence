"""The Policies page: the library, adding a document (the id is read from inside it), the conflict and replace path, and reload."""
import shutil

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from claimtower.api.app import create_app
from claimtower.composition import SYNTHETIC, build_policy_manager, build_registry
from claimtower.core.config import load_settings
from claimtower.ingestion.store import DocumentStore
from claimtower.orchestration.graph import ClaimRunner
from claimtower.ui.client import ApiClient, ApiError

SETTINGS = load_settings()
POLICIES = SYNTHETIC / "policies"


@pytest.fixture
def api(tmp_path) -> ApiClient:
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    for f in POLICIES.glob("*.pdf"):
        if f.stem != "END-01":
            shutil.copy(f, builtin / f.name)
    m = build_policy_manager(SETTINGS, tmp_path / "uploads", builtin)
    store = DocumentStore(tmp_path / "store")
    runner = ClaimRunner(build_registry(SETTINGS, document_store=store, policy_manager=m), settings=SETTINGS)
    return ApiClient(http=TestClient(create_app(runner, store, synthetic_claims=SYNTHETIC / "claims", policies=m)))


def page(api) -> AppTest:
    def app():
        import streamlit as st

        from claimtower.ui.views import policies
        policies.render(st.session_state["_client"])

    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["_client"] = api
    at.run()
    return at


def test_the_client_adds_a_policy_document_and_refuses_a_changed_one_unless_replace(api):
    assert "END-01" not in {d["doc_id"] for d in api.policy_documents()}
    r = api.add_policy_document("anything.pdf", (POLICIES / "END-01.pdf").read_bytes(), uploaded_by="ops-1")
    assert r["outcome"] == "added" and r["document"]["doc_id"] == "END-01" and "END-01" in {d["doc_id"] for d in api.policy_documents()}
    assert api.add_policy_document("again.pdf", (POLICIES / "END-01.pdf").read_bytes())["outcome"] == "unchanged"
    assert api.reload_policies()["total"] >= 1
    with pytest.raises(ApiError) as e:
        api.add_policy_document("junk.pdf", b"nope")
    assert e.value.status == 422


def test_the_page_lists_the_library_and_shows_a_document_in_detail(api):
    at = page(api)
    assert not at.exception and [t.label for t in at.tabs] == ["Library", "Add a document", "Reload", "Policy schedules (mock PAS)"]
    assert {m.label for m in at.metric} == {"Documents", "Built in", "Uploaded", "Clauses left for a person"}
    assert at.selectbox(key="policy_pick").value in {d["doc_id"] for d in api.policy_documents()}


def test_the_reload_button_reports_the_library(api):
    at = page(api)
    next(b for b in at.button if b.label == "Reload the library").click()
    at.run()
    assert not at.exception and any("document(s) in the library" in s.value for s in at.success)


def test_the_page_says_so_when_the_server_cannot_manage_policies(tmp_path):
    store = DocumentStore(tmp_path / "store")
    runner = ClaimRunner(build_registry(SETTINGS, document_store=store), settings=SETTINGS)
    bare = ApiClient(http=TestClient(create_app(runner, store, synthetic_claims=SYNTHETIC / "claims")))
    at = page(bare)
    assert not at.exception and "does not manage policy documents" in at.info[0].value


def test_the_client_adds_many_policy_documents_in_one_request(api):
    res = api.add_policy_documents([("a.pdf", (POLICIES / "END-01.pdf").read_bytes()), ("b.pdf", b"nope")], uploaded_by="ops-1")
    assert (res["added"], res["failed"]) == (1, 1) and [r["outcome"] for r in res["results"]] == ["added", "failed"]
    assert "END-01" in {d["doc_id"] for d in api.policy_documents()}


def test_the_add_tab_takes_many_files(api):
    at = page(api)
    assert not at.exception
    assert any(b.label == "Add to the library" and b.disabled for b in at.button)


def test_the_pas_tab_lists_the_policies_and_waits_for_a_schedule(tmp_path):
    from claimtower.integrations.mock.data import PAS_FILE
    from claimtower.integrations.mock.pas_mock import MockPasAdapter

    store = DocumentStore(tmp_path / "store")
    runner = ClaimRunner(build_registry(SETTINGS, document_store=store), settings=SETTINGS)
    pas = MockPasAdapter(PAS_FILE, tmp_path / "reg.json")
    with_pas = ApiClient(http=TestClient(create_app(runner, store, synthetic_claims=SYNTHETIC / "claims", pas=pas)))
    at = page(with_pas)
    assert not at.exception and any(b.label == "Read the schedule" and b.disabled for b in at.button)
    assert "Policies in the mock PAS" in " ".join(m.value for m in at.markdown)
