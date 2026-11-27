"""The Audit & decision pack page: pack, trail with its chain check, decisions, event log; without a trail it says so."""
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from claimtower.api.app import create_app
from claimtower.audit.trail import AuditTrail
from claimtower.composition import SYNTHETIC, build_registry, policy_library_from_pdfs
from claimtower.core.config import load_settings
from claimtower.ingestion.store import DocumentStore
from claimtower.orchestration.graph import ClaimRunner
from claimtower.ui.client import ApiClient

def run_page(api, claim="CLM-0001") -> AppTest:
    def app():
        import streamlit as st

        from claimtower.ui.views import audit
        audit.render(st.session_state["_client"], lambda: None)

    at = AppTest.from_function(app, default_timeout=120)
    at.session_state["_client"], at.session_state["selected_claim"] = api, claim
    at.run()
    return at


def make(tmp_path, with_trail: bool) -> ApiClient:
    settings = load_settings()
    store = DocumentStore(tmp_path / "store")
    registry = build_registry(settings, document_store=store, policy_library=policy_library_from_pdfs(settings))
    trail = AuditTrail(tmp_path / "trail.jsonl") if with_trail else None
    runner = ClaimRunner(registry, settings=settings, trail=trail)
    api = ApiClient(http=TestClient(create_app(runner, store, synthetic_claims=SYNTHETIC / "claims")))
    api.create_claim(claim_id="CLM-0001", use_synthetic_documents=True)
    return api


@pytest.mark.parametrize("with_trail", [True, False])
def test_the_audit_page_shows_the_pack_the_chain_and_the_log_or_says_there_is_no_trail(tmp_path, with_trail):
    at = run_page(make(tmp_path, with_trail))
    assert not at.exception and [t.label for t in at.tabs][-4:] == ["Decision pack", "Audit trail", "Decisions", "Event log"]
    text = " ".join(x.value for x in [*at.success, *at.info, *at.markdown])
    if with_trail:
        assert "The chain is intact" in text and any(b.label == "Prepare PDF" for b in at.button)
    else:
        assert "runs without" in text


def test_the_audit_page_without_a_claim_points_to_the_claims_page():
    def app():
        from claimtower.ui.views import audit
        audit.render(None, lambda: None)

    at = AppTest.from_function(app, default_timeout=30)
    at.run()
    assert not at.exception and "No claim is open" in at.info[0].value
