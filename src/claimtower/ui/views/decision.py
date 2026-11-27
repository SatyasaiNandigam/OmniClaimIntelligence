"""The Human decision page: what the system recommends and why, what is still in the way, and the decision form. Everything else about the claim is in the workspace; this page is only for deciding.

While the claim waits at the gate it shows a short briefing (what needs attention, how the coverage issues stand) and then the form. Once it is decided it shows the outcome
and every decision recorded for the claim, with who made it, in which role and why.
"""
from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from claimtower.ui.client import ApiClient, ApiError
from claimtower.ui.common import show_error
from claimtower.ui.components import approval, header, overview
from claimtower.ui.views import audit as audit_view


def _outcome(api: ApiClient, v: dict) -> None:
    status, state = v["status"], v["state"]
    if status["phase"] == "completed":
        st.success(f"Decided: **{state['resolution_status'].replace('_', ' ')}**.")
    elif status["phase"] == "halted":
        st.error(f"An agent failed at **{status['paused_at']}**: {status['halted_reason']}. The claim is parked for a human.")
    else:
        st.info("This claim has not reached the human gate yet.")
    audit = api.audit(v["claim_id"])
    st.subheader("Decisions recorded")
    audit_view.decisions_table(audit)


def render(api: ApiClient, open_claims: Callable[[], None]) -> None:
    claim_id = st.session_state.get("selected_claim")
    if not claim_id:
        st.title("Human decision")
        st.info("No claim is open. Open a claim from **Claims** first.")
        if st.button("Go to Claims", key="decision_to_claims"):
            open_claims()
        return
    try:
        v = api.claim(claim_id)
    except ApiError as e:
        show_error(e)
        return
    header.render(v)
    if v["status"]["phase"] == "awaiting_human_decision":
        left, right = st.columns([5, 4], gap="large")                       # the form is on screen at once; the briefing scrolls inside its own box
        with left, st.container(height=560, border=True):
            overview.tabs(v["state"], v["status"])
        with right:
            approval.render(api, v, show_error)
    else:
        _outcome(api, v)
