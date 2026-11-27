"""The Claims page: every claim at a glance, open one, or create a new one from its documents in one request (POST /claims/with-documents)."""
from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from claimtower.ui.client import ApiClient, ApiError
from claimtower.ui.common import PHASE_SHORT, STATUS_ICON, df, show_error

SAMPLE_IDS = "CLM-0001 to CLM-0005"


def _questions(counts: dict[str, int]) -> str:
    return "  ".join(f"{STATUS_ICON.get(s, '')}{n}" for s, n in counts.items()) or "–"


def render(api: ApiClient, open_claim: Callable[[], None]) -> None:
    st.header("Claims")
    claims = api.claims()
    phases = [c["phase"] for c in claims]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Claims", len(claims))
    m2.metric("Awaiting a decision", phases.count("awaiting_human_decision"))
    m3.metric("Decided", phases.count("completed"))
    m4.metric("Halted", phases.count("halted"))

    all_tab, new_tab = st.tabs(["All claims", "New claim"])
    with all_tab:
        _claim_list(claims, open_claim)
    with new_tab:
        _new_claim(api, open_claim)


def _claim_list(claims: list[dict], open_claim: Callable[[], None]) -> None:
    if not claims:
        st.info("No claim yet. Create one in the **New claim** tab: upload its documents, or load a demo claim.")
        return
    st.dataframe(df([{"claim": c["claim_id"], "status": PHASE_SHORT.get(c["phase"], c["phase"]), "tier": c.get("tier") or "–", "version": c.get("state_version"),
                      "business status": (c.get("resolution_status") or "–").replace("_", " "), "questions": _questions(c["open_questions"]),
                      "mock agents": ", ".join(c["mock_agents"])} for c in claims]), hide_index=True, width="stretch")
    ids = [c["claim_id"] for c in claims]
    selected = st.session_state.get("selected_claim")
    c1, c2, _ = st.columns([2, 1, 3], vertical_alignment="bottom")
    choice = c1.selectbox("Claim", ids, index=ids.index(selected) if selected in ids else 0, key="claims_pick")
    if c2.button("Open claim", type="primary", key="claims_open"):
        st.session_state["selected_claim"] = choice
        open_claim()
    st.caption("Question icons: " + "  ".join(f"{i} {s.replace('_', ' ')}" for s, i in STATUS_ICON.items()))


def _new_claim(api: ApiClient, open_claim: Callable[[], None]) -> None:
    created = st.session_state.get("_created")
    if created:
        _created_summary(created, open_claim)
    n = st.session_state.get("_new_claim_n", 0)                                  # a new key set after each creation clears every box
    st.subheader("Create a claim from its documents")
    st.caption("Upload everything the claimant sent: claim form, reports, estimate, invoice, e-mails, photos. The claim is created and assessed once, after all of them are read. "
               "Every box below is optional: whatever is missing (name, policy, loss date, description) is read from the claim form.")
    files = st.file_uploader("Claim documents (PDF, image, text, Word or e-mail)", accept_multiple_files=True, key=f"new_files_{n}")
    c1, c2, c3 = st.columns(3)
    claim_id = c1.text_input("Claim id", placeholder="generated if empty", key=f"new_id_{n}")
    policy_ref = c2.text_input("Policy number", placeholder="read from the claim form", key=f"new_policy_{n}")
    loss_date = c3.date_input("Loss date", value=None, key=f"new_date_{n}")
    c4, c5 = st.columns(2)
    claimant = c4.text_input("Claimant name", placeholder="read from the claim form", key=f"new_name_{n}")
    uploaded_by = c5.selectbox("Documents sent by", ["claimant", "adjuster", "surveyor", "other"], key=f"new_by_{n}")
    description = st.text_input("Loss description", placeholder="read from the claim form", key=f"new_desc_{n}")
    demo = st.checkbox(f"No files? Load the demo documents of the claim id ({SAMPLE_IDS})", key=f"new_demo_{n}", disabled=bool(files))

    if st.button("Create claim and assess", type="primary", disabled=not (files or demo), key=f"new_go_{n}"):
        fields = {"claim_id": claim_id.strip() or None, "policy_ref": policy_ref.strip() or None, "loss_date": loss_date.isoformat() if loss_date else None,
                  "claimant_name": claimant.strip() or None, "loss_description": description.strip() or None, "loss_type": "water_damage" if policy_ref.strip() else None}
        try:
            with st.spinner("Reading the documents and running the claim to the human gate (up to a minute with the model)…"):
                if files:
                    result = api.create_claim_with_documents([(f.name, f.getvalue()) for f in files], uploaded_by=uploaded_by, **fields)
                else:
                    result = {"claim": api.create_claim(use_synthetic_documents=True, **fields), "stored": [], "skipped": []}
        except ApiError as e:
            show_error(e)
            return
        st.session_state["_created"] = result
        st.session_state["selected_claim"] = result["claim"]["claim_id"]
        st.session_state["_new_claim_n"] = n + 1
        st.rerun()


def _created_summary(result: dict, open_claim: Callable[[], None]) -> None:
    claim = result["claim"]
    status = claim["status"]
    st.success(f"Claim **{claim['claim_id']}** was created: {PHASE_SHORT.get(status['phase'], status['phase'])}. {len(result['stored'])} document(s) stored.")
    for s in result.get("skipped", []):
        st.warning(f"Skipped **{s['filename']}**: {s['reason']}")
    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("Open it", type="primary", key="created_open"):
        st.session_state.pop("_created", None)
        open_claim()
    if b2.button("Dismiss", key="created_dismiss"):
        st.session_state.pop("_created", None)
        st.rerun()
    st.divider()
