"""The Streamlit UI over the API. Everything it does goes through `ApiClient`, i.e. through the endpoints in docs/api.md.

    python scripts/run_api.py        # terminal 1
    python scripts/run_ui.py         # terminal 2   (or: python scripts/run_ui.py --start-api)

A sidebar menu of pages, one job each: Claims (list, open, create from documents) and Claim workspace (one claim's tabs). More pages follow: decision, audit and decision pack, policies.
The sidebar also holds the API address and the agents.
"""
from __future__ import annotations

import streamlit as st

from claimtower.ui.client import DEFAULT_URL, ApiClient, ApiError
from claimtower.ui.common import PHASE_TEXT, STATUS_ICON, client, df, show_error
from claimtower.ui.components import approval, calc_trace, counterfactual_view, doc_viewer, fact_table, gap_board, issue_matrix
from claimtower.ui.views import claims

HIDE_ENTER_HINT = """<style>[data-testid="InputInstructions"] {display: none;}</style>"""     # the "Press Enter to apply" text under every text box


# ---------------------------------------------------------------------------------------------------- sidebar
def sidebar(api: ApiClient) -> bool:
    """The API address, its health and the agents. False when the API cannot be reached (the page then stops)."""
    st.sidebar.title("Claim Control Tower")
    st.sidebar.caption("Prototype · SYNTHETIC data only")
    st.sidebar.text_input("API address", value=DEFAULT_URL, key="api_url")
    try:
        health = api.health()
    except ApiError as e:
        show_error(e)
        st.sidebar.info("Start the API first: `python scripts/run_api.py`")
        return False
    st.session_state["low_confidence_below"] = health.get("low_confidence_below")
    with st.sidebar.expander(f"Agents: {len(health['implemented_agents'])} real, {len(health['mock_agents'])} mock"):
        st.dataframe(df(api.agents(), ["node", "status", "task", "will_do"]), hide_index=True, width="stretch")
    return True


# ---------------------------------------------------------------------------------------------------- claim tabs
def overview(api: ApiClient, v: dict) -> None:
    status, state = v["status"], v["state"]
    if status["phase"] == "halted":
        st.error(f"An agent failed at **{status['paused_at']}**: {status['halted_reason']}. The claim is parked for a human.")
    if status["mock_nodes_run"]:
        st.warning("These agents are **YET TO IMPLEMENT** and returned labelled fixtures, not computed results: " + ", ".join(status["mock_nodes_run"]))
    r = state.get("routing")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tier", r["tier"] if r else "–")
    c2.metric("Version", v["state_version"])
    c3.metric("Business status", state["resolution_status"].replace("_", " "))
    c4.metric("Required approver", r["required_approver_role"].replace("_", " ") if r else "–")
    if r:
        st.caption("Reasons: " + ", ".join(r["reason_codes"]))
    if state["open_questions"]:
        st.write("**Open questions**  " + "   ".join(f"{STATUS_ICON.get(q['status'], '')} {q['question_id']} {q['status'].replace('_', ' ')}" for q in state["open_questions"]))
    if state.get("settlement"):
        s = state["settlement"]
        st.write(f"**Settlement{' (mock)' if 'settlement' in status['mock_nodes_run'] else ''}:** {s['status'].replace('_', ' ')}" + (f", payable INR {s['payable']}" if s.get("payable") else ""))

    if status["phase"] == "awaiting_human_decision":
        st.info("To decide, open the **Assessment & decision** tab: the coverage issues with their clauses, the calculation, and the decision form.")
    else:
        st.info(PHASE_TEXT.get(status["phase"], status["phase"]) + (f" — {state['resolution_status'].replace('_', ' ')}" if status["phase"] == "completed" else ""))


def documents_tab(api: ApiClient, v: dict) -> None:
    claim_id = v["claim_id"]
    docs = api.documents(claim_id)
    st.dataframe(df([{"file": d["filename"], "type": d["detected_type"], "pages": d["page_count"], "uploaded by": d.get("uploaded_by"), "OCR needed": d["needs_ocr"],
                      "id": d["document_id"]} for d in docs], ["file", "type", "pages", "uploaded by", "OCR needed", "id"]), hide_index=True, width="stretch")
    st.subheader("Add documents")
    st.caption("Uploading re-assesses the claim from the intake node: the checklist is refreshed and only the new documents are processed. A recommendation waiting at the gate is replaced.")
    files = st.file_uploader("PDF, image, text or e-mail", accept_multiple_files=True, key=f"upload_{claim_id}_{len(docs)}")
    uploader = st.text_input("Uploaded by", value="claimant")
    if st.button("Upload", disabled=not files):
        try:
            with st.spinner("Uploading and re-assessing…"):
                for i, f in enumerate(files):
                    api.upload(claim_id, f.name, f.getvalue(), uploaded_by=uploader or None, reassess=i == len(files) - 1)
            st.rerun()
        except ApiError as e:
            show_error(e)
    if docs:
        with st.expander("Download an original"):
            pick = st.selectbox("Document", [d["document_id"] for d in docs], format_func=lambda i: next(d["filename"] for d in docs if d["document_id"] == i))
            name = next(d["filename"] for d in docs if d["document_id"] == pick)
            st.download_button(f"Download {name}", api.document_file(claim_id, pick), file_name=name)


def policy_tab(v: dict) -> None:
    p = v["state"].get("policy_snapshot")
    if p is None:
        st.info("No policy snapshot: no policy is in force on the loss date, or the policy number or loss date is missing.")
        return
    st.write(f"**{p['policy_id']}** version {p['version']} · {p['pas_status'].replace('_', ' ')} · effective {p['effective_from']} to {p.get('effective_to') or 'open'}"
             + ("  ·  MOCK PAS" if p.get("mock") else ""))
    if p["endorsements"]:
        st.write("Endorsements applied: " + ", ".join(e["endorsement_id"] for e in p["endorsements"]))
    st.dataframe(df([{"clause": c["clause_id"], "type": c["clause_type"], "heading": c["heading"], "document": c["document"], "text": c["text"]} for c in p["clauses"]]),
                 hide_index=True, width="stretch")


def history_tab(api: ApiClient, v: dict) -> None:
    claim_id = v["claim_id"]
    versions = api.versions(claim_id)
    st.dataframe(df([{"version": x["version"], "node": x["node"], "mock": "mock" if x["mock"] else "", "what happened": x["summary"]} for x in versions]),
                 hide_index=True, width="stretch")
    nums = [x["version"] for x in versions]
    if len(nums) > 1:
        c1, c2 = st.columns(2)
        a = c1.selectbox("Compare from version", nums, index=0)
        b = c2.selectbox("to version", nums, index=len(nums) - 1)
        d = api.diff(claim_id, a, b)
        st.caption("Changes: " + (", ".join(c["field"] for c in d["changes"]) or "none"))
        for c in d["changes"]:
            st.write(f"**{c['field']}**: " + (f"{c['before']} → {c['after']}" if c["before"] or c["after"] else
                                              "; ".join(f"{k} {', '.join(x)}" for k, x in (("added", c["added"]), ("removed", c["removed"]), ("changed", c["changed"])) if x)))
        if d["status_changes"]:
            st.write("**Question status**: " + ", ".join(f"{s['question_id']} {s['before'] or 'new'} → {s['after'] or 'gone'}" for s in d["status_changes"]))
    st.subheader("Human decisions and overrides")
    audit = api.audit(claim_id)
    st.caption("Every decision with who made it, in which role, why, and what they changed. A refused decision is shown too.")
    st.dataframe(df([{"#": e["seq"], "decision": e["data"].get("decision") or "refused", "by": e["data"].get("reviewer_id", ""), "role": e["data"].get("reviewer_role", ""),
                      "reason": e["data"].get("reason", "") or ("" if e["kind"] == "human_decision" else e["message"]),
                      "changed inputs": ", ".join(f"{k} = {x}" for k, x in (e["data"].get("edits") or {}).items())} for e in audit["human_decisions"]],
                     ["#", "decision", "by", "role", "reason", "changed inputs"]), hide_index=True, width="stretch")
    st.subheader("Audit log")
    st.dataframe(df([{"#": e["seq"], "event": e["kind"].replace("_", " "), "node": e["node"], "mock": "mock" if e["mock"] else "", "version": e["state_version"],
                      "message": e["message"]} for e in audit["events"]]), hide_index=True, width="stretch")
    st.caption(audit["note"])


# ---------------------------------------------------------------------------------------------------- pages
def workspace(api: ApiClient) -> None:
    claim_id = st.session_state.get("selected_claim")
    st.title("Claim workspace")
    if not claim_id:
        st.info("No claim is open. Go to **Claims** in the menu: open a claim, or create one from its documents (the demo claim ids are CLM-0001 to CLM-0005).")
        return
    try:
        v = api.claim(claim_id)
    except ApiError as e:
        show_error(e)
        return
    s = v["status"]
    st.subheader(f"{claim_id}  ·  {PHASE_TEXT.get(s['phase'], s['phase'])}")
    tabs = st.tabs(["Overview & decision", "Documents", "Document reader", "Facts", "Questions & gaps", "Assessment & decision", "Policy", "History"])
    with tabs[0]:
        overview(api, v)
    with tabs[1]:
        documents_tab(api, v)
    with tabs[2]:
        doc_viewer.render(api, v)
    with tabs[3]:
        fact_table.render(v, st.session_state.get("low_confidence_below"))
    with tabs[4]:
        gap_board.render(v)
        counterfactual_view.render(v)
    with tabs[5]:
        issue_matrix.render(v)
        calc_trace.render(v, (s.get("gate") or {}).get("overrides"))
        approval.render(api, v, show_error)
    with tabs[6]:
        policy_tab(v)
    with tabs[7]:
        history_tab(api, v)


def main() -> None:
    st.set_page_config(page_title="Claim Control Tower", page_icon="🛡", layout="wide")
    st.markdown(HIDE_ENTER_HINT, unsafe_allow_html=True)
    api = client()
    if not sidebar(api):
        st.stop()
    has_claim = bool(st.session_state.get("selected_claim"))

    def workspace_view() -> None:
        workspace(api)

    workspace_page = st.Page(workspace_view, title="Claim workspace", icon="📂", url_path="workspace", default=has_claim)

    def claims_view() -> None:
        claims.render(api, lambda: st.switch_page(workspace_page))

    pages = [st.Page(claims_view, title="Claims", icon="🗂", url_path="claims", default=not has_claim), workspace_page]
    if has_claim:
        st.sidebar.caption(f"Open claim: **{st.session_state['selected_claim']}**")
    st.navigation(pages).run()
    st.caption("Prototype. SYNTHETIC data only. A human decides every approval, denial and payment: the system only recommends.")


main()
