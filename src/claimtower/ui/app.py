"""A very simple Streamlit UI over the API (T7.2, first cut). Everything it does goes through `ApiClient`, i.e. through the endpoints in docs/api.md.

    python scripts/run_api.py        # terminal 1
    python scripts/run_ui.py         # terminal 2   (or: python scripts/run_ui.py --start-api)

Left: the API address, the agents (real or "YET TO IMPLEMENT" mock), the claims, and a form to create one.
Main: for the selected claim, tabs for the overview, documents, document reader, facts, gaps and scenarios, the assessment and the human decision (issue matrix, calculation trace, approval form), policy, and history.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from claimtower.ui.client import DEFAULT_URL, ApiClient, ApiError
from claimtower.ui.components import approval, calc_trace, counterfactual_view, doc_viewer, fact_table, gap_board, issue_matrix

STATUS_ICON = {"established": "✅", "partially_established": "🟡", "possible": "🟠", "unverified": "⚪", "missing": "🔴", "requires_validation": "🟣"}
PHASE_TEXT = {"awaiting_human_decision": "⏸ waiting for a human decision", "completed": "✔ decided", "halted": "⛔ halted (an agent failed)", "not_started": "not started"}
SAMPLE_FILES = ["fnol.pdf", "plumber_report.pdf", "surveyor_report.pdf", "estimate.pdf", "invoice.pdf", "claimant_email.pdf"]


HIDE_ENTER_HINT = """<style>[data-testid="InputInstructions"] {display: none;}</style>"""     # the "Press Enter to apply" text under every text box


def client() -> ApiClient:
    injected = st.session_state.get("_client")                      # tests inject a client wired to an in-process API
    if injected is not None:
        return injected
    return ApiClient(st.session_state.get("api_url", DEFAULT_URL))


def show_error(e: ApiError) -> None:
    st.error(f"{'Cannot reach the API' if e.status == 0 else f'The API said no ({e.status})'}: {e.message}")


def df(rows: list[dict], columns: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns or [])


# ---------------------------------------------------------------------------------------------------- sidebar
def sidebar(api: ApiClient) -> str | None:
    st.sidebar.title("Claim Control Tower")
    st.sidebar.caption("Prototype · SYNTHETIC data only")
    st.sidebar.text_input("API address", value=DEFAULT_URL, key="api_url")
    try:
        health = api.health()
    except ApiError as e:
        show_error(e)
        st.sidebar.info("Start the API first: `python scripts/run_api.py`")
        return None
    st.session_state["low_confidence_below"] = health.get("low_confidence_below")
    with st.sidebar.expander(f"Agents: {len(health['implemented_agents'])} real, {len(health['mock_agents'])} mock"):
        st.dataframe(df(api.agents(), ["node", "status", "task", "will_do"]), hide_index=True, width="stretch")

    claims = api.claims()
    ids = [c["claim_id"] for c in claims]
    st.sidebar.subheader("Claims")
    if st.sidebar.button("Refresh", key="refresh"):
        st.rerun()
    default = st.session_state.get("selected_claim")
    choice = st.sidebar.selectbox("Open a claim", ids, index=ids.index(default) if default in ids else None, placeholder="no claim yet" if not ids else "choose a claim",
                                  key="claim_choice", format_func=lambda i: f"{i}  ·  {next(c['phase'] for c in claims if c['claim_id'] == i).replace('_', ' ')}")
    if choice:
        st.session_state["selected_claim"] = choice

    with st.sidebar.expander("Create a claim", expanded=not ids):
        if st.session_state.pop("_reset_create", False):                    # clear the boxes after a claim was created (before the widgets exist)
            st.session_state.update(new_claim_id="", new_policy="", new_loss_date=None, new_desc="", new_demo=False)
        claim_id = st.text_input("Claim id (optional)", placeholder="CLM-1001", key="new_claim_id")
        policy_ref = st.text_input("Policy number", placeholder="POL-1013", key="new_policy")
        loss_date = st.date_input("Loss date", value=None, key="new_loss_date")
        loss_desc = st.text_input("Loss description", placeholder="Dishwasher inlet hose burst", key="new_desc")
        demo = st.checkbox("Load the demo documents of this claim id (CLM-0001 to CLM-0005)", key="new_demo")
        if st.button("Create and run to the gate", key="create_claim"):
            try:
                with st.spinner("Running the claim through the agents (up to a minute with the model)…"):
                    created = api.create_claim(claim_id=claim_id.strip() or None, policy_ref=policy_ref.strip() or None,
                                               loss_date=loss_date.isoformat() if loss_date else None, loss_type="water_damage" if policy_ref else None,
                                               loss_description=loss_desc.strip() or None, use_synthetic_documents=demo)
                st.session_state["selected_claim"] = created["claim_id"]
                st.session_state["_reset_create"] = True
                st.rerun()
            except ApiError as e:
                show_error(e)
    return st.session_state.get("selected_claim")


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


# ---------------------------------------------------------------------------------------------------- page
def main() -> None:
    st.set_page_config(page_title="Claim Control Tower", page_icon="🛡", layout="wide")
    st.markdown(HIDE_ENTER_HINT, unsafe_allow_html=True)
    api = client()
    claim_id = sidebar(api)
    st.title("Insurance Claim Resolution Control Tower")
    st.caption("Prototype. SYNTHETIC data only. A human decides every approval, denial and payment: the system only recommends.")
    if not claim_id:
        st.info("Create a claim in the sidebar (try the demo claim id **CLM-0001** with *Load the demo documents*), or open one.")
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


main()
