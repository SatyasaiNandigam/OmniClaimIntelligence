"""The Streamlit UI over the API. Everything it does goes through `ApiClient`, i.e. through the endpoints in docs/api.md.

    python scripts/run_api.py        # terminal 1
    python scripts/run_ui.py         # terminal 2   (or: python scripts/run_ui.py --start-api)

A sidebar menu of pages, one job each: Claims (list, open, create from documents) and Claim workspace (one claim's tabs). More pages follow: decision, audit and decision pack, policies.
The sidebar also holds the API address and the agents.
"""
from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from claimtower.ui.client import DEFAULT_URL, ApiClient, ApiError
from claimtower.ui.common import STATUS_ICON, client, df, show_error
from claimtower.ui.components import calc_trace, counterfactual_view, doc_viewer, fact_table, gap_board, header, issue_matrix, overview
from claimtower.ui.views import audit, claims, decision, policies

HIDE_ENTER_HINT = """<style>
[data-testid="InputInstructions"] {display: none;}                                            /* the "Press Enter to apply" text under every text box */
[data-testid="stMetricValue"] {font-size: 1.2rem; line-height: 1.3;}                          /* KPI boxes: a smaller figure that wraps instead of being cut off */
[data-testid="stMetricValue"] > div {white-space: normal; overflow: visible; text-overflow: clip; overflow-wrap: anywhere;}
[data-testid="stMetricLabel"] {font-size: 0.8rem; opacity: 0.75;}
[data-testid="stMetricLabel"] p {white-space: normal; overflow: visible; text-overflow: clip;}
[data-testid="stMainBlockContainer"] {padding-top: 2.5rem;}                                    /* less empty space above the page */
[data-testid="stMetric"] {gap: 0;}
hr {margin: 0.6rem 0 !important;}
</style>"""


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
    return True


# ---------------------------------------------------------------------------------------------------- claim tabs
def reassess(api: ApiClient, claim_id: str, node: str) -> None:
    """Run the claim again from `node` and show the result (the page reloads on success)."""
    try:
        with st.spinner(f"Re-assessing from {node}…"):
            api.reassess(claim_id, node)
        st.rerun()
    except ApiError as e:
        show_error(e)


def documents_tab(api: ApiClient, v: dict) -> None:
    claim_id = v["claim_id"]
    docs = api.documents(claim_id)
    types: dict[str, int] = {}
    for d in docs:
        types[d["detected_type"]] = types.get(d["detected_type"], 0) + 1
    st.caption(f"{len(docs)} document(s): " + ", ".join(f"{n} {t.replace('_', ' ')}" for t, n in sorted(types.items())) if docs else "No document yet.")
    st.dataframe(df([{"file": d["filename"], "type": d["detected_type"].replace("_", " "), "pages": d["page_count"], "uploaded by": d.get("uploaded_by"), "OCR needed": d["needs_ocr"],
                      "id": d["document_id"]} for d in docs], ["file", "type", "pages", "uploaded by", "OCR needed", "id"]), hide_index=True, width="stretch",
                 column_config={"file": st.column_config.TextColumn(width="large"), "pages": st.column_config.NumberColumn(width="small"), "OCR needed": st.column_config.CheckboxColumn(width="small")})
    with st.container(border=True):
        st.subheader("Add evidence")
        st.caption("Uploading re-assesses the claim from the intake node: the checklist is refreshed and only the new documents are processed. A recommendation waiting at the gate is replaced.")
        files = st.file_uploader("PDF, image, text or e-mail", accept_multiple_files=True, key=f"upload_{claim_id}_{len(docs)}")
        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        uploader = c1.text_input("Uploaded by", value="claimant")
        if c2.button("Upload and re-assess", type="primary", disabled=not files):
            try:
                with st.spinner("Uploading and re-assessing…"):
                    for i, f in enumerate(files):
                        api.upload(claim_id, f.name, f.getvalue(), uploaded_by=uploader or None, reassess=i == len(files) - 1)
                st.rerun()
            except ApiError as e:
                show_error(e)
    with st.container(border=True):
        st.subheader("Re-assess without new documents")
        st.caption("Runs the claim again, for example after a policy document was added to the library or when a step failed. A recommendation waiting at the gate is replaced.")
        nodes = [a["node"] for a in api.agents()]
        r1, r2 = st.columns([2, 1], vertical_alignment="bottom")
        start = r1.selectbox("Start from step", nodes, index=nodes.index("policy") if "policy" in nodes else 0, key=f"reassess_node_{claim_id}",
                             help="Steps before it are not run again; later steps whose inputs did not change are skipped.")
        if r2.button("Re-assess", key=f"reassess_{claim_id}"):
            reassess(api, claim_id, start)
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
    c1, c2, c3, c4 = st.columns(4)
    for col, (label, val) in zip((c1, c2, c3, c4), (("Policy", p["policy_id"]), ("Policy version", p["version"]), ("Status", p["pas_status"].replace("_", " ")),
                                                      ("Period", f"{p['effective_from']} to {p.get('effective_to') or 'open'}"))):
        with col, st.container(border=True):
            st.metric(label, val)
    if p.get("mock"):
        st.caption("Policy data comes from the MOCK policy administration system.")
    if p["endorsements"]:
        st.markdown("**Endorsements applied:** " + ", ".join(f"`{e['endorsement_id']}`" for e in p["endorsements"]))
    types = sorted({c["clause_type"] for c in p["clauses"]})
    pick = st.multiselect("Filter by clause type", types, key=f"policy_types_{v['claim_id']}", placeholder="all clause types")
    rows = [c for c in p["clauses"] if not pick or c["clause_type"] in pick]
    st.caption(f"{len(rows)} of {len(p['clauses'])} clauses")
    st.dataframe(df([{"clause": c["clause_id"], "type": c["clause_type"], "heading": c["heading"], "document": c["document"], "text": c["text"]} for c in rows]),
                 hide_index=True, width="stretch", column_config={"text": st.column_config.TextColumn(width="large")})


FIELD_NAME = {"documents": "Documents", "facts": "Facts", "open_questions": "Questions", "missing_evidence": "Evidence requests", "contradictions": "Conflicts", "counterfactuals": "Scenarios",
              "coverage_issues": "Coverage issues", "proposed_actions": "Proposed actions", "drafts": "Drafts", "claim_header": "Claim header", "policy_snapshot": "Policy snapshot",
              "settlement": "Settlement", "routing": "Routing", "qa_report": "Quality checks", "audit_ref": "Audit reference", "human_approval_required": "Human approval required",
              "resolution_status": "Business status", "image_findings": "Image findings"}


def change_summary(c: dict) -> str:
    """One short phrase for a changed field: '+12 added, 2 changed' for a list, 'a → b' for a single value."""
    if c["before"] is not None or c["after"] is not None:
        return f"{c['before']}  →  {c['after']}"
    return ", ".join(f"{len(c[k])} {k}" for k in ("added", "removed", "changed") if c[k]) or "changed"


def diff_view(api: ApiClient, claim_id: str, nums: list[int]) -> None:
    c1, c2, _ = st.columns([1, 1, 3])
    a = c1.selectbox("From version", nums, index=max(len(nums) - 2, 0), key=f"diff_from_{claim_id}")
    b = c2.selectbox("To version", nums, index=len(nums) - 1, key=f"diff_to_{claim_id}")
    if a == b:
        st.caption("Pick two different versions to compare.")
        return
    d = api.diff(claim_id, min(a, b), max(a, b))
    if not d["changes"] and not d["status_changes"]:
        st.info(f"Nothing differs between version {d['from_version']} and version {d['to_version']}.")
        return
    st.markdown(f"**Version {d['from_version']} → {d['to_version']}**: {len(d['changes'])} field(s) changed" + (f", {len(d['status_changes'])} question(s) moved" if d["status_changes"] else ""))
    st.dataframe(df([{"what": FIELD_NAME.get(c["field"], c["field"].replace("_", " ").capitalize()), "change": change_summary(c)} for c in d["changes"]], ["what", "change"]),
                 hide_index=True, width="stretch", column_config={"what": st.column_config.TextColumn(width="small"), "change": st.column_config.TextColumn(width="large")})
    if d["status_changes"]:
        st.markdown("**Questions that moved**")
        st.dataframe(df([{"question": x["question_id"], "from": f"{STATUS_ICON.get(x['before'], '')} {(x['before'] or 'new').replace('_', ' ')}",
                          "to": f"{STATUS_ICON.get(x['after'], '')} {(x['after'] or 'gone').replace('_', ' ')}"} for x in d["status_changes"]], ["question", "from", "to"]),
                     hide_index=True, width="stretch")
    listed = [c for c in d["changes"] if any(c[k] for k in ("added", "removed", "changed"))]
    if listed:
        with st.expander("Which items changed"):
            for c in listed:
                st.markdown(f"**{FIELD_NAME.get(c['field'], c['field'])}**")
                for kind, mark in (("added", "➕"), ("removed", "➖"), ("changed", "✏️")):
                    if c[kind]:
                        st.caption(f"{mark} {kind}: " + ", ".join(f"`{i}`" for i in c[kind][:40]) + (f" … and {len(c[kind]) - 40} more" if len(c[kind]) > 40 else ""))


def history_tab(api: ApiClient, v: dict) -> None:
    claim_id = v["claim_id"]
    versions = api.versions(claim_id)
    st.subheader("Versions")
    st.dataframe(df([{"version": x["version"], "step": x["node"], "mock": "mock" if x["mock"] else "", "what happened": x["summary"]} for x in versions]),
                 hide_index=True, width="stretch", column_config={"version": st.column_config.NumberColumn(width="small"), "step": st.column_config.TextColumn(width="small"),
                                                                  "mock": st.column_config.TextColumn(width="small"), "what happened": st.column_config.TextColumn(width="large")})
    nums = [x["version"] for x in versions]
    if len(nums) > 1:
        st.subheader("Compare two versions")
        diff_view(api, claim_id, nums)
    st.caption("The decisions, the audit log and the decision pack are on the **Audit & decision pack** page.")


def claim_switcher(api: ApiClient) -> None:
    """The claim being looked at, switchable from every page: the workspace and the decision page follow it."""
    claims_now = api.claims()
    ids = [c["claim_id"] for c in claims_now]
    if not ids:
        return
    current = st.session_state.get("selected_claim")
    if current in ids:
        st.session_state["claim_switch"] = current                          # the page may have opened another claim (Claims page): the box follows
    else:
        st.session_state.pop("claim_switch", None)
    kw = {} if current in ids else {"index": None, "placeholder": "choose a claim"}
    st.sidebar.selectbox("Current claim", ids, key="claim_switch", on_change=pick_claim, **kw)


def pick_claim() -> None:
    """The sidebar box changed: that claim is now the one every page shows (runs before the page, so the page already follows it)."""
    if st.session_state.get("claim_switch"):
        st.session_state["selected_claim"] = st.session_state["claim_switch"]


# ---------------------------------------------------------------------------------------------------- pages
def workspace(api: ApiClient, open_decision: Callable[[], None]) -> None:
    claim_id = st.session_state.get("selected_claim")
    if not claim_id:
        st.title("Claim workspace")
        st.info("No claim is open. Go to **Claims** in the menu: open a claim, or create one from its documents (the demo claim ids are CLM-0001 to CLM-0005).")
        return
    try:
        v = api.claim(claim_id)
    except ApiError as e:
        show_error(e)
        return
    s = v["status"]
    header.render(v)
    tabs = st.tabs(["Overview", "Questions & gaps", "Documents", "Document reader", "Facts", "Assessment", "Policy", "History"])
    with tabs[0]:
        overview.render(v, open_decision, lambda node: reassess(api, claim_id, node))
    with tabs[1]:
        gap_board.render(v)
        st.divider()
        counterfactual_view.render(v)
    with tabs[2]:
        documents_tab(api, v)
    with tabs[3]:
        doc_viewer.render(api, v)
    with tabs[4]:
        fact_table.render(v, st.session_state.get("low_confidence_below"))
    with tabs[5]:
        issue_matrix.render(v)
        calc_trace.render(v, (s.get("gate") or {}).get("overrides"))
        if s["phase"] == "awaiting_human_decision":
            st.info("The decision itself is made on the **Human decision** page.")
            if st.button("Go to the decision", key="assessment_to_decision"):
                open_decision()
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
    claim_switcher(api)
    has_claim = bool(st.session_state.get("selected_claim"))

    def workspace_view() -> None:
        workspace(api, lambda: st.switch_page(decision_page))

    def decision_view() -> None:
        decision.render(api, lambda: st.switch_page(claims_page))

    workspace_page = st.Page(workspace_view, title="Claim workspace", icon="📂", url_path="workspace", default=has_claim)
    decision_page = st.Page(decision_view, title="Human decision", icon="⚖️", url_path="decision")

    def claims_view() -> None:
        claims.render(api, lambda: st.switch_page(workspace_page))

    claims_page = st.Page(claims_view, title="Claims", icon="🗂", url_path="claims", default=not has_claim)

    def audit_view() -> None:
        audit.render(api, lambda: st.switch_page(claims_page))

    audit_page = st.Page(audit_view, title="Audit & decision pack", icon="🧾", url_path="audit")

    def policies_view() -> None:
        policies.render(api)

    policies_page = st.Page(policies_view, title="Policies", icon="📚", url_path="policies")
    pages = [claims_page, workspace_page, decision_page, audit_page, policies_page]
    st.navigation(pages).run()
    st.caption("Prototype. SYNTHETIC data only. A human decides every approval, denial and payment: the system only recommends.")


main()
