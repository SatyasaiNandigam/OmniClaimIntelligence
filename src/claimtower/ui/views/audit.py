"""The Audit & decision pack page: the record of the claim. The decision pack (what was decided, on what evidence, by whom) to read and download, the hash-chained audit trail with its check,
every human decision, and the event log. Nothing here changes the claim."""
from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from claimtower.ui.client import ApiClient, ApiError
from claimtower.ui.common import df, show_error
from claimtower.ui.components import header


def decisions_table(audit: dict) -> None:
    st.dataframe(df([{"#": e["seq"], "decision": e["data"].get("decision") or "refused", "by": e["data"].get("reviewer_id", ""), "role": e["data"].get("reviewer_role", ""),
                      "reason": e["data"].get("reason", "") or ("" if e["kind"] == "human_decision" else e["message"]),
                      "changed inputs": ", ".join(f"{k} = {x}" for k, x in (e["data"].get("edits") or {}).items())} for e in audit["human_decisions"]],
                     ["#", "decision", "by", "role", "reason", "changed inputs"]), hide_index=True, width="stretch")


def _pack(api: ApiClient, claim_id: str) -> None:
    try:
        md = api.decision_pack(claim_id, "md").decode("utf-8")
    except ApiError as e:
        st.info("The decision pack needs the audit trail, which this server runs without." if e.status == 404 else "The decision pack could not be built.")
        if e.status != 404:
            show_error(e)
        return
    c1, c2, _ = st.columns([1, 1, 4])
    c1.download_button("Download Markdown", md, file_name=f"decision-pack-{claim_id}.md", mime="text/markdown", key=f"pack_md_{claim_id}")
    if c2.button("Prepare PDF", key=f"pack_pdf_prepare_{claim_id}"):
        try:
            st.session_state[f"pack_pdf_{claim_id}"] = api.decision_pack(claim_id, "pdf")
        except ApiError as e:
            show_error(e)
    if st.session_state.get(f"pack_pdf_{claim_id}"):
        st.download_button("Download the PDF", st.session_state[f"pack_pdf_{claim_id}"], file_name=f"decision-pack-{claim_id}.pdf", mime="application/pdf", key=f"pack_pdf_dl_{claim_id}")
    with st.container(height=520, border=True):
        st.markdown(md)


def _trail(api: ApiClient, claim_id: str) -> None:
    try:
        t = api.trail(claim_id)
    except ApiError as e:
        st.info("This server runs without an audit trail." if e.status == 404 else "The audit trail could not be read.")
        if e.status != 404:
            show_error(e)
        return
    if t["chain_ok"]:
        st.success(f"The chain is intact: {len(t['records'])} records, none changed or removed since they were written.")
    else:
        st.error("The chain is BROKEN: " + "; ".join(t["problems"]))
    st.caption("Each record carries the hash of the one before it, so altering any record breaks every record after it. Head of the chain: " + t["head"][:16] + "…")
    st.dataframe(df([{"#": r["seq"], "time": r["ts"], "kind": r["kind"].replace("_", " "), "what": r["summary"], "hash": r["hash"][:12]} for r in t["records"]]), hide_index=True, width="stretch",
                 column_config={"#": st.column_config.NumberColumn(width="small"), "what": st.column_config.TextColumn(width="large")})


def render(api: ApiClient, open_claims: Callable[[], None]) -> None:
    claim_id = st.session_state.get("selected_claim")
    if not claim_id:
        st.title("Audit & decision pack")
        st.info("No claim is open. Open a claim from **Claims** first.")
        if st.button("Go to Claims", key="audit_to_claims"):
            open_claims()
        return
    try:
        v = api.claim(claim_id)
        audit = api.audit(claim_id)
    except ApiError as e:
        show_error(e)
        return
    header.render(v)
    t_pack, t_trail, t_dec, t_log = st.tabs(["Decision pack", "Audit trail", "Decisions", "Event log"])
    with t_pack:
        _pack(api, claim_id)
    with t_trail:
        _trail(api, claim_id)
    with t_dec:
        st.caption("Every decision with who made it, in which role, why, and what they changed. A refused decision is shown too.")
        decisions_table(audit)
    with t_log:
        st.dataframe(df([{"#": e["seq"], "event": e["kind"].replace("_", " "), "node": e["node"], "version": e["state_version"], "message": e["message"]} for e in audit["events"]]),
                     hide_index=True, width="stretch", column_config={"message": st.column_config.TextColumn(width="large")})
        st.caption(audit["note"])
