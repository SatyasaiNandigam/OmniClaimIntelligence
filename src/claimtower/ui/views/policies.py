"""The Policies page: the library of policy wordings and endorsements the system reads clauses from. Add a document, reload the folders, see what is in the library and how each document was identified.
It belongs to no claim. A document added here changes no claim already assessed: re-assess a claim to use it."""
from __future__ import annotations

import streamlit as st

from claimtower.ui.client import ApiClient, ApiError
from claimtower.ui.common import df, show_error

ID_FROM = {"document": "read from inside the document", "model": "read by the model", "given": "given by the uploader", "file_name": "taken from the file name"}


def _library(api: ApiClient) -> None:
    docs = api.policy_documents()
    built, up = sum(1 for d in docs if d["source"] == "built-in"), sum(1 for d in docs if d["source"] == "uploaded")
    cols = st.columns(4)
    for c, (label, value) in zip(cols, [("Documents", len(docs)), ("Built in", built), ("Uploaded", up), ("Clauses left for a person", sum(d["unclassified"] for d in docs))]):
        with c, st.container(border=True):
            st.metric(label, value)
    if not docs:
        st.info("The library is empty. Add a document in the next tab.")
        return
    kinds = sorted({d["kind"] for d in docs})
    pick = st.multiselect("Kind", kinds, key="policy_kinds", placeholder="all kinds")
    shown = [d for d in docs if not pick or d["kind"] in pick]
    st.dataframe(df([{"document": d["doc_id"], "title": d["title"] or "", "kind": d["kind"].replace("_", " "), "version": d["version"] or "", "clauses": d["clauses"], "unclear": d["unclassified"],
                      "source": d["source"], "identified": ID_FROM.get(d["id_from"], d["id_from"]), "file": d["filename"]} for d in shown]),
                 hide_index=True, width="stretch", column_config={"clauses": st.column_config.NumberColumn(width="small"), "unclear": st.column_config.NumberColumn(width="small")})
    if shown:
        one = st.selectbox("Look at one document", [d["doc_id"] for d in shown], key="policy_pick")
        d = next(x for x in shown if x["doc_id"] == one)
        with st.container(border=True):
            st.markdown(f"**{d['doc_id']}**  :gray-badge[{d['source']}]" + (" :orange-badge[replaces a built-in]" if d["replaces_built_in"] else ""))
            st.caption(f"{ID_FROM.get(d['id_from'], d['id_from'])}" + (f": “{d['id_evidence']}”" if d["id_evidence"] else "")
                       + (f" · uploaded by {d['uploaded_by']}" if d["uploaded_by"] else "") + (f" · {d['uploaded_at']}" if d["uploaded_at"] else ""))
            for w in d["warnings"]:
                st.warning(w)
            if d["details"]:
                st.dataframe(df([{"what": k.replace("_", " "), "value": v} for k, v in d["details"].items()]), hide_index=True, width="stretch")
            st.caption(f"sha256 {d['sha256'][:16]}…")


def _add(api: ApiClient) -> None:
    n = st.session_state.setdefault("_policy_n", 0)
    st.caption("Wordings and endorsements as PDFs, one or many at once. The files may be called anything: each document's id and version are read from inside it.")
    files = st.file_uploader("Policy documents", type=["pdf"], accept_multiple_files=True, key=f"policy_files_{n}")
    c1, c2 = st.columns([1, 2])
    by = c1.text_input("Uploaded by", key=f"policy_by_{n}")
    replace = c2.checkbox("Replace a document with the same id if the content differs", key=f"policy_replace_{n}")
    if st.button(f"Add {len(files)} document(s) to the library" if files else "Add to the library", type="primary", disabled=not files, key=f"policy_add_{n}"):
        try:
            with st.spinner("Reading the documents…"):
                r = api.add_policy_documents([(f.name, f.getvalue()) for f in files], uploaded_by=by.strip() or None, replace=replace)
        except ApiError as e:
            show_error(e)
            return
        st.session_state["_policy_n"] = n + 1                                   # empties the boxes for the next batch
        st.session_state["_policy_done"] = r
        st.rerun()
    r = st.session_state.get("_policy_done")
    if r:
        parts = [f"{r[k]} {k}" for k in ("added", "replaced", "unchanged", "failed") if r[k]]
        (st.error if r["failed"] else st.success)(", ".join(parts) or "nothing to add")
        st.dataframe(df([{"file": x["filename"], "result": x["outcome"], "document": (x["document"] or {}).get("doc_id", ""), "version": (x["document"] or {}).get("version") or "",
                          "clauses": (x["document"] or {}).get("clauses", ""), "why": x["error"]} for x in r["results"]]), hide_index=True, width="stretch",
                     column_config={"why": st.column_config.TextColumn(width="large")})
        if any(x["status"] == 409 for x in r["results"]):
            st.info("A failed file with a 409 has the id of a document already in the library but different content. Tick the replace box and add it again to replace it.")
        st.caption(r["note"])


def _reload(api: ApiClient) -> None:
    st.caption("Reads the built-in and upload folders again: use it after files were added or removed on the server.")
    if st.button("Reload the library", key="policy_reload"):
        try:
            r = api.reload_policies()
        except ApiError as e:
            show_error(e)
            return
        st.success(f"{r['total']} document(s) in the library.")
        for label, items in (("Added", r["added"]), ("Removed", r["removed"]), ("Changed", r["changed"])):
            if items:
                st.markdown(f"**{label}:** " + ", ".join(items))
        for p in r["problems"]:
            st.warning(p)


def policy_from_form(base: dict, holder: str, start: str, end: str, wording: str, endorsements: list[dict], schedule: list[dict]) -> dict:
    """The record to register: the proposal with what the person corrected in the form. Dates are ISO strings; a blank row is ignored; an endorsement without a date starts with the period."""
    v = dict(base["versions"][0])
    v.update(effective_from=start, effective_to=end or None, base_wording_id=wording.strip(),
             endorsements=[{"endorsement_id": str(e["id"]).strip(), "effective_from": str(e.get("effective_from") or start), "effective_to": None}
                           for e in endorsements if str(e.get("id") or "").strip()],
             schedule={str(r["name"]).strip(): str(r["value"]).replace(",", "").strip() for r in schedule if str(r.get("name") or "").strip() and str(r.get("value") or "").strip()})
    seg = dict(base["status_history"][0])
    seg.update(effective_from=start, effective_to=end or None)
    return {**base, "holder_name": holder.strip() or base["holder_name"], "versions": [v], "status_history": [seg]}


def _pas_list(rows: list[dict]) -> None:
    st.markdown("#### Policies in the mock PAS")
    st.dataframe(df([{"policy": r["policy_no"], "holder": r["holder_name"], "from": str(r["effective_from"] or ""), "to": str(r["effective_to"] or ""), "wording": r["base_wording_id"],
                      "endorsements": ", ".join(r["endorsements"]), "deductible": r["schedule"].get("standard_deductible", ""), "source": r["source"]} for r in rows]),
                 hide_index=True, width="stretch")


def _pas(api: ApiClient) -> None:
    try:
        rows = api.pas_policies()
    except ApiError as e:
        if e.status == 404:
            st.info("This server has no mock policy system to write to.")
        else:
            show_error(e)
        return
    st.caption("A real policy system is read-only for us. For the demo, the mock one is filled from a policy schedule: the system reads the schedule, checks every value against the document, "
               "and you confirm the record before it is stored. A policy registered here is used at once by the Policy agent.")
    n = st.session_state.setdefault("_pas_n", 0)
    f = st.file_uploader("Policy schedule (PDF)", type=["pdf"], key=f"pas_file_{n}")
    if st.button("Read the schedule", type="primary", disabled=f is None, key=f"pas_read_{n}"):
        try:
            with st.spinner("Reading the schedule…"):
                st.session_state["_pas_proposal"] = api.read_schedule(f.name, f.getvalue())
            st.rerun()
        except ApiError as e:
            show_error(e)
    pr = st.session_state.get("_pas_proposal")
    if pr:
        with st.container(border=True):
            st.markdown(f"**{pr['filename']}**  :gray-badge[read by {'the model' if pr['read_by'] == 'model' else 'fixed patterns'}]")
            for p in pr["problems"]:
                st.error(p)
            for w in pr["warnings"]:
                st.warning(w)
            if pr["checks"]:
                with st.expander("What was checked against the document"):
                    st.dataframe(df([{"": "✅" if c["found"] else "❌", "value": c["field"], "read": c["value"], "result": c["note"] or "found in the document"} for c in pr["checks"]]),
                                 hide_index=True, width="stretch")
            if pr["policy"]:
                _proposal_form(api, pr, n)
    _pas_list(rows)


def _proposal_form(api: ApiClient, pr: dict, n: int) -> None:
    from datetime import date

    base = pr["policy"]
    v = base["versions"][0]
    st.markdown(f"**Policy {base['policy_no']}**: check, correct if needed, then register.")
    c1, c2, c3 = st.columns([2, 1, 1])
    holder = c1.text_input("Policyholder", base["holder_name"], key=f"pas_holder_{n}")
    start = c2.date_input("Period starts", date.fromisoformat(v["effective_from"]), key=f"pas_start_{n}")
    end = c3.date_input("Period ends", date.fromisoformat(v["effective_to"]) if v["effective_to"] else None, key=f"pas_end_{n}")
    wording = st.text_input("Base wording id (the wording in the policy library)", v["base_wording_id"], key=f"pas_wording_{n}")
    e1, e2 = st.columns(2)
    with e1:
        st.caption("Endorsements attached")
        ends = st.data_editor(df([{"id": e["endorsement_id"], "effective_from": e["effective_from"]} for e in v["endorsements"]], ["id", "effective_from"]), num_rows="dynamic",
                              hide_index=True, width="stretch", key=f"pas_ends_{n}")
    with e2:
        st.caption("Limits and deductible")
        sched = st.data_editor(df([{"name": k, "value": x} for k, x in v["schedule"].items()], ["name", "value"]), num_rows="dynamic", hide_index=True, width="stretch", key=f"pas_sched_{n}")
    replace = False
    if pr["existing"] != "none":
        st.warning(f"Policy {base['policy_no']} is already in the mock PAS ({pr['existing']}).")
        replace = st.checkbox("Replace it with this record", key=f"pas_replace_{n}")
    blocked = not wording.strip() or (pr["existing"] != "none" and not replace) or end is None
    if st.button("Register in the mock PAS", type="primary", disabled=blocked, key=f"pas_register_{n}"):
        rec = policy_from_form(base, holder, start.isoformat(), end.isoformat() if end else "", wording, ends.to_dict("records"), sched.to_dict("records"))
        try:
            r = api.register_pas_policy(rec, replace=replace)
        except ApiError as e:
            show_error(e)
            return
        st.session_state["_pas_n"] = n + 1
        st.session_state.pop("_pas_proposal", None)
        st.session_state["_pas_done"] = r
        st.rerun()
    if blocked:
        st.caption("Enter the base wording id" + (" and tick Replace" if pr["existing"] != "none" else "") + " to register.")


def render(api: ApiClient) -> None:
    st.title("Policy Library")
    st.caption("The wordings and endorsements the system reads clauses from. A document added here changes no claim already assessed: re-assess the claim to use it.")
    try:
        api.policy_documents()
        library_ok = True
    except ApiError as e:
        library_ok = False
        if e.status != 404:
            show_error(e)
    t_lib, t_add, t_reload, t_pas = st.tabs(["Library", "Add a document", "Reload", "Policy schedules (mock PAS)"])
    for tab, body in ((t_lib, _library), (t_add, _add), (t_reload, _reload)):
        with tab:
            if library_ok:
                body(api)
            else:
                st.info("This server does not manage policy documents.")
    with t_pas:
        done = st.session_state.pop("_pas_done", None)
        if done:
            st.success(f"Policy {done['policy_no']}: {done['outcome']}. {done['note']}")
        _pas(api)
