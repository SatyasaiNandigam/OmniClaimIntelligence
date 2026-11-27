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
    st.caption("A wording or an endorsement as a PDF. The file may be called anything: the document id and version are read from inside it.")
    f = st.file_uploader("Policy document", type=["pdf"], key=f"policy_file_{n}")
    c1, c2 = st.columns(2)
    by = c1.text_input("Uploaded by", key=f"policy_by_{n}")
    given = c2.text_input("Document id (only if the document has none inside)", key=f"policy_id_{n}")
    replace = st.checkbox("Replace the document with the same id if the content differs", key=f"policy_replace_{n}")
    if st.button("Add to the library", type="primary", disabled=f is None, key=f"policy_add_{n}"):
        try:
            r = api.add_policy_document(f.name, f.getvalue(), uploaded_by=by.strip() or None, document_id=given.strip() or None, replace=replace)
        except ApiError as e:
            show_error(e)
            if e.status == 409:
                st.info("A document with this id is already in the library with different content. Tick the replace box to replace it.")
            return
        st.session_state["_policy_n"] = n + 1                                   # empties the boxes for the next document
        st.session_state["_policy_done"] = r
        st.rerun()
    r = st.session_state.get("_policy_done")
    if r:
        (st.success if r["outcome"] != "unchanged" else st.info)(f"**{r['document']['doc_id']}**: {r['outcome']}. {r['note']}")


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


def render(api: ApiClient) -> None:
    st.title("Policies")
    st.caption("The wordings and endorsements the system reads clauses from. A document added here changes no claim already assessed: re-assess the claim to use it.")
    try:
        api.policy_documents()
    except ApiError as e:
        if e.status == 404:
            st.info("This server does not manage policy documents.")
        else:
            show_error(e)
        return
    t_lib, t_add, t_reload = st.tabs(["Library", "Add a document", "Reload"])
    with t_lib:
        _library(api)
    with t_add:
        _add(api)
    with t_reload:
        _reload(api)
