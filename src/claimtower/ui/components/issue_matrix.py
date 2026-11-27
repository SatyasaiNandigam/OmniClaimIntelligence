"""The issue matrix (T7.5): the coverage issues, each with the policy clauses behind it, so the adjuster can check the citations instead of trusting a label.

A status says how well an issue is resolved on the evidence; it is NEVER a verdict on the claim. Each row names the clauses cited (with the endorsements that modify them); opening a row shows
the verbatim clause text, its document and page, the facts the reading rests on, and the note. Rows the system declined to conclude are shown as such, with the reason in words.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

STATUS = {"established": ("✅", "resolved on the evidence"), "identified": ("🔎", "identified, nothing concluded"), "unresolved": ("❔", "cannot be concluded yet"),
          "potentially_relevant": ("🟠", "may apply: a person decides"), "incomplete": ("🟡", "evidence still missing"), "abstain": ("⏸", "the system declined to conclude")}


def rows(state: dict) -> list[dict]:
    return [{"": STATUS.get(i["status"], ("", ""))[0], "issue": i["issue"], "status": i["status"].replace("_", " "), "meaning": STATUS.get(i["status"], ("", ""))[1],
             "clauses": ", ".join(c["clause_id"] + (f" (modified by {', '.join(c['modified_by'])})" if c["modified_by"] else "") for c in i["clause_citations"]) or "none cited",
             "facts": len(i["fact_ids"])} for i in state["coverage_issues"]]


def render(v: dict) -> None:
    state = v["state"]
    st.subheader("Coverage issues")
    st.caption("A status says how well an issue is resolved on the evidence. It is never a verdict: the person at the gate decides.")
    if not state["coverage_issues"]:
        st.info("No coverage issues yet.")
        return
    st.dataframe(pd.DataFrame(rows(state)), hide_index=True, width="stretch")
    facts = {f["fact_id"]: f for f in state["facts"]}
    docs = {d["document_id"]: d["filename"] for d in state["documents"]}
    for i in state["coverage_issues"]:
        with st.expander(f"{STATUS.get(i['status'], ('', ''))[0]} {i['issue']}: {i['status'].replace('_', ' ')}"):
            if i.get("note"):
                st.markdown(f"_{i['note']}_")
            for c in i["clause_citations"]:
                st.markdown(f"**{c['clause_id']}** · {c['document']}" + (f", page {c['page']}" if c.get("page") else "") + (f" · modified by {', '.join(c['modified_by'])}" if c["modified_by"] else ""))
                st.markdown(f"> {c['text']}")
            if not i["clause_citations"]:
                st.caption("No clause is cited for this row" + (" (the system declined to conclude)." if i["status"] == "abstain" else "."))
            if i["fact_ids"]:
                st.markdown("**Facts it rests on**")
                for fid in i["fact_ids"]:
                    f = facts.get(fid)
                    if f:
                        st.markdown(f"- {f['proposition']}  \n  “{f['evidence_text']}” ({docs.get(f['source_document_id'], f['source_document_id'])}, page {f['page']})")
