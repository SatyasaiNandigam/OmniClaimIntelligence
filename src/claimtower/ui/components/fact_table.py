"""The facts table (T7.3): every fact the system holds, with where it came from, how sure the reading was, and a way to open it in the document it was read from.

Low-confidence readings are MARKED (never hidden, never promoted), duplicates and verification are shown as they are, and no label anywhere accuses a person: the words are "low confidence",
"conflicting", "unverified". A fact is opened in the Document reader tab through two session keys set by a callback (the reader reads them when its widgets are created).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

LOW = "⚠ low confidence"


def rows(state: dict, floor: float | None) -> list[dict]:
    docs = {d["document_id"]: d for d in state["documents"]}
    return [{"": LOW if floor is not None and f["extraction_confidence"] < floor else "", "fact": f["fact_id"], "document": docs.get(f["source_document_id"], {}).get("filename", f["source_document_id"]),
             "page": f["page"], "proposition": f["proposition"], "evidence (verbatim)": f["evidence_text"], "confidence": f["extraction_confidence"],
             "bears on": ", ".join(f["related_question_ids"]), "checked": f["verification_status"].replace("_", " ")} for f in state["facts"]]


def open_in_reader(claim_id: str, document_id: str, fact_id: str) -> None:
    st.session_state[f"viewer_doc_{claim_id}"] = document_id
    st.session_state[f"viewer_fact_{claim_id}"] = fact_id


def render(v: dict, floor: float | None) -> None:
    state, k = v["state"], v["claim_id"]
    data = rows(state, floor)
    low = sum(1 for r in data if r[""])
    st.caption(f"{len(data)} facts" + (f", {low} read with low confidence (marked, to be checked against the document)" if low else "")
               + ". Every fact quotes the original document verbatim and was re-checked against it.")
    f1, f2, f3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    docs = sorted({r["document"] for r in data})
    qs = sorted({q for r in data for q in r["bears on"].split(", ") if q})
    pick_docs = f1.multiselect("Document", docs, key=f"facts_docs_{k}", placeholder="all documents")
    pick_qs = f2.multiselect("Bears on question", qs, key=f"facts_qs_{k}", placeholder="all questions")
    only_low = f3.toggle("Low confidence only", key=f"facts_low_{k}", disabled=not low)
    shown = [r for r in data if (not pick_docs or r["document"] in pick_docs) and (not pick_qs or set(pick_qs) & set(r["bears on"].split(", "))) and (not only_low or r[""])]
    if len(shown) != len(data):
        st.caption(f"Showing {len(shown)} of {len(data)} facts.")
    st.dataframe(pd.DataFrame(shown) if shown else pd.DataFrame(columns=["fact"]), hide_index=True, width="stretch",
                 column_config={"proposition": st.column_config.TextColumn(width="large"), "evidence (verbatim)": st.column_config.TextColumn(width="large"),
                                "confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f", width="small"), "page": st.column_config.NumberColumn(width="small")})
    if shown:
        by_id = {f["fact_id"]: f for f in state["facts"]}
        pick = st.selectbox("Show a fact in its document", [r["fact"] for r in shown], key=f"facttable_pick_{k}",
                            format_func=lambda i: f"{i}  ·  {by_id[i]['proposition'][:70]}")
        f = by_id[pick]
        st.button("Open in the Document reader tab", key=f"facttable_open_{k}", on_click=open_in_reader, args=(k, f["source_document_id"], pick))
        if st.session_state.get(f"viewer_fact_{k}") == pick:
            st.success(f"{pick} is open in the **Document reader** tab: page {f['page']} of {by_id[pick]['source_document_id']}, with its place marked.")
