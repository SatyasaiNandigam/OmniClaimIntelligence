"""The document reader (T7.3): what the system is reading, next to the document it is reading.

Left: the page picture; choosing a fact jumps to its page and marks where it was read (the box the extractor recorded, or the quoted line found on the page). Documents with no page picture
(text, e-mail, Word) show the stored text of the page instead. Right: the detected type with the reason, page count, uploader and source, whether the text came from OCR, the source excerpt beside
the selected finding, the extracted fields with page and confidence (low confidence marked), and what the agents did with this document. No label accuses anyone of anything.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from claimtower.ui.client import ApiClient, ApiError

HOW = {"region": "the box recorded when the value was read", "text": "the quoted line, found on the page", "none": "no position could be marked"}


def agent_log(api: ApiClient, claim_id: str, doc: dict) -> None:
    mine = [e for e in api.audit(claim_id)["events"] if e["node"] in ("intake", "evidence") and e["kind"] in ("agent_finished", "node_skipped", "agent_failed")]
    if not mine:
        st.caption("No agent has processed this claim yet.")
    for e in mine[-4:]:
        st.markdown(f"- **{e['node']}** ({e['kind'].replace('_', ' ')}, v{e['state_version']}): {e['message'][:300]}")
    same = doc.get("same_content_as") or []
    if same:
        st.caption("Same file content as: " + ", ".join(same))


def fields_table(details: dict) -> None:
    rows = [{"": "⚠ low confidence" if f["low_confidence"] else "", "field": f["name"], "value": f["value"], "normalised": f["normalised"], "page": f["page"],
             "where": "region recorded" if f["region"] else "line only", "confidence": f["confidence"], "how": f["method"], "flags": ", ".join(f["flags"])} for f in details["fields"]]
    st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame(columns=["field"]), hide_index=True, width="stretch")
    if details["missing_required"]:
        st.warning("Required fields not found: " + ", ".join(details["missing_required"]) + ". Nothing was guessed.")
    for w in details["warnings"]:
        st.caption("Note: " + w)


def render(api: ApiClient, v: dict) -> None:
    claim_id, state = v["claim_id"], v["state"]
    docs = state["documents"]
    if not docs:
        st.info("No documents yet. Upload some in the Documents tab.")
        return
    ids = [d["document_id"] for d in docs]
    label = {d["document_id"]: f"{d['filename']}  ·  {d['detected_type']}" for d in docs}
    doc_key, fact_key = f"viewer_doc_{claim_id}", f"viewer_fact_{claim_id}"
    if st.session_state.get(doc_key) not in ids:
        st.session_state[doc_key] = ids[0]
    doc_id = st.selectbox("Document", ids, key=doc_key, format_func=lambda i: label[i])
    try:
        d = api.document_details(claim_id, doc_id)
    except ApiError as e:
        st.error(f"Cannot read the document details: {e.message}")
        return
    facts = [f for f in state["facts"] if f["source_document_id"] == doc_id]
    options = ["(none: show the first page)"] + [f["fact_id"] for f in facts]
    if st.session_state.get(fact_key) not in options:
        st.session_state[fact_key] = options[0]
    fact_id = st.selectbox("Finding to show", options, key=fact_key, format_func=lambda i: i if i == options[0] else f"{i}  ·  {next(f['proposition'] for f in facts if f['fact_id'] == i)[:70]}")
    fact = next((f for f in facts if f["fact_id"] == fact_id), None)
    page = fact["page"] if fact else 1

    left, right = st.columns([3, 2])
    with left:
        if d["page_count"] > 1 and fact is None:
            page = st.number_input("Page", 1, d["page_count"], 1, key=f"viewer_page_{claim_id}_{doc_id}")
        png, how = api.page_image(claim_id, doc_id, int(page), fact["fact_id"] if fact else None)
        if png is not None:
            st.image(png, caption=f"{d['filename']}, page {int(page)} of {d['page_count']}" + (f" · marked: {HOW[how]}" if fact else ""), width="stretch")
        else:
            p = next((p for p in d["pages"] if p["number"] == int(page)), None)
            st.caption(f"{d['filename']} has no page picture ({d['kind']}); the stored text of page {int(page)}:")
            st.text(p["text"] if p else "")
    with right:
        st.markdown(f"**{d['detected_type'].replace('_', ' ')}**  ·  {d['page_count']} page(s)  ·  uploaded by {d['uploaded_by'] or 'unknown'}" + (f" via {d['source_channel']}" if d["source_channel"] else ""))
        st.caption(f"Typed by {d['type_method'] or 'nobody yet'} with confidence {d['type_confidence']:.2f}: {d['type_reason'] or 'no reason recorded'}")
        if d["type_evidence_snippet"]:
            st.caption(f"Evidence for the type (page {d['type_page']}): “{d['type_evidence_snippet']}”")
        ocr = [p["number"] for p in d["pages"] if p["source"] == "ocr"]
        if ocr:
            st.warning(f"Page(s) {', '.join(map(str, ocr))} were read by OCR ({d['ocr_engine']}): the text is a transcription, not the original text.")
        if d["ocr_errors"] or d["load_error"]:
            st.error("; ".join([*d["ocr_errors"], *([d["load_error"]] if d["load_error"] else [])]))
        if fact:
            low = d["low_confidence_below"] is not None and fact["extraction_confidence"] < d["low_confidence_below"]
            st.markdown("**Source excerpt**")
            st.markdown(f"> {fact['evidence_text']}")
            st.caption(f"Page {fact['page']} · confidence {fact['extraction_confidence']:.2f}" + (" · ⚠ low confidence: please check it against the page" if low else "")
                       + f" · bears on {', '.join(fact['related_question_ids']) or 'no question'} · {fact['verification_status'].replace('_', ' ')}")
            st.write("Reads as: " + fact["proposition"])
    st.subheader("Fields extracted from this document")
    fields_table(d)
    st.subheader("What the agents did")
    agent_log(api, claim_id, d)
