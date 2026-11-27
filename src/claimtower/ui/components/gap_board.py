"""The gap board (T7.4): what stands between this claim and a decision, by TYPE of problem, so a reviewer sees at once which kind of work each open item needs.

Three kinds, shown in three separate columns with different icons and wording (never a verdict, never an accusation):
  conflicting   two sources disagree: both sides are shown next to each other with the document, page and verbatim words, and how serious the conflict is
  missing       nothing on file answers the question: what to ask for, from whom
  insufficient  something is on file but not enough to settle it: what is still needed
Below them: questions not yet judged (nothing has assessed them, so nothing is claimed) and the ones that are settled. Every request names who is asked; third parties are asked through the
claimant or an adjuster, never directly. The grouping is a function of the claim state (`group`), so it can be tested without a page.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

STATUS_ICON = {"established": "✅", "partially_established": "🟡", "possible": "🟠", "unverified": "⚪", "missing": "🔴", "requires_validation": "🟣"}
KINDS = {
    "contradictory": ("🟣", "Conflicting evidence", "Two sources disagree. A person decides which to rely on, or asks for more."),
    "missing": ("🔴", "Missing evidence", "Nothing on file answers this yet."),
    "insufficient": ("🟡", "Insufficient evidence", "Something is on file, but not enough to settle it."),
}
UNJUDGED = {"unverified", "possible"}


def group(state: dict) -> dict:
    """{kinds: {type: [card]}, unjudged: [question], settled: [question]} from the claim state. A card is a question with its requests and, for a conflict, its contradictions."""
    questions = {q["question_id"]: q for q in state["open_questions"]}
    facts = {f["fact_id"]: f for f in state["facts"]}
    docs = {d["document_id"]: d["filename"] for d in state["documents"]}

    def side(ids: list[str]) -> list[dict]:
        return [{"fact": i, "text": facts[i]["proposition"], "quote": facts[i]["evidence_text"], "where": f"{docs.get(facts[i]['source_document_id'], facts[i]['source_document_id'])}, page {facts[i]['page']}"}
                for i in ids if i in facts]

    kinds: dict[str, list[dict]] = {k: [] for k in KINDS}
    asked: set[str] = set()
    for m in state["missing_evidence"]:
        q = questions.get(m["question_id"], {"question_id": m["question_id"], "text": "", "status": "unverified", "linked_clause_ids": []})
        asked.add(m["question_id"])
        card = next((c for c in kinds[m["gap_type"]] if c["question"] == q["question_id"]), None)
        if card is None:
            card = {"question": q["question_id"], "text": q["text"], "status": q["status"], "clauses": q.get("linked_clause_ids", []), "requests": [], "contradictions": []}
            card["contradictions"] = [{"kind": c["kind"], "severity": c["severity"], "explanation": c["explanation"], "a": side(c["fact_ids_a"]), "b": side(c["fact_ids_b"])}
                                      for c in state["contradictions"] if c["question_id"] == q["question_id"]]
            kinds[m["gap_type"]].append(card)
        r = m["minimum_evidence_request"]
        card["requests"].append({"to": r["addressee_role"], "items": r["items"], "resolves": m["resolves_question_ids"]})
    rest = [q for q in state["open_questions"] if q["question_id"] not in asked]
    return {"kinds": kinds, "unjudged": [q for q in rest if q["status"] in UNJUDGED], "settled": [q for q in rest if q["status"] not in UNJUDGED]}


def _card(c: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**{c['question']}** · {STATUS_ICON.get(c['status'], '')} {c['status'].replace('_', ' ')}")
        if c["text"]:
            st.caption(c["text"])
        for x in c["contradictions"]:
            st.markdown(f"_{x['kind']} conflict, {x['severity']} severity:_ {x['explanation']}")
            a, b = st.columns(2)
            for col, label, items in ((a, "Side A", x["a"]), (b, "Side B", x["b"])):
                col.markdown(f"**{label}**")
                for it in items:
                    col.markdown(f"- {it['text']}  \n  “{it['quote']}” ({it['where']})")
        for r in c["requests"]:
            st.markdown(f"**Ask {r['to'].replace('_', ' ')}** (through the claimant or an adjuster):")
            st.markdown("\n".join(f"- {i}" for i in r["items"]))
        if c["clauses"]:
            st.caption("Bears on: " + ", ".join(c["clauses"]))


def render(v: dict) -> None:
    g = group(v["state"])
    kinds = g["kinds"]
    cols = st.columns(len(KINDS))
    for col, (key, (icon, title, hint)) in zip(cols, KINDS.items()):
        with col:
            st.markdown(f"### {icon} {title}")
            st.caption(f"{hint} ({len(kinds[key])})")
            if not kinds[key]:
                st.write("None.")
            for c in kinds[key]:
                _card(c)
    st.markdown("### ⚪ Not yet judged")
    st.caption("Nothing has assessed these, so nothing is claimed about them." if g["unjudged"] else "Nothing waiting to be judged.")
    if g["unjudged"]:
        st.dataframe(pd.DataFrame([{"": STATUS_ICON.get(q["status"], ""), "question": q["question_id"], "status": q["status"].replace("_", " "), "text": q["text"]} for q in g["unjudged"]]),
                     hide_index=True, width="stretch")
    st.markdown("### ✅ Settled")
    st.dataframe(pd.DataFrame([{"": STATUS_ICON.get(q["status"], ""), "question": q["question_id"], "status": q["status"].replace("_", " "), "evidence": f"{len(q['linked_fact_ids'])} fact(s)",
                                "clauses": ", ".join(q["linked_clause_ids"])} for q in g["settled"]]) if g["settled"] else pd.DataFrame(columns=["question"]), hide_index=True, width="stretch")
