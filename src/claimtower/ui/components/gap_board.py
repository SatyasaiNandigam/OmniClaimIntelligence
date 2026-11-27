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


SEVERITY = {"high": ("red", 0), "medium": ("orange", 1), "low": ("gray", 2)}
KIND_BADGE = {"contradictory": "violet", "missing": "red", "insufficient": "orange"}
VIEWS = ["Needs attention", "Not yet judged", "Settled", "All"]


def _conflict(x: dict) -> None:
    st.markdown(f"_{x['kind']} conflict, {x['severity']} severity:_ {x['explanation']}")
    a, b = st.columns(2)
    for col, label, items in ((a, "Side A", x["a"]), (b, "Side B", x["b"])):
        col.markdown(f"**{label}**")
        for it in items:
            col.markdown(f"- {it['text']}  " + chr(10) + f"  “{it['quote']}” ({it['where']})")


def _evidence(state: dict, question_id: str) -> None:
    q = next((q for q in state["open_questions"] if q["question_id"] == question_id), None)
    facts = {f["fact_id"]: f for f in state["facts"]}
    docs = {d["document_id"]: d["filename"] for d in state["documents"]}
    linked = [facts[i] for i in (q["linked_fact_ids"] if q else []) if i in facts]
    if not linked:
        return
    with st.expander(f"Evidence behind this question ({len(linked)} facts)"):
        st.dataframe(pd.DataFrame([{"document": docs.get(f["source_document_id"], f["source_document_id"]), "page": f["page"], "fact": f["proposition"], "verbatim": f["evidence_text"]} for f in linked]),
                     hide_index=True, width="stretch")


def _card(c: dict, kind: str, state: dict) -> None:
    icon, title, _ = KINDS[kind]
    with st.container(border=True):
        h1, h2 = st.columns([4, 2], vertical_alignment="center")
        h1.markdown(f"**{c['question']}** · {STATUS_ICON.get(c['status'], '')} {c['status'].replace('_', ' ')}")
        h2.markdown(f":{KIND_BADGE[kind]}-badge[{icon} {title}]")
        if c["text"]:
            st.caption(c["text"])
        for r in c["requests"]:
            with st.container(border=True):
                st.markdown(f"**Ask {r['to'].replace('_', ' ')}** (through the claimant or an adjuster):")
                st.markdown(chr(10).join(f"- {i}" for i in r["items"]))
        ordered = sorted(c["contradictions"], key=lambda x: SEVERITY[x["severity"]][1])
        major, minor = [x for x in ordered if x["severity"] != "low"], [x for x in ordered if x["severity"] == "low"]
        for x in major:
            with st.expander(f":{SEVERITY[x['severity']][0]}-badge[{x['severity']}] {x['kind']} conflict: {x['explanation'][:100]}{'…' if len(x['explanation']) > 100 else ''}"):
                _conflict(x)
        if minor:
            with st.expander(f"{len(minor)} minor difference(s)"):
                for n, x in enumerate(minor):
                    if n:
                        st.divider()
                    _conflict(x)
        _evidence(state, c["question"])
        if c["clauses"]:
            st.caption("Bears on: " + ", ".join(c["clauses"]))


def _plain_card(q: dict, state: dict, badge: str) -> None:
    """A question that is not waiting for evidence (settled, or not judged yet): the same card as the others, with what establishes it and the facts that prove it."""
    facts = {f["fact_id"]: f for f in state["facts"]}
    docs = {facts[i]["source_document_id"] for i in q["linked_fact_ids"] if i in facts}
    with st.container(border=True):
        h1, h2 = st.columns([4, 2], vertical_alignment="center")
        h1.markdown(f"**{q['question_id']}** · {STATUS_ICON.get(q['status'], '')} {q['status'].replace('_', ' ')}")
        h2.markdown(badge)
        if q["text"]:
            st.caption(q["text"])
        if q.get("establishing_criteria"):
            st.markdown("**What establishes it**")
            st.markdown(chr(10).join(f"- {c}" for c in q["establishing_criteria"]))
        n = len([i for i in q["linked_fact_ids"] if i in facts])
        st.caption(f"Rests on {n} fact(s) from {len(docs)} document(s)." if n else "No fact is linked to this question.")
        _evidence(state, q["question_id"])
        if q["linked_clause_ids"]:
            st.caption("Bears on: " + ", ".join(q["linked_clause_ids"]))


def render(v: dict) -> None:
    state, k = v["state"], v["claim_id"]
    g = group(state)
    kinds = g["kinds"]
    counts = [(f"{icon} {title}", len(kinds[key])) for key, (icon, title, _) in KINDS.items()] + [("⚪ Not yet judged", len(g["unjudged"])), ("✅ Settled", len(g["settled"]))]
    for col, (label, n) in zip(st.columns(len(counts)), counts):
        with col, st.container(border=True):
            st.metric(label, n)
    view = st.radio("Show", VIEWS, horizontal=True, key=f"gaps_view_{k}", label_visibility="collapsed")
    if view in ("Needs attention", "All"):
        cards = [(c, key) for key in KINDS for c in kinds[key]]
        if not cards:
            st.success("No question is waiting for evidence.")
        for c, key in cards:
            _card(c, key, state)
    if view in ("Not yet judged", "All"):
        st.markdown("### ⚪ Not yet judged")
        st.caption("Nothing has assessed these, so nothing is claimed about them." if g["unjudged"] else "Nothing waiting to be judged.")
        for q in g["unjudged"]:
            _plain_card(q, state, ":gray-badge[⚪ Not yet judged]")
    if view in ("Settled", "All"):
        st.markdown("### ✅ Settled")
        st.caption("Each one shows what establishes it and the facts, with their verbatim words, that prove it." if g["settled"] else "Nothing is settled yet.")
        for q in g["settled"]:
            _plain_card(q, state, ":green-badge[✅ Settled]" if q["status"] == "established" else f":blue-badge[{STATUS_ICON.get(q['status'], '')} {q['status'].replace('_', ' ')}]")
