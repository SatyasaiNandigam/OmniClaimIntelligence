"""The Overview tab: what needs a person's attention, in order, and where every question stands. It never decides anything: it points to the tab where the work is done."""
from __future__ import annotations

import streamlit as st

from claimtower.ui.components import actions, issue_matrix
from claimtower.ui.components.calc_trace import money
from claimtower.ui.components.header import settlement_figure

STATUS_ICON = {"established": "✅", "partially_established": "🟡", "possible": "🟠", "unverified": "⚪", "missing": "🔴", "requires_validation": "🟣"}
SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def attention(state: dict, status: dict, floor_facts: list[str] | None = None) -> list[tuple[str, str]]:
    """The things in the way of a decision, most serious first: [(level, text)], level one of error, warning, info."""
    out: list[tuple[str, str]] = []
    if status["phase"] == "halted":
        out.append(("error", f"An agent failed at **{status['paused_at']}**: {status['halted_reason']}. The claim is parked for a human."))
    serious = [c for c in state["contradictions"] if c["severity"] in ("high", "medium")]
    if serious:
        qs = sorted({c["question_id"] for c in serious})
        out.append(("warning", f"**{len(serious)} conflict(s)** between documents on {', '.join(qs)}: a person decides which to rely on (see *Questions & gaps*)."))
    for m in state["missing_evidence"]:
        r = m["minimum_evidence_request"]
        item = r["items"][0] if r["items"] else ""
        out.append(("info", f"**Ask {r['addressee_role'].replace('_', ' ')}** ({m['gap_type']}, {', '.join(m['resolves_question_ids'])}): {item[:170]}{'…' if len(item) > 170 else ''}"))
    s = state.get("settlement")
    if s is not None and s["status"] == "cannot_calculate":
        out.append(("warning", "The payable is **held**: " + (", ".join(x.replace("_", " ") for x in s["missing_inputs"]) or "inputs are not validated yet") + ". The forecast is not a payment."))
    qa = state.get("qa_report")
    if qa is not None and not qa["passed"]:
        out.append(("error", "A quality check failed: " + "; ".join(f["detail"] for f in qa["failures"][:2])))
    if floor_facts:
        out.append(("warning", f"{len(floor_facts)} fact(s) were read with low confidence and are unverified: check them in the *Document reader*."))
    return out


def tabs(state: dict, status: dict, coverage: bool = True) -> None:
    """Next steps, what needs attention and (optionally) the coverage issues, as tabs with the next steps first: the same arrangement on the Overview and on the Human decision page."""
    items = attention(state, status, (status.get("gate") or {}).get("low_confidence_facts"))
    names = ["Next steps", "Needs attention"] + (["Coverage issues"] if coverage else [])
    ts = st.tabs(names)
    with ts[0]:
        actions.render(state)
    with ts[1]:
        if not items:
            st.success("Nothing is in the way: the evidence is consistent and the calculation is complete.")
        for level, text in items:
            {"error": st.error, "warning": st.warning, "info": st.info}[level](text)
    if coverage:
        with ts[2]:
            if not state["coverage_issues"]:
                st.caption("No coverage issues yet.")
            for i in state["coverage_issues"]:
                icon, meaning = issue_matrix.STATUS.get(i["status"], ("", ""))
                st.markdown(f"{icon} **{i['issue']}**  " + chr(10) + f"<small>{meaning}</small>", unsafe_allow_html=True)


def render(v: dict, open_decision=None, reassess=None) -> None:
    state, status = v["state"], v["status"]
    left, right = st.columns([3, 2], gap="large")
    with left:
        if status["phase"] == "awaiting_human_decision":
            st.markdown("**Next step:** check the *Assessment* tab, then decide on the **Human decision** page.")
            if open_decision is not None and st.button("Go to the decision", type="primary", key="overview_to_decision"):
                open_decision()
        elif status["phase"] == "completed":
            st.success(f"Decided: {state['resolution_status'].replace('_', ' ')}.")
        elif status["phase"] == "halted" and reassess is not None and status.get("paused_at"):
            st.markdown("**Next step:** fix the cause above (for example add a missing policy document), then run the claim again from the step that failed.")
            if st.button(f"Re-assess from {status['paused_at']}", type="primary", key="overview_reassess"):
                reassess(status["paused_at"])
        tabs(state, status, coverage=False)
    with right:
        st.subheader("Where the questions stand")
        qs = state["open_questions"]
        if qs:
            done = sum(1 for q in qs if q["status"] == "established")
            st.progress(done / len(qs), text=f"{done} of {len(qs)} settled")
            for q in qs:
                st.markdown(f"{STATUS_ICON.get(q['status'], '')} **{q['question_id']}** {q['status'].replace('_', ' ')}  \n<small>{q['text']}</small>", unsafe_allow_html=True)
        else:
            st.caption("No questions yet.")
        r = state.get("routing")
        if r:
            st.subheader("Why this route")
            st.caption("Tier **" + r["tier"] + "** · " + ", ".join(c.replace("_", " ").lower() for c in r["reason_codes"]))
        s = state.get("settlement")
        if s is not None:
            name, value, note = settlement_figure(state)
            st.subheader("Settlement")
            st.markdown(f"**{name}:** {value}" + (f"  \n<small>assessed loss {money(s['assessed_loss'], s['currency'])}</small>" if s.get("assessed_loss") else ""), unsafe_allow_html=True)
