"""The counterfactual view (T7.4): "what would change if the open question were answered each way", for the adjuster to read, never a prediction of a decision.

For every open material question: three scenarios A, B and C (what a new piece of evidence might show), which coverage issues would have to be assessed again because of it and where each stands today, and
what to do next. A matrix below shows at a glance which scenario touches which issue. The scenario wording is checked elsewhere (it may not predict approval, denial or payment); this view says so too.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

STATUS_ICON = {"established": "✅", "identified": "🔎", "unresolved": "❔", "potentially_relevant": "🟠", "incomplete": "🟡", "abstain": "⏸"}


def rows(state: dict) -> list[dict]:
    issues = {i["issue_id"]: i for i in state["coverage_issues"]}
    return [{"question": s["question_id"], "scenario": s["scenario_id"], "if the evidence shows": s["assumed_finding"],
             "issues to re-assess": ", ".join(f"{STATUS_ICON.get(issues.get(i, {}).get('status', ''), '')} {i}" for i in s["affected_issue_ids"]) or "none",
             "what it would mean": s["implication_text"], "next step": s["next_step"]} for s in state["counterfactuals"]]


def matrix(state: dict) -> pd.DataFrame:
    """Issues down, scenarios across: 'to re-assess' where a scenario touches the issue. The current status is in the first column."""
    issues = [i for i in state["coverage_issues"] if any(i["issue_id"] in s["affected_issue_ids"] for s in state["counterfactuals"])]
    cols = [f"{s['question_id']} {s['scenario_id']}" for s in state["counterfactuals"]]
    data = [{"issue": i["issue"], "today": f"{STATUS_ICON.get(i['status'], '')} {i['status'].replace('_', ' ')}",
             **{c: ("re-assess" if i["issue_id"] in s["affected_issue_ids"] else "") for c, s in zip(cols, state["counterfactuals"])}} for i in issues]
    return pd.DataFrame(data, columns=["issue", "today", *cols]) if data else pd.DataFrame(columns=["issue", "today"])


def render(v: dict) -> None:
    state = v["state"]
    st.subheader("What would resolve this claim?")
    st.caption("Scenarios show what each possible answer to an open question would mean for the assessment. They never predict whether the claim is approved, declined or paid: a person decides.")
    if not state["counterfactuals"]:
        reason = ("No open question needs a scenario." if not state["missing_evidence"] else
                  "Scenarios are written when the claim is re-assessed after the first run; re-assess it (upload a document, or request more evidence) to see them.")
        st.info(reason)
        return
    r = rows(state)
    by_q: dict[str, list[dict]] = {}
    for x in r:
        by_q.setdefault(x["question"], []).append(x)
    for q, items in by_q.items():
        with st.container(border=True):
            st.markdown(f"**{q}**")
            cols = st.columns(len(items))
            for col, x in zip(cols, items):
                with col:
                    st.markdown(f"**Scenario {x['scenario']}**")
                    st.markdown(x["if the evidence shows"])
                    st.caption("Re-assess: " + x["issues to re-assess"])
                    st.write(x["what it would mean"])
                    st.caption("Next step: " + x["next step"])
    st.markdown("**Which scenario touches which issue**")
    st.dataframe(matrix(state), hide_index=True, width="stretch")
