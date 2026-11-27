"""Proposed next steps: what the system suggests the adjuster does about the gaps, and the drafts written for it. Nothing here is sent or done: a person reviews every draft, and a task is carried out only after the decision."""
from __future__ import annotations

import streamlit as st

MODE = {"draft": "draft for a person to send", "controlled_api": "carried out through the claims system after the decision"}


def rows(state: dict) -> list[dict]:
    """The live actions (cancelled ones are history), each with its draft when it has one."""
    drafts = {d["draft_id"]: d for d in state.get("drafts", [])}
    return [{"action": a, "draft": drafts.get(a.get("payload_ref") or "")} for a in state.get("proposed_actions", []) if a["status"] != "cancelled"]


def label(a: dict) -> str:
    return a["action_type"].replace("_", " ") + (f" · {a['responsible_party'].replace('_', ' ')}" if a["responsible_party"] and a["action_type"] == "create_task" else "")


def render(state: dict, title: str = "Proposed next steps") -> None:
    st.subheader(title)
    items = rows(state)
    if not items:
        st.caption("No next step is proposed.")
        return
    st.caption("Suggestions only. Nothing is sent or done until a person approves it.")
    for it in items:
        a, d = it["action"], it["draft"]
        mode = MODE.get(a["execution_mode"], a["execution_mode"].replace("_", " "))
        if d is None:
            st.markdown(f"**{label(a)}**  :gray-badge[{a['status']}]  \n<small>to {a['responsible_party'].replace('_', ' ')} · {mode}</small>", unsafe_allow_html=True)
            continue
        with st.expander(f"✉ {label(a)}: {d['subject']}"):
            st.caption(f"To {d['addressee_role'].replace('_', ' ')} · {d['status']} · {mode}")
            st.text(d["body"])
