"""The claim banner: who and what the claim is, where it stands, and the figures a reviewer looks at first. It sits above the workspace tabs, so it stays on screen whichever tab is open."""
from __future__ import annotations

import streamlit as st

from claimtower.ui.components.calc_trace import money

PHASE_BADGE = {"awaiting_human_decision": ("orange", "Awaiting human decision"), "completed": ("green", "Decided"), "halted": ("red", "Halted"), "not_started": ("gray", "Not started")}


def settlement_figure(state: dict) -> tuple[str, str, str]:
    """(label, value, note) for the money shown first: the payable when it is calculated, else the forecast range, else nothing."""
    s = state.get("settlement")
    if s is None:
        return "Payable", "–", "no settlement yet"
    cur = s["currency"]
    if s["status"] == "calculated":
        return "Payable", money(s["payable"], cur), "calculated by the rules"
    if s.get("forecast_low") is not None:
        low, high = s["forecast_low"], s["forecast_high"]
        return "Forecast range", money(low, cur) if low == high else f"{money(low, cur)} to {money(high, cur)}", "held until the amounts are validated"
    return "Payable", "not calculated", ", ".join(m.replace("_", " ") for m in s["missing_inputs"]) or "inputs missing"


def render(v: dict) -> None:
    state, status = v["state"], v["status"]
    h = state.get("claim_header") or {}
    color, label = PHASE_BADGE.get(status["phase"], ("gray", status["phase"]))
    bits = [h.get("claimant_name"), f"policy {h['policy_ref']}" if h.get("policy_ref") else None, f"loss on {h['loss_date']}" if h.get("loss_date") else None,
            (h.get("loss_type") or "").replace("_", " ") or None]
    st.markdown(f"**{v['claim_id']}**  :{color}-badge[{label}]  &nbsp;<small>{'  ·  '.join(b for b in bits if b)}</small>", unsafe_allow_html=True)
    if h.get("loss_description"):
        d = h["loss_description"]
        st.caption(d if len(d) <= 160 else d[:157] + "…", help=d if len(d) > 160 else None)

    r = state.get("routing")
    questions = state["open_questions"]
    settled = sum(1 for q in questions if q["status"] == "established")
    figure, value, note = settlement_figure(state)
    cells = [(figure, value, note), ("Tier", r["tier"] if r else "–", None), ("Required approver", r["required_approver_role"].replace("_", " ") if r else "–", None),
             ("Questions settled", f"{settled} of {len(questions)}", None), ("Documents", str(len(state["documents"])), None), ("Version", str(v["state_version"]), None)]
    for col, (name, val, help_text) in zip(st.columns([2.2, 1, 1.3, 1.2, 1, 0.8]), cells):
        with col:
            st.metric(name, val, help=help_text)
    st.divider()
