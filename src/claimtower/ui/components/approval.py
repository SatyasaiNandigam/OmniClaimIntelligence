"""The approval form (T7.5): the human decision, posted to the API. Approve, edit, request more evidence, or reject, by a named reviewer, with a reason that is MANDATORY for anything but an approval.

An edit changes the settlement INPUTS only (never the payable): leave a box empty to keep the input as it is; the engine calculates again and the gate opens again. Every edit is recorded with the
reviewer, the role and the reason, and appears in the audit view (History tab). The form shows what the gate knows: the quality checks, facts read with low confidence, edits already in force and
the last refusal. Only the Submit button decides: Enter in a box never does. The API has the final say on roles (a refusal is shown as the API gave it).
"""
from __future__ import annotations

import streamlit as st

from claimtower.ui.client import ApiClient, ApiError

ROLES = ["adjuster", "senior_adjuster", "supervisor"]
INPUTS = {"assessed_covered_loss": "Assessed covered loss", "non_covered_items_total": "Non-covered items total", "applicable_deductible": "Applicable deductible",
          "water_damage_sublimit": "Water-damage sublimit", "remaining_policy_limit": "Remaining policy limit", "prior_payments_total": "Prior payments total"}
NEEDS_REASON = ("edit", "reject", "request_more_evidence")


def problem(decision: str, reason: str, edits: dict[str, str]) -> str | None:
    """Why the form may not be submitted yet, or None."""
    if decision in NEEDS_REASON and not reason.strip():
        return f"A '{decision.replace('_', ' ')}' decision needs a reason before it can be submitted: it is recorded with your name in the audit trail."
    if decision == "edit" and not edits:
        return "An 'edit' decision needs at least one settlement input to change."
    return None


def render(api: ApiClient, v: dict, show_error) -> None:
    status, state = v["status"], v["state"]
    gate = status.get("gate")
    k = v["claim_id"]                                                         # keys per claim: one claim's draft decision never leaks into another's
    st.subheader("Human decision")
    if status["phase"] != "awaiting_human_decision":
        st.info({"completed": "✔ decided", "halted": "⛔ halted (an agent failed)", "not_started": "not started"}.get(status["phase"], status["phase"])
                + (f" — {state['resolution_status'].replace('_', ' ')}" if status["phase"] == "completed" else ""))
        return
    st.caption("The system only recommends. Approving here pays or denies nothing.")
    qa = state.get("qa_report")
    if qa is not None and qa["passed"]:
        st.caption("✅ Quality checks passed before this reached you.")
    if gate.get("low_confidence_facts"):
        st.warning(f"{len(gate['low_confidence_facts'])} fact(s) were read with low confidence and are unverified: " + ", ".join(gate["low_confidence_facts"][:6]) + ". Check them in the Document reader.")
    if gate.get("overrides"):
        st.info("Edits already in force: " + ", ".join(f"{INPUTS.get(n, n)} = {x}" for n, x in gate["overrides"].items()))
    if gate.get("error"):
        st.error("The last decision was refused: " + gate["error"])
    col1, col2 = st.columns(2)
    reviewer = col1.text_input("Reviewer id", value="reviewer-1", key=f"dec_reviewer_{k}")
    role = col2.selectbox("Reviewer role", ROLES, index=ROLES.index(gate["required_approver_role"]) if gate["required_approver_role"] in ROLES else 0, key=f"dec_role_{k}")
    decision = st.radio("Decision", gate["allowed_decisions"], horizontal=True, format_func=lambda d: d.replace("_", " "), key=f"dec_decision_{k}")
    edits: dict[str, str] = {}
    if decision == "edit":
        st.caption("Change the inputs of the calculation. Leave a box empty to keep the input as it is. The engine calculates again; you never type the payable.")
        cols = st.columns(3)
        for n, (name, label) in enumerate(INPUTS.items()):
            val = cols[n % 3].text_input(label, key=f"dec_edit_{name}_{k}", placeholder="keep")
            if val.strip():
                edits[name] = val.strip().replace(",", "")
    reason = st.text_input("Reason (needed for edit, reject and request more evidence)", key=f"dec_reason_{k}")
    if st.button("Submit decision", key=f"dec_submit_{k}"):                  # only this button decides: Enter in a box does not
        why = problem(decision, reason, edits)
        if why:
            st.error(why)
            return
        try:
            api.decide(k, decision, reviewer, role, reason, edits if decision == "edit" else None)
            st.rerun()
        except ApiError as e:
            show_error(e)
