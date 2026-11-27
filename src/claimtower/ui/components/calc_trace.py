"""The calculation trace (T7.5): the three figures kept apart, and every step of the calculation.

  assessed loss    what the surveyor assessed (the loss, before anything is taken off)
  payable          what the engine calculates once EVERY input is known (never shown when one is missing)
  reserve forecast an estimate for the reserve when only an optional cap is unknown: a forecast, never a payment
The trace lists each rule in order with the amount before and after it and the change, so the arithmetic can be checked by hand; the explanation below it is generated from the trace and re-verified
(its total equals the engine's). Inputs a reviewer edited at the gate are listed with the reason. A calculation that cannot be made says which inputs are missing and shows no number.
"""
from __future__ import annotations

from decimal import Decimal

import pandas as pd
import streamlit as st

LABEL = {"start": "Assessed covered loss", "less_non_covered": "Less non-covered items", "less_deductible": "Less deductible", "apply_sublimit": "Water-damage sublimit",
         "cap_remaining_limit": "Remaining policy limit", "less_prior_payments": "Less prior payments", "floor_zero": "Floor at zero", "rounding": "Rounding"}


def money(x: str | None, cur: str = "INR") -> str:
    if x is None:
        return "not available"
    d = Decimal(x)
    return f"{cur} {d.quantize(Decimal(1)) if d == d.to_integral_value() else d:,}"


def rows(settlement: dict) -> list[dict]:
    out = []
    for s in settlement["steps"]:
        before, after = Decimal(s["amount_before"]), Decimal(s["amount_after"])
        out.append({"step": LABEL.get(s["rule_id"], s["rule_id"].replace("_", " ")), "input": (s["input_ref"] or "").replace("_", " "), "before": money(s["amount_before"], settlement["currency"]),
                    "after": money(s["amount_after"], settlement["currency"]),
                    "change": "no change" if before == after else f"{'-' if after < before else '+'}{abs(after - before):,.0f}"})
    return out


def render(v: dict, overrides: dict[str, str] | None = None) -> None:
    s = v["state"].get("settlement")
    st.subheader("Calculation")
    if s is None:
        st.info("No settlement was calculated" + (" (no policy is in force)." if v["state"].get("policy_snapshot") is None else " yet."))
        return
    cur = s["currency"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Assessed loss", money(s["assessed_loss"], cur))
    c2.metric("Payable", money(s["payable"], cur) if s["status"] == "calculated" else "not calculated")
    if s.get("forecast_low") is not None:
        low, high = s["forecast_low"], s["forecast_high"]
        c3.metric("Forecast range", money(low, cur) if low == high else f"{money(low, cur)} to {money(high, cur)}")
    else:
        c3.metric("Reserve forecast", money(s["reserve_forecast"], cur) if s["reserve_forecast"] is not None else "none")
    st.caption("Three different things: the loss as assessed, the amount payable under the rules (only when every input is known), and a reserve forecast (an estimate, never a payment). "
               f"Rules {s['rule_version']}.")
    if s["status"] == "cannot_calculate":
        st.warning("The settlement cannot be calculated yet. Missing: " + ", ".join(m.replace("_", " ") for m in s["missing_inputs"]) + ". No amount is shown and none is estimated as a payment.")
    if overrides:
        st.info("Edited by a reviewer at the gate (the engine still calculates): " + "; ".join(f"{k.replace('_', ' ')} = {money(x, cur)}" for k, x in overrides.items()))
    if s["steps"]:
        st.dataframe(pd.DataFrame(rows(s)), hide_index=True, width="stretch")
    if s.get("explanation"):
        st.markdown("**In words**")
        st.write(s["explanation"])
    qa = v["state"].get("qa_report")
    if qa is not None:
        st.caption("Quality checks: " + ("passed (citations valid, the explanation matches the engine's total)." if qa["passed"] else "FAILED: " + "; ".join(f["detail"] for f in qa["failures"][:3])))
