"""Settlement explanation (T5.5): text generated FROM the engine's trace, then re-verified against it. NO LLM imports (import-linter).

    text = explain(result, labels)               # deterministic: every amount in it is taken from the trace
    problems = verify(text, result)              # [] when the text may be shown

The verification is the guarantee, whoever wrote the text (this module's template, or a model asked to reword it): the amount shown as the payable equals the
engine's payable, and no other rupee amount appears unless it is an amount of the trace or the missing-input list. A `cannot_calculate` result never shows a payable.
A text that fails verification is not shown; the caller falls back to the template, and a failing template is a bug that must fail the build.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal

from claimtower.core.schemas import SettlementResult

_AMOUNT = re.compile(r"(?:INR|Rs\.?|₹)\s*([\d,]+(?:\.\d+)?)")
_PAYABLE = re.compile(r"payable[^.\n]{0,40}?(?:INR|Rs\.?|₹)\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)


def money(d: Decimal | None, currency: str = "INR") -> str:
    """INR 98,000 (whole rupees print without decimals; anything else keeps its decimals)."""
    if d is None:
        return "not available"
    q = d.quantize(Decimal(1)) if d == d.to_integral_value() else d
    return f"{currency} {q:,}"


def _amounts(text: str) -> list[Decimal]:
    return [Decimal(m.group(1).replace(",", "")) for m in _AMOUNT.finditer(text)]


def allowed_amounts(result: SettlementResult) -> set[Decimal]:
    out: set[Decimal] = set()
    for s in result.steps:
        out.update({s.amount_before, s.amount_after, abs(s.amount_after - s.amount_before)})
    for v in (result.assessed_loss, result.payable, result.reserve_forecast, result.forecast_low, result.forecast_high):
        if v is not None:
            out.add(v)
    return out


def explain(result: SettlementResult, labels: Mapping[str, str] | None = None, values: Mapping[str, Decimal | None] | None = None,
            cites: Mapping[str, list[str]] | None = None) -> str:
    """The plain-language explanation of a result. `labels`: rule id -> words; `values`: input name -> amount; `cites`: input name -> fact / clause ids."""
    labels, values, cites = labels or {}, values or {}, cites or {}
    cur = result.currency
    if result.status == "cannot_calculate":
        names = ", ".join(result.missing_inputs) or "an input"
        text = (f"cannot_calculate_final_settlement_yet: the settlement cannot be calculated until these are known: {names}. No payable amount is shown; "
                "none is estimated.")
        if result.forecast_low is not None and result.forecast_high is not None:
            if result.forecast_low == result.forecast_high:
                text += f" Forecast only (a reserve, not a payment): about {money(result.forecast_low, cur)}."
            else:
                text += f" Forecast only (a reserve, not a payment): between {money(result.forecast_low, cur)} and {money(result.forecast_high, cur)}."
            text += (f" The low figure is the engine's result at the assessed loss of {money(result.assessed_loss, cur)}; the high figure is its result at the highest amount claimed or estimated. "
                     "The repair cost has not been validated, so no final amount is calculated.")
        elif result.reserve_forecast is not None:
            text += (f" Forecast only (a reserve, not a payment): about {money(result.reserve_forecast, cur)} if the missing limit does not reduce it "
                     f"further. Assessed loss {money(result.assessed_loss, cur)}.")
        return text
    parts: list[str] = []
    for s in result.steps:
        if s.rule_id == "rounding":
            continue
        what = labels.get(s.rule_id, s.rule_id.replace("_", " "))
        ref = f" [{', '.join(cites[s.input_ref])}]" if s.input_ref and cites.get(s.input_ref) else ""
        if s.amount_before == s.amount_after and s.rule_id != "start":
            parts.append(f"{what}{ref}: no change ({money(s.amount_after, cur)})")
        else:
            parts.append(f"{what}{ref}: {money(s.amount_before, cur)} -> {money(s.amount_after, cur)}")
    return f"Payable: {money(result.payable, cur)} (assessed loss {money(result.assessed_loss, cur)}; rules {result.rule_version}). " + "; ".join(parts) + "."


def verify(text: str, result: SettlementResult) -> list[str]:
    """Why `text` must not be shown, or [] if it may. The displayed payable must equal the engine's; no amount may appear that the trace does not contain."""
    problems: list[str] = []
    shown = _PAYABLE.findall(text)
    if result.status == "cannot_calculate":
        if shown:
            problems.append("a payable is shown although the settlement cannot be calculated")
        if "cannot_calculate" not in text:
            problems.append("the text does not say the settlement cannot be calculated yet")
        missing = [m for m in result.missing_inputs if m not in text]
        if missing:
            problems.append(f"the missing inputs are not listed: {missing}")
        allowed = allowed_amounts(result)
        problems += [f"amount {a:,} is not in the engine output" for a in _amounts(text) if a not in allowed]
        return problems
    if not shown:
        problems.append("the text does not state the payable")
    for s in shown:
        if Decimal(s.replace(",", "")) != result.payable:
            problems.append(f"the displayed payable {s} differs from the engine's {result.payable}")
    allowed = allowed_amounts(result)
    problems += [f"amount {a:,} is not in the engine output" for a in _amounts(text) if a not in allowed]
    return problems


__all__ = ["allowed_amounts", "explain", "money", "verify"]
