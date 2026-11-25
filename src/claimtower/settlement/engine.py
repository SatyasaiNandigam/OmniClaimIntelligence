"""Deterministic settlement engine (T5.3). NO LLM IMPORTS (import-linter), Decimal only, never float.

The LLM extracts and cites inputs; this module computes. It interprets the ordered, versioned step list from a rule file
(packs/*/calculation/*.yaml) so formula order is configuration, not code.

Behaviour contract
  * identical inputs + identical rules -> identical output (pure function, no clock, no randomness, inputs never mutated)
  * a step input that is None - or listed in `unavailable_inputs` even when the step is `optional` - makes the result
    `cannot_calculate` with the sorted list of missing inputs; a number is never fabricated
  * negative / float / non-finite amounts raise InvalidAmount (a data error, not something to round away)
  * the trace has one entry per rule plus a final 'rounding' entry. `input_ref=None` on a cap means the optional step did not apply
  * the engine knows nothing about policy status: callers must not ask for a settlement on a policy that is not in force
    (S-03); a cancelled policy whose limit cannot be determined arrives as an unavailable `remaining_policy_limit`
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal, localcontext

from claimtower.core.config import SETTLEMENT_INPUTS, CalculationRules
from claimtower.core.errors import ConfigError
from claimtower.core.schemas import SettlementInputs, SettlementResult, SettlementStepTrace
from claimtower.settlement.money import PRECISION, InvalidAmount, non_negative, quantize

__all__ = ["SettlementCalculator", "InvalidAmount", "calculate", "calculate_values"]


def calculate_values(rules: CalculationRules, values: Mapping[str, Decimal | None], unavailable: Iterable[str] = ()) -> SettlementResult:
    """Generic core: `values` maps input name -> amount (None = not known). Any input name the rules reference may be used."""
    unavailable = set(unavailable)
    checked: dict[str, Decimal | None] = {name: None if v is None else non_negative(v, name) for name, v in values.items()}

    missing = sorted({
        s.from_ for s in rules.steps
        if s.from_ and (s.from_ in unavailable or (checked.get(s.from_) is None and not s.optional))
    })
    if missing:
        return SettlementResult(status="cannot_calculate", missing_inputs=missing, rule_version=rules.version,
                                currency=rules.currency, assessed_loss=checked.get("assessed_covered_loss"))

    amount = Decimal(0)
    steps: list[SettlementStepTrace] = []
    with localcontext() as ctx:
        ctx.prec = PRECISION
        for step in rules.steps:
            before, ref = amount, step.from_
            val = checked.get(ref) if ref else None
            if step.op == "set":
                amount = val  # type: ignore[assignment]
            elif step.op == "subtract":
                amount = amount - val  # type: ignore[operator]
            elif step.op == "cap":
                if val is None:                     # optional step with no value: does not apply
                    ref = None
                else:
                    amount = min(amount, val)
            else:                                   # floor (validated to be 0)
                amount = max(amount, Decimal(str(step.value)))
            steps.append(SettlementStepTrace(rule_id=step.id, input_ref=ref, amount_before=before, amount_after=amount))
        payable = quantize(amount, rules.rounding["places"])
        steps.append(SettlementStepTrace(rule_id="rounding", input_ref=None, amount_before=amount, amount_after=payable))
    return SettlementResult(status="calculated", assessed_loss=checked.get("assessed_covered_loss"), payable=payable, steps=steps,
                            rule_version=rules.version, currency=rules.currency)


def calculate(rules: CalculationRules, inputs: SettlementInputs) -> SettlementResult:
    """Typed entry point: SettlementInputs (what the Settlement agent extracted) -> SettlementResult."""
    values = {name: getattr(inputs, name) for name in SETTLEMENT_INPUTS}
    return calculate_values(rules, values, inputs.unavailable_inputs)


class SettlementCalculator:
    """Implements the frozen `SettlementEngine` protocol (core/interfaces.py): compute_settlement(inputs, rules_version)."""

    def __init__(self, rules_by_version: Mapping[str, CalculationRules]) -> None:
        self._rules = dict(rules_by_version)

    @property
    def versions(self) -> list[str]:
        return sorted(self._rules)

    def compute_settlement(self, inputs: SettlementInputs, rules_version: str) -> SettlementResult:
        try:
            rules = self._rules[rules_version]
        except KeyError:
            raise ConfigError(f"unknown settlement rules version '{rules_version}'; available: {self.versions}") from None
        return calculate(rules, inputs)
