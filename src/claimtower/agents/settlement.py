"""Settlement agent (T5.5): extract -> compute -> explain. Owns exactly `settlement` (core.state.OWNERSHIP).

    extract   the inputs of the calculation, each with its source (agents/settlement_inputs.py): amounts from the facts (rules, plus the model's checked reading),
              the sublimit from the policy snapshot, the deductible and limit from the schedule, payments from the claims system. The model only copies and
              classifies amounts; it never computes and its amount must appear in the cited fact.
    compute   the deterministic engine (settlement/engine.py, no LLM imports) with the pack's versioned rules. An input that is not known makes the result
              `cannot_calculate` with the missing inputs listed; a number is never invented.
    explain   a text generated from the engine's trace (settlement/explain.py), optionally reworded by the model, and RE-VERIFIED: the displayed payable equals
              the engine's and no other amount appears. A reworded text that fails is dropped for the template; a template that fails is a bug (it raises).

Three separate figures (BUILD_PLAN 6.1): `assessed_loss` (the surveyor's amount), `payable` (the engine's, only when every input is known) and `reserve_forecast`
(a forecast for the reserve, set only when nothing but an optional cap is unavailable; it is never a payment). Output is HUMAN-REVIEWED. Pure: the same state
gives the same result, nothing is mutated.
"""
from __future__ import annotations

import re
from decimal import Decimal
from typing import Any, ClassVar

from pydantic import BaseModel

from claimtower.agents.settlement_inputs import (DeductibleReader, Extracted, LLMAmountReader, PaymentsSource, ScheduleSource, combine, deductible_candidates, rule_amounts,
                                                 sublimit_from_snapshot)
from claimtower.agents.settlement_inputs import _money_of
from claimtower.core.config import Pack
from claimtower.core.schemas import ClaimState, SettlementInputs, SettlementResult
from claimtower.core.state import OWNERSHIP, StatePatch
from claimtower.settlement.engine import SettlementCalculator, calculate_values
from claimtower.settlement.explain import explain, verify

LABELS = {"start": "Assessed covered loss", "less_non_covered": "Less non-covered items", "less_deductible": "Less deductible", "apply_sublimit": "Water-damage sublimit",
          "cap_remaining_limit": "Remaining policy limit", "less_prior_payments": "Less prior payments", "floor_zero": "Floor at zero"}
# The evidence for the repair cost was JUDGED and found lacking (or contested): no final amount. `unverified` (the judge did not run or abstained, e.g. rules-only mode) keeps the old behaviour.
NOT_VALIDATED = frozenset({"partially_established", "possible", "missing", "requires_validation"})


INPUT_OF = {"start": "assessed_covered_loss", "less_non_covered": "non_covered_items_total", "less_deductible": "applicable_deductible",
            "apply_sublimit": "water_damage_sublimit", "cap_remaining_limit": "remaining_policy_limit", "less_prior_payments": "prior_payments_total"}


class SettlementExplanationError(AssertionError):
    """The deterministic explanation failed its own verification: a bug, not a data problem."""


class Rewrite(BaseModel):
    text: str


class ExplanationRewriter:
    def __init__(self, llm: Any, pack: Pack) -> None:
        self._llm, self._pack = llm, pack

    def rewrite(self, text: str) -> str | None:
        from claimtower.core.llm import load_prompt

        try:
            return self._llm.structured(Rewrite, load_prompt("settlement_explanation", self._pack, explanation=text)).text
        except Exception:
            return None


class SettlementAgent:
    agent_id: ClassVar[str] = "settlement"
    owns: ClassVar[frozenset[str]] = OWNERSHIP["settlement"]

    def __init__(self, calculator: SettlementCalculator, rules_version: str, schedule: ScheduleSource | None = None, payments: PaymentsSource | None = None,
                 reader: LLMAmountReader | None = None, rewriter: ExplanationRewriter | None = None, overrides: Any | None = None,
                 deductible_reader: DeductibleReader | None = None) -> None:
        self.deductible_reader = deductible_reader
        self.calculator, self.rules_version, self.overrides = calculator, rules_version, overrides
        self.schedule, self.payments, self.reader, self.rewriter = schedule, payments, reader, rewriter
        self.last: dict[str, Any] = {}

    # ------------------------------------------------------------------ extract
    def extract(self, state: ClaimState) -> tuple[SettlementInputs, dict[str, Extracted], list[str]]:
        notes: list[str] = []
        qmap = {q.question_id: q for q in state.open_questions}
        ids = set(qmap["Q-AMOUNT"].linked_fact_ids) if "Q-AMOUNT" in qmap else set()
        facts = [f for f in state.facts if f.verification_status != "rejected" and (f.fact_id in ids or "Q-AMOUNT" in f.related_question_ids)]
        assessed, non_cov = rule_amounts(facts)
        if self.reader is not None and facts:
            m_a, m_n, refused = self.reader.read(facts)
            if refused:
                notes.append(f"model amount reading refused: {refused}")
            assessed, non_cov = combine(assessed, m_a, "assessed loss"), combine(non_cov, m_n, "non-covered items")
        if any((c.question_id == "Q-AMOUNT" or c.kind == "amount") and c.severity in ("medium", "high") for c in state.contradictions):          # a low-severity difference (an estimate above the assessment) is noted, not blocking
            assessed = Extracted(None, assessed.cites, assessed.how, "the amounts on file conflict (see the contradiction); not used", unavailable=True)
        if assessed.value is not None and non_cov.value is None and not non_cov.unavailable:
            non_cov = Extracted(Decimal(0), [], "assumed", "no fact states a non-covered item, so the total is 0 (assumption Q-S6)")
        parts: dict[str, Extracted] = {"assessed_covered_loss": assessed, "non_covered_items_total": non_cov,
                                       "water_damage_sublimit": sublimit_from_snapshot(state.policy_snapshot, state.open_questions)}
        h = state.claim_header
        sched = self.schedule.get_schedule(h.policy_ref, h.loss_date) if self.schedule and h and h.policy_ref and h.loss_date else None
        paid_here = self.payments.paid_on_claim(state.claim_id) if self.payments else None
        paid_policy = self.payments.paid_against_policy(h.policy_ref) if self.payments and h and h.policy_ref else None
        sch = "schedule of the policy " + (h.policy_ref if h and h.policy_ref else "")
        parts["applicable_deductible"] = Extracted(Decimal(sched["standard_deductible"]), [sch], "schedule") if sched and "standard_deductible" in sched else \
            Extracted(None, [], "schedule", "the deductible is not available from the policy system")
        parts["remaining_policy_limit"] = Extracted(Decimal(sched["building_limit"]) - (paid_policy - paid_here), [sch, "payments on other claims"], "schedule") \
            if sched and "building_limit" in sched and paid_policy is not None and paid_here is not None else Extracted(None, [], "schedule", "the limit is not available")
        parts["prior_payments_total"] = Extracted(paid_here, ["payments on this claim"], "payments") if paid_here is not None else \
            Extracted(None, [], "payments", "payments on this claim are not available")
        cands = deductible_candidates(state.policy_snapshot, h.loss_date) if h else []
        if cands:                                                               # an endorsement in force on the loss date amends the deductible: the schedule's figure is not used silently
            std = Decimal(sched["standard_deductible"]) if sched and "standard_deductible" in sched else None
            parts["applicable_deductible"] = self.deductible_reader.read(cands, h.loss_date, std) if self.deductible_reader is not None else \
                Extracted(None, [c.clause_id for c in cands], "clause", "an endorsement amends the deductible; choosing the one in force on the loss date needs the model", unavailable=True)
        rec = self.overrides.last(state.claim_id) if self.overrides is not None else None
        if rec is not None:                                                     # a reviewer's edit at the gate: the INPUT is replaced, the engine still computes
            for name, value in self.overrides.get(state.claim_id).items():
                parts[name] = Extracted(value, [f"human override by {rec.by}"], "human", f"set by {rec.by}: {rec.reason}", review=True)
        for name, e in parts.items():
            if e.note:
                notes.append(f"{name}: {e.note}")
        values = {n: e.value for n, e in parts.items()}
        inputs = SettlementInputs(**values, unavailable_inputs=sorted(n for n, e in parts.items() if e.unavailable),
                                  input_citations={n: e.cites for n, e in parts.items() if e.cites})
        return inputs, parts, notes

    # ------------------------------------------------------------------ compute + explain
    def settle(self, state: ClaimState) -> tuple[SettlementResult, dict[str, Extracted], list[str]]:
        inputs, parts, notes = self.extract(state)
        result = self.calculator.compute_settlement(inputs, self.rules_version)
        if result.status == "cannot_calculate":
            optional = {s.from_ for s in self.calculator.rules_for(self.rules_version).steps if s.optional and s.from_}
            if result.missing_inputs and set(result.missing_inputs) <= optional and result.assessed_loss is not None:
                values = {n: getattr(inputs, n) for n in INPUT_OF.values()}
                forecast = calculate_values(self.calculator.rules_for(self.rules_version), values)         # the optional cap treated as not applying: a forecast only
                if forecast.status == "calculated":
                    result = result.model_copy(update={"reserve_forecast": forecast.payable})
        amount_q = next((q for q in state.open_questions if q.question_id == "Q-AMOUNT"), None)
        if result.status == "calculated" and amount_q is not None and amount_q.status in NOT_VALIDATED:
            result = self._hold_for_validation(result, inputs, state)                    # the evidence was judged and the repair cost is not validated: no final amount, a forecast range instead
        text = self._cited_text(result, parts)
        if (bad := verify(text, result)):
            raise SettlementExplanationError("the generated explanation fails its own verification: " + "; ".join(bad))
        if self.rewriter is not None:
            reworded = self.rewriter.rewrite(text)
            if reworded and not verify(reworded, result):
                text = reworded
            elif reworded:
                notes.append("a model rewording of the explanation was dropped: " + "; ".join(verify(reworded, result)))
        return result.model_copy(update={"explanation": text}), parts, notes

    def _hold_for_validation(self, result: SettlementResult, inputs: SettlementInputs, state: ClaimState) -> SettlementResult:
        """No payable while the repair cost is unvalidated. The forecast range is the deterministic engine's own result at the assessed loss (low) and at the highest amount the
        pack names as claimed or estimated (high); nothing here is a model's number."""
        rules = self.calculator.rules_for(self.rules_version)
        low = result.payable
        high = low
        patterns = [re.compile(p) for p in rules.forecast_upper_sources]
        figures = [v for f in state.facts if f.verification_status != "rejected" for rx in patterns if (m := rx.match(f.proposition)) and (v := _money_of(m.group("value"))) is not None]
        if figures and inputs.assessed_covered_loss is not None and max(figures) > inputs.assessed_covered_loss:
            values = {n: getattr(inputs, n) for n in INPUT_OF.values()}
            values["assessed_covered_loss"] = max(figures)
            top = calculate_values(rules, values)
            if top.status == "calculated" and top.payable is not None and low is not None and top.payable >= low:
                high = top.payable
        return SettlementResult(status="cannot_calculate", assessed_loss=result.assessed_loss, payable=None, steps=[], missing_inputs=["validated_repair_cost"],
                                forecast_low=low, forecast_high=high, rule_version=result.rule_version, currency=result.currency)

    @staticmethod
    def _cited_text(result: SettlementResult, parts: dict[str, Extracted]) -> str:
        labels = {rid: LABELS.get(rid, rid) for rid in (s.rule_id for s in result.steps)}
        cites = {name: parts[name].cites for name in parts if parts[name].cites}
        # explain() looks cites up by the step's input name, which the trace carries as input_ref
        return explain(result, labels, None, cites)

    def run(self, state: ClaimState) -> StatePatch:
        result, parts, notes = self.settle(state)
        self.last = {"parts": parts, "notes": notes}
        bits = [f"{result.status}: " + (f"payable {result.payable:,} {result.currency} (assessed loss {result.assessed_loss:,})" if result.status == "calculated"
                                        else "missing " + ", ".join(result.missing_inputs))]
        if result.forecast_low is not None:
            bits.append(f"forecast range {result.forecast_low:,} to {result.forecast_high:,} (the repair cost is not validated; a forecast, never a payment)")
        if result.reserve_forecast is not None:
            bits.append(f"reserve forecast {result.reserve_forecast:,} (a forecast, never a payment)")
        bits.append("sources: " + ", ".join(f"{n.replace('_', ' ')}={e.how or 'n/a'}" for n, e in parts.items() if e.value is not None or e.unavailable))
        review = [n for n, e in parts.items() if e.review or e.how in ("model", "rules+model", "assumed")]
        if review:
            bits.append("REVIEW: " + ", ".join(f"{n} ({parts[n].how})" for n in review))
        if notes:
            bits.append("notes: " + " | ".join(notes))
        return StatePatch(agent_id=self.agent_id, updates={"settlement": result}, rationale="; ".join(bits))


def make_settlement_agent(pack: Pack, calculator: SettlementCalculator, rules_version: str, schedule: ScheduleSource | None, payments: PaymentsSource | None,
                          llm: Any | None = None, overrides: Any | None = None) -> SettlementAgent:
    return SettlementAgent(calculator, rules_version, schedule, payments, LLMAmountReader(llm, pack) if llm is not None else None,
                           ExplanationRewriter(llm, pack) if llm is not None else None, overrides, DeductibleReader(llm, pack) if llm is not None else None)


__all__ = ["ExplanationRewriter", "SettlementAgent", "SettlementExplanationError", "make_settlement_agent"]
