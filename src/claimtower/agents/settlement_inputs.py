"""Where the settlement inputs come from (T5.5). The model extracts and classifies the amounts a document states; it never computes. Checked, never trusted.

    assessed_covered_loss, non_covered_items_total   from the facts (the surveyor's assessment; items stated as not covered). Rules read the amount out of the
                                                     fact's own one-line statement (its evidence line may carry several amounts); the model proposes the same with cited fact ids and a verbatim quote. Both must agree
                                                     (a disagreement makes the input UNAVAILABLE, never a guess); a model-only value is accepted only when its
                                                     amount appears in the cited fact's own text, and is flagged for review.
    water_damage_sublimit                            from the sublimit clause of the policy snapshot (an exact amount in the clause); the Gap agent's Q-SUBLIMIT
                                                     `missing` means the wording is not available -> unavailable, not zero and not absent
    applicable_deductible, remaining_policy_limit    from the policy schedule held by the policy system (building limit less payments on other claims)
    prior_payments_total                             payments already made on THIS claim (the claims system)
A contractor's or claimant's estimate alone is not an assessment: with no assessed amount the input is missing. When an assessed amount exists and no fact states a
non-covered item, the total is 0 (an assumption stated in the explanation, docs/assumptions.md Q-S6).
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from pydantic import BaseModel

from claimtower.core.config import Pack
from claimtower.core.schemas import Fact, OpenQuestion, PolicySnapshot

AMOUNT = re.compile(r"(?:INR|Rs\.?|₹)\s*([\d,]+(?:\.\d+)?)")
NON_COVERED_WORDS = ("upgrade", "not caused by the loss", "not covered", "betterment", "non-covered", "non covered")
PRICE_FACT = re.compile(r"-(?:item-\d+|total_amount)$")           # a line item or a total of an estimate or invoice is a PRICE, never a statement about what is covered


class ScheduleSource(Protocol):
    def get_schedule(self, policy_no: str, as_of: Any) -> dict[str, str] | None: ...


class PaymentsSource(Protocol):
    def paid_on_claim(self, claim_id: str) -> Decimal: ...
    def paid_against_policy(self, policy_no: str) -> Decimal: ...


def parse_amount(text: str) -> list[Decimal]:
    return [Decimal(m.group(1).replace(",", "")) for m in AMOUNT.finditer(text)]


def _norm(s: str) -> str:
    return " ".join(s.split())


@dataclass
class Extracted:
    """One input: its value (None = not known), the ids it was taken from, and how."""

    value: Decimal | None = None
    cites: list[str] = field(default_factory=list)
    how: str = ""                                   # rules | model | rules+model | assumed | schedule | clause | payments
    note: str = ""
    unavailable: bool = False                       # applies to this claim but cannot be determined (conflict, wording missing)
    review: bool = False


class AmountItem(BaseModel):
    amount: str
    fact_id: str
    quote: str


class AmountsAnswer(BaseModel):
    assessed: AmountItem | None = None
    non_covered: list[AmountItem] = []
    note: str = ""


# ------------------------------------------------------------------------------------------------------------- rules
def rule_amounts(facts: Sequence[Fact]) -> tuple[Extracted, Extracted]:
    """(assessed loss, non-covered items) read from the verbatim text of the facts. No model."""
    assessed: dict[Decimal, list[str]] = {}
    items: dict[str, tuple[Decimal, str]] = {}
    for f in facts:
        text = f.proposition.lower()                           # the proposition is the fact's own one-line claim; the evidence line may carry several amounts
        amts = parse_amount(f.proposition) or parse_amount(f.evidence_text)
        if len(amts) != 1:
            continue
        if any(w in text for w in NON_COVERED_WORDS):
            items[f.fact_id] = (amts[0], f.fact_id)
        elif "assessed" in text:
            assessed.setdefault(amts[0], []).append(f.fact_id)
    a = Extracted()
    if len(assessed) == 1:
        (v, ids), = assessed.items()
        a = Extracted(v, ids, "rules")
    elif len(assessed) > 1:
        a = Extracted(None, [i for ids in assessed.values() for i in ids], "rules", "the assessed amounts differ: " + ", ".join(f"{v:,}" for v in assessed), unavailable=True)
    n = Extracted()
    if items:
        n = Extracted(sum((v for v, _ in items.values()), Decimal(0)), [i for _, i in items.values()], "rules")
    return a, n


def _money_of(s: str) -> Decimal | None:
    try:
        d = Decimal(s.replace(",", "").replace("INR", "").replace("Rs.", "").replace("₹", "").strip())
    except InvalidOperation:
        return None
    return d if d.is_finite() and d >= 0 else None


class LLMAmountReader:
    """The model's proposal of the two amounts, with its own checks (a cited, offered fact whose text contains the amount and the quote)."""

    def __init__(self, llm: Any, pack: Pack) -> None:
        self._llm, self._pack = llm, pack

    def read(self, facts: Sequence[Fact]) -> tuple[Extracted, Extracted, str | None]:
        from claimtower.core.llm import load_prompt

        block = "\n".join(f"- {f.fact_id} ({f.source_document_id}): {_norm(f.proposition)} | evidence: {_norm(f.evidence_text)}" for f in facts)
        try:
            ans = self._llm.structured(AmountsAnswer, load_prompt("settlement_inputs", self._pack, facts_block=block))
        except Exception as e:                              # model down, bad JSON
            return Extracted(), Extracted(), f"{type(e).__name__}: {e}"
        return self.check(ans, facts)

    @staticmethod
    def check(ans: AmountsAnswer, facts: Sequence[Fact]) -> tuple[Extracted, Extracted, str | None]:
        by_id = {f.fact_id: f for f in facts}
        refused: list[str] = []
        assessed_ids = {ans.assessed.fact_id} if ans.assessed is not None else set()

        def ok(item: AmountItem, must_say_excluded: bool = False) -> Decimal | None:
            f = by_id.get(item.fact_id)
            v = _money_of(item.amount)
            if f is None or v is None:
                refused.append(f"{item.fact_id}: not an offered fact or not an amount")
                return None
            text = _norm(f"{f.proposition} {f.evidence_text}")
            if v not in parse_amount(text) or len(_norm(item.quote)) < 4 or _norm(item.quote) not in text:
                refused.append(f"{item.fact_id}: the amount or quote is not in the fact's own text")
                return None
            if must_say_excluded and PRICE_FACT.search(f.fact_id):                                  # a priced line of an estimate or invoice is not 'non-covered' because a model says so
                refused.append(f"{item.fact_id}: a line item or total of an estimate or invoice is a price, not a statement that the amount is not covered")
                return None
            if must_say_excluded and (f.fact_id in assessed_ids or "assessed" in f.proposition.lower()):    # the fact that STATES the assessment cannot also be the source of a deduction from it:
                refused.append(f"{item.fact_id}: it states the assessed loss; an amount it says is 'not included' is already left out of that loss, not a deduction from it")
                return None
            return v

        a = Extracted()
        if ans.assessed is not None and (v := ok(ans.assessed)) is not None:
            a = Extracted(v, [ans.assessed.fact_id], "model")
        n = Extracted()
        vals = [(ok(i, must_say_excluded=True), i.fact_id) for i in ans.non_covered]
        good = [(v, i) for v, i in vals if v is not None]
        if good:
            n = Extracted(sum((v for v, _ in good), Decimal(0)), [i for _, i in good], "model")
            if a.value is not None and n.value is not None and n.value > a.value:              # what is left out of the loss can never exceed the assessed loss
                refused.append(f"the non-covered amounts ({n.value:,}) exceed the assessed loss ({a.value:,}): not used")
                n = Extracted()
        return a, n, "; ".join(refused) or None


def combine(rules: Extracted, model: Extracted | None, name: str) -> Extracted:
    """Rules and the model's reading of the same input: agree -> that; one only -> it (model-only flagged for review); disagree -> unavailable."""
    if model is None or (model.value is None and not model.unavailable):
        return rules
    if rules.unavailable:
        return rules
    if rules.value is None:
        return Extracted(model.value, model.cites, "model", f"{name}: found by the model only; check it against the document", review=True)
    if model.value is None:
        return rules
    if rules.value == model.value:
        return Extracted(rules.value, sorted(set(rules.cites) | set(model.cites)), "rules+model")
    return Extracted(None, sorted(set(rules.cites) | set(model.cites)), "rules+model", f"{name}: rules read {rules.value:,}, the model read {model.value:,}; not used",
                     unavailable=True)


# ------------------------------------------------------------------------------------------------------------- policy-side inputs
def sublimit_from_snapshot(snapshot: PolicySnapshot | None, questions: Sequence[OpenQuestion]) -> Extracted:
    q = next((q for q in questions if q.question_id == "Q-SUBLIMIT"), None)
    if q is not None and q.status == "missing":
        return Extracted(None, list(q.linked_clause_ids), "clause", "the sublimit wording is listed on the schedule but was not provided", unavailable=True)
    for c in snapshot.clauses if snapshot else []:
        if c.clause_type == "sublimit":
            amts = parse_amount(c.text)
            if len(amts) == 1:
                return Extracted(amts[0], [c.clause_id], "clause")
            return Extracted(None, [c.clause_id], "clause", "the sublimit clause does not state one amount", unavailable=True)
    return Extracted(None, [], "clause", "no sublimit applies to this policy")


# ------------------------------------------------------------------------------------------------------------- the deductible of an endorsement
class DeductibleAnswer(BaseModel):
    amount: str = ""
    clause_id: str = ""
    quote: str = ""
    note: str = ""


def deductible_candidates(snapshot: PolicySnapshot | None, loss_date: Any) -> list[Any]:
    """Deductible clauses of the endorsements in force on the loss date (an endorsement can amend the deductible of the wording or the schedule)."""
    if snapshot is None or loss_date is None:
        return []
    in_force = [e for e in snapshot.endorsements if e.effective_from <= loss_date and (e.effective_to is None or e.effective_to >= loss_date)]
    ids = {cid for e in in_force for cid in e.clause_ids}
    return [c for c in snapshot.clauses if c.clause_type == "deductible" and c.clause_id in ids]


class DeductibleReader:
    """Which deductible applies on the loss date when an endorsement amends it. The model reads the clauses and picks one; the pick is checked, never trusted: the clause must be one
    offered, its quote verbatim, its amount inside the quote. A refusal makes the input UNAVAILABLE (the schedule's figure is never used silently when an endorsement amends it)."""

    def __init__(self, llm: Any, pack: Pack) -> None:
        self._llm, self._pack = llm, pack

    def read(self, candidates: Sequence[Any], loss_date: Any, standard: Decimal | None) -> Extracted:
        from claimtower.core.llm import load_prompt

        block = "\n".join(f"- {c.clause_id}: {_norm(c.heading)} {_norm(c.text)}" for c in candidates)
        try:
            ans = self._llm.structured(DeductibleAnswer, load_prompt("deductible_choice", self._pack, loss_date=str(loss_date), standard=f"{standard:,}" if standard is not None else "not stated", clauses=block))
        except Exception as e:
            return Extracted(None, [c.clause_id for c in candidates], "model", f"an endorsement amends the deductible and the model could not read it ({type(e).__name__}: {e})", unavailable=True)
        return self.check(ans, candidates)

    @staticmethod
    def check(ans: DeductibleAnswer, candidates: Sequence[Any]) -> Extracted:
        ids = [c.clause_id for c in candidates]
        c = next((x for x in candidates if x.clause_id == ans.clause_id), None)
        v = _money_of(ans.amount) if ans.amount else None
        quote = _norm(ans.quote)
        if c is None or v is None or len(quote) < 4 or quote not in _norm(f"{c.heading} {c.text}") or v not in parse_amount(quote):
            return Extracted(None, ids, "model", "an endorsement amends the deductible but the model's choice could not be checked against the clause text" + (f" ({ans.note})" if ans.note else ""), unavailable=True)
        return Extracted(v, [c.clause_id], "model", f"the deductible in force on the loss date is taken from {c.clause_id}", review=True)

__all__ = ["AmountsAnswer", "DeductibleAnswer", "DeductibleReader", "Extracted", "LLMAmountReader", "deductible_candidates", "PaymentsSource", "ScheduleSource", "combine", "rule_amounts", "sublimit_from_snapshot"]
