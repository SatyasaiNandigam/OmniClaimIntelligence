"""Contradiction detector (T4.3): facts that cannot all be true, each conflict citing the facts on both sides.

    detector = ContradictionDetector(pack.question_templates.contradictions, judge=LLMConflictJudge(llm, pack))      # judge optional
    result = detector.detect(facts, questions, links)           # questions: T4.1 register; links: T4.2 (else the facts' own tags are used)
    result.contradictions                                       # list[Contradiction] (contract): question, both sides, kind, severity, explanation

Two sources, one list:
  * RULES (config: `contradictions:` in question_templates.yaml) compare what is stated as a date or an amount, by the wording of the facts:
      date rule    facts stating the same date (the date of loss) with different values
      order rule   a date that cannot come before another (an invoice, a surveyor visit, a plumber inspection dated before the loss)
      amount rule  two amounts that should agree (claimed vs assessed, estimate vs assessed, invoice vs estimate); the gap sets the severity
    A value that cannot be read is not compared: a rule never guesses.
  * MODEL (llm_contradiction.py) looks at the facts linked to each open question of the configured categories (cause, timing, extent) and
    proposes narrative conflicts ("sudden burst" against "long-term corrosion"). Each is checked: two disjoint sides of real facts of that
    question, a verbatim quote from each side, an explanation, a confidence (capped), an allowed kind and severity.
No contradiction is ever returned without at least one cited fact on each side (the contract requires it and the detector enforces it).
If the model is down the rule contradictions are still returned, with a warning that narrative conflicts were not checked.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal, Protocol

from claimtower.core.config import AmountRule, ContradictionConfig, DateRule, OrderRule
from claimtower.core.schemas import Contradiction, Fact, OpenQuestion
from claimtower.evidence.values import parse_amount, parse_date

Kind = Literal["date", "amount", "cause", "extent", "other"]
KINDS: tuple[str, ...] = ("date", "amount", "cause", "extent", "other")
SEVERITIES: tuple[str, ...] = ("low", "medium", "high")


@dataclass(frozen=True)
class ModelConflict:
    """A conflict the model proposed that passed the checks."""

    question_id: str
    fact_ids_a: tuple[str, ...]
    fact_ids_b: tuple[str, ...]
    kind: str
    severity: str
    explanation: str
    quote_a: str
    quote_b: str
    confidence: float


@dataclass(frozen=True)
class RefusedConflict:
    question_id: str
    explanation: str
    reason: str


@dataclass(frozen=True)
class Retraction:
    """The model says new evidence resolves an EARLIER contradiction, and cites it (checked: a new fact, quoted verbatim)."""

    contradiction_id: str
    fact_ids: tuple[str, ...]
    quote: str
    explanation: str


@dataclass
class ConflictOutcome:
    conflicts: list[ModelConflict] = field(default_factory=list)
    refused: list[RefusedConflict] = field(default_factory=list)
    retractions: list[Retraction] = field(default_factory=list)
    unavailable: bool = False
    detail: str = ""


class ConflictJudge(Protocol):
    def judge(self, question: OpenQuestion, facts: Sequence[Fact], earlier: Sequence[Contradiction] = (),
              new_fact_ids: Collection[str] = ()) -> ConflictOutcome: ...


@dataclass(frozen=True)
class ConflictDetail:
    source: Literal["rule", "model"]
    rule_id: str | None = None
    confidence: float | None = None
    quote_a: str = ""
    quote_b: str = ""


@dataclass
class DetectionResult:
    contradictions: list[Contradiction]
    details: dict[str, ConflictDetail] = field(default_factory=dict)          # contradiction_id -> where it came from
    refused: list[RefusedConflict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    retained: list[str] = field(default_factory=list)                         # earlier contradictions kept because no checked evidence resolved them
    retracted: list[Retraction] = field(default_factory=list)                 # earlier contradictions removed, with the evidence that resolved them

    def for_question(self, question_id: str) -> list[Contradiction]:
        return [c for c in self.contradictions if c.question_id == question_id]

    def by_kind(self, kind: str) -> list[Contradiction]:
        return [c for c in self.contradictions if c.kind == kind]


def contradiction_id(question_id: str, kind: str, a: Sequence[str], b: Sequence[str]) -> str:
    sides = sorted([",".join(sorted(a)), ",".join(sorted(b))])
    return "C-" + hashlib.sha1("|".join([question_id, kind, *sides]).encode("utf-8")).hexdigest()[:8].upper()


def _fmt(d: Decimal) -> str:
    return f"{d:,.0f}" if d == d.to_integral_value() else f"{d:,.2f}"


class ContradictionDetector:
    def __init__(self, cfg: ContradictionConfig, judge: ConflictJudge | None = None) -> None:
        self.cfg = cfg
        self.judge = judge if cfg.enabled else None
        self._rx: dict[str, re.Pattern[str]] = {}

    # ------------------------------------------------------------------ rules
    def _find(self, facts: Sequence[Fact], patterns: Sequence[str]) -> list[tuple[Fact, str]]:
        out = []
        for f in facts:
            for p in patterns:
                rx = self._rx.setdefault(p, re.compile(p, re.IGNORECASE))
                if m := rx.search(f.proposition.strip()):
                    out.append((f, m.group("value").strip()))
                    break
        return out

    def _date_rule(self, r: DateRule, facts: Sequence[Fact]) -> list[tuple[Contradiction, ConflictDetail]]:
        found = [(f, parse_date(v)) for f, v in self._find(facts, r.sources)]
        found = [(f, d) for f, d in found if d is not None]
        out = []
        for i, (fa, da) in enumerate(found):
            for fb, db in found[i + 1:]:
                if da != db:
                    out.append(self._make(r.question, "date", r.severity, [fa.fact_id], [fb.fact_id],
                                          f"The date is stated as {da.isoformat()} ({fa.proposition!r}) and as {db.isoformat()} ({fb.proposition!r}).", r.id))
        return out

    def _order_rule(self, r: OrderRule, facts: Sequence[Fact]) -> list[tuple[Contradiction, ConflictDetail]]:
        earlier = [(f, parse_date(v)) for f, v in self._find(facts, r.earlier)]
        later = [(f, parse_date(v)) for f, v in self._find(facts, r.later)]
        out = []
        for fe, de in earlier:
            for fl, dl in later:
                if de is not None and dl is not None and dl < de:
                    out.append(self._make(r.question, "date", r.severity, [fe.fact_id], [fl.fact_id],
                                          f"{fl.proposition!r} ({dl.isoformat()}) is dated before {fe.proposition!r} ({de.isoformat()}).", r.id))
        return out

    def _amount_rule(self, r: AmountRule, facts: Sequence[Fact]) -> list[tuple[Contradiction, ConflictDetail]]:
        a_side = [(f, parse_amount(v)) for f, v in self._find(facts, r.a)]
        b_side = [(f, parse_amount(v)) for f, v in self._find(facts, r.b)]
        out = []
        for fa, va in a_side:
            for fb, vb in b_side:
                if va is None or vb is None or fa.fact_id == fb.fact_id or max(va, vb) == 0:
                    continue
                gap = abs(va - vb) / max(va, vb) * 100
                band = [b for b in r.bands if gap >= Decimal(str(b.min_gap_pct))]
                if band:
                    out.append(self._make(r.question, "amount", band[-1].severity, [fa.fact_id], [fb.fact_id],
                                          f"The amounts differ by {gap:.0f}%: INR {_fmt(va)} ({fa.proposition!r}) against INR {_fmt(vb)} ({fb.proposition!r}).", r.id))
        return out

    @staticmethod
    def _make(question_id: str, kind: str, severity: str, a: list[str], b: list[str], explanation: str,
              rule_id: str) -> tuple[Contradiction, ConflictDetail]:
        c = Contradiction(contradiction_id=contradiction_id(question_id, kind, a, b), question_id=question_id, fact_ids_a=a, fact_ids_b=b,
                          kind=kind, severity=severity, explanation=explanation)  # type: ignore[arg-type]
        return c, ConflictDetail("rule", rule_id)

    # ------------------------------------------------------------------ detect
    def detect(self, facts: Sequence[Fact], questions: Sequence[OpenQuestion], links: Sequence[object] | None = None,
               only: Collection[str] | None = None, previous: Sequence[Contradiction] = (), changed_fact_ids: Collection[str] = ()) -> DetectionResult:
        """`only`: restrict to these question ids (an incremental run): rule conflicts on other questions are not reported and the model is not asked about them.
        `previous` / `changed_fact_ids`: contradictions found earlier, and the facts that are new or changed since. An earlier narrative contradiction whose facts
        are all still present and unchanged is KEPT unless the model retracts it with checked evidence (a new fact, quoted): a conflict never disappears just
        because a later model call did not repeat it."""
        usable = [f for f in facts if f.verification_status != "rejected"]
        qids = {q.question_id for q in questions}
        result = DetectionResult(contradictions=[])
        seen: set[str] = set()

        def add(c: Contradiction, d: ConflictDetail) -> None:
            if c.contradiction_id not in seen and c.question_id in qids and c.fact_ids_a and c.fact_ids_b and (only is None or c.question_id in only):
                seen.add(c.contradiction_id)
                result.contradictions.append(c)
                result.details[c.contradiction_id] = d

        for r in self.cfg.date_rules:
            for c, d in self._date_rule(r, usable):
                add(c, d)
        for r in self.cfg.order_rules:
            for c, d in self._order_rule(r, usable):
                add(c, d)
        for r in self.cfg.amount_rules:
            for c, d in self._amount_rule(r, usable):
                add(c, d)

        if self.judge is not None:
            linked = self._linked(usable, questions, links)
            for q in questions:
                facts_q = linked.get(q.question_id, [])
                if q.category not in self.cfg.model_categories or len(facts_q) < 2 or (only is not None and q.question_id not in only):
                    continue
                ids_q = {f.fact_id for f in facts_q}
                earlier = [c for c in previous if c.question_id == q.question_id and {*c.fact_ids_a, *c.fact_ids_b} <= ids_q
                           and not {*c.fact_ids_a, *c.fact_ids_b} & set(changed_fact_ids)]
                outcome = self.judge.judge(q, facts_q) if not earlier else self.judge.judge(q, facts_q, earlier=earlier, new_fact_ids=frozenset(changed_fact_ids))
                result.refused += outcome.refused
                retracted = {r.contradiction_id: r for r in outcome.retractions}
                for c in earlier:
                    if c.contradiction_id in retracted:
                        result.retracted.append(retracted[c.contradiction_id])
                    elif c.contradiction_id not in seen:
                        add(c, ConflictDetail("model", None, None, "", ""))
                        result.retained.append(c.contradiction_id)
                if outcome.unavailable:
                    result.warnings.append(f"narrative conflicts were not checked for {q.question_id}: the model was unavailable ({outcome.detail or 'no detail'})")
                for m in outcome.conflicts:
                    cid = contradiction_id(m.question_id, m.kind, m.fact_ids_a, m.fact_ids_b)
                    if cid in seen:
                        continue
                    add(Contradiction(contradiction_id=cid, question_id=m.question_id, fact_ids_a=list(m.fact_ids_a), fact_ids_b=list(m.fact_ids_b),
                                      kind=m.kind, severity=m.severity, explanation=m.explanation),  # type: ignore[arg-type]
                        ConflictDetail("model", None, m.confidence, m.quote_a, m.quote_b))
        return result

    @staticmethod
    def _linked(facts: Sequence[Fact], questions: Sequence[OpenQuestion], links: Sequence[object] | None) -> dict[str, list[Fact]]:
        """The facts linked to each question: from the T4.2 links when given (any object with fact_id and question_id), else the facts' own tags."""
        by_id = {f.fact_id: f for f in facts}
        out: dict[str, list[Fact]] = {q.question_id: [] for q in questions}
        if links is not None:
            for x in links:
                f = by_id.get(getattr(x, "fact_id", ""))
                if f is not None and getattr(x, "question_id", "") in out and f not in out[x.question_id]:  # type: ignore[attr-defined]
                    out[x.question_id].append(f)  # type: ignore[attr-defined]
        else:
            for f in facts:
                for q in f.related_question_ids:
                    if q in out:
                        out[q].append(f)
        return out
