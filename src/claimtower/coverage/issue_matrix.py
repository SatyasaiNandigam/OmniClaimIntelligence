"""Coverage issue matrix (T5.1): clauses x facts x open questions -> one row per issue, with citations and a status. Never a coverage verdict.

    builder = IssueMatrixBuilder(pack.coverage, judge=LLMReadingJudge(llm, pack), min_citations=1)
    result = builder.build(snapshot, not_in_force, facts, questions, contradictions)
    result.issues       # list[CoverageIssue] (contract): issue, status, clause citations (verbatim text, modified_by), fact ids, question ids, note

Rows come from the pack (`coverage_issues.yaml`): an issue, the open questions that control it, the clauses it cites. A row exists only when its controlling
questions are in the register. How a row's status follows is a DECISION TABLE in the pack, read top to bottom, from:
  kind     condition | cover | exclusion | amount
  control  the aggregate state of the controlling questions: contested > missing > partial > open > settled
  reading  for cover and exclusion rows whose questions are settled: does the evidence favour the claim as presented under the clause? (favourable | adverse |
           unclear | not_judged), judged by a model that must cite facts and clauses and quote a fact verbatim (llm_reading.py)
  flags    not_in_force (the policy system answered: nothing in force), no_citation (no clause could be cited)
An ADVERSE reading never makes a row `established` (an exclusion that may be engaged is `potentially_relevant`, a condition that may not be met is
`unresolved`): the system never concludes against the claimant, a person does (guardrail 2). Every note passes a wording check: none says a claim is
covered, excluded, approved or denied. A row is cited from the policy snapshot, so its clauses are the ones in force on the date of loss, with the
endorsements that modify them.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from claimtower.core.config import ClauseLink, CoverageConfig, IssueRow, IssueSpec
from claimtower.core.schemas import ClauseCitation, Contradiction, CoverageIssue, Fact, OpenQuestion, PolicyClause, PolicySnapshot

ReadingValue = Literal["favourable", "adverse", "unclear", "ambiguous", "not_judged"]


@dataclass(frozen=True)
class Reading:
    value: ReadingValue = "not_judged"
    fact_ids: tuple[str, ...] = ()
    clause_ids: tuple[str, ...] = ()
    quote: str = ""
    rationale: str = ""
    confidence: float | None = None
    alternatives: tuple[str, ...] = ()            # ambiguous: each reasonable reading, with the clause words it rests on


@dataclass
class IssueContext:
    """What the judge is shown for one row."""

    spec: IssueSpec
    clauses: Sequence[PolicyClause]
    facts: Sequence[Fact]
    questions: Sequence[OpenQuestion]
    contradictions: Sequence[Contradiction]


@dataclass
class ReadingOutcome:
    reading: Reading = field(default_factory=Reading)
    refused: str | None = None                  # why a model answer was refused (the reading is then `unclear`)
    unavailable: bool = False
    detail: str = ""


class ReadingJudge(Protocol):
    def read(self, ctx: IssueContext) -> ReadingOutcome: ...


@dataclass(frozen=True)
class RowDetail:
    issue_id: str
    row_id: str                                 # the decision-table row that decided
    control: str
    reading: Reading
    in_force: bool
    reason: str = ""                            # the abstention / non-conclusion reason code of the decision-table row


@dataclass
class MatrixResult:
    issues: list[CoverageIssue]
    details: dict[str, RowDetail] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)                    # issues whose controlling questions are not in the register
    refused: list[tuple[str, str]] = field(default_factory=list)        # (issue id, why a model reading was refused)
    warnings: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)                   # self-check failures: must be empty

    def get(self, issue_id: str) -> CoverageIssue:
        return next(i for i in self.issues if i.issue_id == issue_id)

    def abstentions(self) -> dict[str, str]:
        """Rows where the system declines to conclude, by reason code (T5.2): the abstain rows, and those left open for want of an answer."""
        return {i: d.reason for i, d in self.details.items() if d.reason}


def control_state(statuses: Sequence[str]) -> str:
    """The aggregate state of an issue's controlling questions."""
    s = set(statuses)
    if "requires_validation" in s:
        return "contested"
    if "missing" in s:
        return "missing"
    if "partially_established" in s:
        return "partial"
    return "settled" if s and s == {"established"} else "open"


class IssueMatrixBuilder:
    def __init__(self, cfg: CoverageConfig, judge: ReadingJudge | None = None, min_citations: int = 1) -> None:
        self.cfg = cfg
        self.judge = judge if cfg.reading.enabled else None
        self.min_citations = min_citations
        self._forbidden = [re.compile(p, re.IGNORECASE) for p in cfg.forbidden_patterns]
        self._rx: dict[int, list[re.Pattern[str]]] = {}
        self._cache: dict[str, tuple[Reading, str | None]] = {}      # a reading is kept while its inputs are identical: unchanged rows are never re-read
        self.model_reads = 0
        self.cache_hits = 0

    # ------------------------------------------------------------------ wording
    def lint(self, text: str) -> list[str]:
        return [f"states a coverage conclusion: {m.group(0)!r}" for rx in self._forbidden for m in [rx.search(text)] if m]

    # ------------------------------------------------------------------ citations
    def _select(self, links: Sequence[ClauseLink], clauses: Sequence[PolicyClause]) -> list[PolicyClause]:
        found: list[PolicyClause] = []
        for link in links:
            if link.fallback and found:
                continue
            rx = self._rx.setdefault(id(link), [re.compile(p, re.IGNORECASE) for p in link.match])
            for c in clauses:
                if c in found or c.clause_type not in link.types:
                    continue
                if rx and not any(r.search(f"{c.heading or ''} {c.text}") for r in rx):
                    continue
                found.append(c)
        return found

    @staticmethod
    def _citation(c: PolicyClause, snapshot: PolicySnapshot) -> ClauseCitation:
        modified_by = [r.source_clause_id for r in snapshot.relations if r.relation == "modifies" and r.target_clause_id == c.clause_id]
        return ClauseCitation(clause_id=c.clause_id, document=c.document, page=c.page, span=c.span, text=c.text, modified_by=modified_by)

    # ------------------------------------------------------------------ the table
    def _row(self, facts: dict[str, object]) -> IssueRow:
        for row in self.cfg.table:
            ok = True
            for key, want in row.when.items():
                have = facts[key]
                ok = ok and (have == want if isinstance(want, bool) else have in (want if isinstance(want, list) else [want]))
            if ok:
                return row
        raise AssertionError("the last row of the status table has no condition")           # CoverageConfig guarantees it

    # ------------------------------------------------------------------ build
    def build(self, snapshot: PolicySnapshot | None, not_in_force: bool, facts: Sequence[Fact], questions: Sequence[OpenQuestion],
              contradictions: Sequence[Contradiction] = ()) -> MatrixResult:
        result = MatrixResult(issues=[])
        qmap = {q.question_id: q for q in questions}
        fact_by_id = {f.fact_id: f for f in facts if f.verification_status != "rejected"}
        clauses = list(snapshot.clauses) if snapshot is not None else []
        for spec in self.cfg.issues:
            if any(q not in qmap for q in spec.questions):
                result.skipped.append(spec.id)
                continue
            controlling = [qmap[q] for q in spec.questions]
            control = control_state([q.status for q in controlling])
            picked = self._select(spec.clauses, clauses)
            citations = [self._citation(c, snapshot) for c in picked] if snapshot is not None else []
            linked = [fact_by_id[i] for q in controlling for i in q.linked_fact_ids if i in fact_by_id] if spec.kind != "amount" else []
            linked = list({f.fact_id: f for f in linked}.values())
            reading, why = self._reading(spec, control, picked, linked, controlling, contradictions, result)
            if why:
                result.refused.append((spec.id, why))
            row = self._row({"kind": spec.kind, "control": control, "reading": reading.value, "not_in_force": not_in_force,
                             "no_citation": len(citations) < self.min_citations})
            note = row.note
            if reading.value in ("favourable", "adverse") and reading.rationale and not self.lint(reading.rationale):
                note = f"{note} Reading: {reading.rationale}"
            if reading.value == "ambiguous":
                note = f"{note} " + " ".join(f"Reading {chr(65 + n)}: {a}" for n, a in enumerate(reading.alternatives))
            cited_facts = list(reading.fact_ids) or [f.fact_id for f in linked]
            result.issues.append(CoverageIssue(issue_id=spec.id, issue=spec.issue, status=row.status, clause_citations=citations, fact_ids=cited_facts,
                                               question_ids=list(spec.questions), note=note))
            result.details[spec.id] = RowDetail(spec.id, row.id, control, reading, not not_in_force, row.reason)
        result.problems = self.self_check(result)
        return result

    def _reading(self, spec: IssueSpec, control: str, picked: Sequence[PolicyClause], facts: Sequence[Fact], controlling: Sequence[OpenQuestion],
                 contradictions: Sequence[Contradiction], result: MatrixResult) -> tuple[Reading, str | None]:
        if spec.kind not in ("cover", "exclusion") or control != "settled" or not picked or not facts:
            return Reading(), None                                                       # nothing to read, or not yet answerable: the table decides
        if self.judge is None:
            return Reading(), None
        qids = {q.question_id for q in controlling}
        conflicts = [c for c in contradictions if c.question_id in qids]
        key = hashlib.sha256(json.dumps([spec.id, [(c.clause_id, c.text) for c in picked], [(f.fact_id, f.proposition) for f in facts],
                                         [(q.question_id, q.status) for q in controlling], sorted(c.contradiction_id for c in conflicts)]).encode()).hexdigest()
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.model_reads += 1
        outcome = self.judge.read(IssueContext(spec, picked, facts, controlling, conflicts))
        if outcome.unavailable:
            result.warnings.append(f"{spec.id}: the model was unavailable ({outcome.detail or 'no detail'}); the facts were not read against the clause")
            return Reading(), None
        self._cache[key] = (outcome.reading, outcome.refused)
        return outcome.reading, outcome.refused

    # ------------------------------------------------------------------ self-check
    def self_check(self, result: MatrixResult) -> list[str]:
        """Every row is cited unless it is the not-in-force row or an abstention; no note states a coverage conclusion; an adverse reading is never `established`."""
        problems: list[str] = []
        for i in result.issues:
            d = result.details[i.issue_id]
            if len(i.clause_citations) < self.min_citations and i.status != "abstain" and d.row_id != "R01-not-in-force":
                problems.append(f"{i.issue_id}: status {i.status} without a clause citation")
            if i.status == "established" and d.reading.value == "adverse":
                problems.append(f"{i.issue_id}: established on an adverse reading")
            problems += [f"{i.issue_id}: {p}" for p in self.lint(i.note or "")]
        return problems
