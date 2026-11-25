"""Counterfactual engine (T4.6): "what would resolve this claim?" without ever saying what the decision will be.

    engine = CounterfactualEngine(pack.question_templates, writer=LLMScenarioWriter(llm, pack))          # writer optional
    result = engine.generate(questions, sufficiency, coverage_issues, snapshot.clauses, requests, facts)
    result.scenarios         # list[CounterfactualScenario] (contract): A, B, C for each unresolved material question

For every MATERIAL question (framed in the pack: `counterfactuals.framings`) that is not established, three scenarios:
  A  a finding that supports the claim as presented     B  a finding that points away from it     C  an inconclusive finding
Each lists the coverage issues it would re-assess (those that share a clause with the question and are not already established), what the finding
would do to each (`effects`, by kind of issue: cover / exclusion / amount, read from the clause types of its citations) and the next step (B goes
to a senior adjuster with the cited clauses, C to an independent expert, A to collecting what is outstanding then re-running coverage and settlement).
The text is built from these rules. A model may REWRITE it for the claim (llm_counterfactual.py); a rewrite is kept only if it passes `lint`
(no wording that predicts or recommends a decision), is conditional, names only issues that are affected, mentions each affected issue and keeps each
effect, and fits the length limits. Anything else reverts to the rule text. Nothing here ever returns an outcome: `lint` is also run on the rule text.
"""
from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from claimtower.core.config import SCENARIOS, CounterfactualFraming, QuestionTemplates
from claimtower.core.schemas import CoverageIssue, CounterfactualScenario, Fact, MissingEvidence, OpenQuestion, PolicyClause
from claimtower.evidence.questions import overlap
from claimtower.evidence.sufficiency import SufficiencyResult

_ID = re.compile(r"\b(?:ISS|CL|Q|F|C)-[A-Za-z0-9][A-Za-z0-9-]*")
AMOUNT_TYPES = {"limit", "sublimit", "deductible"}


@dataclass(frozen=True)
class AffectedIssue:
    issue_id: str
    title: str                                     # the issue's name for display; the issue id when the name itself would pass for a decision
    kind: Literal["cover", "exclusion", "amount"]


@dataclass
class ScenarioDraft:
    scenario_id: str
    implication_text: str
    next_step: str


@dataclass
class WriterContext:
    question: OpenQuestion
    framing: CounterfactualFraming
    issues: Sequence[AffectedIssue]
    base: Mapping[str, ScenarioDraft]              # the rule-built text per scenario, to be rewritten
    effects: Mapping[str, Mapping[str, str]]       # issue id -> scenario -> effect phrase that must be kept
    clauses: Mapping[str, str]                     # clause id -> text of the clauses behind the issues
    known: Sequence[str]                           # facts on file about the question
    max_implication_chars: int
    max_next_step_chars: int


@dataclass
class WriterOutcome:
    drafts: list[ScenarioDraft] = field(default_factory=list)
    unavailable: bool = False
    detail: str = ""


class ScenarioWriter(Protocol):
    def write(self, ctx: WriterContext) -> WriterOutcome: ...


@dataclass(frozen=True)
class RefusedRewrite:
    question_id: str
    scenario_id: str
    reason: str


@dataclass
class CounterfactualResult:
    scenarios: list[CounterfactualScenario]
    source: dict[tuple[str, str], Literal["rule", "model"]] = field(default_factory=dict)       # (question id, scenario id) -> where the text came from
    refused: list[RefusedRewrite] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)                                             # material questions with no affected issue
    problems: list[str] = field(default_factory=list)                                            # self-check failures: must be empty
    warnings: list[str] = field(default_factory=list)

    def for_question(self, question_id: str) -> list[CounterfactualScenario]:
        return [s for s in self.scenarios if s.question_id == question_id]


class CounterfactualEngine:
    def __init__(self, templates: QuestionTemplates, writer: ScenarioWriter | None = None) -> None:
        if templates.counterfactuals is None:
            raise ValueError("the pack has no `counterfactuals` section in question_templates.yaml")
        self.templates, self.cfg = templates, templates.counterfactuals
        self.writer = writer if self.cfg.enabled else None
        self._forbidden = [re.compile(p, re.IGNORECASE) for p in self.cfg.forbidden_patterns]
        self._conditional = [re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in self.cfg.conditional_markers]

    # ------------------------------------------------------------------ the linter
    def lint(self, text: str) -> list[str]:
        """Wording problems in a scenario text: a prediction or recommendation of a decision. Empty means clean."""
        return [f"predicts or recommends a decision: {m.group(0)!r}" for rx in self._forbidden for m in [rx.search(text)] if m]

    def lint_implication(self, text: str) -> list[str]:
        problems = self.lint(text)
        if self._conditional and not any(rx.search(text) for rx in self._conditional):
            problems.append("is not conditional (it must say what a finding WOULD change, not what is)")
        return problems

    # ------------------------------------------------------------------ affected issues
    @staticmethod
    def kind_of(issue: CoverageIssue, clause_types: Mapping[str, str]) -> Literal["cover", "exclusion", "amount"]:
        types = {clause_types.get(c.clause_id, "") for c in issue.clause_citations}
        return "exclusion" if "exclusion" in types else "amount" if types & AMOUNT_TYPES else "cover"

    def affected(self, q: OpenQuestion, issues: Sequence[CoverageIssue], clauses: Sequence[PolicyClause], kinds: Sequence[str] | None = None) -> list[AffectedIssue]:
        types = {c.clause_id: c.clause_type for c in clauses}
        linked = set(q.linked_clause_ids)
        out = [AffectedIssue(i.issue_id, i.issue, self.kind_of(i, types)) for i in issues
               if i.status != "established" and (q.question_id in i.question_ids or linked & {c.clause_id for c in i.clause_citations})]
        out = [AffectedIssue(a.issue_id, a.title if not self.lint(a.title) else a.issue_id, a.kind) for a in out]       # a title that reads like a decision is not repeated
        return [a for a in out if kinds is None or a.kind in kinds]

    # ------------------------------------------------------------------ generate
    def generate(self, questions: Sequence[OpenQuestion], sufficiency: SufficiencyResult, issues: Sequence[CoverageIssue],
                 clauses: Sequence[PolicyClause] = (), requests: Sequence[MissingEvidence] = (), facts: Sequence[Fact] = (),
                 links: Sequence[object] = (), only_questions: Collection[str] | None = None) -> CounterfactualResult:
        """`only_questions`: write the scenarios of these questions only (an incremental run keeps the others as they were)."""
        result = CounterfactualResult(scenarios=[])
        status = {a.question_id: a for a in sufficiency.assessments}
        qmap = {q.question_id: q for q in questions}
        fact_by_id = {f.fact_id: f for f in facts}
        clause_text = {c.clause_id: " ".join(c.text.split()) for c in clauses}
        for framing in self.cfg.framings:
            if only_questions is not None and framing.question not in only_questions:
                continue
            q = qmap.get(framing.question)
            a = status.get(framing.question)
            if q is None or a is None or a.status == "established":
                continue                                                           # nothing unresolved to ask "what if" about
            issues_q = self.affected(q, issues, clauses, framing.issue_kinds)
            if not issues_q:
                result.skipped.append(q.question_id)
                result.warnings.append(f"{q.question_id}: no open coverage issue shares a clause with it, so there is nothing for a scenario to re-assess")
                continue
            base = {s: self._rule_draft(s, q, issues_q, requests) for s in SCENARIOS}
            drafts = dict(base)
            if self.writer is not None:
                known = [" ".join(fact_by_id[x.fact_id].proposition.split()) for x in links
                         if getattr(x, "question_id", None) == q.question_id and getattr(x, "fact_id", None) in fact_by_id][:6]
                ctx = WriterContext(q, framing, issues_q, base, {i.issue_id: {s: self.cfg.effects[i.kind][s] for s in SCENARIOS} for i in issues_q},
                                    {cid: clause_text[cid] for i in issues if i.issue_id in {x.issue_id for x in issues_q}
                                     for cid in (c.clause_id for c in i.clause_citations) if cid in clause_text},
                                    known, self.cfg.max_implication_chars, self.cfg.max_next_step_chars)
                outcome = self.writer.write(ctx)
                if outcome.unavailable:
                    result.warnings.append(f"{q.question_id}: the model was unavailable ({outcome.detail or 'no detail'}); the scenarios use the rule text")
                for d in outcome.drafts:
                    why = self._rewrite_problem(d, q, issues_q, base)
                    if why:
                        result.refused.append(RefusedRewrite(q.question_id, d.scenario_id, why))
                    else:
                        drafts[d.scenario_id] = ScenarioDraft(d.scenario_id, " ".join(d.implication_text.split()), " ".join(d.next_step.split()))
            for s in SCENARIOS:
                d = drafts[s]
                result.scenarios.append(CounterfactualScenario(
                    scenario_id=s, question_id=q.question_id, assumed_finding=getattr(framing, s.lower()),  # type: ignore[arg-type]
                    affected_issue_ids=[i.issue_id for i in issues_q], implication_text=d.implication_text, next_step=d.next_step))
                result.source[(q.question_id, s)] = "rule" if d is base[s] else "model"
            result.problems += self.self_check(q, issues_q, [x for x in result.scenarios if x.question_id == q.question_id])
        return result

    # ------------------------------------------------------------------ rule text
    def _rule_draft(self, s: str, q: OpenQuestion, issues: Sequence[AffectedIssue], requests: Sequence[MissingEvidence]) -> ScenarioDraft:
        parts = [f"{i.title} {self.cfg.effects[i.kind][s]}" for i in issues]
        text = "; ".join(parts) + "."
        suffix = self.cfg.review_suffix.get(s)
        implication = f"{text} {suffix}" if suffix else text
        roles = [self.templates.requests.roles[m.minimum_evidence_request.addressee_role].label
                 for m in requests if q.question_id in m.resolves_question_ids
                 and m.minimum_evidence_request.addressee_role in (self.templates.requests.roles if self.templates.requests else {})]
        outstanding = ", ".join(dict.fromkeys(roles)) or "the outstanding requests"
        return ScenarioDraft(s, implication, self.cfg.next_steps[s].format(outstanding=outstanding))

    # ------------------------------------------------------------------ checks on a rewrite
    def _rewrite_problem(self, d: ScenarioDraft, q: OpenQuestion, issues: Sequence[AffectedIssue], base: Mapping[str, ScenarioDraft]) -> str | None:
        if d.scenario_id not in SCENARIOS:
            return f"'{d.scenario_id}' is not a scenario"
        imp, nxt = " ".join(d.implication_text.split()), " ".join(d.next_step.split())
        if not imp or len(imp) > self.cfg.max_implication_chars:
            return f"the implication is empty or longer than {self.cfg.max_implication_chars} characters"
        if not nxt or len(nxt) > self.cfg.max_next_step_chars:
            return f"the next step is empty or longer than {self.cfg.max_next_step_chars} characters"
        problems = self.lint_implication(imp) + self.lint(nxt)
        if problems:
            return "; ".join(problems)
        allowed = {i.issue_id for i in issues} | {q.question_id} | set(q.linked_clause_ids)
        stray = sorted({m.group(0) for m in _ID.finditer(f"{imp} {nxt}")} - allowed)
        if stray:
            return "it names ids that are not part of this scenario: " + ", ".join(stray)
        for i in issues:
            if i.issue_id not in imp and overlap(i.title, imp) < 0.5:
                return f"it does not mention the affected issue '{i.title}'"
            if overlap(self.cfg.effects[i.kind][d.scenario_id], imp) < 0.5:
                return f"it does not keep the effect on '{i.title}' ({self.cfg.effects[i.kind][d.scenario_id]})"
        return None

    # ------------------------------------------------------------------ self-check
    def self_check(self, q: OpenQuestion, issues: Sequence[AffectedIssue], scenarios: Sequence[CounterfactualScenario]) -> list[str]:
        problems: list[str] = []
        if [s.scenario_id for s in scenarios] != list(SCENARIOS):
            problems.append(f"{q.question_id}: expected scenarios A, B and C, got {[s.scenario_id for s in scenarios]}")
        known = {i.issue_id for i in issues}
        for s in scenarios:
            if not set(s.affected_issue_ids) <= known:
                problems.append(f"{q.question_id}/{s.scenario_id}: affected issues are not all open issues of this question")
            problems += [f"{q.question_id}/{s.scenario_id}: {p}" for p in self.lint_implication(s.implication_text) + self.lint(s.next_step)]
        return problems
