"""Sufficiency classifier (T4.4): is each open question answered, and if not, what kind of gap is it?

    classifier = SufficiencyClassifier(pack.question_templates.sufficiency, judge=LLMCriterionJudge(llm, pack))      # judge optional
    result = classifier.classify(questions, facts, links, contradictions, snapshot)
    result.apply(questions)                  # -> questions with `status` set
    result.get("Q-START").gap_type           # "contradictory" | "missing" | "insufficient" | None (established, or abstained)

For every question, each establishing criterion (from T4.1) gets a verdict: met / partly / unmet / unclear.
  * a criterion about the POLICY is decided from the snapshot (config `criteria_rules`): was the PAS answer obtained, is there sublimit wording
  * a criterion no linked fact bears on is `unmet` without asking a model
  * any other is judged by the model (llm_sufficiency.py): it must cite the facts and quote them, and may say `unclear`
The verdicts, the question's links and its contradictions go through the DECISION TABLE in the pack (`sufficiency.table`): the first matching
row gives the status and the gap type, and its id is recorded, so every status can be traced to a row and to the facts behind it:
  contradictory   a contradiction at a blocking severity (T4.3) beats everything else
  abstain         a criterion is unclear (the model was unsure, refused, or unavailable): status unverified, NO gap type guessed
  established     every criterion met
  insufficient    evidence exists but does not establish it (some criteria met, or addressed only partly)
  missing         nothing bears on any criterion
A person asked to resolve an abstained question sees which criteria were unclear and why.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

from claimtower.core.config import SufficiencyConfig
from claimtower.core.schemas import Contradiction, Fact, NotInForce, OpenQuestion, PolicySnapshot
from claimtower.core.schemas.common import GapType, QuestionStatus

Verdict = Literal["met", "partly", "unmet", "unclear"]
VERDICTS: tuple[str, ...] = ("met", "partly", "unmet", "unclear")


@dataclass(frozen=True)
class CriterionAssessment:
    index: int
    criterion: str
    verdict: Verdict
    source: Literal["policy", "rule", "model"]
    rationale: str
    fact_ids: tuple[str, ...] = ()
    quote: str = ""
    confidence: float | None = None


@dataclass(frozen=True)
class RefusedVerdict:
    question_id: str
    index: int | None
    reason: str


@dataclass
class JudgeOutcome:
    verdicts: dict[int, CriterionAssessment] = field(default_factory=dict)
    refused: list[RefusedVerdict] = field(default_factory=list)
    unavailable: bool = False
    detail: str = ""


@dataclass(frozen=True)
class PurposeVerdict:
    """The model's answer to "do the facts together answer what the question is for?", after the checks (None settled = no, with the reason)."""

    settled: bool
    fact_ids: tuple[str, ...] = ()
    quote: str = ""
    rationale: str = ""
    confidence: float | None = None


class CriterionJudge(Protocol):
    def judge(self, question: OpenQuestion, criteria: dict[int, str], facts: Sequence[Fact], directions: dict[str, str],
              sources: dict[str, str]) -> JudgeOutcome: ...


@dataclass(frozen=True)
class QuestionAssessment:
    question_id: str
    status: QuestionStatus
    gap_type: GapType | None
    row_id: str                                   # the decision-table row that decided
    criteria: tuple[CriterionAssessment, ...]
    contradiction_ids: tuple[str, ...]            # blocking contradictions (when row R1 applies)
    noted_contradiction_ids: tuple[str, ...]      # contradictions below the blocking severity: noted, not blocking
    abstained: bool
    reason: str
    settled_on_evidence: bool = False             # row P1: the facts together answered the question although a listed example was not met word for word
    settled_fact_ids: tuple[str, ...] = ()
    settled_quote: str = ""

    def unresolved(self) -> list[CriterionAssessment]:
        """The criteria that still need evidence (what T4.5 turns into a request): everything not met."""
        return [c for c in self.criteria if c.verdict != "met"]


@dataclass
class SufficiencyResult:
    assessments: list[QuestionAssessment]
    refused: list[RefusedVerdict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def get(self, question_id: str) -> QuestionAssessment:
        return next(a for a in self.assessments if a.question_id == question_id)

    def apply(self, questions: Sequence[OpenQuestion]) -> list[OpenQuestion]:
        by = {a.question_id: a for a in self.assessments}
        return [q.model_copy(update={"status": by[q.question_id].status}) if q.question_id in by else q for q in questions]

    def gaps(self) -> dict[str, GapType]:
        return {a.question_id: a.gap_type for a in self.assessments if a.gap_type is not None}


class SufficiencyClassifier:
    def __init__(self, cfg: SufficiencyConfig, judge: CriterionJudge | None = None) -> None:
        self.cfg = cfg
        self.judge = judge if cfg.enabled else None

    def classify(self, questions: Sequence[OpenQuestion], facts: Sequence[Fact], links: Sequence[object],
                 contradictions: Sequence[Contradiction], snapshot: PolicySnapshot | NotInForce | None,
                 documents: Mapping[str, str] | None = None, modes: Mapping[str, str] | None = None) -> SufficiencyResult:
        """`documents`: document id -> type (fnol, plumber_report, surveyor_report ...), shown to the model so it knows who a fact comes from.
        `modes`: question id -> 'all' | 'any' (from the question templates); questions not listed need every criterion."""
        usable = {f.fact_id: f for f in facts if f.verification_status != "rejected"}
        result = SufficiencyResult(assessments=[])
        for q in questions:
            linked = [(getattr(x, "fact_id"), getattr(x, "direction", "neutral")) for x in links if getattr(x, "question_id", None) == q.question_id
                      and getattr(x, "fact_id", None) in usable]
            directions = dict(linked)
            q_facts = [usable[i] for i in directions]
            sources = {f.fact_id: documents[f.source_document_id] for f in q_facts if documents and f.source_document_id in documents}
            criteria = self._assess_criteria(q, q_facts, directions, sources, snapshot, result)
            a = self._decide(q, criteria, directions, [c for c in contradictions if c.question_id == q.question_id], (modes or {}).get(q.question_id, "all"))
            result.assessments.append(self._settle(q, a, q_facts, directions, sources, result))
        return result

    # ------------------------------------------------------------------ settling on the evidence as a whole (config `purpose`)
    def _settle(self, q: OpenQuestion, a: QuestionAssessment, facts: Sequence[Fact], directions: dict[str, str], sources: dict[str, str],
                result: SufficiencyResult) -> QuestionAssessment:
        """A question about HOW something happened (its category is listed in `purpose.categories`) that its criteria only partly establish gets a second look: do the facts TOGETHER answer
        what it is for? Only a question with evidence for it and an `insufficient` gap qualifies: never one with a blocking contradiction, one the model could not judge (abstained), or one
        with nothing to go on (missing). A fact that points the other way stops it, and so does a model that is down: the status stays as the decision table gave it."""
        p = self.cfg.purpose
        settle = getattr(self.judge, "settle", None)
        if not (p.enabled and settle is not None and q.category in p.categories and a.gap_type == "insufficient" and not a.abstained and facts):
            return a
        if any(d == "contradicting" for d in directions.values()):
            return a
        v = settle(q, a.criteria, facts, directions, sources)
        if v is None or not v.settled:
            return a
        return replace(a, status="established", gap_type=None, row_id="P1-settled-on-evidence", abstained=False, settled_on_evidence=True, settled_fact_ids=v.fact_ids,
                       settled_quote=v.quote, reason=f"row P1: the facts together answer the question: {v.rationale}")

    # ------------------------------------------------------------------ criteria
    def _assess_criteria(self, q: OpenQuestion, facts: Sequence[Fact], directions: dict[str, str], sources: dict[str, str],
                         snapshot: PolicySnapshot | NotInForce | None, result: SufficiencyResult) -> list[CriterionAssessment]:
        rules = {r.criterion: r for r in self.cfg.criteria_rules if r.question == q.question_id}
        out: dict[int, CriterionAssessment] = {}
        to_judge: dict[int, str] = {}
        for i, text in enumerate(q.establishing_criteria):
            r = rules.get(text)
            if r is not None:
                if r.met_if == "policy_answer_known":
                    ok, why = snapshot is not None, ("the policy system answered" if snapshot is not None else "no policy answer was obtained")
                else:
                    types = set(r.types)
                    ids = [c for c in q.linked_clause_ids]
                    clauses = {c.clause_id: c for c in snapshot.clauses} if isinstance(snapshot, PolicySnapshot) else {}
                    hit = [c for c in ids if c in clauses and clauses[c].clause_type in types]
                    ok, why = bool(hit), (f"policy clause(s) {', '.join(hit)} apply" if hit else f"no {'/'.join(sorted(types))} clause is in the policy snapshot")
                out[i] = CriterionAssessment(i, text, "met" if ok else "unmet", "policy", why)
            elif not facts:
                out[i] = CriterionAssessment(i, text, "unmet", "rule", "no fact bears on this question")
            else:
                to_judge[i] = text
        if to_judge:
            if self.judge is None:
                for i, text in to_judge.items():
                    out[i] = CriterionAssessment(i, text, "unclear", "rule", "no model was available to judge the facts against this criterion")
                result.warnings.append(f"{q.question_id}: fact-based criteria were not judged (no model available)")
            else:
                outcome = self.judge.judge(q, to_judge, facts, directions, sources)
                result.refused += outcome.refused
                if outcome.unavailable:
                    result.warnings.append(f"{q.question_id}: the model was unavailable ({outcome.detail or 'no detail'}); its criteria are unclear")
                for i, text in to_judge.items():
                    out[i] = outcome.verdicts.get(i) or CriterionAssessment(
                        i, text, "unclear", "model", "the model's judgement was unavailable or refused" if outcome.unavailable or outcome.refused
                        else "the model did not judge this criterion")
        return [out[i] for i in sorted(out)]

    # ------------------------------------------------------------------ decision table
    def _decide(self, q: OpenQuestion, criteria: Sequence[CriterionAssessment], directions: dict[str, str],
                contradictions: Sequence[Contradiction], mode: str = "all") -> QuestionAssessment:
        blocking = [c for c in contradictions if c.severity in self.cfg.blocking_severities]
        noted = [c for c in contradictions if c.severity not in self.cfg.blocking_severities]
        v = [c.verdict for c in criteria]
        flags = {"blocking_contradiction": bool(blocking), "unclear_criteria": "unclear" in v, "all_met": bool(v) and ("met" in v if mode == "any" else all(x == "met" for x in v)),
                 "any_met": "met" in v, "any_partly": "partly" in v, "supporting_link": "supporting" in directions.values(),
                 "linked_fact": bool(directions)}
        row = next(r for r in self.cfg.table if all(flags[k] == want for k, want in r.when.items()))
        met = sum(x == "met" for x in v)
        reason = f"row {row.id}: " + (f"{len(blocking)} contradiction(s) at {'/'.join(self.cfg.blocking_severities)} severity" if flags["blocking_contradiction"]
                                      else f"{met} of {len(v)} criteria met")
        return QuestionAssessment(q.question_id, row.status, row.gap, row.id, tuple(criteria), tuple(c.contradiction_id for c in blocking),
                                  tuple(c.contradiction_id for c in noted), row.abstain, reason)
