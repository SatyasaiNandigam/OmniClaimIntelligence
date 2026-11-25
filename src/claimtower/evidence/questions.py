"""Open-question register (T4.1): what must be known to decide this claim. The first step of the evidence-sufficiency engine.

    register = QuestionRegister(pack.question_templates, proposer=LLMQuestionProposer(llm, pack))      # proposer optional
    result = register.build(header, snapshot, facts, review_clause_ids=...)       # snapshot: PolicySnapshot | NotInForce | None
    result.questions        # list[OpenQuestion] (contract), template questions first, then the model's claim-specific ones

Two sources, one register:
  * TEMPLATE questions (question_templates.yaml) are the floor: always asked, never dropped or reworded by a model. Each is linked to the clauses
    of the snapshot it bears on by the pack's `clause_links` (clause types + word patterns). With no policy snapshot only the questions that do
    not need one are asked (was the policy in force?), and the register says why the others are missing.
  * MODEL questions (`model_questions.enabled`) are claim-specific additions a model proposes from the claim header, the facts and the clauses in
    force. They are CHECKED, never trusted (see llm_questions.py): each cites clauses / facts that exist, quotes them verbatim, uses an allowed
    category, is not a repeat, does not pre-judge the decision, and is capped in number and confidence. They carry `origin == "model"` here
    (not in the frozen OpenQuestion contract: a deliberate contract change, if wanted, is a separate decision) so a reader can tell them apart.
A model that is down or answers badly never breaks the register: it is the template questions plus a warning.

Every question starts `unverified`; T4.2-T4.4 link facts, detect contradictions and set the status.
"""
from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from claimtower.core.config import ClauseLink, ModelQuestionsConfig, QuestionTemplate, QuestionTemplates
from claimtower.core.schemas import ClaimHeader, Fact, NotInForce, OpenQuestion, PolicyClause, PolicySnapshot


@dataclass(frozen=True)
class ProposedQuestion:
    """A claim-specific question the model proposed and the checks accepted."""

    question_id: str
    text: str
    category: str
    clause_ids: tuple[str, ...]
    fact_ids: tuple[str, ...]
    evidence_quote: str
    criteria: tuple[str, ...]
    confidence: float                     # already capped


@dataclass(frozen=True)
class RejectedProposal:
    text: str
    reason: str


@dataclass
class ProposalContext:
    header: ClaimHeader
    facts: Sequence[Fact]
    clauses: Sequence[PolicyClause]
    existing: Sequence[OpenQuestion]      # the template questions: a proposal must not repeat them


@dataclass
class ProposalOutcome:
    accepted: list[ProposedQuestion] = field(default_factory=list)
    rejected: list[RejectedProposal] = field(default_factory=list)
    unavailable: bool = False             # the model could not be reached or its output was unusable (not a judgement about the claim)
    detail: str = ""


class QuestionProposer(Protocol):
    def propose(self, ctx: ProposalContext) -> ProposalOutcome: ...


@dataclass
class RegisterResult:
    questions: list[OpenQuestion]
    origin: dict[str, str]                                         # question_id -> "template" | "model"
    proposed: dict[str, ProposedQuestion] = field(default_factory=dict)       # the model questions, with quote / confidence
    rejected: list[RejectedProposal] = field(default_factory=list)            # what the checks refused, with the reason
    review_links: dict[str, list[str]] = field(default_factory=dict)          # question_id -> linked clauses whose TYPE a model decided
    warnings: list[str] = field(default_factory=list)

    def get(self, question_id: str) -> OpenQuestion:
        return next(q for q in self.questions if q.question_id == question_id)


_WORD = re.compile(r"[a-z0-9]+")
STOP = frozenset({"a", "an", "and", "any", "are", "as", "at", "be", "been", "by", "did", "do", "does", "for", "from", "has", "have", "in", "is", "it",
                  "of", "on", "or", "the", "this", "that", "to", "was", "were", "what", "when", "which", "who", "whether", "with"})


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in STOP}


def overlap(a: str, b: str) -> float:
    """Share of a's content words that b also has (0..1). 0 when a has no content words."""
    wa, wb = content_words(a), content_words(b)
    return len(wa & wb) / len(wa) if wa else 0.0


class QuestionRegister:
    def __init__(self, templates: QuestionTemplates, proposer: QuestionProposer | None = None) -> None:
        self.templates = templates
        self.cfg: ModelQuestionsConfig = templates.model_questions
        self.proposer = proposer if templates.model_questions.enabled else None
        self._rx: dict[int, list[re.Pattern[str]]] = {}

    # ------------------------------------------------------------------ links
    def _patterns(self, link: ClauseLink) -> list[re.Pattern[str]]:
        key = id(link)
        if key not in self._rx:
            self._rx[key] = [re.compile(p, re.IGNORECASE) for p in link.match]
        return self._rx[key]

    def linked_clauses(self, template: QuestionTemplate, clauses: Sequence[PolicyClause]) -> list[str]:
        """Ids of the clauses the question bears on, in snapshot order. A fallback link is used only while nothing has been found."""
        found: list[str] = []
        for link in template.clause_links:
            if link.fallback and found:
                continue
            rx = self._patterns(link)
            for c in clauses:
                if c.clause_id in found or c.clause_type not in link.types:
                    continue
                if rx and not any(r.search(f"{c.heading or ''} {c.text}") for r in rx):
                    continue
                found.append(c.clause_id)
        return found

    # ------------------------------------------------------------------ build
    def build(self, header: ClaimHeader, snapshot: PolicySnapshot | NotInForce | None, facts: Sequence[Fact] = (),
              review_clause_ids: Collection[str] = ()) -> RegisterResult:
        policy = snapshot if isinstance(snapshot, PolicySnapshot) else None
        clauses: Sequence[PolicyClause] = policy.clauses if policy else ()
        review = set(review_clause_ids)
        result = RegisterResult(questions=[], origin={})
        skipped: list[str] = []

        for t in self.templates.questions:
            if policy is None and t.needs_policy:
                skipped.append(t.id)
                continue
            q = OpenQuestion(question_id=t.id, text=t.text, category=t.category, status="unverified",
                             establishing_criteria=list(t.establishing_criteria), linked_clause_ids=self.linked_clauses(t, clauses))
            result.questions.append(q)
            result.origin[t.id] = "template"
        if skipped:
            why = snapshot.reason if isinstance(snapshot, NotInForce) else "no policy snapshot was available"
            result.warnings.append(f"no policy applies ({why}): {len(skipped)} question(s) that need the policy were not asked: " + ", ".join(skipped))

        if policy is not None and self.proposer is not None and (clauses or facts):
            outcome = self.proposer.propose(ProposalContext(header, list(facts), list(clauses), list(result.questions)))
            result.rejected.extend(outcome.rejected)
            if outcome.unavailable:
                result.warnings.append(f"claim-specific questions are missing: the model was unavailable ({outcome.detail or 'no detail'}); "
                                       "only the standard questions are listed")
            for p in outcome.accepted:
                result.questions.append(OpenQuestion(question_id=p.question_id, text=p.text, category=p.category,  # type: ignore[arg-type]
                                                     status="unverified", establishing_criteria=list(p.criteria),
                                                     linked_clause_ids=list(p.clause_ids), linked_fact_ids=list(p.fact_ids)))
                result.origin[p.question_id] = "model"
                result.proposed[p.question_id] = p

        for q in result.questions:
            flagged = [c for c in q.linked_clause_ids if c in review]
            if flagged:
                result.review_links[q.question_id] = flagged
        if result.review_links:
            result.warnings.append("REVIEW: some linked clauses were typed by a language model, not by rules: "
                                   + "; ".join(f"{q}: {', '.join(c)}" for q, c in result.review_links.items()))
        return result
