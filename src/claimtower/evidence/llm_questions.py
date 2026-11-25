"""Claim-specific open questions proposed by a model (T4.1). Checked, never trusted.

    proposer = LLMQuestionProposer(llm, pack)             # llm: core.llm.LLMClient; prompt: packs/<pack>/prompts/question_proposal.md
    register = QuestionRegister(pack.question_templates, proposer)

The model sees the claim header, the facts, the clauses in force and the standard questions, and may ADD questions the standard ones cannot
know to ask. A proposal is accepted only if ALL hold (each refusal is recorded with its reason, nothing is dropped silently):
  * a clear question (not empty, within max_text_chars) that does not pre-judge the decision (forbidden_patterns: approve / deny / pay ...)
  * its category is one of the contract's categories
  * it cites at least one clause or fact, and every cited id exists in this claim's input
  * its `evidence_quote` is verbatim text of a cited clause or fact (so a person can see what prompted the question)
  * it has 1..max_criteria establishing criteria (a question with no way to be answered cannot be judged later)
  * it is not a repeat: its content words are not already covered by an existing or earlier-accepted question (duplicate_overlap)
  * its confidence reaches min_confidence; the confidence kept is capped at confidence_cap
  * at most max_questions are kept (highest confidence first)
The model can add questions; it can never remove or change a template question. If it is unavailable the register is just the templates.
Question ids are `Q-M-<hash of the text>`: the same question gets the same id on every run, so a later re-assessment can match it.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any, get_args

from pydantic import BaseModel, Field

from claimtower.core.config import ModelQuestionsConfig, Pack
from claimtower.core.schemas.common import QuestionCategory
from claimtower.evidence.questions import (
    ProposalContext, ProposalOutcome, ProposedQuestion, RejectedProposal, overlap,
)

CATEGORIES: tuple[str, ...] = get_args(QuestionCategory)
MAX_FACTS = 40                                           # facts shown to the model (the rest are summarised as a count)


class ProposedItem(BaseModel):
    text: str
    category: str
    clause_ids: list[str] = []
    fact_ids: list[str] = []
    evidence_quote: str = ""
    establishing_criteria: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)


class ProposalAnswer(BaseModel):
    questions: list[ProposedItem] = []


class DuplicateVerdict(BaseModel):
    verdict: str                                         # "duplicate" | "new"
    duplicate_of: str = "none"
    reason: str = ""


def _norm(s: str) -> str:
    return " ".join(s.split())


def question_id(text: str) -> str:
    return "Q-M-" + hashlib.sha1(_norm(text).lower().encode("utf-8")).hexdigest()[:8].upper()


class LLMQuestionProposer:
    def __init__(self, llm: Any, pack: Pack, cfg: ModelQuestionsConfig | None = None) -> None:
        self.cfg = cfg or pack.question_templates.model_questions
        self._llm = llm.derive(reasoning_effort=self.cfg.reasoning_effort) if self.cfg.reasoning_effort and hasattr(llm, "derive") else llm
        self._pack = pack
        self._forbidden = [re.compile(p, re.IGNORECASE) for p in self.cfg.forbidden_patterns]
        self.asked = 0

    # ------------------------------------------------------------------ prompt
    def _prompt(self, ctx: ProposalContext) -> Any:
        from claimtower.core.llm import load_prompt

        h, c = ctx.header, self.cfg
        claim = "\n".join([f"loss date: {h.loss_date or 'not stated'}", f"loss type: {h.loss_type or 'not stated'}",
                           f"description: {h.loss_description or 'not stated'}"])
        existing = "\n".join(f"- {q.question_id}: {q.text} (established by: {'; '.join(q.establishing_criteria) or 'n/a'})"
                             for q in ctx.existing) or "(none)"
        facts = [f"- {f.fact_id}: {_norm(f.proposition)[:c.max_fact_chars]}" for f in ctx.facts[:MAX_FACTS]]
        if len(ctx.facts) > MAX_FACTS:
            facts.append(f"(and {len(ctx.facts) - MAX_FACTS} more facts not shown)")
        clauses = [f"- {x.clause_id} [{x.clause_type}] {x.heading or ''}: {_norm(x.text)[:c.max_clause_chars]}" for x in ctx.clauses]
        return load_prompt("question_proposal", self._pack, max_questions=c.max_questions, allowed_categories=", ".join(CATEGORIES),
                           claim_block=claim, existing_block=existing, facts_block="\n".join(facts) or "(none)",
                           clauses_block="\n".join(clauses) or "(none)")

    def propose(self, ctx: ProposalContext) -> ProposalOutcome:
        self.asked += 1
        try:
            answer = self._llm.structured(ProposalAnswer, self._prompt(ctx))
        except Exception as e:                           # model down, bad JSON, TBD model name ...
            return ProposalOutcome(unavailable=True, detail=f"{type(e).__name__}: {e}")
        out = self.check(answer, ctx)
        return self._second_look(out, ctx) if self.cfg.duplicate_check else out

    def _second_look(self, out: ProposalOutcome, ctx: ProposalContext) -> ProposalOutcome:
        """A word-overlap test cannot see that 'are there earlier repair records?' IS the standard question about when the damage began, so each
        surviving proposal is put to the model again, next to the standard questions and what establishes each. 'duplicate' of a real standard
        question refuses it. A proposal the model cannot be asked about is refused too: it is not accepted unchecked."""
        from claimtower.core.llm import load_prompt

        ids = {q.question_id for q in ctx.existing}
        standard = "\n".join(f"- {q.question_id}: {q.text} (established by: {'; '.join(q.establishing_criteria) or 'n/a'})" for q in ctx.existing)
        kept: list[ProposedQuestion] = []
        for p in out.accepted:
            try:
                v = self._llm.structured(DuplicateVerdict, load_prompt("question_duplicate_check", self._pack, standard_block=standard, proposed=p.text))
            except Exception as e:
                out.rejected.append(RejectedProposal(p.text, f"it could not be checked against the standard questions ({type(e).__name__}: {e})"))
                out.unavailable = True
                out.detail = out.detail or f"{type(e).__name__}: {e}"
                continue
            verdict, dup = v.verdict.strip().lower(), v.duplicate_of.strip()
            if verdict == "duplicate":                       # a duplicate of something unnamed is refused too: when unsure, do not add
                out.rejected.append(RejectedProposal(p.text, f"it repeats standard question {dup if dup in ids else '(unnamed)'}: {v.reason.strip()}"))
            elif verdict == "new":
                kept.append(p)
            else:
                out.rejected.append(RejectedProposal(p.text, f"the duplicate check gave no usable verdict ({v.verdict!r})"))
        out.accepted = kept
        return out

    # ------------------------------------------------------------------ checks
    def check(self, answer: ProposalAnswer, ctx: ProposalContext) -> ProposalOutcome:
        out = ProposalOutcome()
        known_text = [q.text for q in ctx.existing]
        for item in sorted(answer.questions, key=lambda i: -i.confidence):
            problem = self._problem(item, ctx, known_text)
            if problem:
                out.rejected.append(RejectedProposal(item.text, problem))
                continue
            if len(out.accepted) >= self.cfg.max_questions:
                out.rejected.append(RejectedProposal(item.text, f"over the limit of {self.cfg.max_questions} claim-specific questions"))
                continue
            text = _norm(item.text)
            out.accepted.append(ProposedQuestion(
                question_id=question_id(text), text=text, category=item.category.strip().lower(),
                clause_ids=tuple(dict.fromkeys(item.clause_ids)), fact_ids=tuple(dict.fromkeys(item.fact_ids)),
                evidence_quote=_norm(item.evidence_quote), criteria=tuple(_norm(c) for c in item.establishing_criteria),
                confidence=round(min(item.confidence, self.cfg.confidence_cap), 3)))
            known_text.append(text)
        return out

    def _problem(self, item: ProposedItem, ctx: ProposalContext, known_text: Sequence[str]) -> str | None:
        cfg, text = self.cfg, _norm(item.text)
        if len(text) < 10:
            return "the question is empty or too short"
        if not text.endswith("?"):
            return "it is not phrased as a question (a request or instruction is not an open question)"
        if len(text) > cfg.max_text_chars:
            return f"the question is longer than {cfg.max_text_chars} characters"
        if any(rx.search(text) for rx in self._forbidden):
            return "the question pre-judges the decision (approve / deny / pay) instead of asking what is known"
        if item.category.strip().lower() not in CATEGORIES:
            return f"'{item.category}' is not an allowed category"
        if not item.clause_ids and not item.fact_ids:
            return "it cites no clause and no fact"
        clauses = {c.clause_id: c for c in ctx.clauses}
        facts = {f.fact_id: f for f in ctx.facts}
        unknown = [i for i in item.clause_ids if i not in clauses] + [i for i in item.fact_ids if i not in facts]
        if unknown:
            return "it cites ids that do not exist: " + ", ".join(unknown)
        quote = _norm(item.evidence_quote)
        visible = [_norm(f"{clauses[i].heading or ''} {clauses[i].text}") for i in item.clause_ids]
        visible += [_norm(f"{facts[i].proposition} {facts[i].evidence_text}") for i in item.fact_ids]
        if len(quote) < 4 or not any(quote in v for v in visible):
            return "its evidence quote is not verbatim text of a cited clause or fact"
        crit = [_norm(c) for c in item.establishing_criteria if _norm(c)]
        if not 1 <= len(crit) <= cfg.max_criteria:
            return f"it needs 1 to {cfg.max_criteria} establishing criteria"
        if any(len(c) > 200 for c in crit):
            return "an establishing criterion is longer than 200 characters"
        repeat = next((t for t in known_text if overlap(text, t) >= cfg.duplicate_overlap), None)
        if repeat is not None:
            return f"it repeats an existing question ({repeat!r})"
        if item.confidence < cfg.min_confidence:
            return f"confidence {item.confidence:.2f} is below {cfg.min_confidence:.2f}"
        return None


def make_register(pack: Pack, llm: Any | None):
    """The register to use for a claim: the template questions, plus the model's claim-specific ones when the pack enables them and a client is given."""
    from claimtower.evidence.questions import QuestionRegister

    cfg = pack.question_templates.model_questions
    proposer = LLMQuestionProposer(llm, pack, cfg) if cfg.enabled and llm is not None else None
    return QuestionRegister(pack.question_templates, proposer)


__all__ = ["CATEGORIES", "LLMQuestionProposer", "ProposalAnswer", "ProposedItem", "make_register", "question_id"]
