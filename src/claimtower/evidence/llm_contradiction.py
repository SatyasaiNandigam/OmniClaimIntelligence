"""Model detection of narrative conflicts between facts (T4.3). Checked, never trusted.

    judge = LLMConflictJudge(llm, pack)                  # llm: core.llm.LLMClient; prompt: packs/<pack>/prompts/contradiction_check.md
    detector = ContradictionDetector(pack.question_templates.contradictions, judge)

Asked once per open question (of the configured categories) that has at least two linked facts. A proposed conflict is accepted only if ALL hold:
  * each side names at least one fact, every id is one of the facts offered for THAT question, and no fact is on both sides
  * its `kind` is one of date / amount / cause / extent / other, its `severity` one of low / medium / high
  * `quote_a` is verbatim text of a side-A fact and `quote_b` of a side-B fact (so a person can see the two statements that clash)
  * it has an explanation within the length limit and a confidence that reaches `min_confidence` (the one kept is capped)
  * it is not a repeat of a conflict already accepted for the question, and at most `max_per_question` are kept (most confident first)
Anything else is refused with its reason. The model never changes a rule contradiction and cannot raise one without two cited sides.
"""
from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

from pydantic import BaseModel, Field

from claimtower.core.config import ContradictionConfig, Pack
from claimtower.core.schemas import Contradiction, Fact, OpenQuestion
from claimtower.evidence.contradiction import KINDS, SEVERITIES, ConflictOutcome, ModelConflict, RefusedConflict, Retraction


class ConflictItem(BaseModel):
    fact_ids_a: list[str] = []
    fact_ids_b: list[str] = []
    kind: str = "other"
    severity: str = "medium"
    quote_a: str = ""
    quote_b: str = ""
    explanation: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class ResolvedItem(BaseModel):
    contradiction_id: str
    resolving_fact_ids: list[str] = []
    evidence_quote: str = ""
    explanation: str = ""


class ConflictAnswer(BaseModel):
    conflicts: list[ConflictItem] = []


class ConflictAnswerWithResolved(ConflictAnswer):
    """The answer when EARLIER conflicts were shown to the model: it may also say which of them new facts resolve (a separate class so that the
    first assessment of a question keeps exactly the schema, and so the recorded answers, it always had)."""

    resolved: list[ResolvedItem] = []


def _norm(s: str) -> str:
    return " ".join(s.split())


class LLMConflictJudge:
    def __init__(self, llm: Any, pack: Pack, cfg: ContradictionConfig | None = None) -> None:
        self.cfg = cfg or pack.question_templates.contradictions
        self._llm = llm.derive(reasoning_effort=self.cfg.reasoning_effort) if self.cfg.reasoning_effort and hasattr(llm, "derive") else llm
        self._pack = pack
        self.asked = 0

    def judge(self, question: OpenQuestion, facts: Sequence[Fact], earlier: Sequence[Contradiction] = (), new_fact_ids: Collection[str] = ()) -> ConflictOutcome:
        self.asked += 1
        try:
            answer = self._llm.structured(ConflictAnswerWithResolved if earlier else ConflictAnswer, self._prompt(question, facts, earlier, new_fact_ids))
        except Exception as e:                              # model down, bad JSON, TBD model name ...
            return ConflictOutcome(unavailable=True, detail=f"{type(e).__name__}: {e}")
        out = self.check(answer, question, facts)
        out.retractions = self._retractions(answer, earlier, facts, new_fact_ids)
        return out

    def _retractions(self, answer: ConflictAnswer, earlier: Sequence[Contradiction], facts: Sequence[Fact], new_fact_ids: Collection[str]) -> list[Retraction]:
        """An earlier conflict is retracted only if the model names it, cites a NEW fact that is on file, quotes that fact verbatim and explains."""
        known = {c.contradiction_id for c in earlier}
        by_id = {f.fact_id: f for f in facts}
        out = []
        for r in getattr(answer, "resolved", []):
            ids = [i for i in dict.fromkeys(r.resolving_fact_ids) if i in by_id]
            quote = _norm(r.evidence_quote)
            if (r.contradiction_id in known and ids and set(ids) & set(new_fact_ids) and len(quote) >= 4 and _norm(r.explanation)
                    and len(_norm(r.explanation)) <= self.cfg.max_explanation_chars
                    and any(quote in _norm(f"{by_id[i].proposition} {by_id[i].evidence_text}") for i in ids)):
                out.append(Retraction(r.contradiction_id, tuple(ids), quote, _norm(r.explanation)))
        return out

    def _prompt(self, question: OpenQuestion, facts: Sequence[Fact], earlier: Sequence[Contradiction] = (), new_fact_ids: Collection[str] = ()) -> Any:
        from dataclasses import replace

        from claimtower.core.llm import load_prompt

        c = self.cfg
        qb = f"{question.question_id}: {question.text} (established by: {'; '.join(question.establishing_criteria) or 'n/a'})"
        fb = "\n".join(f"- {f.fact_id}: {_norm(f.proposition)[:c.max_fact_chars]}" for f in facts)
        prompt = load_prompt("contradiction_check", self._pack, question_block=qb, facts_block=fb, max_conflicts=c.max_per_question,
                             rubric_block="\n".join(f"- {r}" for r in c.severity_rubric) or "- (none)", allowed_kinds=", ".join(KINDS))
        if not earlier:
            return prompt                                    # unchanged text: the first assessment of a question is the same prompt as before
        shown = "\n".join(f"- {e.contradiction_id}: {_norm(e.explanation)[:c.max_explanation_chars]} (sides: {', '.join(e.fact_ids_a)} against {', '.join(e.fact_ids_b)})"
                          for e in earlier)
        extra = ("\n\nEarlier conflicts on this question (found before the newest facts arrived):\n" + shown +
                 "\nNew facts since then: " + (", ".join(sorted(new_fact_ids)) or "none") +
                 "\nStill report any of them you consider a conflict only if you see a NEW one. For an earlier conflict that a NEW fact resolves "
                 "(for example a later report that explains the corrosion away), list it under `resolved` with its contradiction_id, resolving_fact_ids (new facts only), "
                 "an evidence_quote copied exactly from one of them, and an explanation. Do NOT resolve a conflict merely because you were not asked about it again: "
                 "an earlier conflict stays open unless a new fact resolves it.")
        return replace(prompt, text=prompt.text + extra)

    def check(self, answer: ConflictAnswer, question: OpenQuestion, facts: Sequence[Fact]) -> ConflictOutcome:
        cfg = self.cfg
        by_id = {f.fact_id: f for f in facts}
        out = ConflictOutcome()
        seen: set[tuple[frozenset[str], frozenset[str]]] = set()
        for item in sorted(answer.conflicts, key=lambda i: -i.confidence):
            why = self._problem(item, by_id, seen)
            if why:
                out.refused.append(RefusedConflict(question.question_id, _norm(item.explanation)[:200], why))
                continue
            if len(out.conflicts) >= cfg.max_per_question:
                out.refused.append(RefusedConflict(question.question_id, _norm(item.explanation)[:200], f"over the limit of {cfg.max_per_question} per question"))
                continue
            a, b = tuple(dict.fromkeys(item.fact_ids_a)), tuple(dict.fromkeys(item.fact_ids_b))
            seen.add((frozenset(a), frozenset(b)))
            out.conflicts.append(ModelConflict(question.question_id, a, b, item.kind.strip().lower(), item.severity.strip().lower(), _norm(item.explanation),
                                               _norm(item.quote_a), _norm(item.quote_b), round(min(item.confidence, cfg.confidence_cap), 3)))
        return out

    def _problem(self, item: ConflictItem, by_id: dict[str, Fact], seen: set[tuple[frozenset[str], frozenset[str]]]) -> str | None:
        cfg = self.cfg
        a, b = set(item.fact_ids_a), set(item.fact_ids_b)
        if not a or not b:
            return "it does not cite at least one fact on each side"
        if a & b:
            return "a fact is on both sides"
        unknown = sorted((a | b) - set(by_id))
        if unknown:
            return "it cites facts that were not offered for this question: " + ", ".join(unknown)
        if item.kind.strip().lower() not in KINDS:
            return f"'{item.kind}' is not an allowed kind"
        if item.severity.strip().lower() not in SEVERITIES:
            return f"'{item.severity}' is not an allowed severity"
        if not self._quoted(item.quote_a, a, by_id) or not self._quoted(item.quote_b, b, by_id):
            return "a quote is not verbatim text of a fact on its side"
        expl = _norm(item.explanation)
        if not expl or len(expl) > cfg.max_explanation_chars:
            return f"it needs an explanation of at most {cfg.max_explanation_chars} characters"
        if (frozenset(a), frozenset(b)) in seen or (frozenset(b), frozenset(a)) in seen:
            return "it repeats a conflict already accepted"
        if item.confidence < cfg.min_confidence:
            return f"confidence {item.confidence:.2f} is below {cfg.min_confidence:.2f}"
        return None

    @staticmethod
    def _quoted(quote: str, ids: set[str], by_id: dict[str, Fact]) -> bool:
        q = _norm(quote)
        return len(q) >= 4 and any(q in _norm(f"{by_id[i].proposition} {by_id[i].evidence_text}") for i in ids)


def make_detector(pack: Pack, llm: Any | None):
    """The detector to use for a claim: the rules, plus the model's narrative conflicts when enabled and a client is given."""
    from claimtower.evidence.contradiction import ContradictionDetector

    cfg = pack.question_templates.contradictions
    return ContradictionDetector(cfg, LLMConflictJudge(llm, pack, cfg) if cfg.enabled and llm is not None else None)


__all__ = ["ConflictAnswer", "ConflictAnswerWithResolved", "ConflictItem", "LLMConflictJudge", "make_detector"]
