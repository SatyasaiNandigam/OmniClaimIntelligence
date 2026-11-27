"""Model reading of the facts against the clauses of one coverage issue (T5.1). Checked, never trusted.

    judge = LLMReadingJudge(llm, pack)                  # llm: core.llm.LLMClient; prompt: packs/<pack>/prompts/coverage_reading.md
    builder = IssueMatrixBuilder(pack.coverage, judge)

Asked only for cover and exclusion rows whose controlling questions are settled (the table decides everything else without a model). The model says which way
the facts point under the clause: supports (the facts do not point against the claim as presented), against, or unclear. An answer is accepted only if ALL hold:
  * the reading is one of supports / against / unclear
  * supports and against cite at least one offered clause and one offered fact, and quote a cited fact verbatim
  * the confidence reaches `min_confidence` (the one kept is capped at `confidence_cap`)
Anything else is refused and the reading is `unclear`, so the row abstains: the model can never push a row to `established` on an invented quote. A rationale that
states a coverage conclusion is dropped (the reading stays, the note shows the table's plain text instead). An adverse reading is mapped by the table to
`unresolved` or `potentially_relevant`, never to a verdict.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from claimtower.core.config import CoverageConfig, Pack
from claimtower.coverage.issue_matrix import IssueContext, Reading, ReadingOutcome

KIND_TEXT = {"cover": "an insuring clause the claim must satisfy", "exclusion": "an exclusion that may take cover away", "condition": "a condition of the policy",
             "amount": "a limit"}
SUPPORTS = {"cover": "the condition appears satisfied", "exclusion": "the exclusion does not appear to be engaged"}
AGAINST = {"cover": "the condition appears not to be satisfied", "exclusion": "the exclusion appears to be engaged"}


class ReadingAnswer(BaseModel):
    reading: str
    clause_ids: list[str] = []
    fact_ids: list[str] = []
    evidence_quote: str = ""
    rationale: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class AmbiguousReading(BaseModel):
    interpretation: str
    clause_quote: str
    points: str


class AmbiguityAnswer(BaseModel):
    ambiguous: bool
    readings: list[AmbiguousReading] = []
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class CauseCheck(BaseModel):
    states_cause: str
    what_it_states: str = ""


def _norm(s: str) -> str:
    return " ".join(s.split())


class LLMReadingJudge:
    def __init__(self, llm: Any, pack: Pack, cfg: CoverageConfig | None = None) -> None:
        self.cfg = cfg or pack.coverage
        assert self.cfg is not None
        self.rc = self.cfg.reading
        self._llm = llm.derive(reasoning_effort=self.rc.reasoning_effort) if self.rc.reasoning_effort and hasattr(llm, "derive") else llm
        self._pack = pack
        self._forbidden = [re.compile(p, re.IGNORECASE) for p in self.cfg.forbidden_patterns]
        self.asked = 0

    def read(self, ctx: IssueContext) -> ReadingOutcome:
        from claimtower.core.llm import load_prompt

        self.asked += 1
        k = ctx.spec.kind
        clauses = "\n".join(f"- {c.clause_id} [{c.clause_type}] {c.heading or ''}: {_norm(c.text)[:self.rc.max_clause_chars]}" for c in ctx.clauses)
        facts = "\n".join(f"- {f.fact_id} ({f.source_document_id}): {_norm(f.proposition)[:self.rc.max_fact_chars]}" for f in ctx.facts)
        qs = "\n".join(f"- {q.question_id} ({q.status.replace('_', ' ')}): {q.text}" for q in ctx.questions)
        if ctx.contradictions:
            qs += "\nConflicts on file: " + " | ".join(_norm(c.explanation)[:200] for c in ctx.contradictions)
        if self.rc.check_ambiguity and self.has_exception(ctx):
            amb = self._ambiguity(ctx, clauses, facts)
            if amb is not None:
                return amb
        try:
            prompt = load_prompt("coverage_reading", self._pack, issue=ctx.spec.issue, kind_text=KIND_TEXT.get(k, k), clauses_block=clauses, facts_block=facts,
                                 questions_block=qs, supports_text=SUPPORTS.get(k, "the claim as presented stands"), against_text=AGAINST.get(k, "it does not"))
            answer = self._llm.structured(ReadingAnswer, prompt)
        except Exception as e:                              # model down, bad JSON, TBD model name ...
            return ReadingOutcome(unavailable=True, detail=f"{type(e).__name__}: {e}")
        out = self.check(answer, ctx)
        if out.reading.value == "favourable" and self.rc.verify_supports:      # a favourable reading must rest on a quote that states the cause
            try:
                chk = self._llm.structured(CauseCheck, load_prompt("coverage_cause_check", self._pack, quote=out.reading.quote))
            except Exception as e:
                return ReadingOutcome(unavailable=True, detail=f"cause check {type(e).__name__}: {e}")
            if chk.states_cause.strip().lower() != "yes":
                return self._refuse("its quote does not itself state how or when the water escaped")
        return out

    def has_exception(self, ctx: IssueContext) -> bool:
        """The rule that decides whether the ambiguity question is worth asking: some offered clause of a trigger type carries exception wording."""
        rx = [re.compile(p, re.IGNORECASE) for p in self.rc.ambiguity_trigger_patterns]
        return any(c.clause_type in self.rc.ambiguity_trigger_types and any(r.search(c.text) for r in rx) for c in ctx.clauses)

    def _ambiguity(self, ctx: IssueContext, clauses: str, facts: str) -> ReadingOutcome | None:
        """T5.2: do the clauses admit two reasonable readings that point opposite ways for these facts? Returns the `ambiguous` outcome, or None to go on to the reading."""
        from claimtower.core.llm import load_prompt

        try:
            prompt = load_prompt("coverage_ambiguity", self._pack, issue=ctx.spec.issue, kind_text=KIND_TEXT.get(ctx.spec.kind, ctx.spec.kind), clauses_block=clauses,
                                 facts_block=facts)
            a = self._llm.structured(AmbiguityAnswer, prompt)
        except Exception as e:
            return ReadingOutcome(unavailable=True, detail=f"ambiguity check {type(e).__name__}: {e}")
        return self.check_ambiguity(a, ctx)

    def check_ambiguity(self, a: AmbiguityAnswer, ctx: IssueContext) -> ReadingOutcome | None:
        if not a.ambiguous or a.confidence < self.rc.min_confidence or len(a.readings) < 2:
            return None
        texts = {c.clause_id: _norm(f"{c.heading or ''} {c.text}").replace(":", "") for c in ctx.clauses}      # the model may include the heading it was shown
        found, alts, quoted = [], [], []
        for r in a.readings:
            quote = _norm(r.clause_quote).replace(":", "")
            cid = next((i for i, t in texts.items() if len(quote) >= 4 and quote in t), None)
            if cid is None or r.points.strip().lower() not in ("favourable", "adverse"):
                return None                                  # a reading not grounded in the wording, or with no direction: not accepted as an ambiguity
            why = _norm(r.interpretation)
            if not why or len(why) > self.rc.max_rationale_chars or any(rx.search(why) for rx in self._forbidden):
                why = "a reading of the clause words quoted here"          # wording that states a conclusion is not shown
            found.append((cid, r.points.strip().lower()))
            quoted.append(quote)
            alts.append(f"{why} (\"{quote}\")")
        quotes = {p: {q for (_, pp), q in zip(found, quoted, strict=True) if pp == p} for p in ("favourable", "adverse")}
        if quotes["favourable"] == quotes["adverse"]:
            return None                                      # both readings rest on the very same words: that is a re-reading of the facts, not an ambiguity of the wording
        if {p for _, p in found} != {"favourable", "adverse"}:
            return None                                      # all readings point the same way: nothing for a person to decide between
        return ReadingOutcome(Reading("ambiguous", (), tuple(dict.fromkeys(c for c, _ in found)), "", "", round(min(a.confidence, self.rc.confidence_cap), 3), tuple(alts)))

    def check(self, a: ReadingAnswer, ctx: IssueContext) -> ReadingOutcome:
        rc = self.rc
        r = a.reading.strip().lower()
        if r == "unclear":
            return ReadingOutcome(Reading("unclear", rationale=""))
        if r not in ("supports", "against"):
            return self._refuse(f"'{a.reading}' is not an allowed reading")
        clauses = {c.clause_id for c in ctx.clauses}
        facts = {f.fact_id: f for f in ctx.facts}
        if not a.clause_ids or not a.fact_ids or any(c not in clauses for c in a.clause_ids) or any(i not in facts for i in a.fact_ids):
            return self._refuse("it cites no clause or fact, or one that was not offered")
        quote = _norm(a.evidence_quote)
        if len(quote) < 4 or not any(quote in _norm(f"{facts[i].proposition} {facts[i].evidence_text}") for i in a.fact_ids):
            return self._refuse("its quote is not verbatim text of a cited fact")
        if a.confidence < rc.min_confidence:
            return self._refuse(f"confidence {a.confidence:.2f} is below {rc.min_confidence:.2f}")
        why = _norm(a.rationale)
        if not why or len(why) > rc.max_rationale_chars or any(rx.search(why) for rx in self._forbidden):
            why = ""                                         # the reading stands on its quote; the table's plain note is shown instead
        return ReadingOutcome(Reading("favourable" if r == "supports" else "adverse", tuple(dict.fromkeys(a.fact_ids)), tuple(dict.fromkeys(a.clause_ids)),
                                      quote, why, round(min(a.confidence, rc.confidence_cap), 3)))

    @staticmethod
    def _refuse(why: str) -> ReadingOutcome:
        return ReadingOutcome(Reading("unclear"), refused=why)


def make_builder(pack: Pack, llm: Any | None, min_citations: int = 1):
    """The matrix builder for a pack: the decision table, plus the model's reading when the pack enables it and a client is given."""
    from claimtower.coverage.issue_matrix import IssueMatrixBuilder

    cfg = pack.coverage
    if cfg is None:
        raise ValueError("the pack has no coverage_issues.yaml")
    return IssueMatrixBuilder(cfg, LLMReadingJudge(llm, pack, cfg) if cfg.reading.enabled and llm is not None else None, min_citations)


__all__ = ["LLMReadingJudge", "ReadingAnswer", "make_builder"]
