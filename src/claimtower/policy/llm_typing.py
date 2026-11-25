"""Optional model typing of policy clauses the rules could not type (T3.1 follow-up, 4a). Checked, never trusted.

    typer = LLMClauseTyper(llm, pack)                       # llm: core.llm.LLMClient; prompt: packs/<pack>/prompts/clause_typing.md
    decomposer = PolicyDecomposer(pack.policy_structure, typer=typer)

Asked ONLY for clauses that no rule typed; a clause the rules typed is never sent to the model. A model answer is accepted only if ALL hold:
  * the type is one of the seven clause types (not 'unknown', not invented, never a type outside the contract)
  * its `evidence_snippet` is verbatim text of the clause (heading or body), so a person can see what decided it
  * its confidence reaches `min_confidence`; the confidence kept is capped at `confidence_cap`
  * a limit / sublimit / deductible states an amount or points to one (INR 80,000, 'the Sum Insured', 'shown in the Schedule'); a
    clause that merely mentions payment ("payments are rounded") is not a payout rule, whatever the model says
Anything else - an 'unknown', a made-up quote, a low confidence, an unavailable model, unparseable output - leaves the clause UNCLASSIFIED
with the reason. The model can add a type; it can never remove or change one the rules decided.

Typed clauses carry `type_source == "llm"` (and the capped confidence) in the decomposition and the relational store, and the policy
snapshot names them for human review: a model's reading of a limit or an exclusion must be visible to whoever relies on it.
"""
from __future__ import annotations

import re
from typing import Any, get_args

from pydantic import BaseModel, Field

from claimtower.core.config import LLMTypingConfig, Pack
from claimtower.core.schemas.policy import ClauseType
from claimtower.policy.decompose import PolicyDecomposer, TypeVerdict, Unclassified, normalise

ALLOWED: tuple[str, ...] = get_args(ClauseType)


class TypingAnswer(BaseModel):
    clause_type: str
    evidence_snippet: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class LLMClauseTyper:
    def __init__(self, llm: Any, pack: Pack, cfg: LLMTypingConfig | None = None) -> None:
        self.cfg = cfg or pack.policy_structure.llm_typing
        self._llm = llm.derive(reasoning_effort=self.cfg.reasoning_effort) if self.cfg.reasoning_effort and hasattr(llm, "derive") else llm
        self._pack = pack
        self._amount = [re.compile(p, re.IGNORECASE) for p in self.cfg.amount_cues]
        self.asked = 0                                     # how many clauses went to the model (never the rule-typed ones)

    def type_clause(self, clause: Unclassified, document_title: str | None) -> TypeVerdict:
        from claimtower.core.llm import load_prompt        # local: the prompt machinery is only needed when a model is used

        self.asked += 1
        try:
            prompt = load_prompt("clause_typing", self._pack, doc_title=document_title or "(untitled)", section=clause.section or "(none)",
                                 heading=clause.heading, text=clause.text[: self.cfg.max_clause_chars])
            answer = self._llm.structured(TypingAnswer, prompt)
        except Exception as e:                              # model down, bad JSON, TBD model name ...
            return TypeVerdict(None, reason=f"the model was unavailable ({type(e).__name__}: {e})", unavailable=True)
        return self.check(answer, clause)

    def check(self, answer: TypingAnswer, clause: Unclassified) -> TypeVerdict:
        t = answer.clause_type.strip().lower()
        snippet = normalise(answer.evidence_snippet)
        visible = normalise(f"{clause.heading} {clause.text}")
        if t == "unknown":
            return TypeVerdict(None, reason="the model answered unknown")
        if t not in ALLOWED:
            return TypeVerdict(None, reason=f"'{answer.clause_type}' is not an allowed clause type")
        if len(snippet) < 4 or snippet not in visible:
            return TypeVerdict(None, reason="its evidence snippet is not verbatim text of the clause")
        if answer.confidence < self.cfg.min_confidence:
            return TypeVerdict(None, reason=f"confidence {answer.confidence:.2f} is below {self.cfg.min_confidence:.2f}")
        if t in self.cfg.types_needing_amount and not any(rx.search(clause.text) for rx in self._amount):
            return TypeVerdict(None, reason=f"typed '{t}' but the clause states no amount and points to none")
        return TypeVerdict(t, round(min(answer.confidence, self.cfg.confidence_cap), 3))  # type: ignore[arg-type]


def make_decomposer(pack: Pack, llm: Any | None) -> PolicyDecomposer:
    """The decomposer to use wherever policy documents are ingested: the rules, plus the model typing when the pack enables it
    (`llm_typing.enabled`) and a client is given. Without a client the rules run alone and untyped clauses are reported, never guessed."""
    cfg = pack.policy_structure.llm_typing
    typer = LLMClauseTyper(llm, pack, cfg) if cfg.enabled and llm is not None else None
    return PolicyDecomposer(pack.policy_structure, typer=typer)
