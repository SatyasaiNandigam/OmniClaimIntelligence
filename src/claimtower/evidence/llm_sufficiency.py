"""Model judgement of whether facts meet a question's establishing criteria (T4.4). Checked, never trusted.

    judge = LLMCriterionJudge(llm, pack)                 # llm: core.llm.LLMClient; prompt: packs/<pack>/prompts/sufficiency_check.md
    classifier = SufficiencyClassifier(pack.question_templates.sufficiency, judge)

One call per open question, covering the criteria that policy rules did not decide. A verdict is accepted only if ALL hold:
  * its criterion index is one that was asked, and it is one of met / partly / unmet / unclear
  * `met` and `partly` cite at least one of the facts offered for THIS question and quote one of them verbatim (so a person can see what
    the verdict rests on); the model cannot call a criterion met on nothing
  * it has a rationale within the length limit, and a confidence that reaches `min_confidence` (the one kept is capped)
A verdict that fails a check is refused and the criterion becomes `unclear` (the question then abstains); a criterion the model skipped is
`unclear` too. A model that is down leaves the criteria `unclear`: it never makes a question look established.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from claimtower.core.config import Pack, SufficiencyConfig
from claimtower.core.schemas import Fact, OpenQuestion
from claimtower.evidence.sufficiency import VERDICTS, CriterionAssessment, JudgeOutcome, RefusedVerdict


class VerdictItem(BaseModel):
    criterion_index: int
    verdict: str
    fact_ids: list[str] = []
    evidence_quote: str = ""
    rationale: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class VerdictAnswer(BaseModel):
    criteria: list[VerdictItem] = []


def _norm(s: str) -> str:
    return " ".join(s.split())


class LLMCriterionJudge:
    def __init__(self, llm: Any, pack: Pack, cfg: SufficiencyConfig | None = None) -> None:
        self.cfg = cfg or pack.question_templates.sufficiency
        assert self.cfg is not None
        self._llm = llm.derive(reasoning_effort=self.cfg.reasoning_effort) if self.cfg.reasoning_effort and hasattr(llm, "derive") else llm
        self._pack = pack
        self.asked = 0

    def judge(self, question: OpenQuestion, criteria: dict[int, str], facts: Sequence[Fact], directions: dict[str, str],
              sources: dict[str, str] | None = None) -> JudgeOutcome:
        self.asked += 1
        try:
            answer = self._llm.structured(VerdictAnswer, self._prompt(question, criteria, facts, directions, sources or {}))
        except Exception as e:                              # model down, bad JSON, TBD model name ...
            return JudgeOutcome(unavailable=True, detail=f"{type(e).__name__}: {e}")
        return self.check(answer, question, criteria, facts)

    def _prompt(self, question: OpenQuestion, criteria: dict[int, str], facts: Sequence[Fact], directions: dict[str, str],
                sources: dict[str, str]) -> Any:
        from claimtower.core.llm import load_prompt

        cb = "\n".join(f"{i}: {t}" for i, t in criteria.items())
        fb = "\n".join(f"- {f.fact_id} [{directions.get(f.fact_id, 'neutral')}] (from the {sources.get(f.fact_id, 'unknown')} document): "
                       f"{_norm(f.proposition)[:self.cfg.max_fact_chars]}" for f in facts)
        return load_prompt("sufficiency_check", self._pack, question_text=f"{question.question_id}: {question.text}", criteria_block=cb, facts_block=fb)

    def check(self, answer: VerdictAnswer, question: OpenQuestion, criteria: dict[int, str], facts: Sequence[Fact]) -> JudgeOutcome:
        cfg = self.cfg
        by_id = {f.fact_id: f for f in facts}
        out = JudgeOutcome()
        for item in sorted(answer.criteria, key=lambda i: -i.confidence):
            i, v = item.criterion_index, item.verdict.strip().lower()

            def refuse(why: str, idx: int | None = i) -> None:
                out.refused.append(RefusedVerdict(question.question_id, idx, why))

            if i not in criteria:
                refuse("it judges a criterion that was not asked")
            elif i in out.verdicts:
                refuse("the criterion was already judged with a higher confidence")
            elif v not in VERDICTS:
                refuse(f"'{item.verdict}' is not an allowed verdict")
            elif item.confidence < cfg.min_confidence:
                refuse(f"confidence {item.confidence:.2f} is below {cfg.min_confidence:.2f}")
            elif not _norm(item.rationale) or len(_norm(item.rationale)) > cfg.max_rationale_chars:
                refuse(f"it needs a rationale of at most {cfg.max_rationale_chars} characters")
            elif v in ("met", "partly") and not self._grounded(item, by_id):
                refuse("it cites no offered fact, or its quote is not verbatim text of a cited fact")
            else:
                cited = tuple(dict.fromkeys(item.fact_ids)) if v in ("met", "partly") else ()
                out.verdicts[i] = CriterionAssessment(i, criteria[i], v, "model", _norm(item.rationale), cited,  # type: ignore[arg-type]
                                                      _norm(item.evidence_quote) if cited else "", round(min(item.confidence, cfg.confidence_cap), 3))
        return out

    @staticmethod
    def _grounded(item: VerdictItem, by_id: dict[str, Fact]) -> bool:
        if not item.fact_ids or any(f not in by_id for f in item.fact_ids):
            return False
        quote = _norm(item.evidence_quote)
        return len(quote) >= 4 and any(quote in _norm(f"{by_id[f].proposition} {by_id[f].evidence_text}") for f in item.fact_ids)


def make_classifier(pack: Pack, llm: Any | None):
    """The classifier to use for a claim: the policy rules and decision table, plus the model's criterion judgements when enabled and a client is given."""
    from claimtower.evidence.sufficiency import SufficiencyClassifier

    cfg = pack.question_templates.sufficiency
    if cfg is None:
        raise ValueError("the pack has no `sufficiency` section in question_templates.yaml")
    return SufficiencyClassifier(cfg, LLMCriterionJudge(llm, pack, cfg) if cfg.enabled and llm is not None else None)


__all__ = ["LLMCriterionJudge", "VerdictAnswer", "VerdictItem", "make_classifier"]
