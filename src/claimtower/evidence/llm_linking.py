"""Model judgement of fact <-> question links (T4.2). Checked, never trusted.

    judge = LLMLinkJudge(llm, pack)                      # llm: core.llm.LLMClient; prompt: packs/<pack>/prompts/fact_linking.md
    linker = FactQuestionLinker(pack.question_templates.fact_links, judge)

The model sees the open questions (with what would establish each), the facts and the questions each fact is tagged with, and returns for
each pair a direction (supporting / contradicting / neutral) or `unrelated`. A judgement is accepted only if ALL hold:
  * the fact and the question are real (the fact in this batch, the question in the register); the direction is one of the four allowed
  * unless `unrelated`: its `evidence_quote` is verbatim text of the fact (proposition or source text), and it has a short rationale
  * its confidence reaches `min_confidence` (the one kept is capped at `confidence_cap`)
  * a pair is judged once (the most confident wins) and a fact keeps at most `max_links_per_fact` links
Anything else is refused with its reason and the tagged link stays neutral. Facts are sent in batches of `batch_size`.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from claimtower.core.config import FactLinkConfig, Pack
from claimtower.core.schemas import Fact, OpenQuestion
from claimtower.evidence.linker import DIRECTIONS, JudgeOutcome, LinkVerdict, RefusedJudgement

ALLOWED: tuple[str, ...] = (*DIRECTIONS, "unrelated")


class LinkItem(BaseModel):
    fact_id: str
    question_id: str
    direction: str
    evidence_quote: str = ""
    rationale: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class LinkAnswer(BaseModel):
    links: list[LinkItem] = []


def _norm(s: str) -> str:
    return " ".join(s.split())


class LLMLinkJudge:
    def __init__(self, llm: Any, pack: Pack, cfg: FactLinkConfig | None = None) -> None:
        self.cfg = cfg or pack.question_templates.fact_links
        self._llm = llm.derive(reasoning_effort=self.cfg.reasoning_effort) if self.cfg.reasoning_effort and hasattr(llm, "derive") else llm
        self._pack = pack
        self.asked = 0

    def judge(self, facts: Sequence[Fact], questions: Sequence[OpenQuestion], tagged: dict[str, list[str]]) -> JudgeOutcome:
        out = JudgeOutcome()
        for i in range(0, len(facts), self.cfg.batch_size):
            batch = facts[i:i + self.cfg.batch_size]
            self.asked += 1
            try:
                answer = self._llm.structured(LinkAnswer, self._prompt(batch, questions, tagged))
            except Exception as e:                          # model down, bad JSON, TBD model name ...
                out.unavailable, out.detail = True, out.detail or f"{type(e).__name__}: {e}"
                continue
            verdicts, refused = self.check(answer, batch, questions)
            out.verdicts += verdicts
            out.refused += refused
        return out

    def _prompt(self, facts: Sequence[Fact], questions: Sequence[OpenQuestion], tagged: dict[str, list[str]]) -> Any:
        from claimtower.core.llm import load_prompt

        qs = "\n".join(f"- {q.question_id}: {q.text} (established by: {'; '.join(q.establishing_criteria) or 'n/a'})" for q in questions)
        fs = "\n".join(f"- {f.fact_id}: {_norm(f.proposition)[:self.cfg.max_fact_chars]} | suggested questions: {', '.join(tagged.get(f.fact_id, [])) or 'none'}"
                       for f in facts)
        return load_prompt("fact_linking", self._pack, max_links=self.cfg.max_links_per_fact, questions_block=qs, facts_block=fs)

    def check(self, answer: LinkAnswer, facts: Sequence[Fact], questions: Sequence[OpenQuestion]) -> tuple[list[LinkVerdict], list[RefusedJudgement]]:
        cfg = self.cfg
        by_fact = {f.fact_id: f for f in facts}
        qids = {q.question_id for q in questions}
        good: dict[tuple[str, str], LinkVerdict] = {}
        refused: list[RefusedJudgement] = []

        def refuse(item: LinkItem, why: str) -> None:
            refused.append(RefusedJudgement(item.fact_id, item.question_id, why))

        for item in sorted(answer.links, key=lambda i: -i.confidence):
            d = item.direction.strip().lower()
            fact = by_fact.get(item.fact_id)
            if fact is None or item.question_id not in qids:
                refuse(item, "it names a fact or question that does not exist")
            elif d not in ALLOWED:
                refuse(item, f"'{item.direction}' is not an allowed direction")
            elif (item.fact_id, item.question_id) in good:
                refuse(item, "the pair was already judged with a higher confidence")
            elif item.confidence < cfg.min_confidence:
                refuse(item, f"confidence {item.confidence:.2f} is below {cfg.min_confidence:.2f}")
            elif not _norm(item.rationale) or len(_norm(item.rationale)) > cfg.max_rationale_chars:
                refuse(item, f"it needs a rationale of at most {cfg.max_rationale_chars} characters")
            elif d != "unrelated" and not self._quoted(item, fact):
                refuse(item, "its evidence quote is not verbatim text of the fact")
            else:
                good[(item.fact_id, item.question_id)] = LinkVerdict(
                    item.fact_id, item.question_id, d, _norm(item.rationale), round(min(item.confidence, cfg.confidence_cap), 3),  # type: ignore[arg-type]
                    "" if d == "unrelated" else _norm(item.evidence_quote))
        per_fact: dict[str, int] = {}
        kept: list[LinkVerdict] = []
        for v in good.values():                              # most confident first (dict keeps the sorted insertion order)
            if v.direction != "unrelated":
                per_fact[v.fact_id] = per_fact.get(v.fact_id, 0) + 1
                if per_fact[v.fact_id] > cfg.max_links_per_fact:
                    refused.append(RefusedJudgement(v.fact_id, v.question_id, f"more than {cfg.max_links_per_fact} links for one fact"))
                    continue
            kept.append(v)
        return kept, refused

    @staticmethod
    def _quoted(item: LinkItem, fact: Fact) -> bool:
        quote = _norm(item.evidence_quote)
        return len(quote) >= 4 and quote in _norm(f"{fact.proposition} {fact.evidence_text}")


def make_linker(pack: Pack, llm: Any | None):
    """The linker to use for a claim: the fact-template tags, plus the model's direction judgements when enabled and a client is given."""
    from claimtower.evidence.linker import FactQuestionLinker

    cfg = pack.question_templates.fact_links
    return FactQuestionLinker(cfg, LLMLinkJudge(llm, pack, cfg) if cfg.enabled and llm is not None else None)


__all__ = ["ALLOWED", "LLMLinkJudge", "LinkAnswer", "LinkItem", "make_linker"]
