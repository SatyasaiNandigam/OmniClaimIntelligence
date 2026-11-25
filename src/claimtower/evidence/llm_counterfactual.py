"""Model rewrite of counterfactual scenario text (T4.6). Checked, never trusted.

    writer = LLMScenarioWriter(llm, pack)                # llm: core.llm.LLMClient; prompt: packs/<pack>/prompts/counterfactual_rewrite.md
    engine = CounterfactualEngine(pack.question_templates, writer)

One call per material question, for its three scenarios. The writer only PROPOSES a rewrite; the engine accepts it for a scenario only if it
passes the linter (no wording that predicts or recommends a decision), is conditional, fits the length limits, names no issue or clause that is
not part of the scenario, mentions each affected issue and keeps each effect. Otherwise the scenario keeps its rule text, and the refusal is recorded.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from claimtower.core.config import Pack
from claimtower.evidence.counterfactual import ScenarioDraft, WriterContext, WriterOutcome


class ScenarioModel(BaseModel):
    scenario_id: str
    implication_text: str
    next_step: str


class RewriteAnswer(BaseModel):
    scenarios: list[ScenarioModel] = []


class LLMScenarioWriter:
    def __init__(self, llm: Any, pack: Pack) -> None:
        cfg = pack.question_templates.counterfactuals
        assert cfg is not None
        self._llm = llm.derive(reasoning_effort=cfg.reasoning_effort) if cfg.reasoning_effort and hasattr(llm, "derive") else llm
        self._pack = pack
        self.asked = 0

    def write(self, ctx: WriterContext) -> WriterOutcome:
        from claimtower.core.llm import load_prompt

        self.asked += 1
        fr = ctx.framing
        scenarios = "\n".join(f"- {s}: assumed finding: {getattr(fr, s.lower())} | implication: {d.implication_text} | next step: {d.next_step}"
                              for s, d in ctx.base.items())
        effects = "\n".join(f"- {i.issue_id} \"{i.title}\": " + "; ".join(f"{s}: {ctx.effects[i.issue_id][s]}" for s in ctx.base) for i in ctx.issues)
        clauses = "\n".join(f"- {cid}: {text[:300]}" for cid, text in ctx.clauses.items()) or "- (none)"
        facts = "\n".join(f"- {t[:300]}" for t in ctx.known) or "- (nothing yet)"
        try:
            prompt = load_prompt("counterfactual_rewrite", self._pack, question_text=f"{ctx.question.question_id}: {ctx.question.text}",
                                 scenarios_block=scenarios, effects_block=effects, clauses_block=clauses, facts_block=facts,
                                 max_implication=ctx.max_implication_chars, max_next_step=ctx.max_next_step_chars)
            answer = self._llm.structured(RewriteAnswer, prompt)
        except Exception as e:                              # model down, bad JSON, TBD model name ...
            return WriterOutcome(unavailable=True, detail=f"{type(e).__name__}: {e}")
        return WriterOutcome([ScenarioDraft(s.scenario_id.strip().upper(), s.implication_text, s.next_step) for s in answer.scenarios])


def make_engine(pack: Pack, llm: Any | None):
    """The engine to use for a claim: the rule-built scenarios, plus the model's claim-specific rewrite when enabled and a client is given."""
    from claimtower.evidence.counterfactual import CounterfactualEngine

    cfg = pack.question_templates.counterfactuals
    if cfg is None:
        raise ValueError("the pack has no `counterfactuals` section in question_templates.yaml")
    return CounterfactualEngine(pack.question_templates, LLMScenarioWriter(llm, pack) if cfg.enabled and llm is not None else None)


__all__ = ["LLMScenarioWriter", "RewriteAnswer", "ScenarioModel", "make_engine"]
