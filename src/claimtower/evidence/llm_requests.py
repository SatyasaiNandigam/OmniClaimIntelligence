"""Model wording of minimum-evidence requests (T4.5). Checked, never trusted.

    writer = LLMRequestWriter(llm, pack)                 # llm: core.llm.LLMClient; prompt: packs/<pack>/prompts/min_evidence_request.md
    generator = MinimumEvidenceGenerator(pack.question_templates, writer)

One call per addressee. The writer only PROPOSES items (which needs each serves, what it asks for); the generator checks every one (serves a need,
within the length limit, no verdict language, no repeat, no document type already on file) and writes an item itself for any need left uncovered.
A model that is down returns no drafts: the request is then made of rule-written items, still complete.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from claimtower.core.config import Pack
from claimtower.evidence.min_evidence import ASKS_FOR, DraftItem, WriterContext, WriterOutcome


class DraftModel(BaseModel):
    need_ids: list[int] = []
    text: str
    asks_for: str = "information"
    document_type: str = ""


class DraftAnswer(BaseModel):
    items: list[DraftModel] = []


class LLMRequestWriter:
    def __init__(self, llm: Any, pack: Pack) -> None:
        cfg = pack.question_templates.requests
        assert cfg is not None
        self._llm = llm.derive(reasoning_effort=cfg.reasoning_effort) if cfg.reasoning_effort and hasattr(llm, "derive") else llm
        self._pack = pack
        self.asked = 0

    @staticmethod
    def _feedback(ctx: WriterContext) -> str:
        if not ctx.feedback:
            return ""
        return ("Your previous attempt had items that were refused. Write them again for the needs that still have no item (ids: "
                + ", ".join(str(i) for i in ctx.uncovered) + "), fixing the problem named for each, and keep every item within the character limit.\n"
                + "\n".join(f"- {line}" for line in ctx.feedback))

    def write(self, ctx: WriterContext) -> WriterOutcome:
        from claimtower.core.llm import load_prompt

        self.asked += 1
        lines = []
        for n in ctx.needs:
            q = ctx.questions[n.question_id]
            what = f"CONFLICT between statements on file: {n.explanation}" if n.kind == "conflict" else f"{n.criterion} [{n.verdict}: {n.rationale or 'not shown'}]"
            known = "; ".join(ctx.known.get(n.index, [])) or "nothing yet"
            lines.append(f"[{n.index}] {q.question_id} {q.text} | missing: {what} | known: {known}")
        internal = ("This is an internal request from the claims system to a team that holds the records: ask for the exact document or data."
                    if ctx.role_cfg.internal else "")
        try:
            prompt = load_prompt("min_evidence_request", self._pack, role_label=ctx.role_cfg.label, internal_note=internal, needs_block="\n".join(lines),
                                 received_block=", ".join(sorted(ctx.received_types)) or "none", max_items=ctx.max_items, max_chars=ctx.max_item_chars,
                                 allowed_types=", ".join(ctx.role_cfg.document_types) or "none", feedback_block=self._feedback(ctx))
            answer = self._llm.structured(DraftAnswer, prompt)
        except Exception as e:                              # model down, bad JSON, TBD model name ...
            return WriterOutcome(unavailable=True, detail=f"{type(e).__name__}: {e}")
        return WriterOutcome([DraftItem(tuple(d.need_ids), d.text, d.asks_for.strip().lower() if d.asks_for else "information", d.document_type.strip())
                              for d in answer.items])


def make_generator(pack: Pack, llm: Any | None):
    """The generator to use for a claim: the rules for who is asked and what is covered, plus the model's wording when enabled and a client is given."""
    from claimtower.evidence.min_evidence import MinimumEvidenceGenerator

    cfg = pack.question_templates.requests
    if cfg is None:
        raise ValueError("the pack has no `requests` section in question_templates.yaml")
    return MinimumEvidenceGenerator(pack.question_templates, LLMRequestWriter(llm, pack) if cfg.enabled and llm is not None else None)


__all__ = ["ASKS_FOR", "DraftAnswer", "DraftModel", "LLMRequestWriter", "make_generator"]
