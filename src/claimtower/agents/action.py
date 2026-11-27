"""Action agent (T6.4): the gaps and the decision state -> proposed next steps and their drafts. Owns exactly `proposed_actions` and `drafts` (core.state.OWNERSHIP).

The planner (actions/planner.py) decides WHAT is needed, from the evidence requests and the settlement state, with the execution mode from the autonomy matrix; the draft
writer (actions/drafts.py) writes the text from pack templates, and a model may reword it (checked: every item kept, no new amount, no conclusion). Nothing is sent: a draft
waits for a person, a `controlled_api` action is carried out only through the claims adapter after the gate.

Idempotent: an action is identified by its key `<claim>:<type>[:<party>]`. Running again keeps an existing action as it is (its status too: approved, executed ...), adds only
what is new, and cancels a still-`proposed` action whose need has gone (its unsent draft goes with it). It never creates a second task for the same need.
"""
from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel

from claimtower.actions.drafts import DraftWriter, draft_id
from claimtower.actions.planner import ActionPlanner, ActionTemplates
from claimtower.core.config import AutonomyMatrix, Pack
from claimtower.core.schemas import ClaimState, Draft, ProposedAction
from claimtower.core.state import OWNERSHIP, StatePatch


class Reworded(BaseModel):
    text: str


class DraftRewriter:
    def __init__(self, llm: Any, pack: Pack) -> None:
        self._llm, self._pack = llm, pack

    def __call__(self, text: str, kind: str, items: list[str]) -> str | None:
        from claimtower.core.llm import load_prompt

        try:
            return self._llm.structured(Reworded, load_prompt("action_draft", self._pack, kind=kind, text=text, items="\n".join(f"- {i}" for i in items) or "(none)")).text
        except Exception:
            return None


class ActionAgent:
    agent_id: ClassVar[str] = "action"
    owns: ClassVar[frozenset[str]] = OWNERSHIP["action"]

    def __init__(self, planner: ActionPlanner, writer: DraftWriter) -> None:
        self.planner, self.writer = planner, writer
        self.last_plan: list = []

    def run(self, state: ClaimState) -> StatePatch:
        plan = self.planner.plan(state)
        self.last_plan = plan
        self.writer.dropped.clear()
        wanted = {p.key for p in plan}
        existing = {a.idempotency_key: a for a in state.proposed_actions}
        actions: list[ProposedAction] = []
        for a in state.proposed_actions:
            if a.idempotency_key not in wanted and a.status == "proposed":
                a = a.model_copy(update={"status": "cancelled"})                  # the need has gone; kept for the audit trail, never deleted
            actions.append(a)
        n = max([int(a.action_id.rsplit("-", 1)[-1]) for a in state.proposed_actions if a.action_id.rsplit("-", 1)[-1].isdigit()] or [0])
        added = []
        for p in plan:
            if p.key in existing:
                continue
            n += 1
            payload = draft_id(p.key) if p.kind == "draft" else f"task:{p.key}" if p.kind == "task" else None
            act = ProposedAction(action_id=f"ACT-{state.claim_id}-{n}", action_type=p.action_type, responsible_party=p.party, execution_mode=p.mode, status="proposed",
                                 idempotency_key=p.key, payload_ref=payload)
            actions.append(act)
            added.append(p.key)
        # drafts: keep any a person has moved on (approved / sent); regenerate the rest from the plan; drop unsent drafts of cancelled actions
        keep = {d.draft_id: d for d in state.drafts if d.status != "draft"}
        drafts: dict[str, Draft] = dict(keep)
        for p in plan:
            if p.kind == "draft" and draft_id(p.key) not in keep:
                d = self.writer.write(p, state)
                drafts[d.draft_id] = d
        cancelled = [a.idempotency_key for a in actions if a.status == "cancelled" and existing.get(a.idempotency_key, a).status == "proposed"]
        live = [a for a in actions if a.status != "cancelled"]
        parts = [f"{len(live)} action(s) proposed ({len(added)} new): " + (", ".join(f"{a.action_type}:{a.responsible_party}" if a.action_type == "create_task" else a.action_type
                                                                                 for a in live) or "none")]
        parts.append(f"{len(drafts)} draft(s) for a person to review; nothing is sent")
        if cancelled:
            parts.append("cancelled (no longer needed): " + ", ".join(cancelled))
        if self.writer.dropped:
            parts.append("model rewording dropped: " + " | ".join(self.writer.dropped))
        return StatePatch(agent_id=self.agent_id, updates={"proposed_actions": actions, "drafts": sorted(drafts.values(), key=lambda d: d.draft_id)}, rationale="; ".join(parts))


def make_action_agent(pack: Pack, matrix: AutonomyMatrix, llm: Any | None = None) -> ActionAgent:
    templates = ActionTemplates.load(pack)
    patterns = pack.coverage.forbidden_patterns if pack.coverage else []
    return ActionAgent(ActionPlanner(matrix, templates), DraftWriter(templates, patterns, DraftRewriter(llm, pack) if llm is not None else None))


__all__ = ["ActionAgent", "DraftRewriter", "make_action_agent"]
