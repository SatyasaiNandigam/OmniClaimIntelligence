"""Action planner (T6.4): the gaps and the decision state -> the next steps, each with its execution mode and an idempotency key. Rules only.

    plan = ActionPlanner(matrix, templates).plan(state)         # list[PlannedAction], in a stable order

What is planned (templates in the pack, packs/<pack>/action_templates.yaml; the modes come from configs/autonomy_matrix.yaml):
  * one CONSOLIDATED information-request draft per addressee that receives drafts (claimant, plumber ...): every item every request asks of that role, once
  * a TASK for a role handled inside or by an appointed professional (surveyor follow-up, request the endorsement wording from policy administration)
  * a status-message draft to the claimant (an acknowledgement instead when no policy is in force, with the claim registered and routed to an operator)
  * the settlement put in front of a person when it is calculated; the settlement held while a missing wording is awaited
Nothing is sent or paid here. The idempotency key of an action is `<claim>:<type>[:<party>]`: planning twice yields the same keys, so the agent never duplicates a task.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

import yaml

from claimtower.core.config import AutonomyMatrix, Pack
from claimtower.core.schemas import ClaimState

POLICY_SIDE_INPUTS = {"water_damage_sublimit"}


@dataclass
class ActionTemplates:
    draft_roles: list[str]
    task_roles: dict[str, dict[str, Any]]
    operations: dict[str, str]
    drafts: dict[str, dict[str, str]]

    @classmethod
    def load(cls, pack: Pack) -> ActionTemplates:
        raw = yaml.safe_load((pack.root / "action_templates.yaml").read_text(encoding="utf-8"))
        return cls(raw["draft_roles"], raw["task_roles"], raw["operations"], raw["drafts"])


@dataclass
class PlannedAction:
    action_type: str
    party: str
    mode: Literal["draft", "controlled_api"]
    key: str
    kind: Literal["draft", "task", "step"]
    draft_kind: str | None = None
    items: list[str] = field(default_factory=list)
    question_ids: list[str] = field(default_factory=list)
    title: str = ""


def execution_mode(matrix: AutonomyMatrix, operation: str) -> Literal["draft", "controlled_api"]:
    """The matrix decides: controlled_api -> a controlled call; draft_only / human_reviewed / autonomous -> a draft. A human-only operation is never planned."""
    op = matrix.operations[operation]
    if op.mode == "authorised_human_only":
        raise ValueError(f"operation {operation} is for an authorised person only; an agent cannot plan it")
    return "controlled_api" if op.mode == "controlled_api" else "draft"


class ActionPlanner:
    def __init__(self, matrix: AutonomyMatrix, templates: ActionTemplates) -> None:
        self.matrix, self.t = matrix, templates

    def _mode(self, op_key: str) -> Literal["draft", "controlled_api"]:
        return execution_mode(self.matrix, self.t.operations[op_key])

    def plan(self, state: ClaimState) -> list[PlannedAction]:
        cid = state.claim_id
        h = state.claim_header
        lapsed = state.policy_snapshot is None and h is not None and bool(h.policy_ref) and h.loss_date is not None
        out: list[PlannedAction] = []
        if lapsed:
            out += [PlannedAction("register_claim", "adjuster", self._mode("register_claim"), f"{cid}:register_claim", "step"),
                    PlannedAction("route_to_operator", "adjuster", self._mode("route_to_operator"), f"{cid}:route_to_operator", "step")]
            out.append(PlannedAction("draft_acknowledgement", "claimant", self._mode("acknowledgement"), f"{cid}:draft_acknowledgement:claimant", "draft", "acknowledgement"))
            return out
        by_role: dict[str, list] = {}
        for m in state.missing_evidence:
            by_role.setdefault(m.minimum_evidence_request.addressee_role, []).append(m)
        asked = [i.description for i in state.intake_checklist if i.required and not i.received and i.item_id.startswith(('field:', 'doc:'))]      # registration: asked once, together
        for role in (['claimant'] if asked and 'claimant' not in by_role else []):
            by_role[role] = []
        for role, reqs in by_role.items():
            items = list(dict.fromkeys([*(asked if role == 'claimant' else []), *(i for r in reqs for i in r.minimum_evidence_request.items)]))
            qids = list(dict.fromkeys(q for r in reqs for q in r.resolves_question_ids))
            if role in self.t.draft_roles:
                kind = "draft_consolidated_request" if len(qids) > 1 or len(reqs) > 1 or (role == "claimant" and asked and len(items) > 1) else "draft_request"
                out.append(PlannedAction(kind, role, self._mode("information_request"), f"{cid}:{kind}:{role}", "draft", "information_request", items, qids))
            elif role in self.t.task_roles:
                spec = self.t.task_roles[role]
                name = spec["task"]
                if spec.get("id_pattern"):
                    ids = list(dict.fromkeys(re.findall(spec["id_pattern"], " ".join(items))))
                    name = "_".join([spec["task"], *ids, "from", role]) if ids else f"{spec['task']}_from_{role}"
                out.append(PlannedAction("create_task", name, execution_mode(self.matrix, spec["operation"]), f"{cid}:create_task:{name}", "task", None, items, qids,
                                         spec.get("title", "")))
        s = state.settlement
        if s is not None and s.status == "cannot_calculate" and s.missing_inputs and set(s.missing_inputs) <= POLICY_SIDE_INPUTS:
            out.append(PlannedAction("hold_settlement_until_endorsement_received", "adjuster", self._mode("hold_settlement"),
                                     f"{cid}:hold_settlement_until_endorsement_received", "step"))
        out.append(PlannedAction("draft_status_message", "claimant", self._mode("status_message"), f"{cid}:draft_status_message:claimant", "draft", "status_message",
                                 question_ids=[q for r in by_role.get("claimant", []) for q in r.resolves_question_ids]))
        if s is not None and s.status == "calculated":
            out.append(PlannedAction("present_settlement_for_human_approval", "adjuster", self._mode("present_settlement"),
                                     f"{cid}:present_settlement_for_human_approval", "step"))
        return out


__all__ = ["ActionPlanner", "ActionTemplates", "PlannedAction", "execution_mode"]
