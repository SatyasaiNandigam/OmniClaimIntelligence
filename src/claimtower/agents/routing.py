"""Routing agent (T5.6): the claim state -> the tier, reason codes and required approver role (`routing`), and whether it waits for evidence or for a person.

Owned fields are the supervisor's (core.state.OWNERSHIP["supervisor"]): `routing`, `human_approval_required`, `resolution_status`. Deterministic: the features come from the
claim state (routing/features.py), the decision from the rules file (routing/tiering.py). No model and no extraction confidence take part, by design: a tier must be
the same whatever a model thought. `human_approval_required` stays True in the prototype (guardrail 2). The status is `awaiting_evidence` while an evidence request is open or a material question is not settled,
otherwise `awaiting_human`. Its output is a recommendation for the person at the gate.
"""
from __future__ import annotations

from collections.abc import Collection
from typing import ClassVar

from claimtower.core.schemas import ClaimState
from claimtower.core.state import OWNERSHIP, StatePatch
from claimtower.routing.features import derive_features
from claimtower.routing.tiering import TieringEngine


class RoutingAgent:
    agent_id: ClassVar[str] = "supervisor"
    owns: ClassVar[frozenset[str]] = OWNERSHIP["supervisor"]

    def __init__(self, engine: TieringEngine, material_categories: Collection[str] = ("cause", "timing"), amount_issue_ids: Collection[str] = ()) -> None:
        self.engine, self.material, self.amounts = engine, tuple(material_categories), tuple(amount_issue_ids)
        self.last_features: dict = {}

    def run(self, state: ClaimState) -> StatePatch:
        features, assumed = derive_features(state, self.material, self.amounts)
        self.last_features = features
        decision = self.engine.route(features)
        status = "awaiting_evidence" if state.missing_evidence or features["evidence_completeness"] == "major_gap" else "awaiting_human"
        shown = ", ".join(f"{k}={v}" for k, v in features.items())
        rationale = (f"tier {decision.tier} ({', '.join(decision.reason_codes)}); approver {decision.required_approver_role}; rules {decision.rules_version}; features: {shown}; "
                     f"assumed: {'; '.join(assumed)}")
        return StatePatch(agent_id=self.agent_id, updates={"routing": decision, "human_approval_required": True, "resolution_status": status}, rationale=rationale)


__all__ = ["RoutingAgent"]
