"""Tiering engine (T5.6): the routing rules (configs/routing_rules.yaml) applied to a claim's features -> tier, reason codes, required approver role.

    engine = TieringEngine(settings.routing)
    decision = engine.route({"policy_status": "in_force", ..., "exposure_inr": 98000})        # -> RoutingDecision (a frozen contract)

Deterministic, no model and no clock: the same features and rules give the same decision, and changing the YAML changes the routing (nothing here is a rule).
Evaluation order, the first stage that matches decides: 1 mandatory referral -> complex | 2 complex escalation triggers -> complex | 3 simple (ALL hold) | 4 medium
(ANY holds) | 5 otherwise complex with the reason code DEFAULT_COMPLEX. A feature whose value is None is UNKNOWN: no condition on it holds (a claim whose exposure
is unknown can therefore never be simple). Extraction confidence is not a feature (the rules loader rejects it, guardrail 4).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from claimtower.core.config import Condition, RoutingRules
from claimtower.core.schemas import RoutingDecision

DEFAULT_CODE = "DEFAULT_COMPLEX"


def holds(c: Condition, features: Mapping[str, Any]) -> bool:
    """Does this condition hold? An unknown (missing or None) feature never satisfies a condition."""
    v = features.get(c.feature)
    if v is None:
        return False
    ok = True
    if c.is_ is not None:
        ok = ok and v == c.is_
    if c.in_ is not None:
        ok = ok and v in c.in_
    for op, test in (("gt", lambda a, b: a > b), ("gte", lambda a, b: a >= b), ("lt", lambda a, b: a < b), ("lte", lambda a, b: a <= b)):
        bound = getattr(c, op)
        if bound is not None:
            ok = ok and test(v, bound)
    return bool(ok)


class TieringEngine:
    def __init__(self, rules: RoutingRules) -> None:
        self.rules = rules

    def route(self, features: Mapping[str, Any]) -> RoutingDecision:
        r = self.rules
        stages = [("complex", r.mandatory_referral.any_of, "any"), ("complex", r.tiers.complex.any_of, "any"), ("simple", r.tiers.simple.all_of, "all"),
                  ("medium", r.tiers.medium.any_of, "any")]
        for tier, conds, mode in stages:
            hits = [c for c in conds if holds(c, features)]
            if (mode == "any" and hits) or (mode == "all" and len(hits) == len(conds)):
                return RoutingDecision(tier=tier, reason_codes=[c.code for c in hits], required_approver_role=r.approver_roles[tier], rules_version=r.rules_version)
        return RoutingDecision(tier="complex", reason_codes=[DEFAULT_CODE], required_approver_role=r.approver_roles["complex"], rules_version=r.rules_version)


__all__ = ["DEFAULT_CODE", "TieringEngine", "holds"]
