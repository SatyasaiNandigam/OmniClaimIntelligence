"""Quality & Audit agent (T8.5): the finished assessment -> `qa_report` and `audit_ref`. Owns exactly those two fields (core.state.OWNERSHIP).

It runs before routing and the human gate and checks, with plain rules (audit/checks.py; no model: a safety net must not depend on the thing it checks), that every assertion has a valid
citation, that the settlement explanation matches the engine's result, that nothing written for a person states an outcome, and that nothing was carried out before a human approval.
A FAILING report blocks the release: the supervisor parks the claim for a person with the reasons (the claim does not reach the gate with a defective assessment), and the report stays in the
state for the reviewer. `audit_ref` is stable per claim (`AUDIT-<claim id>`), so the same reference is used by every version of the dossier.
Idempotent and pure.
"""
from __future__ import annotations

from typing import ClassVar

from claimtower.audit.checks import QaConfig, run_checks
from claimtower.core.schemas import ClaimState
from claimtower.core.state import OWNERSHIP, StatePatch


class AuditAgent:
    agent_id: ClassVar[str] = "audit"
    owns: ClassVar[frozenset[str]] = OWNERSHIP["audit"]

    def __init__(self, cfg: QaConfig) -> None:
        self.cfg = cfg

    def run(self, state: ClaimState) -> StatePatch:
        report = run_checks(state, self.cfg)
        if report.passed:
            rationale = "quality checks passed: citations valid, settlement explanation matches the engine, no outcome wording, nothing carried out before approval"
        else:
            by_check: dict[str, int] = {}
            for f in report.failures:
                by_check[f.check] = by_check.get(f.check, 0) + 1
            rationale = (f"QUALITY CHECKS FAILED ({len(report.failures)}): " + ", ".join(f"{k} x{n}" for k, n in sorted(by_check.items())) + ". "
                         + "; ".join(f"{f.check}: {f.detail}" for f in report.failures[:5]) + (" ..." if len(report.failures) > 5 else ""))
        return StatePatch(agent_id=self.agent_id, updates={"qa_report": report, "audit_ref": f"AUDIT-{state.claim_id}"}, rationale=rationale)


__all__ = ["AuditAgent"]
