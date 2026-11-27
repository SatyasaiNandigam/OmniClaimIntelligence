"""Coverage agent (T5.1): the policy snapshot, the facts and the open questions -> the coverage issue matrix (`coverage_issues`).

Owns (writes) exactly `coverage_issues` (core.state.OWNERSHIP). It reads the policy snapshot (the clauses in force on the date of loss), the facts, the open
questions and the contradictions. It is a thin wrapper: what each row says, and why, is decided by `coverage.issue_matrix.IssueMatrixBuilder` (a decision table
in the pack, plus a checked model reading of the facts against the clause). Its output is HUMAN-REVIEWED (BUILD_PLAN section 5): a row says how well an issue
is resolved on the evidence, never that the claim is covered, excluded, approved or denied.

Idempotent and incremental in the cheap way: a row whose clauses, facts and question states are unchanged is not read by the model again, so a re-run with
nothing new returns identical rows. A policy that is not in force gets the one row that says so (no further coverage reasoning); a claim with no policy
number or loss date yet gets no snapshot-based rows (they cannot be cited).
"""
from __future__ import annotations

from typing import Any, ClassVar

from claimtower.core.schemas import ClaimState
from claimtower.core.state import OWNERSHIP, StatePatch
from claimtower.coverage.issue_matrix import IssueMatrixBuilder


def not_in_force(state: ClaimState) -> bool:
    """The policy system answered and nothing is in force on the loss date (the Policy agent's convention: no snapshot although the claim has a policy
    number and a loss date)."""
    h = state.claim_header
    return state.policy_snapshot is None and h is not None and bool(h.policy_ref) and h.loss_date is not None


class CoverageAgent:
    agent_id: ClassVar[str] = "coverage"
    owns: ClassVar[frozenset[str]] = OWNERSHIP["coverage"]

    def __init__(self, builder: IssueMatrixBuilder) -> None:
        self.builder = builder
        self.last_result = None

    def run(self, state: ClaimState) -> StatePatch:
        result = self.builder.build(state.policy_snapshot, not_in_force(state), state.facts, state.open_questions, state.contradictions)
        self.last_result = result
        counts: dict[str, int] = {}
        for i in result.issues:
            counts[i.status] = counts.get(i.status, 0) + 1
        read = sorted(i for i, d in result.details.items() if d.reading.value in ("favourable", "adverse"))
        parts = [f"{len(result.issues)} issue(s): " + (", ".join(f"{n} {s.replace('_', ' ')}" for s, n in sorted(counts.items())) or "none")]
        if result.skipped:
            parts.append(f"not raised (their questions are not in the register): {', '.join(result.skipped)}")
        if read:
            parts.append(f"REVIEW: read by a model against the clause: {', '.join(read)}")
        reasons = result.abstentions()
        if reasons:                                                                      # T5.2: where the system declines to conclude, and why
            parts.append("not concluded: " + ", ".join(f"{i} ({r})" for i, r in sorted(reasons.items())))
        amb = sorted(i for i, d in result.details.items() if d.reading.value == "ambiguous")
        if amb:
            parts.append(f"REVIEW: the wording can be read two ways, both shown: {', '.join(amb)}")
        if result.refused:
            parts.append("model readings refused: " + "; ".join(f"{i}: {w}" for i, w in result.refused))
        if result.warnings:
            parts.append("warnings: " + " | ".join(result.warnings))
        if result.problems:
            parts.append("SELF-CHECK PROBLEMS: " + " | ".join(result.problems))
        return StatePatch(agent_id=self.agent_id, updates={"coverage_issues": result.issues}, rationale="; ".join(parts))


def make_coverage_agent(pack: Any, llm: Any | None = None, min_citations: int = 1) -> CoverageAgent:
    """The Coverage agent for a pack, with the model reading switched on when `llm` is given and the pack enables it (otherwise settled rows are `identified`)."""
    from claimtower.coverage.llm_reading import make_builder

    return CoverageAgent(make_builder(pack, llm, min_citations))
