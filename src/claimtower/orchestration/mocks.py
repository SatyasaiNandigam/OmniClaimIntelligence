"""Mock agents: the agents that are YET TO IMPLEMENT, so the graph can run end to end today.

A mock is a fixture stub (BUILD_PLAN section 11 and data/fixtures/README.md): it returns the labelled slice of the claim's fixture for the fields it
owns, wrapped in a StatePatch, and its rationale says plainly that the agent is not implemented. For a claim that has no fixture it returns a NEUTRAL
slice (the owned fields at their defaults). A mock never computes anything: its output is illustrative, not derived from the other agents' output
(so it can disagree with a real agent upstream, for example in clause ids). Every event of a mock node is flagged `mock`, and the human gate lists the
mock agents so a reviewer cannot mistake the dossier for a complete one.

To replace a mock, implement the agent (task id in the rationale) and put it in the registry in `claimtower.composition`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from claimtower.core.schemas import ClaimState
from claimtower.core.state import OWNERSHIP, StatePatch
from claimtower.orchestration.registry import ROSTER

REPO = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURES = REPO / "data" / "fixtures"
MARK = "YET TO IMPLEMENT"


class FixtureSource:
    """The labelled fixtures, found by claim id (data/fixtures/<scenario>/<agent>.json, one slice per agent)."""

    def __init__(self, root: Path | str = DEFAULT_FIXTURES) -> None:
        self.root = Path(root)
        self._by_claim: dict[str, Path] = {}
        for state_file in sorted(self.root.glob("*/claim_state.json")) if self.root.is_dir() else []:
            claim_id = json.loads(state_file.read_text(encoding="utf-8")).get("claim_id")
            if claim_id:
                self._by_claim[claim_id] = state_file.parent

    def scenario_of(self, claim_id: str) -> str | None:
        d = self._by_claim.get(claim_id)
        return d.name if d else None

    def slice(self, claim_id: str, agent_id: str) -> dict[str, Any] | None:
        d = self._by_claim.get(claim_id)
        f = d / f"{agent_id}.json" if d else None
        return json.loads(f.read_text(encoding="utf-8")) if f and f.is_file() else None


def neutral_slice(agent_id: str) -> dict[str, Any]:
    """The owned fields at their defaults (what a mock would return for an empty claim)."""
    blank = ClaimState(claim_id="neutral")
    return {f: getattr(blank, f) for f in OWNERSHIP[agent_id]}


class MockAgent:
    """Implements the Agent protocol for an agent that is not built yet."""

    def __init__(self, node: str, fixtures: FixtureSource | None = None) -> None:
        self.agent_id = ROSTER[node][0]
        self.owns = OWNERSHIP[self.agent_id]
        self.node, self.fixtures = node, fixtures or FixtureSource()
        _, self.task, self.purpose = ROSTER[node]

    def run(self, state: ClaimState) -> StatePatch:
        raw = self.fixtures.slice(state.claim_id, self.agent_id)
        scenario = self.fixtures.scenario_of(state.claim_id)
        source = f"the fixture of scenario {scenario}" if raw is not None else f"no change (no fixture for claim {state.claim_id}: its fields were left as they are)"
        if raw is None:
            raw = {f: getattr(state, f) for f in self.owns}                  # nothing to return: what the claim already has (a supplied header stays)
        holder = ClaimState.model_validate({**state.model_dump(), **raw})            # validates the slice against the frozen contract
        updates = {f: getattr(holder, f) for f in self.owns}
        return StatePatch(agent_id=self.agent_id, updates=updates,
                          rationale=f"{MARK} (mock, {self.task}): the {self.node} agent will {self.purpose}. Returned {source}; nothing was computed.")
