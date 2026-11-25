"""The agent roster of the graph (BUILD_PLAN section 5) and which of them are real.

Every agent of the pipeline has a SLOT. A slot holds either the real agent or a MOCK (orchestration.mocks) that returns the labelled fixture for the
claim and says `YET TO IMPLEMENT`. Replacing a mock is one line in the composition (`claimtower.composition.build_registry`); the graph does not change.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

from claimtower.agents.base import Agent

# node -> (agent id that writes, task that implements it, what the real agent will do). Order = the order of the new-claim flow (BUILD_PLAN section 4).
ROSTER: dict[str, tuple[str, str, str]] = {
    "intake": ("intake", "T6.7", "normalise the FNOL, look the policy up in the PAS, build the required-field checklist and the duplicate candidates"),
    "policy": ("policy", "T3.5", "build the policy snapshot in force on the date of loss"),
    "evidence": ("evidence", "T2.4", "classify the documents and turn them into facts with provenance"),
    "gap": ("gap", "T4.7", "open questions, contradictions, minimum-evidence requests and counterfactual scenarios"),
    "coverage": ("coverage", "T5.1", "combine clauses, facts and open questions into the coverage issue matrix, abstaining when ambiguous"),
    "settlement": ("settlement", "T5.5", "extract the calculation inputs and call the deterministic settlement engine"),
    "action": ("action", "T6.4", "plan the next actions and draft the notifications and tasks"),
    "audit": ("audit", "T8.5", "check citations, permissions and consistency, and block a release to the human gate"),
    "routing": ("supervisor", "T5.6", "assign the tier and the reason codes from the routing rules and name the approver role"),
}
PIPELINE: tuple[str, ...] = tuple(ROSTER)

Status = Literal["implemented", "mock"]


@dataclass(frozen=True)
class AgentSlot:
    node: str
    agent: Agent
    status: Status
    task: str
    purpose: str

    @property
    def is_mock(self) -> bool:
        return self.status == "mock"


class AgentRegistry:
    """node name -> slot. Build it with `claimtower.composition.build_registry`, or directly in tests."""

    def __init__(self, agents: Mapping[str, tuple[Agent, Status]]) -> None:
        missing = [n for n in PIPELINE if n not in agents]
        extra = [n for n in agents if n not in ROSTER]
        if missing or extra:
            raise ValueError(f"the registry needs exactly the roster {list(PIPELINE)}; missing {missing}, unknown {extra}")
        self._slots: dict[str, AgentSlot] = {}
        for node, (agent, status) in agents.items():
            expected, task, purpose = ROSTER[node]
            if agent.agent_id != expected:
                raise ValueError(f"node '{node}' is written by agent '{expected}', not '{agent.agent_id}'")
            self._slots[node] = AgentSlot(node, agent, status, task, purpose)

    def __getitem__(self, node: str) -> AgentSlot:
        return self._slots[node]

    def __iter__(self) -> Iterator[AgentSlot]:
        return (self._slots[n] for n in PIPELINE)

    @property
    def mocks(self) -> list[str]:
        return [s.node for s in self if s.is_mock]

    @property
    def implemented(self) -> list[str]:
        return [s.node for s in self if not s.is_mock]

    def describe(self) -> list[dict[str, str]]:
        """One row per agent: the status table the API and the UI show ('implemented' or 'YET TO IMPLEMENT (mock)')."""
        return [{"node": s.node, "agent": s.agent.agent_id, "status": "implemented" if not s.is_mock else "YET TO IMPLEMENT (mock)",
                 "task": s.task, "will_do": s.purpose} for s in self]
