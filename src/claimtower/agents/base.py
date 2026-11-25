"""Agent protocol (BUILD_PLAN A.1). StatePatch lives in core.state and is re-exported here."""
from typing import ClassVar, Protocol

from claimtower.core.schemas import ClaimState
from claimtower.core.state import StatePatch

__all__ = ["Agent", "StatePatch"]


class Agent(Protocol):
    agent_id: ClassVar[str]          # key into core.state.OWNERSHIP
    owns: ClassVar[frozenset[str]]   # must equal OWNERSHIP[agent_id]

    def run(self, state: ClaimState) -> StatePatch: ...
