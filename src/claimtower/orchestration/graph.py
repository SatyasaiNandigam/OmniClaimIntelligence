"""The claim graph (T6.1): the LangGraph state machine for one claim, and the runner the API will call.

    registry = build_registry(settings, ...)             # claimtower.composition: real agents where they exist, mocks ("YET TO IMPLEMENT") elsewhere
    runner = ClaimRunner(registry)
    status = runner.start(ClaimState(claim_id="CLM-0001"))       # runs until the human gate pauses it (or a node fails)
    status = runner.decide("CLM-0001", HumanDecision(decision="approve", reviewer_id="u1", reviewer_role="senior_adjuster"))

The flow of a new claim (BUILD_PLAN section 4):

    start -> intake -> policy -> evidence -> gap -> coverage -> settlement -> action -> audit -> routing -> HUMAN GATE -> commit
                                               \\_______ no policy in force ______/^                        (approve only)

  * every agent node is wrapped by the supervisor: ownership is enforced, the version bumped, an event logged; a failing agent parks the claim
    (`awaiting_human`) and the flow halts with the reason;
  * a claim with no policy in force skips coverage and settlement;
  * the gate pauses the graph (`interrupt`) and resumes with a named reviewer's decision; only an approval reaches the (mock) commit step;
  * the thread id is the claim id, so every claim has its own checkpoint history (audit-friendly, replayable).
Only this package imports langgraph (ADR-0001): agents stay framework-free.
"""
from __future__ import annotations

import threading
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel

from claimtower.core.schemas import ClaimState
from claimtower.core.state import StatePatch, apply_patch
from claimtower.core.state_diff import StateDiff, diff_states
from claimtower.orchestration.checkpoints import make_checkpointer
from claimtower.orchestration.events import EventLog, RunEvent
from claimtower.orchestration.registry import PIPELINE, AgentRegistry
from claimtower.orchestration.supervisor import HALT, HumanDecision, Supervisor

Phase = Literal["not_started", "awaiting_human_decision", "completed", "halted"]


def build_graph(registry: AgentRegistry, supervisor: Supervisor, checkpointer: Any | None = None):
    g = StateGraph(ClaimState)
    g.add_node("start", supervisor.start)
    for slot in registry:
        g.add_node(slot.node, supervisor.node(slot))
    g.add_node("human_gate", supervisor.human_gate)
    g.add_node("commit", supervisor.commit)

    g.add_edge(START, "start")
    g.add_edge("start", PIPELINE[0])
    for here, there in zip(PIPELINE, PIPELINE[1:]):
        if here == "gap":
            g.add_conditional_edges("gap", supervisor.after_gap, {"continue": "coverage", "skip": "action", HALT: END})
        elif here == "routing":
            continue
        else:
            g.add_conditional_edges(here, supervisor.after_agent, {"continue": there, HALT: END})
    g.add_conditional_edges("routing", supervisor.after_routing, {"continue": "human_gate", HALT: END})
    g.add_conditional_edges("human_gate", supervisor.after_gate, {"commit": "commit", "end": END})
    g.add_edge("commit", END)
    return g.compile(checkpointer=checkpointer or make_checkpointer("memory"))


class RunStatus(BaseModel):
    """Where a claim stands, in the words the API and the UI need."""

    claim_id: str
    phase: Phase
    state_version: int | None = None
    resolution_status: str | None = None
    paused_at: str | None = None
    gate: dict[str, Any] | None = None             # what the reviewer is shown (while paused at the gate)
    halted_reason: str | None = None               # why a node failed (phase halted)
    nodes_run: list[str] = []
    mock_nodes_run: list[str] = []                 # the nodes that ran a mock ("yet to implement") agent in this claim's runs
    agents: list[dict[str, str]] = []              # the roster: which agent is implemented and which is a mock


class VersionInfo(BaseModel):
    """One committed version of a claim's state and what produced it."""

    version: int
    kind: str
    node: str
    agent_id: str | None = None
    mock: bool = False
    summary: str = ""


class ClaimRunner:
    def __init__(self, registry: AgentRegistry, *, checkpointer: Any | None = None, events: EventLog | None = None) -> None:
        self.registry = registry
        self.events = events or EventLog()
        self._lock = threading.RLock()                    # one run at a time: the agents hold per-run attributes (and the model client counters)
        self._claims: dict[str, None] = {}               # claims in the order they were started
        self.supervisor = Supervisor(registry, self.events)
        self.graph = build_graph(registry, self.supervisor, checkpointer)

    @staticmethod
    def _cfg(claim_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": claim_id}}          # thread id = claim id: one checkpoint history per claim

    # ------------------------------------------------------------------ commands
    def start(self, state: ClaimState) -> RunStatus:
        with self._lock:
            return self._start(state)

    def _start(self, state: ClaimState) -> RunStatus:
        if self.graph.get_state(self._cfg(state.claim_id)).values:
            raise ValueError(f"claim {state.claim_id} has already been started: use reassess() for new evidence")
        self._claims[state.claim_id] = None
        if not state.correlation_id:                                  # written only at claim creation, never by an agent: ids every event and adapter call of this claim
            state = state.model_copy(update={"correlation_id": f"corr-{state.claim_id}"})
        self.graph.invoke(state, self._cfg(state.claim_id))
        return self.status(state.claim_id)

    def decide(self, claim_id: str, decision: HumanDecision | dict[str, Any]) -> RunStatus:
        """Resume the paused gate with a reviewer's decision. An invalid decision is refused and the gate stays open (see status().gate['error'])."""
        with self._lock:
            snap = self.graph.get_state(self._cfg(claim_id))
            if "human_gate" not in snap.next:
                raise ValueError(f"claim {claim_id} is not waiting for a human decision (next: {snap.next or 'finished'})")
            payload = decision.model_dump() if isinstance(decision, HumanDecision) else dict(decision)
            self.graph.invoke(Command(resume=payload), self._cfg(claim_id))
            return self.status(claim_id)

    def reassess(self, claim_id: str, from_node: str = "evidence") -> RunStatus:
        """Run the claim again from `from_node` (new evidence arrived): the nodes before it are NOT re-run (first cut of T6.2; the dependency map comes later).
        If the claim is waiting at the human gate, the new evidence SUPERSEDES the pending recommendation: nobody had decided yet, the gate is presented again
        after the re-assessment (the event log says so)."""
        with self._lock:
            return self._reassess(claim_id, from_node)

    def _reassess(self, claim_id: str, from_node: str) -> RunStatus:
        if from_node not in PIPELINE:
            raise ValueError(f"unknown node '{from_node}'; nodes are {', '.join(PIPELINE)}")
        cfg = self._cfg(claim_id)
        snap = self.graph.get_state(cfg)
        if not snap.values:
            raise ValueError(f"claim {claim_id} has not been started")
        superseded = "human_gate" in snap.next
        state = ClaimState.model_validate(snap.values)
        new = apply_patch(state, StatePatch(agent_id="supervisor", updates={"resolution_status": "assessing"}))
        before = PIPELINE[PIPELINE.index(from_node) - 1] if PIPELINE.index(from_node) > 0 else "start"
        self.events.add(claim_id, "reassessment", from_node, f"re-assessing from '{from_node}' (nodes before it are not re-run)"
                        + ("; the pending recommendation at the human gate is superseded by the new evidence" if superseded else ""),
                        agent_id="supervisor", state_version=new.state_version, from_node=from_node, superseded_gate=superseded)
        self.graph.update_state(cfg, {"resolution_status": "assessing", "state_version": new.state_version}, as_node=before)
        self.graph.invoke(None, cfg)
        return self.status(claim_id)

    # ------------------------------------------------------------------ read
    def claims(self) -> list[str]:
        return list(self._claims)

    def knows(self, claim_id: str) -> bool:
        return claim_id in self._claims or bool(self.graph.get_state(self._cfg(claim_id)).values)

    def state_at(self, claim_id: str, version: int) -> ClaimState:
        """The state as it was at `version` (the latest checkpoint with that version)."""
        for snap in self.graph.get_state_history(self._cfg(claim_id)):
            if snap.values and int(snap.values.get("state_version", -1)) == version:
                return ClaimState.model_validate(snap.values)
        raise KeyError(f"claim {claim_id} has no version {version}")

    def versions(self, claim_id: str) -> list[VersionInfo]:
        """Every committed version with the node that produced it (from the event log), oldest first."""
        kinds = ("run_started", "agent_finished", "agent_failed", "human_decision", "reassessment")
        seen: dict[int, VersionInfo] = {}
        for e in self.events.of_kind(claim_id, *kinds):
            if e.state_version is not None and e.state_version not in seen:
                seen[e.state_version] = VersionInfo(version=e.state_version, kind=e.kind, node=e.node, agent_id=e.agent_id, mock=e.mock, summary=e.message[:240])
        return [seen[v] for v in sorted(seen)]

    def diff(self, claim_id: str, from_version: int, to_version: int) -> StateDiff:
        return diff_states(self.state_at(claim_id, from_version), self.state_at(claim_id, to_version))

    def state(self, claim_id: str) -> ClaimState:
        snap = self.graph.get_state(self._cfg(claim_id))
        if not snap.values:
            raise KeyError(f"unknown claim {claim_id}")
        return ClaimState.model_validate(snap.values)

    def status(self, claim_id: str) -> RunStatus:
        snap = self.graph.get_state(self._cfg(claim_id))
        if not snap.values:
            return RunStatus(claim_id=claim_id, phase="not_started", agents=self.registry.describe())
        st = ClaimState.model_validate(snap.values)
        ran = self.events.of_kind(claim_id, "agent_finished")
        base = dict(claim_id=claim_id, state_version=st.state_version, resolution_status=st.resolution_status, nodes_run=[e.node for e in ran],
                    mock_nodes_run=sorted({e.node for e in ran if e.mock}), agents=self.registry.describe())
        if "human_gate" in snap.next:
            gate = next((i.value for t in snap.tasks for i in t.interrupts), None)
            return RunStatus(phase="awaiting_human_decision", paused_at="human_gate", gate=gate, **base)
        if not snap.next:
            fail = self.events.last_failure(claim_id)
            last_start = max((e.seq for e in self.events.of_kind(claim_id, "run_started", "reassessment")), default=0)
            if fail is not None and fail.seq > last_start:
                return RunStatus(phase="halted", halted_reason=fail.data.get("reason", fail.message), paused_at=fail.node, **base)
            return RunStatus(phase="completed", **base)
        return RunStatus(phase="halted", paused_at=snap.next[0], **base)

    def events_for(self, claim_id: str) -> list[RunEvent]:
        return self.events.for_claim(claim_id)

    def history(self, claim_id: str) -> list[tuple[int, tuple[str, ...]]]:
        """(state_version, the nodes that were next) for every checkpoint, oldest first: the replayable trace."""
        out = [(int(s.values.get("state_version", 0)), tuple(s.next)) for s in self.graph.get_state_history(self._cfg(claim_id)) if s.values]
        return list(reversed(out))

    def diagram(self) -> str:
        """The graph as a Mermaid diagram (docs, UI)."""
        return self.graph.get_graph().draw_mermaid()
