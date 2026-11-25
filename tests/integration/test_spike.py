"""T0.1 framework spike: does LangGraph (pinned in pyproject.toml) give us what BUILD_PLAN section 3 needs?

Proves, against our real ClaimState contract:
  1. typed state            - the Pydantic ClaimState is the graph state
  2. ownership enforcement  - nodes write only owned fields (core.state.apply_patch); a rogue node is stopped
  3. human gate             - interrupt() pauses with a payload; Command(resume=...) continues
  4. partial re-run         - new evidence re-enters at a chosen node; upstream nodes are NOT re-executed
  5. audit / replay         - checkpoints give step-level history we can time-travel through

Findings worth knowing (also recorded in docs/adr/0001-framework.md):
  * a node that calls interrupt() is re-executed from its first line on resume -> code before interrupt() must be idempotent
  * graph.update_state(config, values, as_node=X) injects a state change "as if node X had run", so the NEXT node is X's successor
"""
from collections import Counter

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from claimtower.core.schemas import ClaimState, Fact, OpenQuestion
from claimtower.core.state import OwnershipViolation, StatePatch, apply_patch

CALLS: Counter = Counter()


def wrap(agent_id: str, fn):
    """What supervisor.wrap() will become: run the agent, enforce ownership, bump the version, count the call."""

    def node(state: ClaimState) -> dict:
        CALLS[agent_id] += 1
        patch = StatePatch(agent_id=agent_id, updates=fn(state))
        new = apply_patch(state, patch)                      # raises OwnershipViolation on illegal writes
        return {**patch.updates, "state_version": new.state_version}

    node.__name__ = agent_id
    return node


def make_fact(n: int) -> Fact:
    return Fact(fact_id=f"F{n}", claim_id="c1", proposition="p", source_document_id="D1", page=1,
                evidence_text="x", evidence_type="neutral", extraction_confidence=0.9)


def human_gate(state: ClaimState) -> dict:
    CALLS["human_gate"] += 1
    decision = interrupt({"claim_id": state.claim_id, "ask": "approve, edit or request evidence?"})
    status = "approved" if decision["decision"] == "approve" else "awaiting_evidence"
    patch = StatePatch(agent_id="supervisor", updates={"resolution_status": status})
    new = apply_patch(state, patch)
    return {**patch.updates, "state_version": new.state_version}


def build(rogue: bool = False):
    g = StateGraph(ClaimState)
    g.add_node("policy", wrap("policy", lambda s: {}))
    g.add_node("evidence", wrap("evidence", lambda s: {"facts": [*s.facts, make_fact(len(s.facts) + 1)]}))
    # the Gap agent only owns open_questions & co.; the rogue variant also tries to write settlement
    g.add_node("gap", wrap("gap", lambda s: {"open_questions": [OpenQuestion(question_id=f"Q{len(s.facts)}", text="t", category="cause")],
                                             **({"settlement": None} if rogue else {})}))
    g.add_node("coverage", wrap("coverage", lambda s: {}))
    g.add_node("human_gate", human_gate)
    g.add_edge(START, "policy")
    for a, b in [("policy", "evidence"), ("evidence", "gap"), ("gap", "coverage"), ("coverage", "human_gate"), ("human_gate", END)]:
        g.add_edge(a, b)
    return g.compile(checkpointer=InMemorySaver())


@pytest.fixture(autouse=True)
def _reset():
    CALLS.clear()


def cfg(thread="t1"):
    return {"configurable": {"thread_id": thread}}


def test_typed_state_pauses_at_human_gate_then_resumes():
    graph, c = build(), cfg()
    out = graph.invoke(ClaimState(claim_id="c1"), c)

    assert "__interrupt__" in out, "graph must pause at the human gate"
    assert out["__interrupt__"][0].value["claim_id"] == "c1"
    snap = graph.get_state(c)
    assert snap.next == ("human_gate",)
    assert isinstance(snap.values["facts"][0], Fact), "typed models survive checkpointing"
    assert snap.values["resolution_status"] == "registered", "nothing is approved before a human decides"

    final = graph.invoke(Command(resume={"decision": "approve"}), c)
    assert final["resolution_status"] == "approved"
    assert final["state_version"] == 5                       # policy, evidence, gap, coverage, gate each committed once
    assert graph.get_state(c).next == ()


def test_interrupt_node_is_reexecuted_on_resume():
    """Documents the LangGraph semantic that shapes how the real human gate must be written."""
    graph, c = build(), cfg()
    graph.invoke(ClaimState(claim_id="c1"), c)
    assert CALLS["human_gate"] == 1
    graph.invoke(Command(resume={"decision": "approve"}), c)
    assert CALLS["human_gate"] == 2                          # runs again from its first line


def test_ownership_violation_inside_a_node_stops_the_run():
    graph = build(rogue=True)
    with pytest.raises(OwnershipViolation, match="gap tried to write"):
        graph.invoke(ClaimState(claim_id="c1"), cfg())


def test_new_evidence_reenters_at_chosen_node_and_skips_upstream():
    graph, c = build(), cfg()
    graph.invoke(ClaimState(claim_id="c1"), c)
    graph.invoke(Command(resume={"decision": "request_more_evidence"}), c)
    assert graph.get_state(c).values["resolution_status"] == "awaiting_evidence"
    assert CALLS == Counter(policy=1, evidence=1, gap=1, coverage=1, human_gate=2)
    v_before = graph.get_state(c).values["state_version"]

    # late evidence arrives: record it "as if evidence had just run", so the graph continues at gap -> coverage -> gate
    state = graph.get_state(c).values
    graph.update_state(c, {"facts": [*state["facts"], make_fact(2)]}, as_node="evidence")
    assert graph.get_state(c).next == ("gap",)
    out = graph.invoke(None, c)

    assert "__interrupt__" in out
    assert CALLS["policy"] == 1 and CALLS["evidence"] == 1, "unchanged upstream nodes must NOT re-run"
    assert CALLS["gap"] == 2 and CALLS["coverage"] == 2, "affected nodes re-run"
    values = graph.get_state(c).values
    assert len(values["facts"]) == 2 and values["open_questions"][0].question_id == "Q2"
    assert values["state_version"] > v_before

    # even narrower: only coverage is affected -> re-enter after gap
    graph.invoke(Command(resume={"decision": "approve"}), c)
    graph.update_state(c, {}, as_node="gap")
    assert graph.get_state(c).next == ("coverage",)
    graph.invoke(None, c)
    assert CALLS["gap"] == 2 and CALLS["coverage"] == 3


def test_checkpoint_history_supports_step_level_replay():
    graph, c = build(), cfg()
    graph.invoke(ClaimState(claim_id="c1"), c)
    graph.invoke(Command(resume={"decision": "approve"}), c)

    history = list(graph.get_state_history(c))               # newest first
    assert len(history) >= 6
    versions = [h.values.get("state_version", 0) for h in reversed(history)]
    assert versions == sorted(versions), "state_version is monotonic across checkpoints"
    after_gap = next(h for h in history if h.next == ("coverage",))
    assert after_gap.values["state_version"] == 3 and after_gap.values["open_questions"], "can read the state as of any step"


def test_threads_are_isolated():
    graph = build()
    graph.invoke(ClaimState(claim_id="c1"), cfg("a"))
    graph.invoke(ClaimState(claim_id="c2"), cfg("b"))
    assert graph.get_state(cfg("a")).values["claim_id"] == "c1"
    assert graph.get_state(cfg("b")).values["claim_id"] == "c2"
