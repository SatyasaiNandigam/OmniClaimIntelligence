# ADR-0001: Agent framework

- **Status:** Accepted (2026-11-25)
- **Decision:** LangGraph `1.2.24` (pinned in `pyproject.toml`) with an in-process checkpointer for the prototype.
- **Evidence:** `tests/integration/test_spike.py` (6 tests, run against the pinned version).

## Context

The control tower needs (BUILD_PLAN §3): a typed, versioned `ClaimState`; a supervisor plus specialist agents with *explicit* transitions
(no free-form agent chat); a human approval gate; re-entry at a chosen node when new evidence arrives; and a replayable trace for audit.
BUILD_PLAN §3 asked for these to be proven against current APIs before building on them.

## Decision

Use **LangGraph**. The spike proved each requirement against our real `ClaimState` contract:

| Requirement | Verified behaviour | API used |
|---|---|---|
| Typed state | The Pydantic `ClaimState` is the graph state; nested models (`Fact`, `OpenQuestion`) survive checkpointing as typed objects. `invoke()` returns a `dict`, `get_state().values` holds typed values. | `StateGraph(ClaimState)` |
| Ownership enforcement | A node wrapper calls `core.state.apply_patch`; a node writing a field it does not own raises `OwnershipViolation` and the run stops. Version bumps once per committed node. | `apply_patch`, node wrapper |
| Human gate | The graph pauses with a payload (`__interrupt__`), `get_state().next == ("human_gate",)`; nothing is approved until a human resumes. | `interrupt()`, `Command(resume=...)` |
| Partial re-run | `update_state(config, values, as_node="evidence")` re-enters at `gap`; `policy` and `evidence` are **not** re-executed (call counters prove it). Re-entering `as_node="gap"` runs only `coverage`. | `update_state(..., as_node=)`, `invoke(None, config)` |
| Audit / replay | Every step is a checkpoint; history is monotonic in `state_version` and any step's state can be read back. | `get_state_history()` |
| Isolation | Separate `thread_id`s keep claims independent (use `thread_id = claim_id`). | `config["configurable"]["thread_id"]` |

## Consequences / rules for the real implementation (T6.1-T6.3)

1. **A node that calls `interrupt()` is re-executed from its first line on resume** (verified: the gate ran twice). Everything before `interrupt()` must be
   idempotent and free of side effects (no payment instruction, no task creation, no LLM call). Put side effects *after* the decision, in the next node.
2. **Incremental reassessment = `update_state(as_node=<last unaffected node>)`** followed by `invoke(None, config)`. The supervisor decides the node from the
   dependency map (T6.2); the spike shows the mechanism works, not the map.
3. **Use `thread_id = claim_id`.** State history per claim is then the audit-friendly checkpoint log, complementing our own `InMemoryVersionStore` / audit table
   (which also records agent, rationale and prompt/model versions that checkpoints do not).
4. **Checkpointer:** the spike uses `InMemorySaver`. A persistent saver (SQLite/Postgres) is a separate package and must be added and pinned in T6.1.
   Until then, state does not survive a process restart.
5. Keep agents framework-free: they receive a `ClaimState` and return a `StatePatch` (`agents/base.py`). Only `orchestration/` imports `langgraph`, so
   a framework swap stays inside that folder.

## Alternatives considered (not chosen)

| Option | Why not (for now) |
|---|---|
| Microsoft Agent Framework / Semantic Kernel | Less example code for explicit state-machine flows; choose if the team becomes Azure/.NET-led. |
| CrewAI | Role/task model gives weaker control over state ownership and approval interrupts. |
| OpenAI Agents SDK | Vendor-tied hand-offs, no checkpointed state graph. |

## Risks

- LangGraph APIs move quickly: versions are pinned and `test_spike.py` is the canary; re-run it before any upgrade.
- On Python 3.14 the transitive `langsmith` package emits a `DeprecationWarning` (`asyncio.iscoroutinefunction`); filtered in `pyproject.toml`, harmless.
