"""Tasks (T6.4): the request a planned `create_task` action would send to the claims system. Built here, sent only by the claims adapter (T6.5-T6.6).

    req = task_request(action, planned, state)         # -> TaskRequest (a frozen integration contract), idempotent on the action's key

The task description lists exactly what is needed (the items of the evidence requests), so the receiver does not have to read the claim.
"""
from __future__ import annotations

from claimtower.actions.planner import ActionTemplates, PlannedAction
from claimtower.core.schemas import ClaimState, ProposedAction, TaskRequest


def task_request(action: ProposedAction, planned: PlannedAction, templates: ActionTemplates, state: ClaimState) -> TaskRequest:
    role = planned.party.split("_from_")[-1] if "_from_" in planned.party else planned.party
    role = next((r for r, spec in templates.task_roles.items() if planned.party.startswith(spec["task"])), role)
    title = planned.title or planned.party.replace("_", " ")
    body = f"{title} for claim {state.claim_id}. Needed:\n" + "\n".join(f"- {i}" for i in planned.items) if planned.items else f"{title} for claim {state.claim_id}."
    return TaskRequest(claim_id=state.claim_id, task_type=planned.party, assignee_role=role, description=body, idempotency_key=action.idempotency_key,
                       correlation_id=state.correlation_id)


__all__ = ["task_request"]
