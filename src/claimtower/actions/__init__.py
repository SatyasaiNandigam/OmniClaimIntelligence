"""actions

OWNER   : A (Lead: orchestration + integrations)
TASKS   : T6.4
PURPOSE : Next-best-action planner, consolidated request drafts, idempotent tasks.

Rules: depend only on claimtower.core (contracts). Never import another
capability's internals - use the signatures in BUILD_PLAN.md section 11.
"""
