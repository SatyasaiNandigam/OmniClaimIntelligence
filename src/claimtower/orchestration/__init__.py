"""orchestration

OWNER   : A (Lead: orchestration + integrations)
TASKS   : T6.1-T6.3, T6.9
PURPOSE : LangGraph graph, supervisor, incremental reassessment, human gate, event bus.

Rules: depend only on claimtower.core (contracts). Never import another
capability's internals - use the signatures in BUILD_PLAN.md section 11.
"""
