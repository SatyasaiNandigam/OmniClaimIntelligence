"""core

OWNER   : A (Lead: orchestration + integrations)
TASKS   : T0.1-T0.6
PURPOSE : Contracts, state patching, LLM wrapper, config/pack loader. Everyone imports this; changes need Lead approval.

Rules: depend only on claimtower.core (contracts). Never import another
capability's internals - use the signatures in BUILD_PLAN.md section 11.
"""
