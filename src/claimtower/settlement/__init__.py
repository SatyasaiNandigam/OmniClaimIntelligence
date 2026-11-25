"""settlement

OWNER   : C (Policy + coverage + settlement + routing)
TASKS   : T5.3-T5.5
PURPOSE : Deterministic Decimal engine driven by YAML. NO LLM IMPORTS in this package (lint-enforced).

Rules: depend only on claimtower.core (contracts). Never import another
capability's internals - use the signatures in BUILD_PLAN.md section 11.
"""
