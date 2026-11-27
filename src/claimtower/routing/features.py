"""The routing features of a claim (T5.6): what the claim state says, in the vocabulary of configs/routing_rules.yaml. Rules only; no model, no extraction confidence.

    features, assumed = derive_features(state, material_categories, amount_issue_ids)

  policy_status / policy_match   no snapshot although the claim has a policy reference -> not_in_force; no policy reference at all -> mismatch (it cannot be matched)
  policy_certainty               partial when the policy side says a wording is not available (snapshot note or an evidence request to the policy system)
  evidence_completeness          major_gap: a material question (categories `cause`, `timing`: SAMPLE) is not settled; minor_gap: another question is not settled or
                                 an evidence request is open; complete otherwise. `unverified` / `possible` count as not settled: nobody has judged them yet
  contradiction_severity         the highest severity among the recorded contradictions
  coverage_certainty             not_assessed (no policy in force); resolved only when every cover / exclusion / condition issue is `established` (limits and amounts
                                 are settlement matters, not coverage certainty); unresolved otherwise
  duplicate_status               clear_or_benign until duplicate detection exists (T6.8): listed in `assumed`
  specialist_required            false until a source exists (fraud / vision flags): listed in `assumed`
  exposure_inr                   the assessed loss when settlement has one, else the largest single amount among the amount-question facts, else unknown (None)
"""
from __future__ import annotations

import re
from collections.abc import Collection

from claimtower.core.schemas import ClaimState

SEVERITY_ORDER = ("none", "low", "medium", "high")
SETTLED = {"established"}
AMOUNT = re.compile(r"(?:INR|Rs\.?|₹)\s*([\d,]+(?:\.\d+)?)")
UNAVAILABLE = re.compile(r"not available|wording is missing|missing wording", re.IGNORECASE)


def derive_features(state: ClaimState, material_categories: Collection[str] = ("cause", "timing"), amount_issue_ids: Collection[str] = ()) -> tuple[dict, list[str]]:
    h, snap = state.claim_header, state.policy_snapshot
    in_force = snap is not None
    f: dict = {"policy_status": "in_force" if in_force else "not_in_force",
               "policy_match": "mismatch" if (not in_force and not (h and h.policy_ref)) or any(i.item_id == "policy:exact_match" and not i.received for i in state.intake_checklist) else "exact"}
    partial = bool(snap and any(UNAVAILABLE.search(r) for r in snap.precedence_rules)) or any(m.minimum_evidence_request.addressee_role == "policy_administration"
                                                                                               for m in state.missing_evidence)
    f["policy_certainty"] = "partial" if partial else "exact"
    material_open = any(q.category in material_categories and q.status not in SETTLED for q in state.open_questions)
    other_open = any(q.category not in material_categories and q.status not in SETTLED for q in state.open_questions) or bool(state.missing_evidence)
    f["evidence_completeness"] = "major_gap" if material_open else "minor_gap" if other_open else "complete"
    sev = [c.severity for c in state.contradictions]
    f["contradiction_severity"] = max(sev, key=SEVERITY_ORDER.index) if sev else "none"
    judged = [i for i in state.coverage_issues if i.issue_id not in amount_issue_ids]
    f["coverage_certainty"] = "not_assessed" if not in_force or not judged else "resolved" if all(i.status in SETTLED for i in judged) else "unresolved"
    f["duplicate_status"] = "clear_or_benign"
    f["specialist_required"] = False
    exposure = state.settlement.assessed_loss if state.settlement and state.settlement.assessed_loss is not None else None
    if exposure is None:
        amounts = [float(m.group(1).replace(",", "")) for x in state.facts if "Q-AMOUNT" in x.related_question_ids for m in [AMOUNT.search(x.proposition)] if m]
        exposure = max(amounts) if amounts else None
    f["exposure_inr"] = float(exposure) if exposure is not None else None
    assumed = ["duplicate_status (duplicate detection T6.8 is not built: assumed clear_or_benign)", "specialist_required (no source yet: assumed false)"]
    if f["exposure_inr"] is None:
        assumed.append("exposure_inr (no amount known: unknown, so the claim cannot be simple)")
    return f, assumed


__all__ = ["derive_features"]
