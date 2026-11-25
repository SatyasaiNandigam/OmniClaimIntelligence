"""Policy agent (T3.5): the claim's policy and loss date -> `policy_snapshot`, the contract in force on the date of loss.

Owns (writes) exactly `policy_snapshot` (core.state.OWNERSHIP). It reads `claim_header` (policy number, loss date) and, only to name an
endorsement whose wording is missing, the facts the evidence agent took from the policy schedule. It is a thin wrapper: everything about
WHAT is in force is decided by `policy.snapshot.SnapshotBuilder` (PAS status, versions, effective dates, precedence).

What the patch says
  * covered       `policy_snapshot` = the snapshot; the rationale names the version, the PAS answer it rests on, the endorsements applied,
                  the endorsements listed but missing, the clauses to abstain on and any unresolved conflict.
  * not covered   `policy_snapshot` = None (the fixture convention, S-03) and the rationale starts `NOT_IN_FORCE: <PAS reason>`. ClaimState
                  has no field for the reason, so the routing step reads it from the rationale / the PAS fact; nothing is invented here.
  * not runnable  no policy number or no loss date yet (a missing date is tracked, not blocking, S-17): `policy_snapshot` = None, so a
                  snapshot built for an OLD date is never mistaken for the current one; the rationale says what is missing.
Errors are not hidden: a PAS failure or a missing BASE wording (`PolicyDocumentMissing`) propagates to the orchestrator, which retries or
sends it to a human. An agent that swallowed them would let the claim go on without a contract.

Idempotent. Re-running with nothing new returns the snapshot it already had (compared without its retrieval timestamp), so reassessment
does not churn the state. A new endorsement wording in the library, a new date or a new policy number gives a new snapshot.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import ClassVar

from claimtower.core.schemas import ClaimState, Fact
from claimtower.core.schemas.policy import NotInForce, PolicySnapshot
from claimtower.core.state import OWNERSHIP, StatePatch
from claimtower.policy.snapshot import Outcome, SnapshotBuilder

_ID = r"[A-Z][A-Z0-9]*-[A-Z0-9]+"
_LISTED = re.compile(rf"\b(?P<id>{_ID})\b[\s:,-]*(?P<title>[A-Za-z][^,;]*?)(?=\s*(?:[,;]|\b{_ID}\b|$))")


def titles_from_facts(facts: Sequence[Fact]) -> dict[str, str]:
    """Endorsement titles as the policy schedule names them: {'END-01': 'Water Damage Sublimit Endorsement'}.

    Read from the verbatim evidence of facts that mention endorsements ('Endorsements attached: END-01 Water Damage Sublimit Endorsement').
    Only used to say which clauses an endorsement WITHOUT wording may affect; a wrong title can at worst add an abstention."""
    titles: dict[str, str] = {}
    for f in facts:
        text = f.evidence_text
        if "endorsement" not in text.lower():
            continue
        after = re.split(r"endorsements?(?: attached)?\s*:", text, maxsplit=1, flags=re.IGNORECASE)
        for m in _LISTED.finditer(after[-1].strip()):
            titles.setdefault(m.group("id"), m.group("title").strip())
    return titles


class PolicyAgent:
    agent_id: ClassVar[str] = "policy"
    owns: ClassVar[frozenset[str]] = OWNERSHIP["policy"]

    def __init__(self, snapshots: SnapshotBuilder) -> None:
        self._snapshots = snapshots
        self.last_outcome: Outcome | None = None            # rich result of the last run (effective set, graph) for in-process callers

    def run(self, state: ClaimState) -> StatePatch:
        self.last_outcome = None
        header = state.claim_header
        policy_no, loss_date = (header.policy_ref, header.loss_date) if header else (None, None)
        if not policy_no or loss_date is None:
            missing = " and ".join(n for n, v in (("policy number", policy_no), ("loss date", loss_date)) if not v)
            return StatePatch(agent_id=self.agent_id, updates={"policy_snapshot": None},
                              rationale=f"NOT_RUN: the claim has no {missing} yet, so no snapshot can be built; any earlier one is cleared")
        outcome = self._snapshots.build(policy_no, loss_date, titles_from_facts(state.facts))
        self.last_outcome = outcome
        if isinstance(outcome.result, NotInForce):
            return StatePatch(agent_id=self.agent_id, updates={"policy_snapshot": None},
                              rationale=f"NOT_IN_FORCE: {outcome.result.reason}")
        new = self._reuse(state.policy_snapshot, outcome.result)
        return StatePatch(agent_id=self.agent_id, updates={"policy_snapshot": new},
                          rationale=self._rationale(outcome, new is state.policy_snapshot))

    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def _reuse(old: PolicySnapshot | None, new: PolicySnapshot) -> PolicySnapshot:
        """Keep the existing snapshot (and its original timestamp) when the new one differs only by when it was retrieved."""
        if old is not None and old.model_copy(update={"retrieved_at": None}) == new.model_copy(update={"retrieved_at": None}):
            return old
        return new

    @staticmethod
    def _rationale(outcome: Outcome, unchanged: bool) -> str:
        snap, eff = outcome.result, outcome.effective
        assert isinstance(snap, PolicySnapshot) and eff is not None
        parts = [f"{'unchanged: ' if unchanged else ''}{snap.policy_id} version {snap.version} ({snap.pas_status}) on {eff.as_of}; "
                 f"{len(snap.clauses)} clause(s) in force; PAS ref {snap.source_system_response_ref}"]
        parts.append("endorsements applied: " + (", ".join(e.endorsement_id for e in snap.endorsements) or "none"))
        if eff.missing_documents:
            parts.append("listed but wording missing: " + ", ".join(eff.missing_documents))
        if eff.abstain_clause_ids:
            parts.append("abstain on: " + ", ".join(eff.abstain_clause_ids))
        if eff.conflicts:
            parts.append("unresolved: " + "; ".join(f"{c.kind} ({', '.join(c.documents)})" for c in eff.conflicts))
        if outcome.model_typed:
            parts.append(f"{len(outcome.model_typed)} clause type(s) decided by a model: needs review")
        if outcome.unclassified:
            parts.append(f"{len(outcome.unclassified)} clause(s) NOT in the snapshot (type undecided)")
        return " | ".join(parts)
