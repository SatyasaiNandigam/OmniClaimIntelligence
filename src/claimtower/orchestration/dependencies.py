"""The dependency map (T6.2): which parts of the claim state each node READS, so a re-assessment re-runs only what the new evidence can change.

    digest = input_digest("coverage", state)            # {"policy_snapshot": "ab12...", ...}, one short hash per field the node reads
    changed_fields(previous_digest, digest)             # the fields that differ: the reason a node re-runs, or [] = "skipped (unchanged)"

A node whose inputs are identical to the inputs it left behind after its last successful run (the digest is taken AFTER the node ran, so its own output is not mistaken for a change) would produce
the same output, so the supervisor skips it and says so in the run log
("skipped (unchanged)"); a node whose inputs changed re-runs and the log names the fields that changed ("re-run (reason: facts, documents)").

Two kinds of node are never skipped (value `None`): `intake` and `evidence`, whose real input is outside the claim state (the stored documents), so a hash of the state cannot say
whether they have anything new. Both are incremental inside (an unchanged document is not read again). `policy` is skipped when the header and the facts it reads are unchanged;
what the policy system or the wording library says is outside the state, so a changed policy needs `reassess(from_node="policy")` (nodes before `from_node` are not run, and that node is
re-run only if its inputs changed OR it is the starting node, which always runs).

The sets below are declarations, and `tests/unit/test_dependencies.py` proves them: clearing any field a node does NOT declare never changes what the node returns.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from claimtower.core.schemas import ClaimState

NODE_INPUTS: dict[str, frozenset[str] | None] = {
    "intake": None,
    "policy": frozenset({"claim_header", "facts"}),
    "evidence": None,
    "gap": frozenset({"claim_header", "policy_snapshot", "documents", "facts", "coverage_issues"}),
    "coverage": frozenset({"claim_header", "policy_snapshot", "facts", "open_questions", "contradictions"}),
    "settlement": frozenset({"claim_header", "policy_snapshot", "facts", "open_questions", "contradictions"}),
    "action": frozenset({"claim_header", "policy_snapshot", "missing_evidence", "settlement", "intake_checklist", "proposed_actions", "drafts"}),
    "audit": frozenset(),
    "routing": frozenset({"claim_header", "policy_snapshot", "intake_checklist", "facts", "open_questions", "missing_evidence", "contradictions", "coverage_issues", "settlement"}),
}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:12]


def input_digest(node: str, state: ClaimState, extra: Mapping[str, Any] | None = None) -> dict[str, str] | None:
    """One hash per field the node reads (plus any `extra` input held outside the state, for example a human override), or None when the node is never skipped."""
    fields = NODE_INPUTS.get(node)
    if fields is None:
        return None
    dumped = state.model_dump(mode="json")
    out = {f: _hash(dumped.get(f)) for f in sorted(fields)}
    for k, v in (extra or {}).items():
        out[k] = _hash(v)
    return out


def changed_fields(previous: Mapping[str, str] | None, current: Mapping[str, str]) -> list[str]:
    """The inputs that differ from the last run (all of them when there was no last run)."""
    if previous is None:
        return sorted(current)
    return sorted(k for k in set(previous) | set(current) if previous.get(k) != current.get(k))


def downstream(node: str, order: Iterable[str]) -> list[str]:
    """The nodes after `node` in the pipeline order (what a change at `node` can reach)."""
    nodes = list(order)
    return nodes[nodes.index(node) + 1:]


__all__ = ["NODE_INPUTS", "changed_fields", "downstream", "input_digest"]
