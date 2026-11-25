"""Snapshot metrics (T3.3): does `get_snapshot` return the contract in force on the loss date - or an explicit NotInForce?

Expected answers, both fixed before the builder was written:
  * `data/synthetic/policy_eval/snapshot_cases.yaml`: lapse boundary, cancellation and reinstatement, a mid-term version change, an unknown
    policy number, a listed endorsement without wording, an endorsement that is not yet effective or is replaced by a later one;
  * the scenario ground truth (S-01..S-05): the effective clause set on the scenario catalogue, and for S-03 the exact fixture `NotInForce`.
A case passes only if every expectation holds (outcome, version, status, endorsements, reason / precedence text).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

from claimtower.core.schemas.policy import NotInForce, PolicySnapshot
from claimtower.integrations.mock.data import load_pas_db
from claimtower.policy.decompose import PolicyDecomposer
from claimtower.policy.relations import RelationExtractor
from claimtower.policy.snapshot import PolicyLibrary, SnapshotBuilder
from evals.ground_truth import load_all_ground_truth, load_scenarios
from evals.metrics.policy_clauses import FRESH
from evals.metrics.policy_relations import all_pools, decompose_pool

ROOT = Path(__file__).resolve().parents[2]
CASES = FRESH / "snapshot_cases.yaml"
FIXED_NOW = datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)
SCENARIOS = ("S-01", "S-02", "S-03", "S-04", "S-05")


@dataclass
class CaseResult:
    case_id: str
    why: str
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


class MockPas:
    """The PAS read path over the synthetic database (the real adapter arrives with T6.5)."""

    def __init__(self) -> None:
        self._db = load_pas_db()

    def get_policy_version(self, policy_no: str, as_of: date):
        return self._db.lookup(policy_no, as_of)


def corpus_library(decomposer: PolicyDecomposer) -> tuple[PolicyLibrary, dict[str, str]]:
    """Every corpus document decomposed, plus {clause id -> labelled id} so results can be compared with the ground truth."""
    docs, to_label = [], {}
    for pool in (p for p in all_pools() if p.split == "corpus"):
        outs, label_of = decompose_pool(decomposer, pool)
        for o in outs:
            if all(o.doc_id != d.doc_id for d in docs):
                docs.append(o)
        to_label.update(label_of)
    return PolicyLibrary(docs), to_label


def _builder(library: PolicyLibrary, extractor: RelationExtractor) -> SnapshotBuilder:
    return SnapshotBuilder(MockPas(), library, extractor, clock=lambda: FIXED_NOW)


def run_cases(decomposer: PolicyDecomposer, extractor: RelationExtractor) -> list[CaseResult]:
    library, to_label = corpus_library(decomposer)
    out = []
    for c in yaml.safe_load(CASES.read_text(encoding="utf-8"))["cases"]:
        res = CaseResult(c["id"], c["why"])
        out.append(res)
        lib = library.without(*c.get("unavailable", []))
        got = _builder(lib, extractor).get_snapshot(c["policy"], date.fromisoformat(c["as_of"]), c.get("titles"))
        if c["outcome"] == "not_in_force":
            if not isinstance(got, NotInForce):
                res.failures.append(f"expected NotInForce, got a snapshot (version {got.version})")
                continue
            res.failures += [f"reason lacks '{t}': {got.reason}" for t in c.get("reason", []) if t not in got.reason]
            continue
        if not isinstance(got, PolicySnapshot):
            res.failures.append(f"expected a snapshot, got NotInForce: {got.reason}")
            continue
        for name, want, have in (("version", c["version"], got.version), ("pas_status", c["status"], got.pas_status),
                                 ("endorsements", sorted(c["endorsements"]), sorted(e.endorsement_id for e in got.endorsements))):
            if want != have:
                res.failures.append(f"{name}: expected {want}, got {have}")
        rules = " | ".join(got.precedence_rules)
        res.failures += [f"precedence_rules lack '{t}'" for t in c.get("rules", []) if t not in rules]
        if not got.source_system_response_ref or got.retrieved_at is None or got.mock is not True:
            res.failures.append("the PAS response reference, timestamp or mock label is missing")
    return out


def run_scenarios(decomposer: PolicyDecomposer, extractor: RelationExtractor) -> list[CaseResult]:
    library, to_label = corpus_library(decomposer)
    sc, gt = load_scenarios(), load_all_ground_truth()
    catalogue = {c["clause_id"] if isinstance(c, dict) else c.clause_id for c in sc.clause_catalogue}
    out = []
    for s in sc.scenarios:
        if s.id not in SCENARIOS:
            continue
        res = CaseResult(s.id, s.title)
        out.append(res)
        snap = gt[s.id].snapshot
        avail = s.pas_expectation.get("endorsement_documents_available")
        lib = library.without(*[e for e in ("END-01", "END-02", "END-03", "END-04", "END-05") if avail is not None and e not in avail])
        titles = {"END-01": "Water Damage Sublimit Endorsement"} if avail == [] else None
        got = _builder(lib, extractor).get_snapshot(s.claim["policy_ref"], date.fromisoformat(s.claim["loss_date"]), titles)
        if snap.status != "in_force":
            fixture = json.loads((ROOT / f"data/fixtures/{s.id}/get_snapshot_result.json").read_text(encoding="utf-8"))["not_in_force"]
            if not isinstance(got, NotInForce) or got.model_dump(mode="json") != fixture:
                res.failures.append(f"expected the fixture NotInForce {fixture}, got {got!r}")
            continue
        if not isinstance(got, PolicySnapshot):
            res.failures.append(f"expected a snapshot, got NotInForce: {got.reason}")
            continue
        labels = {to_label.get(c.clause_id, c.clause_id) for c in got.clauses} & catalogue
        if labels != set(snap.effective_clause_ids):
            res.failures.append(f"effective clauses (catalogue): expected {sorted(snap.effective_clause_ids)}, got {sorted(labels)}")
        if got.pas_status != snap.status:
            res.failures.append(f"pas_status: expected {snap.status}, got {got.pas_status}")
        rules = " | ".join(got.precedence_rules)
        for doc in snap.missing_documents:
            if f"Endorsement {doc} is listed in the schedule but its wording is not available" not in rules:
                res.failures.append(f"the missing endorsement {doc} is not reported in precedence_rules")
    return out

