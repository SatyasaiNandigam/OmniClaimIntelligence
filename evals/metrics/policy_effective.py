"""Effective-clause-set metrics (T3.2): does the engine return the clauses really in force on the loss date?

Two sources of expected answers, both fixed before the engine was run:
  * the ground truth of scenarios S-01, S-02, S-04 and S-05 (`snapshot.effective_clause_ids`, `not_effective_clause_ids`,
    `missing_documents`), compared on the scenario catalogue's clauses;
  * `data/synthetic/policy_eval/effective_cases.yaml`: boundary dates, expiry, a missing endorsement, unstated precedence, a date
    disagreement - corpus policies and an unseen one (HARBOUR).
A case passes only if EVERY expectation holds: clauses in and out, who modifies what, what is abstained on, what is missing, the conflict kinds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from claimtower.policy.decompose import PolicyDecomposer
from claimtower.policy.effective import EffectiveSet, EffectiveSetBuilder, EndorsementListing
from claimtower.policy.relations import PolicyGraph, RelationExtractor
from evals.ground_truth import load_all_ground_truth, load_scenarios
from evals.metrics.policy_clauses import FRESH
from evals.metrics.policy_relations import Pool, all_pools, decompose_pool

ROOT = Path(__file__).resolve().parents[2]
CASES = FRESH / "effective_cases.yaml"
PAS = ROOT / "data" / "synthetic" / "pas_mock_db.json"
SCENARIOS = ("S-01", "S-02", "S-04", "S-05")


@dataclass
class CaseResult:
    case_id: str
    why: str
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass
class Built:
    graph: PolicyGraph
    to_label: dict[str, str]
    to_id: dict[str, str]


def _build(decomposer: PolicyDecomposer, extractor: RelationExtractor, pool: Pool, without: set[str] = frozenset()) -> Built:
    keep = Pool(pool.split, pool.pool_id, [d for d in pool.documents if d.doc_id not in without], pool.relations)
    outs, label_of = decompose_pool(decomposer, keep)
    return Built(extractor.build(pool.pool_id, outs), label_of, {v: k for k, v in label_of.items()})


def _labels(b: Built, ids) -> list[str]:
    return sorted(b.to_label.get(i, i) for i in ids)


def _check(res: CaseResult, name: str, got, want) -> None:
    if sorted(got) != sorted(want):
        res.failures.append(f"{name}: expected {sorted(want)}, got {sorted(got)}")


def _modified_by(b: Built, eff: EffectiveSet) -> dict[str, list[str]]:
    return {b.to_label.get(k, k): _labels(b, v) for k, v in eff.modified_by.items()}


def run_cases(decomposer: PolicyDecomposer, extractor: RelationExtractor) -> list[CaseResult]:
    pools = {p.pool_id: p for p in all_pools()}
    engine = EffectiveSetBuilder(extractor)
    results = []
    for case in yaml.safe_load(CASES.read_text(encoding="utf-8"))["cases"]:
        res = CaseResult(case["id"], case["why"])
        results.append(res)
        pool = pools[case["pool"]]
        b = _build(decomposer, extractor, pool)
        base = next(d.doc_id for d in pool.documents if b.graph.kinds[d.doc_id] == "policy_wording")
        listings = [EndorsementListing(x["id"], date.fromisoformat(x["from"]), date.fromisoformat(x["to"]) if x.get("to") else None, x.get("title"))
                    for x in case["listings"]]
        eff = engine.build(b.graph, base, listings, date.fromisoformat(case["as_of"]))
        got = set(_labels(b, eff.clause_ids))
        for key in case["include"]:
            if key not in got:
                res.failures.append(f"missing from the effective set: {key}")
        for key in case["exclude"]:
            if key in got:
                res.failures.append(f"must not be in force: {key}")
        _check(res, "modified_by", [f"{k}<-{','.join(v)}" for k, v in sorted(_modified_by(b, eff).items())],
               [f"{k}<-{','.join(sorted(v))}" for k, v in sorted(case["modified_by"].items()) if v])
        _check(res, "abstain", _labels(b, eff.abstain_clause_ids), case["abstain"])
        _check(res, "missing documents", eff.missing_documents, case["missing"])
        _check(res, "conflict kinds", [c.kind for c in eff.conflicts], case["conflicts"])
    return results


def run_scenarios(decomposer: PolicyDecomposer, extractor: RelationExtractor) -> list[CaseResult]:
    pools = {p.pool_id: p for p in all_pools()}
    sc = load_scenarios()
    gt = load_all_ground_truth()
    pas = {p["policy_no"]: p for p in json.loads(PAS.read_text(encoding="utf-8"))["policies"]}
    catalogue = {c["clause_id"] if isinstance(c, dict) else c.clause_id for c in sc.clause_catalogue}
    engine = EffectiveSetBuilder(extractor)
    titles = {d["doc_id"]: d["title"] for d in json.loads((ROOT / "data/synthetic/policies/clause_index.json").read_text(encoding="utf-8"))["documents"]}
    out = []
    for s in sc.scenarios:
        if s.id not in SCENARIOS:
            continue
        res = CaseResult(s.id, s.title)
        out.append(res)
        snap = gt[s.id].snapshot
        as_of = date.fromisoformat(s.claim["loss_date"])
        version = next(v for v in pas[s.claim["policy_ref"]]["versions"] if v["effective_from"] <= as_of.isoformat() <= v["effective_to"])
        available = s.pas_expectation.get("endorsement_documents_available")
        missing_docs = {e["endorsement_id"] for e in version["endorsements"]} - set(available) if available is not None else set()
        b = _build(decomposer, extractor, pools[version["base_wording_id"]], without=missing_docs)
        listings = [EndorsementListing(e["endorsement_id"], date.fromisoformat(e["effective_from"]),
                                       date.fromisoformat(e["effective_to"]) if e["effective_to"] else None,
                                       titles[e["endorsement_id"]] if e["endorsement_id"] in missing_docs else None)
                    for e in version["endorsements"]]       # a title is known only where the schedule text lists it (S-05)
        eff = engine.build(b.graph, version["base_wording_id"], listings, as_of)
        in_force = set(_labels(b, eff.clause_ids))
        _check(res, "effective clauses (scenario catalogue)", in_force & catalogue, snap.effective_clause_ids)
        for cid in snap.not_effective_clause_ids:
            if cid in in_force:
                res.failures.append(f"not yet effective but returned: {cid}")
        _check(res, "missing documents", eff.missing_documents, snap.missing_documents)
        wanted_abstain = [c for i in gt[s.id].issues if i.status == "abstain" for c in i.clause_ids]
        _check(res, "abstained clauses", _labels(b, eff.abstain_clause_ids), wanted_abstain)
    return out
