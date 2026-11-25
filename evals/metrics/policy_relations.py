"""Relation metrics (T3.2): precision / recall of extracted clause relations against labelled ones, per policy.

A policy ("pool") = one wording plus the endorsements attached to it. Predicted clauses are aligned to labelled clauses by document,
page and text (the same rule as the decomposition metrics), so a relation is judged on WHICH clauses it joins, not on the ids we invented.

Three kinds of prediction:
  * matched      a labelled relation was found                         -> true positive
  * audited      not labelled, but a person judged it correct          -> reported separately, see evals/policy_relation_audit.yaml
  * unlabelled   not labelled and not audited as correct               -> counted against precision
Labels are minimal by design (the relations that matter for the water-damage scenarios), so unlabelled predictions are expected; the
audit file is how they are judged, openly, not by loosening the metric. An audit entry can also say `invalid`.

Splits: `corpus` (policies the extractor's conventions were first written against) and `fresh` (unseen documents and labels, written
before the extractor was run on them).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from claimtower.ingestion.loaders import load
from claimtower.policy.decompose import Decomposition, PolicyDecomposer
from claimtower.policy.relations import PolicyGraph, RelationExtractor
from evals.metrics.policy_clauses import CORPUS, FRESH, DocumentCase, corpus_cases, fresh_cases

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "evals" / "policy_relation_audit.yaml"

Triple = tuple[str, str, str]


@dataclass
class Pool:
    split: str
    pool_id: str
    documents: list[DocumentCase]
    relations: list[Triple]                               # labelled (source label, target label, relation)


def corpus_pools() -> list[Pool]:
    idx = json.loads((CORPUS / "clause_index.json").read_text(encoding="utf-8"))
    cases = {c.doc_id: c for c in corpus_cases()}
    pools = []
    for d in idx["documents"]:
        if d["kind"] != "base_wording":
            continue
        docs = [d["doc_id"], *[e["doc_id"] for e in idx["documents"] if d["doc_id"] in e.get("applies_to", [])]]
        ids = {c.label_id for k in docs for c in cases[k].clauses}
        rels = [(r["source"], r["target"], r["relation"]) for r in idx["relations"] if r["source"] in ids and r["target"] in ids]
        pools.append(Pool("corpus", d["doc_id"], [cases[k] for k in docs], rels))
    return pools


def fresh_pools() -> list[Pool]:
    cases = {c.doc_id: c for c in fresh_cases()}
    out = []
    for p in yaml.safe_load((FRESH / "labels.yaml").read_text(encoding="utf-8"))["pools"]:
        out.append(Pool("fresh", p["id"], [cases[k] for k in p["documents"]], [(r["source"], r["target"], r["relation"]) for r in p["relations"]]))
    return out


def all_pools() -> list[Pool]:
    return [*corpus_pools(), *fresh_pools()]


def load_audit(path: Path = AUDIT) -> dict[Triple, str]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {(a["source"], a["target"], a["relation"]): a["verdict"] for a in data.get("audited", [])}


def decompose_pool(decomposer: PolicyDecomposer, pool: Pool) -> tuple[list[Decomposition], dict[str, str]]:
    """Decompose the pool's documents; also return {predicted clause id -> labelled clause label} by (document, page, text)."""
    outs, label_of = [], {}
    for case in pool.documents:
        out = decomposer.decompose(case.doc_id, load(case.data, case.filename).pages)
        outs.append(out)
        want = {(c.page, c.text): c.label_id for c in case.clauses}
        for c in out.clauses:
            if (c.page, c.text) in want:
                label_of[c.clause_id] = want[(c.page, c.text)]
    return outs, label_of


@dataclass
class Report:
    split: str
    labelled: int = 0
    predicted: int = 0
    matched: int = 0
    missed: list[str] = field(default_factory=list)
    audited_valid: list[str] = field(default_factory=list)
    unlabelled: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)
    by_type: dict[str, dict[str, int]] = field(default_factory=dict)

    def _t(self, rel: str) -> dict[str, int]:
        return self.by_type.setdefault(rel, {"labelled": 0, "matched": 0, "predicted": 0})

    @property
    def recall(self) -> float | None:
        return self.matched / self.labelled if self.labelled else None

    @property
    def precision(self) -> float | None:
        return self.matched / self.predicted if self.predicted else None

    @property
    def lenient_precision(self) -> float | None:
        """Precision if the audited-valid extras are counted as correct."""
        return (self.matched + len(self.audited_valid)) / self.predicted if self.predicted else None

    def summary(self) -> dict:
        r = lambda x: None if x is None else round(x, 4)  # noqa: E731
        return {"labelled": self.labelled, "predicted": self.predicted, "matched": self.matched, "recall": r(self.recall),
                "precision": r(self.precision), "precision_with_audited": r(self.lenient_precision), "missed": self.missed,
                "audited_valid": self.audited_valid, "unlabelled": self.unlabelled, "ambiguous": self.ambiguous, "by_type": self.by_type}


def evaluate(decomposer: PolicyDecomposer, extractor: RelationExtractor, pools: list[Pool] | None = None,
             audit: dict[Triple, str] | None = None) -> dict[str, Report]:
    audit = load_audit() if audit is None else audit
    reports: dict[str, Report] = {}
    for pool in pools if pools is not None else all_pools():
        rep = reports.setdefault(pool.split, Report(pool.split))
        outs, label_of = decompose_pool(decomposer, pool)
        graph: PolicyGraph = extractor.build(pool.pool_id, outs)
        predicted: dict[Triple, str] = {}
        for r in graph.relations:
            if r.source in label_of and r.target in label_of:
                predicted[(label_of[r.source], label_of[r.target], r.relation)] = f"{r.basis}: {r.evidence}"
        labelled = set(pool.relations)
        rep.labelled += len(labelled)
        rep.predicted += len(predicted)
        for t in labelled:
            rep._t(t[2])["labelled"] += 1
        for t, why in predicted.items():
            rep._t(t[2])["predicted"] += 1
            line = f"{t[0]} -{t[2]}-> {t[1]}   [{why}]"
            if t in labelled:
                rep.matched += 1
                rep._t(t[2])["matched"] += 1
            elif audit.get(t) == "valid":
                rep.audited_valid.append(line)
            else:
                rep.unlabelled.append(line + ("  (audited: invalid)" if audit.get(t) == "invalid" else ""))
        rep.missed += [f"{a} -{rel}-> {b}" for a, b, rel in sorted(labelled - set(predicted))]
        inv = {v: k for k, v in label_of.items()}
        rep.ambiguous += [f"{label_of.get(a.source, a.source)} ({a.relation}): {a.reason}" for a in graph.ambiguities]
        del inv
    return reports
