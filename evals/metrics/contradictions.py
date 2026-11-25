"""Metrics for the contradiction detector (T4.3), reported by kind (date, amount, cause, extent, other).

Cases: `dev` = the five scenario ground-truth files (S-01 has one labelled cause contradiction, S-02..S-05 none) plus the dev cases in
data/synthetic/contradiction_eval/cases.yaml; `fresh` = the unseen cases of the same file.
A prediction MATCHES a label when its kind is one of the label's acceptable kinds and it cites at least one fact from each of the label's two
sides. A label is matched at most once. Precision = matched predictions / predictions; recall = matched labels / labels.
Also counted: predictions that cite fewer than two facts in total (must be 0: the contract and the detector forbid them).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from claimtower.core.schemas import ClaimHeader, Fact, NotInForce, OpenQuestion, PolicySnapshot
from claimtower.evidence.contradiction import ContradictionDetector
from claimtower.evidence.questions import QuestionRegister

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "data" / "synthetic" / "contradiction_eval" / "cases.yaml"
KINDS = ("date", "amount", "cause", "extent", "other")


@dataclass
class Label:
    kinds: list[str]
    a: set[str]
    b: set[str]


@dataclass
class Case:
    id: str
    split: str
    facts: list[Fact]
    labels: list[Label]
    questions: list[OpenQuestion]


def _scenario_cases(register: QuestionRegister) -> list[Case]:
    out = []
    for s in ["S-01", "S-02", "S-03", "S-04", "S-05"]:
        gt = json.loads((ROOT / "data" / "ground_truth" / f"{s}.json").read_text(encoding="utf-8"))
        d = ROOT / "data" / "fixtures" / s
        header = ClaimHeader(**json.loads((d / "intake.json").read_text(encoding="utf-8"))["claim_header"])
        pol = json.loads((d / "policy.json").read_text(encoding="utf-8")).get("policy_snapshot")
        snap = PolicySnapshot(**pol) if pol else NotInForce(policy_id="?", loss_date=header.loss_date or "2026-01-01", reason="not in force")
        facts = [Fact(**{**f, "evidence_type": "neutral"}) for f in gt["facts"]]
        labels = [Label([c["kind"]], set(c["fact_ids_a"]), set(c["fact_ids_b"])) for c in gt["contradictions"]]
        out.append(Case(s, "dev", facts, labels, register.build(header, snap).questions))
    return out


def _synthetic_cases(register: QuestionRegister) -> list[Case]:
    raw = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    questions = register.build(ClaimHeader(claim_id="X"), PolicySnapshot(policy_id="P", version="1", effective_from="2026-01-01")).questions
    out = []
    for c in raw["cases"]:
        fid = lambda i, cid=c["id"]: f"{cid}/{i}"  # noqa: E731
        facts = [Fact(fact_id=fid(f["id"]), claim_id=c["id"], proposition=f["text"], source_document_id="DOC-1", page=1, evidence_text=f["text"],
                      evidence_type="neutral", related_question_ids=f["tags"], extraction_confidence=0.9) for f in c["facts"]]
        labels = [Label(x["kinds"], {fid(i) for i in x["a"]}, {fid(i) for i in x["b"]}) for x in c["contradictions"]]
        out.append(Case(c["id"], c["split"], facts, labels, questions))
    return out


def load_cases(register: QuestionRegister) -> list[Case]:
    return _scenario_cases(register) + _synthetic_cases(register)


@dataclass
class Tally:
    cases: int = 0
    labels: dict[str, int] = field(default_factory=lambda: defaultdict(int))          # by primary kind
    matched_labels: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    predictions: dict[str, int] = field(default_factory=lambda: defaultdict(int))     # by predicted kind
    matched_predictions: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    false_alarms: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    under_cited: int = 0
    refused: int = 0
    severities: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def summary(self) -> dict:
        def pr(num, den):
            return num / den if den else None

        kinds = {k: {"labelled": self.labels[k], "predicted": self.predictions[k],
                     "precision": pr(self.matched_predictions[k], self.predictions[k]), "recall": pr(self.matched_labels[k], self.labels[k])}
                 for k in KINDS if self.labels[k] or self.predictions[k]}
        tp_p, n_p = sum(self.matched_predictions.values()), sum(self.predictions.values())
        tp_l, n_l = sum(self.matched_labels.values()), sum(self.labels.values())
        return {"cases": self.cases, "labelled": n_l, "predicted": n_p, "precision": pr(tp_p, n_p), "recall": pr(tp_l, n_l), "by_kind": kinds,
                "under_cited": self.under_cited, "refused": self.refused, "severities": dict(self.severities),
                "false_alarms": self.false_alarms, "missed": self.missed}


def evaluate(detector: ContradictionDetector, cases: list[Case]) -> dict[str, Tally]:
    out: dict[str, Tally] = {}
    for case in cases:
        t = out.setdefault(case.split, Tally())
        t.cases += 1
        r = detector.detect(case.facts, case.questions)
        t.refused += len(r.refused)
        for lab in case.labels:
            t.labels[lab.kinds[0]] += 1
        used: set[int] = set()
        for c in r.contradictions:
            t.predictions[c.kind] += 1
            t.severities[c.severity] += 1
            a, b = set(c.fact_ids_a), set(c.fact_ids_b)
            if len(a | b) < 2:
                t.under_cited += 1
            hit = next((i for i, lab in enumerate(case.labels) if i not in used and c.kind in lab.kinds
                        and ((a & lab.a and b & lab.b) or (a & lab.b and b & lab.a))), None)
            if hit is None:
                t.false_alarms.append(f"{case.id}: [{c.kind}/{c.severity}] {c.fact_ids_a} vs {c.fact_ids_b}: {c.explanation[:120]}")
            else:
                used.add(hit)
                t.matched_predictions[c.kind] += 1
                t.matched_labels[case.labels[hit].kinds[0]] += 1
        t.missed += [f"{case.id}: {sorted(lab.a)} vs {sorted(lab.b)} ({lab.kinds[0]})" for i, lab in enumerate(case.labels) if i not in used]
    return out
