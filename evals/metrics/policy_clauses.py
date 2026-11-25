"""Clause-decomposition metrics (T3.1): detection F1 against labelled clauses, type accuracy, and citation correctness.

A predicted clause MATCHES a labelled one when it is on the same document and page and has the same (whitespace-normalised) text.
Detection precision / recall / F1 count matches; type accuracy is measured among matches only, because a clause that was not found has
no type to judge. `unclassified` clauses (the decomposer abstained on the type) count as DETECTED but not as correctly typed, and are
reported separately: abstaining is the safe failure.

Splits: `corpus` = data/synthetic/policies (the documents the decomposer's conventions were first written against),
`fresh` = data/synthetic/policy_eval (unseen documents in other styles, labelled before the decomposer was run on them),
`fresh_llm` = a document whose wording the type rules do not recognise (for the optional LLM typing, 4a). A label of `none` means the
clause is none of the seven types: leaving it unclassified is right and ANY type is wrong.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from claimtower.ingestion.loaders import load, normalise
from claimtower.policy.decompose import PolicyDecomposer, verify_clauses

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "synthetic" / "policies"
FRESH = ROOT / "data" / "synthetic" / "policy_eval"


@dataclass(frozen=True)
class Labelled:
    doc_id: str
    page: int
    heading: str
    type: str
    text: str
    label_id: str


@dataclass
class DocumentCase:
    split: str
    doc_id: str
    filename: str
    data: bytes
    clauses: list[Labelled]


def corpus_cases() -> list[DocumentCase]:
    idx = json.loads((CORPUS / "clause_index.json").read_text(encoding="utf-8"))
    cases = []
    for d in idx["documents"]:
        labelled = [Labelled(c["doc_id"], c["page"], c["heading"], c["clause_type"], normalise(c["text"]), c["clause_id"])
                    for c in idx["clauses"] if c["doc_id"] == d["doc_id"]]
        cases.append(DocumentCase("corpus", d["doc_id"], d["pdf"], (CORPUS / d["pdf"]).read_bytes(), labelled))
    return cases


def fresh_cases() -> list[DocumentCase]:
    cases = []
    for d in yaml.safe_load((FRESH / "labels.yaml").read_text(encoding="utf-8"))["documents"]:
        labelled = [Labelled(d["doc_id"], c["page"], c["heading"], c["type"], normalise(c["text"]), f"{d['doc_id']}:{c['heading']}")
                    for c in d["clauses"]]
        cases.append(DocumentCase(d["split"], d["doc_id"], d["file"], (FRESH / d["file"]).read_bytes(), labelled))
    return cases


def all_cases() -> list[DocumentCase]:
    return [*corpus_cases(), *fresh_cases()]


@dataclass
class Report:
    split: str
    labelled: int = 0
    predicted: int = 0                                  # classified clauses
    unclassified: int = 0
    matched: int = 0                                    # classified AND matching a label
    detected: int = 0                                   # matching a label, classified or not
    type_correct: int = 0
    wrong_types: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    spurious: list[str] = field(default_factory=list)
    abstained: list[str] = field(default_factory=list)
    citation_problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def precision(self) -> float | None:
        n = self.predicted + self.unclassified
        return self.detected / n if n else None

    @property
    def recall(self) -> float | None:
        return self.detected / self.labelled if self.labelled else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        return None if not p or not r else 2 * p * r / (p + r)

    @property
    def type_accuracy(self) -> float | None:
        return self.type_correct / self.detected if self.detected else None

    def summary(self) -> dict:
        r = lambda x: None if x is None else round(x, 4)  # noqa: E731
        return {"labelled": self.labelled, "predicted": self.predicted + self.unclassified, "detected": self.detected,
                "precision": r(self.precision), "recall": r(self.recall), "f1": r(self.f1), "type_accuracy": r(self.type_accuracy),
                "wrong_types": self.wrong_types, "abstained_type": self.abstained, "missed": self.missed, "spurious": self.spurious,
                "citation_problems": self.citation_problems}


def evaluate(decomposer: PolicyDecomposer, cases: list[DocumentCase] | None = None) -> dict[str, Report]:
    reports: dict[str, Report] = {}
    for case in cases if cases is not None else all_cases():
        rep = reports.setdefault(case.split, Report(case.split))
        pages = load(case.data, case.filename).pages
        out = decomposer.decompose(case.doc_id, pages)
        rep.warnings += [f"{case.doc_id}: {w}" for w in out.warnings]
        rep.citation_problems += [f"{case.doc_id}: {p}" for p in verify_clauses(out.clauses, pages)]
        rep.labelled += len(case.clauses)
        rep.predicted += len(out.clauses)
        rep.unclassified += len(out.unclassified)
        want = {(c.page, c.text): c for c in case.clauses}
        seen: set[tuple[int, str]] = set()
        for c in out.clauses:
            key = (c.page or 0, c.text)
            lab = want.get(key)
            if lab is None:
                rep.spurious.append(f"{case.doc_id} p{c.page}: {c.heading} [{c.clause_type}]")
                continue
            seen.add(key)
            rep.detected += 1
            rep.matched += 1
            if lab.type == "none":
                rep.wrong_types.append(f"{lab.label_id}: expected no type, got {c.clause_type} ({out.info[c.clause_id].type_source})")
            elif c.clause_type == lab.type:
                rep.type_correct += 1
            else:
                rep.wrong_types.append(f"{lab.label_id}: expected {lab.type}, got {c.clause_type} ({out.info[c.clause_id].type_source})")
        for u in out.unclassified:
            key = (u.page, u.text)
            lab = want.get(key)
            if lab is None:
                rep.spurious.append(f"{case.doc_id} p{u.page}: {u.heading} [unclassified]")
            else:
                seen.add(key)
                rep.detected += 1
                if lab.type == "none":
                    rep.type_correct += 1                       # a clause that is none of the types, correctly left unclassified
                else:
                    rep.abstained.append(f"{lab.label_id}: expected {lab.type}, abstained")
        rep.missed += [c.label_id for key, c in want.items() if key not in seen]
    return reports
