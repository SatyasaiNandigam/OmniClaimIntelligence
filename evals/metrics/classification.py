"""Document-type classification eval (T2.2; reused by the T8.1 harness).

Every case goes through the REAL pipeline: ingest into a temporary DocumentStore, then classify through the store.
Splits:  corpus (rendered T1.1/T1.3 files)  |  dev / holdout / fresh (data/synthetic/classification_set/items.yaml)

Metrics (all counts are over the cases of one split)
  accuracy              correct / n, where `unknown` is the correct answer for must-abstain cases
  answered_precision    of the answers that were NOT `unknown`, the share that were right   <- the "never guessed" number
  wrong_answers         answered with a type, and wrong (the dangerous failure: a confident guess)
  abstain_recall        of the cases whose right answer is `unknown`, the share the classifier abstained on
  over_abstained        right answer was a type but the classifier said `unknown` (safe, but costs a human a click)
"""
from __future__ import annotations

import json
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import yaml

from claimtower.ingestion.store import DocumentStore
from evals.ground_truth import load_scenarios

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data" / "synthetic"
ITEMS_FILE = SYN / "classification_set" / "items.yaml"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Case:
    id: str
    split: str
    filename: str
    data: bytes
    expected: str
    note: str = ""
    ocr_text: str | None = None          # simulated (perfect) OCR transcription, for the scanned-then-OCR case


@dataclass
class Outcome:
    case: Case
    got: str
    confidence: float
    code: str
    reason: str
    snippet: str | None

    @property
    def correct(self) -> bool:
        return self.got == self.case.expected


@dataclass
class Report:
    split: str
    outcomes: list[Outcome] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.outcomes)

    def _ratio(self, num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    @property
    def accuracy(self) -> float | None:
        return self._ratio(sum(o.correct for o in self.outcomes), self.n)

    @property
    def answered(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.got != UNKNOWN]

    @property
    def answered_precision(self) -> float | None:
        return self._ratio(sum(o.correct for o in self.answered), len(self.answered))

    @property
    def wrong_answers(self) -> list[Outcome]:
        return [o for o in self.answered if not o.correct]

    @property
    def abstain_recall(self) -> float | None:
        must = [o for o in self.outcomes if o.case.expected == UNKNOWN]
        return self._ratio(sum(o.got == UNKNOWN for o in must), len(must))

    @property
    def over_abstained(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.got == UNKNOWN and o.case.expected != UNKNOWN]

    def per_type(self) -> dict[str, dict[str, Any]]:
        types = sorted({o.case.expected for o in self.outcomes} | {o.got for o in self.outcomes})
        out = {}
        for t in types:
            tp = sum(1 for o in self.outcomes if o.got == t and o.case.expected == t)
            got, exp = sum(1 for o in self.outcomes if o.got == t), sum(1 for o in self.outcomes if o.case.expected == t)
            out[t] = {"support": exp, "precision": self._ratio(tp, got), "recall": self._ratio(tp, exp)}
        return out

    def confusion(self) -> dict[str, dict[str, int]]:
        m: dict[str, Counter] = defaultdict(Counter)
        for o in self.outcomes:
            m[o.case.expected][o.got] += 1
        return {k: dict(v) for k, v in sorted(m.items())}

    def confidence_summary(self) -> dict[str, float | None]:
        right = [o.confidence for o in self.answered if o.correct]
        wrong = [o.confidence for o in self.wrong_answers]
        return {"mean_confidence_correct": round(sum(right) / len(right), 3) if right else None,
                "min_confidence_correct": round(min(right), 3) if right else None,
                "mean_confidence_wrong": round(sum(wrong) / len(wrong), 3) if wrong else None}

    def summary(self) -> dict[str, Any]:
        return {"n": self.n, "accuracy": self.accuracy, "answered_precision": self.answered_precision,
                "wrong_answers": [f"{o.case.id}: expected {o.case.expected}, got {o.got} ({o.confidence:.2f})" for o in self.wrong_answers],
                "abstain_recall": self.abstain_recall,
                "over_abstained": [f"{o.case.id}: expected {o.case.expected} [{o.code}]" for o in self.over_abstained],
                **self.confidence_summary()}


# ------------------------------------------------------------------------------------------------ cases
def corpus_cases(root: Path = ROOT) -> list[Case]:
    syn, cases = root / "data" / "synthetic", []
    for sc in load_scenarios().scenarios:
        for d in sc.documents:
            data = (syn / "claims" / sc.claim_id / d.filename).read_bytes()
            cases.append(Case(f"{sc.id}/{d.filename}", "corpus", d.filename, data, d.doc_type))
    index = json.loads((syn / "policies" / "clause_index.json").read_text(encoding="utf-8"))
    for d in index["documents"]:
        expected = "policy_wording" if d["kind"] == "base_wording" else "endorsement"
        for fname in (d["pdf"], d["docx"]):
            cases.append(Case(f"policy/{fname}", "corpus", fname, (syn / "policies" / fname).read_bytes(), expected))
    tpl = json.loads((syn / "templates" / "manifest.json").read_text(encoding="utf-8"))
    for d in tpl["documents"]:
        data = (syn / "templates" / d["filename"]).read_bytes()
        cases.append(Case(f"template/{d['filename']}", "corpus", d["filename"], data, d["doc_type"]))
    for p in sorted([*syn.glob("claims/*/photos/*.jpg"), *syn.glob("photos/quality_set/*.jpg")]):
        cases.append(Case(f"photo/{p.parent.parent.name if p.parent.name == 'photos' else p.parent.name}/{p.name}", "corpus", p.name,
                          p.read_bytes(), "photo"))
    manifest = json.loads((syn / "claims" / "CLM-0002" / "manifest.json").read_text(encoding="utf-8"))
    v = manifest["variants"][0]
    scan = (syn / "claims" / "CLM-0002" / v["filename"]).read_bytes()
    source = next(d for s in load_scenarios().scenarios for d in s.documents if d.document_id == v["variant_of"])
    cases.append(Case("scan/before-ocr", "corpus", v["filename"], scan, UNKNOWN, "image-only page: must wait for OCR"))
    # simulated OCR returns the page text WITH its line structure, as a real engine / the transcription prompt would
    cases.append(Case("scan/after-simulated-ocr", "corpus", v["filename"], scan, "surveyor_report", "perfect OCR text, capped confidence",
                      ocr_text=source.pages[0].strip()))
    return cases


def item_cases(splits: set[str] | None = None, path: Path = ITEMS_FILE) -> list[Case]:
    cases = []
    for it in yaml.safe_load(path.read_text(encoding="utf-8"))["items"]:
        if splits and it["split"] not in splits:
            continue
        if it["kind"] == "email":
            m = EmailMessage()
            for k, v in it["headers"].items():
                m[k] = v
            m.set_content(it["body"])
            data, fname = bytes(m), f"{it['id']}.eml"
        else:
            data, fname = it["text"].encode("utf-8"), f"{it['id']}.txt"
        cases.append(Case(it["id"], it["split"], fname, data, it["expected"], it.get("note", "")))
    return cases


def all_cases() -> list[Case]:
    return [*corpus_cases(), *item_cases()]


# ------------------------------------------------------------------------------------------------ evaluation
class _FixedOcr:
    name = "simulated-perfect-ocr"

    def __init__(self, text: str) -> None:
        self.text = text

    def recognise(self, image: bytes, mime: str) -> str:
        return self.text


def evaluate(classifier: Any, cases: list[Case]) -> dict[str, Report]:
    """Run every case through ingest -> classify in a throw-away store. Returns one Report per split."""
    reports: dict[str, Report] = {}
    with tempfile.TemporaryDirectory() as tmp:
        store = DocumentStore(Path(tmp) / "store")
        for c in cases:
            rec = store.ingest(f"EVAL-{c.split.upper()}", c.filename, c.data, document_id=f"E-{len(store._docs) + 1:04d}",
                               ocr=_FixedOcr(c.ocr_text) if c.ocr_text else None, classifier=classifier)
            s = store.stored(rec.document_id)
            reports.setdefault(c.split, Report(c.split)).outcomes.append(
                Outcome(c, rec.detected_type, rec.type_confidence, s.type_code or "", s.type_reason or "", rec.type_evidence_snippet))
    return reports

