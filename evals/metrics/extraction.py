"""Field and table extraction eval (T2.3; reused by the T8.1 harness).

Every case goes through the real pipeline: ingest into a temporary DocumentStore, a human-style type assignment (so extraction is
measured on
its own, not through the classifier), then `store.extract`, reading the persisted result back.

Splits: corpus (rendered T1.1/T1.3 files, data/synthetic/extraction_set/corpus_labels.yaml) | dev | holdout (items.yaml)

Metrics
  field_score      correct / expected: the field-level NORMALISED-MATCH score (ISO dates, plain decimals, canonical ids)
  wrong_values     a value was extracted but differs from the label
  silent_wrong     ...and it was NOT flagged low-confidence: the dangerous failure, target 0
  missed           expected but not extracted
  spurious         extracted although the label says it must be absent
  flag_recall      of the fields that must come out flagged (ambiguous / unparseable / unlabelled), the share that were
  tables           status, items, stated total, tax and flags compared with the label
  required         the reported `missing_required` list equals the label's
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from claimtower.ingestion.extractors.values import plain
from claimtower.ingestion.store import DocumentStore
from evals.ground_truth import load_scenarios

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data" / "synthetic"
SET = SYN / "extraction_set"


@dataclass(frozen=True)
class ECase:
    id: str
    split: str
    filename: str
    data: bytes
    doc_type: str
    expected: dict[str, str | None] = field(default_factory=dict)
    flagged: tuple[str, ...] = ()
    absent: tuple[str, ...] = ()
    missing_required: tuple[str, ...] | None = None
    table: dict[str, Any] | None = None
    note: str = ""


def corpus_cases(root: Path = ROOT) -> list[ECase]:
    syn = root / "data" / "synthetic"
    labels = yaml.safe_load((syn / "extraction_set" / "corpus_labels.yaml").read_text(encoding="utf-8"))["documents"]
    files: dict[str, tuple[str, bytes]] = {}
    for sc in load_scenarios().scenarios:
        for d in sc.documents:
            files[d.document_id] = (d.filename, (syn / "claims" / sc.claim_id / d.filename).read_bytes())
    files["DOC-TPL-1"] = ("invoice_sample.pdf", (syn / "templates" / "invoice_sample.pdf").read_bytes())
    out = []
    for doc_id, lab in labels.items():
        name, data = files[doc_id]
        out.append(ECase(doc_id, "corpus", name, data, lab["doc_type"], lab["expected"], tuple(lab.get("flagged", [])),
                         tuple(lab.get("absent", [])), None, lab.get("table")))
    return out


def item_cases(splits: set[str] | None = None, path: Path = SET / "items.yaml") -> list[ECase]:
    out = []
    for it in yaml.safe_load(path.read_text(encoding="utf-8"))["items"]:
        if splits and it["split"] not in splits:
            continue
        mr = it.get("missing_required")
        out.append(ECase(it["id"], it["split"], f"{it['id']}.txt", it["text"].encode("utf-8"), it["doc_type"], it.get("expected") or {},
                         tuple(it.get("flagged", [])), tuple(it.get("absent", [])), tuple(mr) if mr is not None else None,
                         it.get("table"), it.get("note", "")))
    return out


def all_cases() -> list[ECase]:
    return [*corpus_cases(), *item_cases()]


@dataclass
class Outcome:
    case: str
    field: str
    kind: str                       # correct | wrong | missed | spurious | flag_ok | flag_missed | table_<what> | required_*
    expected: Any = None
    got: Any = None
    low_confidence: bool = False
    detail: str = ""


@dataclass
class Report:
    split: str
    docs: int = 0
    outcomes: list[Outcome] = field(default_factory=list)

    def _k(self, *kinds: str) -> list[Outcome]:
        return [o for o in self.outcomes if o.kind in kinds]

    @staticmethod
    def _ratio(n: int, d: int) -> float | None:
        return round(n / d, 4) if d else None

    @property
    def field_score(self) -> float | None:
        return self._ratio(len(self._k("correct")), len(self._k("correct", "wrong", "missed")))

    @property
    def silent_wrong(self) -> list[Outcome]:
        return [o for o in self._k("wrong") if not o.low_confidence]

    @property
    def flag_recall(self) -> float | None:
        return self._ratio(len(self._k("flag_ok")), len(self._k("flag_ok", "flag_missed")))

    def table_score(self) -> float | None:
        t = [o for o in self.outcomes if o.kind.startswith("table_")]
        return self._ratio(sum(o.kind.endswith("_ok") for o in t), len(t))

    def summary(self) -> dict[str, Any]:
        def fmt(o: Outcome) -> str:
            return f"{o.case}.{o.field}: expected {o.expected!r}, got {o.got!r}" + (f" [{o.detail}]" if o.detail else "")

        return {
            "docs": self.docs, "field_score": self.field_score, "expected_fields": len(self._k("correct", "wrong", "missed")),
            "wrong_values": [fmt(o) + (" (flagged)" if o.low_confidence else " (SILENT)") for o in self._k("wrong")],
            "silent_wrong": len(self.silent_wrong), "missed": [fmt(o) for o in self._k("missed")],
            "spurious": [fmt(o) for o in self._k("spurious")], "flag_recall": self.flag_recall,
            "flag_missed": [fmt(o) for o in self._k("flag_missed")], "table_score": self.table_score(),
            "table_failures": [fmt(o) for o in self.outcomes if o.kind.startswith("table_") and not o.kind.endswith("_ok")],
            "required_failures": [fmt(o) for o in self._k("required_wrong")],
        }


def _table_outcomes(case: ECase, result: Any, out: list[Outcome]) -> None:
    exp, tab = case.table, result.table
    if exp is None:
        return
    if tab is None:
        out.append(Outcome(case.id, "table", "table_status_wrong", exp.get("status"), None))
        return

    def add(what: str, ok: bool, e: Any, g: Any) -> None:
        out.append(Outcome(case.id, f"table.{what}", f"table_{'ok' if ok else 'wrong'}", e, g))

    add("status", tab.status == exp["status"], exp["status"], tab.status)
    if "items" in exp:
        got = [[i.description, plain(i.amount)] for i in tab.items]
        add("items", got == [list(x) for x in exp["items"]], exp["items"], got)
    if "total" in exp:
        add("total", tab.stated_total is not None and plain(tab.stated_total) == exp["total"], exp["total"],
            plain(tab.stated_total) if tab.stated_total is not None else None)
    if "tax" in exp:
        add("tax", plain(tab.tax_total) == exp["tax"], exp["tax"], plain(tab.tax_total))
    if "flags" in exp:
        add("flags", all(f in tab.flags for f in exp["flags"]), exp["flags"], tab.flags)


def evaluate(extractor: Any, cases: list[ECase], allowed_types: list[str]) -> dict[str, Report]:
    reports: dict[str, Report] = {}
    with tempfile.TemporaryDirectory() as tmp:
        store = DocumentStore(Path(tmp) / "store")
        for n, c in enumerate(cases, start=1):
            doc_id = f"E-{n:04d}"
            store.ingest(f"EVAL-{c.split.upper()}", c.filename, c.data, document_id=doc_id)
            store.set_type(doc_id, c.doc_type, by="eval", allowed=allowed_types)
            result = store.extract(doc_id, extractor)
            rep = reports.setdefault(c.split, Report(c.split))
            rep.docs += 1
            got = {f.name: f for f in result.fields}
            for name, exp in c.expected.items():
                f = got.get(name)
                if f is None:
                    rep.outcomes.append(Outcome(c.id, name, "missed", exp))
                elif f.normalised == exp:
                    rep.outcomes.append(Outcome(c.id, name, "correct", exp, f.normalised, f.low_confidence))
                else:
                    rep.outcomes.append(Outcome(c.id, name, "wrong", exp, f.normalised, f.low_confidence, ",".join(f.flags)))
            for name in c.absent:
                if name in got:
                    rep.outcomes.append(Outcome(c.id, name, "spurious", None, got[name].normalised, got[name].low_confidence))
            for name in c.flagged:
                f = got.get(name)
                rep.outcomes.append(Outcome(c.id, name, "flag_ok" if f is not None and f.low_confidence else "flag_missed", True,
                                            f.low_confidence if f else None))
            if c.missing_required is not None:
                ok = sorted(result.missing_required) == sorted(c.missing_required)
                rep.outcomes.append(Outcome(c.id, "missing_required", "required_ok" if ok else "required_wrong",
                                            sorted(c.missing_required), sorted(result.missing_required)))
            _table_outcomes(c, result, rep.outcomes)
    return reports

