"""Rule-based field extraction (T2.3): 'Label: value' lines -> typed, located, scored values.

For each field the pack declares (document_types.yaml) the extractor looks for a logical line whose label is one of the field's labels,
types the remainder with the parsers in `values.py`, and records WHERE it came from (page, region, the verbatim line).

Never silent
  * a required field that is not found is listed in `missing_required`
  * a label whose value will not parse is KEPT (raw value, normalised=None, confidence lowered, flag 'unparseable')
  * a document that states two different values for one field keeps the first, lists the others in `alternatives`, flags it and lowers
    the confidence (the contradiction detector, T4.3, needs them)
  * every value below `low_confidence` is flagged `low_confidence=True` and named in `low_confidence_fields`; nothing is dropped

Confidence is EXTRACTION confidence (how sure we are that this text is the field's value), not a probability about coverage:
    canonical label: label_confidence | synonym label: synonym_confidence | unlabelled body text: body_confidence
    | unparseable: unparseable_confidence
    x ambiguous_factor for an ambiguous date, x conflict_factor for conflicting values, capped by ocr_confidence_cap for OCR text.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from claimtower.core.config import DocumentTypes, FieldSpec
from claimtower.core.schemas.common import BBox
from claimtower.ingestion.extractors import values as V
from claimtower.ingestion.extractors.lines import Line, PageLike, logical_lines
from claimtower.ingestion.extractors.models import ExtractedField, ExtractionResult
from claimtower.ingestion.extractors.table import extract_table

_LABEL_LINE = re.compile(r"^(?P<label>[^:]{1,60}):\s*(?P<value>.*)$")


# A reading that involved a judgement is ALWAYS flagged for review, whatever the numeric confidence ends up being.
ALWAYS_FLAG = frozenset({"ambiguous_day_month", "assumed_month_first", "two_digit_year", "unparseable", "conflicting_values",
                         "range_not_single_value", "unrecognised_suffix"})


def is_low(confidence: float, flags: list[str], threshold: float) -> bool:
    return confidence < threshold or any(f in ALWAYS_FLAG for f in flags)


def norm_label(label: str) -> str:
    """Lower-cased, whitespace-collapsed label without a trailing parenthetical qualifier: 'Date of loss (approx)' -> 'date of loss'."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", " ".join(label.lower().split())).rstrip(".*").strip()


def union(a: BBox | None, b: BBox | None) -> BBox | None:
    if a is None or b is None:
        return a or b
    return BBox(x0=min(a.x0, b.x0), y0=min(a.y0, b.y0), x1=max(a.x1, b.x1), y1=max(a.y1, b.y1))


class RuleExtractor:
    def __init__(self, document_types: DocumentTypes) -> None:
        self.cfg = document_types.extractor
        self.types = document_types.types
        self._boilerplate = [re.compile(p, re.IGNORECASE) for p in self.cfg.boilerplate_patterns]
        self._labels = {t: {name: {norm_label(lab): i for i, lab in enumerate(fs.labels)} for name, fs in spec.fields.items()}
                        for t, spec in self.types.items()}
        self._id = {(t, name): re.compile(fs.pattern, re.IGNORECASE) for t, spec in self.types.items()
                    for name, fs in spec.fields.items() if fs.pattern}

    def extractable_types(self) -> list[str]:
        return [t for t, s in self.types.items() if s.fields or s.table]

    # ---------------------------------------------------------------------------------------------- public
    def extract(self, *, doc_type: str, kind: str | None, pages: Sequence[PageLike],
                original: Callable[[], bytes] | None = None) -> ExtractionResult:
        if doc_type == "unknown":
            reason = "document type is unknown: it must be classified (or typed by a human) before extraction"
            return ExtractionResult(document_type=doc_type, status="skipped", skipped_reason=reason)
        spec = self.types.get(doc_type)
        if spec is None or not (spec.fields or spec.table):
            reason = f"no field specification for type '{doc_type}'"
            return ExtractionResult(document_type=doc_type, status="skipped", skipped_reason=reason)
        lines = logical_lines(pages, kind=kind, original=original, cfg=self.cfg)
        fields: list[ExtractedField] = []
        missing: list[str] = []
        used: set[int] = set()
        for name, fs in spec.fields.items():
            if fs.from_body:
                continue
            found = self._labelled(doc_type, name, fs, lines, used)
            if found:
                fields.append(found)
            elif fs.required:
                missing.append(name)
        for name, fs in spec.fields.items():                                  # body fields last: they take what labelled fields did not use
            if fs.from_body:
                found = self._body(name, lines, used)
                if found:
                    fields.append(found)
                elif fs.required:
                    missing.append(name)
        order = list(spec.fields)
        fields.sort(key=lambda f: order.index(f.name))
        table = None
        if spec.table:
            table, table_fields = extract_table(lines, self.cfg, expects_tax=spec.table_expects_tax)
            fields.extend(table_fields)
        low = [f.name for f in fields if f.low_confidence]
        if table is not None and (table.status != "reconciled" or any(i.low_confidence for i in table.items)):
            low.append("table")
        return ExtractionResult(document_type=doc_type, status="ok", fields=fields, table=table, missing_required=missing,
                                low_confidence_fields=low)

    # ---------------------------------------------------------------------------------------------- labelled fields
    def _labelled(self, doc_type: str, name: str, fs: FieldSpec, lines: list[Line], used: set[int]) -> ExtractedField | None:
        labels = self._labels[doc_type][name]
        cands: list[tuple[int, Line, str, str, int, list[str]]] = []
        for i, ln in enumerate(lines):
            m = _LABEL_LINE.match(ln.text)
            if not m or norm_label(m["label"]) not in labels:
                continue
            rank, value, extra, evidence_line = labels[norm_label(m["label"])], m["value"].strip(), [], ln
            next_is_value = i + 1 < len(lines) and not _LABEL_LINE.match(lines[i + 1].text)
            if not value and next_is_value:                                       # 'Description:' with the text on the next line
                nxt = lines[i + 1]
                value, extra = nxt.text, ["value_on_next_line"]
                evidence_line = Line(ln.page, f"{ln.text} {nxt.text}", union(ln.bbox, nxt.bbox), ln.source)
                used.add(i + 1)
            if value:
                cands.append((i, evidence_line, m["label"].strip(), value, rank, extra))
        if not cands:
            return None
        built = [self._build(doc_type, name, fs, ln, label, value, rank, extra) for _, ln, label, value, rank, extra in cands]
        used.update(i for i, *_ in cands)
        first = built[0]
        distinct, seen = [], {first.normalised or first.value}
        for b in built[1:]:
            key = b.normalised or b.value
            if key not in seen:
                seen.add(key)
                distinct.append(key)
        if not distinct:
            return first
        confidence = round(first.confidence * self.cfg.conflict_factor, 3)
        return first.model_copy(update={"alternatives": distinct, "flags": [*first.flags, "conflicting_values"], "confidence": confidence,
                                        "low_confidence": True})

    def _build(self, doc_type: str, name: str, fs: FieldSpec, ln: Line, label: str, remainder: str, rank: int,
               extra: list[str]) -> ExtractedField:
        flags = list(extra)
        if rank > 0:
            flags.append("synonym_label")
        value, normalised, currency, ambiguous, ok = self._typed(doc_type, name, fs, remainder, flags)
        confidence = self.cfg.label_confidence if rank == 0 else self.cfg.synonym_confidence
        if not ok:
            confidence = self.cfg.unparseable_confidence
            flags.append("unparseable")
        if ambiguous:
            confidence *= self.cfg.ambiguous_factor
        if ln.source == "ocr":
            confidence = min(confidence, self.cfg.ocr_confidence_cap)
            flags.append("ocr")
        confidence = round(confidence, 3)
        return ExtractedField(name=name, value=value, normalised=normalised, currency=currency, label=label, page=ln.page, region=ln.bbox,
                              evidence_text=ln.text, confidence=confidence, flags=flags,
                              low_confidence=is_low(confidence, flags, self.cfg.low_confidence))

    def _typed(self, doc_type: str, name: str, fs: FieldSpec, remainder: str, flags: list[str]):
        """-> (value text, normalised, currency, ambiguous, parsed ok)"""
        raw = V.collapse(remainder)
        if fs.type in ("text", "narrative"):
            return raw, raw, None, False, bool(raw)
        if fs.type == "date" and V.date_is_range(remainder) or fs.type == "money" and V.money_is_range(remainder):
            flags.append("range_not_single_value")                          # kept as written; no single value is picked
            return raw, None, None, False, False
        if fs.type == "date":
            d = V.parse_date(remainder)
            if d:
                flags.extend(d.flags)
                return d.text, d.iso, None, d.ambiguous, True
        elif fs.type == "date_range":
            r = V.parse_date_range(remainder)
            if r:
                a, b, span = r
                flags.extend(sorted({*a.flags, *b.flags}))
                return span, f"{a.iso}/{b.iso}", None, a.ambiguous or b.ambiguous, True
        elif fs.type == "money":
            m = V.parse_money(remainder)
            if m and "unrecognised_suffix" in m.flags:                       # '40abc': the unit is unknown, so the amount is too
                flags.extend(m.flags)
                return raw, None, None, False, False
            if m:
                flags.extend(m.flags)
                return m.text, V.plain(m.amount), m.currency, False, True
        elif fs.type == "id":
            rx = self._id.get((doc_type, name))
            found = V.parse_id(remainder, rx) if rx else (raw or None)
            if found:
                return found, found.upper().replace(" ", ""), None, False, True
        return raw, None, None, False, False

    # ---------------------------------------------------------------------------------------------- unlabelled body text
    def _body(self, name: str, lines: list[Line], used: set[int]) -> ExtractedField | None:
        """Sentence-like lines that no labelled field used (e.g. a surveyor's observations). Lower confidence by construction."""
        def sentence(i: int, ln: Line) -> bool:
            if i in used or ln.text.rstrip()[-1:] not in ".!":
                return False
            m = _LABEL_LINE.match(ln.text)
            if m and len(m["label"].split()) <= 5:
                return False
            if any(rx.search(ln.text) for rx in self._boilerplate):
                return False
            return len(ln.text.split()) >= self.cfg.body_min_words

        idx = [i for i, ln in enumerate(lines) if sentence(i, ln)]
        runs: list[list[int]] = []
        for i in idx:                                                   # consecutive lines form one passage
            if runs and i == runs[-1][-1] + 1 and lines[i].page == lines[runs[-1][-1]].page:
                runs[-1].append(i)
            else:
                runs.append([i])
        if not runs:
            return None
        first = runs[0]
        used.update(first)
        text = " ".join(lines[i].text for i in first)
        region = None
        for i in first:
            region = union(region, lines[i].bbox)
        confidence = self.cfg.body_confidence
        flags = ["unlabelled_body"]
        if lines[first[0]].source == "ocr":
            confidence, flags = min(confidence, self.cfg.ocr_confidence_cap), [*flags, "ocr"]
        others = [" ".join(lines[i].text for i in r) for r in runs[1:]]
        return ExtractedField(name=name, value=text, normalised=text, label="", page=lines[first[0]].page, region=region,
                              evidence_text=text, confidence=round(confidence, 3), flags=flags, alternatives=others,
                              low_confidence=confidence < self.cfg.low_confidence)
