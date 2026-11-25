"""Result models of the field and table extractors (T2.3). Internal to ingestion: T2.4 turns them into `Fact`s with provenance.

Everything keeps WHERE it came from: page, region (normalised box, when the PDF geometry is available) and `evidence_text`, the verbatim
logical line the value was read from. `evidence_text` is a substring of the whitespace-normalised page text, which is the basis the
provenance check in T2.4 uses. Money is `Decimal` (never float); dates are ISO strings in `normalised`.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Literal

from claimtower.core.schemas.common import BBox, Confidence, Contract, Money

TableStatus = Literal["reconciled", "mismatch", "no_total", "no_items", "ambiguous_total", "mixed_currency"]


class ExtractedField(Contract):
    name: str
    value: str                          # as written in the document (the matched text for typed fields, the remainder of the line for text)
    normalised: str | None = None       # ISO date | 'YYYY-MM-DD/YYYY-MM-DD' | plain decimal | canonical id | text; None = would not parse
    currency: str | None = None         # money fields
    label: str = ""                     # the label as written ('' for unlabelled body text)
    page: int
    region: BBox | None = None          # None when the page has no geometry (OCR, DOCX, text, e-mail)
    evidence_text: str                  # verbatim logical line (or consecutive lines) the value was read from
    confidence: Confidence              # EXTRACTION confidence: NOT a probability of coverage
    low_confidence: bool                # flagged, never hidden
    method: Literal["rule", "llm"] = "rule"
    flags: list[str] = []               # synonym_label, ambiguous_date, unparseable, conflicting_values, unlabelled_body, ocr, llm, ...
    alternatives: list[str] = []        # other values the document states for the same field (kept for contradiction detection, T4.3)


class LineItem(Contract):
    description: str
    amount: Money
    currency: str | None = None
    page: int
    region: BBox | None = None
    evidence_text: str
    confidence: Confidence
    low_confidence: bool


class TableResult(Contract):
    """Line items plus the arithmetic self-check, computed in code (never by a model)."""

    items: list[LineItem] = []
    subtotal: Money | None = None
    tax_total: Money = Decimal(0)
    discount_total: Money = Decimal(0)
    stated_total: Money | None = None
    computed_total: Money | None = None            # sum(items) + tax - discount
    difference: Money | None = None                # stated - computed; 0 when reconciled
    currency: str | None = None
    status: TableStatus
    flags: list[str] = []                          # no_tax_line, subtotal_mismatch, duplicate_line_items, ...


class ExtractionResult(Contract):
    document_type: str
    status: Literal["ok", "skipped"]
    skipped_reason: str | None = None
    fields: list[ExtractedField] = []
    table: TableResult | None = None
    missing_required: list[str] = []               # required fields that were not found (reported, not silently dropped)
    low_confidence_fields: list[str] = []          # names of fields (and 'table') flagged low-confidence
    warnings: list[str] = []                       # e.g. the optional LLM filler was unavailable

    def field(self, name: str) -> ExtractedField | None:
        return next((f for f in self.fields if f.name == name), None)

    def values(self) -> dict[str, str | None]:
        """{field name: normalised value} - convenient for evaluation."""
        return {f.name: f.normalised for f in self.fields}
