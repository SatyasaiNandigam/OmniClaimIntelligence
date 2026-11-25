"""Field and table extractors (T2.3).

    ex = HybridExtractor(RuleExtractor(pack.document_types))
    result = ex.extract(doc_type="invoice", kind=doc.kind, pages=doc.pages, original=lambda: store.original(doc_id))
    result.fields, result.table, result.missing_required, result.low_confidence_fields

Modules: models (results) · values (date / money / id parsers) · lines (logical lines + regions) · table (line items and the arithmetic
self-check) · fields (rule extractor) · llm (optional filler for missing REQUIRED fields, verified). Specs live in the pack's
document_types.yaml. Results feed T2.4, which turns them into `Fact`s with provenance.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

from claimtower.ingestion.extractors.fields import RuleExtractor
from claimtower.ingestion.extractors.lines import PageLike
from claimtower.ingestion.extractors.llm import LLMFieldFiller
from claimtower.ingestion.extractors.models import ExtractedField, ExtractionResult, LineItem, TableResult


class HybridExtractor:
    """Rules first; the LLM filler (if configured) is consulted only for REQUIRED fields the rules could not find."""

    def __init__(self, rules: RuleExtractor, llm: LLMFieldFiller | None = None) -> None:
        self.rules, self.llm = rules, llm

    def extract(self, *, doc_type: str, kind: str | None, pages: Sequence[PageLike],
                original: Callable[[], bytes] | None = None) -> ExtractionResult:
        result = self.rules.extract(doc_type=doc_type, kind=kind, pages=pages, original=original)
        if self.llm is not None and result.status == "ok" and result.missing_required:
            return self.llm.fill(result, kind=kind, pages=pages, original=original)
        return result


__all__ = ["ExtractedField", "ExtractionResult", "HybridExtractor", "LLMFieldFiller", "LineItem", "RuleExtractor", "TableResult"]
