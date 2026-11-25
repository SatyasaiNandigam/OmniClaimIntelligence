"""Table extraction with an arithmetic self-check done in CODE (T2.3, guardrail 1: a model never does the arithmetic).

Rows are logical lines of the form 'Label: amount' where the remainder is ONLY an amount. Each row is classified by its label using the
pack's patterns (total / subtotal / tax / discount / skip / otherwise a line item). Then, with Decimal arithmetic:

    computed = sum(items) + tax - discount          reconciled  <=>  |stated total - computed| <= tolerance

Nothing is ever hidden: a mismatch keeps every item and the stated total, records the difference, flags `low_confidence`, and lowers the
confidence. Ambiguous situations (two different totals, mixed currencies, no total, no items) get their own status.
"""
from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal

from claimtower.core.config import ExtractorConfig
from claimtower.ingestion.extractors.lines import Line
from claimtower.ingestion.extractors.models import ExtractedField, LineItem, TableResult
from claimtower.ingestion.extractors.values import plain, whole_line_money

_ROW = re.compile(r"^(?P<label>.{1,80}):\s*(?P<rest>[^:]+)$")            # greedy: the LAST colon separates label from amount


def _matches(label: str, patterns: list[str]) -> bool:
    return any(re.fullmatch(p, label.strip().lower().rstrip(":"), re.IGNORECASE) for p in patterns)


def extract_table(lines: list[Line], cfg: ExtractorConfig, *, expects_tax: bool) -> tuple[TableResult, list[ExtractedField]]:
    """Returns the table and the fields derived from it (`total_amount`, from the stated total row)."""
    t = cfg.table
    items: list[tuple[LineItem, str | None]] = []
    totals, subtotals, taxes, discounts = [], [], [], []
    bare = 0
    for ln in lines:
        m = _ROW.match(ln.text)
        money = whole_line_money(m["rest"]) if m else None
        if not m or money is None:
            continue
        label = m["label"].strip()
        if _matches(label, t.skip_labels):
            continue
        if money.currency is None:                      # a bare number is not evidence of an amount ('Invoice number: 5521')
            bare += 1
            continue
        row = (ln, label, money)
        if _matches(label, t.total_labels):
            totals.append(row)
        elif _matches(label, t.subtotal_labels):
            subtotals.append(row)
        elif _matches(label, t.tax_labels):
            taxes.append(row)
        elif _matches(label, t.discount_labels):
            discounts.append(row)
        else:
            items.append((LineItem(description=label, amount=money.amount, currency=money.currency, page=ln.page, region=ln.bbox,
                                   evidence_text=ln.text, confidence=cfg.label_confidence, low_confidence=False), money.currency))

    flags: list[str] = [f"amounts_without_currency_ignored:{bare}"] if bare else []
    currencies = {c for c in [i[1] for i in items] + [r[2].currency for r in totals + subtotals + taxes + discounts] if c}
    if len(currencies) > 1:
        return TableResult(items=[i for i, _ in items], status="mixed_currency", currency=None,
                           flags=["mixed_currency:" + "/".join(sorted(currencies))]), []
    currency = next(iter(currencies), None)

    items_sum = sum((i.amount for i, _ in items), Decimal(0))
    tax_sum = sum((r[2].amount for r in taxes), Decimal(0))
    discount_sum = sum((r[2].amount for r in discounts), Decimal(0))
    computed = items_sum + tax_sum - discount_sum
    counts = Counter((i.description.lower(), i.amount) for i, _ in items)
    if any(n > 1 for n in counts.values()):
        flags.append("duplicate_line_items")
    if expects_tax and not taxes:
        flags.append("no_tax_line")

    subtotal = subtotals[-1][2].amount if subtotals else None
    if subtotal is not None and subtotal != items_sum:
        flags.append("subtotal_mismatch")
    distinct_totals = {r[2].amount for r in totals}
    stated: Decimal | None = None
    if not items:
        status = "no_items"
        stated = totals[-1][2].amount if len(distinct_totals) == 1 else None
    elif not totals:
        status = "no_total"
    elif len(distinct_totals) > 1:
        status = "ambiguous_total"
        flags.append("ambiguous_total:" + "/".join(sorted(plain(a) for a in distinct_totals)))
    else:
        stated = totals[-1][2].amount
        status = "reconciled" if abs(stated - computed) <= cfg.table_tolerance else "mismatch"
    difference = (stated - computed) if stated is not None and items else None
    if status == "no_total":
        flags.append("no_total_row")

    ok = status == "reconciled" and "subtotal_mismatch" not in flags
    confidence = cfg.label_confidence if ok else cfg.table_mismatch_confidence
    low = confidence < cfg.low_confidence
    final_items = [i.model_copy(update={"confidence": confidence, "low_confidence": low}) for i, _ in items]

    fields: list[ExtractedField] = []
    if totals:
        ln, label, money = totals[-1]
        field_flags = {"mismatch": "total_mismatch", "ambiguous_total": "ambiguous_total"}
        fields.append(ExtractedField(
            name="total_amount", value=money.text, normalised=plain(money.amount), currency=money.currency, label=label, page=ln.page,
            region=ln.bbox, evidence_text=ln.text, confidence=confidence, low_confidence=low,
            flags=([field_flags[status]] if status in field_flags else []) + (["ocr"] if ln.source == "ocr" else [])))
    return (TableResult(items=final_items, subtotal=subtotal, tax_total=tax_sum, discount_total=discount_sum, stated_total=stated,
                        computed_total=computed if items else None, difference=difference, currency=currency, status=status, flags=flags),
            fields)
