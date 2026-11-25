"""T2.3 field and table extractors: typed values, geometry, the arithmetic self-check, never-silent behaviour, store integration."""
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from email.message import EmailMessage
from pathlib import Path

import pymupdf
import pytest
import yaml
from pydantic import ValidationError

from claimtower.core.config import DocumentTypes, FieldSpec, TableConfig, load_settings
from claimtower.core.llm import LLMClient, LLMResponse
from claimtower.ingestion.classifier import HybridClassifier, RuleClassifier
from claimtower.ingestion.extractors import HybridExtractor, LLMFieldFiller, RuleExtractor
from claimtower.ingestion.extractors import values as V
from claimtower.ingestion.extractors.fields import ALWAYS_FLAG, is_low
from claimtower.ingestion.extractors.lines import logical_lines
from claimtower.ingestion.extractors.table import extract_table
from claimtower.ingestion.loaders import load, normalise
from claimtower.ingestion.store import DocumentStore

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data/synthetic"
PACK = load_settings().pack
DT = PACK.document_types
EXT = DT.extractor
RULES = RuleExtractor(DT)
D = Decimal
RUPEE = chr(0x20B9)


@dataclass
class P:
    number: int
    text: str
    source: str = "text_layer"
    needs_ocr: bool = False


def extract(doc_type, text, **kw):
    return RULES.extract(doc_type=doc_type, kind="text", pages=[P(1, text)], **kw)


def lines_of(text, source="text_layer"):
    return logical_lines([P(1, text, source)], kind="text", original=None, cfg=EXT)


def table_of(text, expects_tax=False):
    return extract_table(lines_of(text), EXT, expects_tax=expects_tax)


def pdf_bytes(draw):
    pdf = pymupdf.open()
    draw(pdf.new_page())
    return pdf.tobytes()


def pdf_lines(draw):
    data = pdf_bytes(draw)
    d = load(data, "x.pdf")
    return logical_lines(d.pages, kind="pdf", original=lambda: data, cfg=EXT)


# ================================================================== typed value parsers
@pytest.mark.parametrize("text, iso", [
    ("14 March 2026", "2026-03-14"), ("3rd June 2026", "2026-06-03"), ("3rd of June, 2026", "2026-06-03"), ("3 june 2026", "2026-06-03"),
    ("March 14, 2026", "2026-03-14"), ("2026-03-14", "2026-03-14"), ("02-Aug-2026", "2026-08-02"), ("Tue, 10 Mar 2026 10:00:00 +0000", "2026-03-10"),
    ("14 March 2026 at 10:30", "2026-03-14"), ("01.01.2026", "2026-01-01"), ("31/12/2026", "2026-12-31"),
])
def test_dates_are_parsed_to_iso(text, iso):
    d = V.parse_date(text)
    assert d is not None and d.iso == iso and not d.ambiguous


@pytest.mark.parametrize("text, iso, flag", [
    ("03/04/2026", "2026-04-03", "ambiguous_day_month"), ("2/8/2026", "2026-08-02", "ambiguous_day_month"),
    ("25/12/26", "2026-12-25", "two_digit_year"), ("12/25/2026", "2026-12-25", "assumed_month_first"),
])
def test_ambiguous_dates_are_read_day_first_and_marked_ambiguous(text, iso, flag):
    d = V.parse_date(text)
    assert d.iso == iso and d.ambiguous and flag in d.flags


@pytest.mark.parametrize("text", ["31 February 2026", "last Tuesday", "3 June", "June 2026", "", "99/99/2026"])
def test_impossible_or_incomplete_dates_are_not_repaired(text):
    assert V.parse_date(text) is None


@pytest.mark.parametrize("text", [
    "between 3 and 5 May 2026", "3-5 May 2026", "3 - 5 May 2026", "from 3 May to 5 May 2026", "night of 3/4 May 2026", "3rd to 5th May 2026",
    "3 June 2026 / 4 June 2026", "3 June 2026 - 5 June 2026", "6 June 2026 and 7 June 2026",
])
def test_date_ranges_are_detected_so_no_single_date_is_picked(text):
    assert V.date_is_range(text)


@pytest.mark.parametrize("text", ["14 March 2026", "14 March 2026 (reported on 16 March 2026)", "2026-05-05", "3 June 2026."])
def test_single_dates_are_not_ranges(text):
    assert not V.date_is_range(text)


def test_date_range_parser():
    a, b, span = V.parse_date_range("1 January 2026 to 31 December 2026")
    assert (a.iso, b.iso, span) == ("2026-01-01", "2026-12-31", "1 January 2026 to 31 December 2026")
    assert V.parse_date_range("1 Jan 2026 - 31 Dec 2026")[2] == "1 Jan 2026 - 31 Dec 2026"
    assert V.parse_date_range("1 Jan 2026 until 31 Dec 2026") is not None and V.parse_date_range("1 Jan 2026") is None


@pytest.mark.parametrize("text, amount, currency", [
    ("INR 180,000", "180000", "INR"), ("Rs. 1,50,000", "150000", "INR"), (RUPEE + " 5000.50", "5000.5", "INR"), ("INR 1.5 lakh", "150000", "INR"),
    ("INR 5,00,00,000", "50000000", "INR"), ("INR 1.2 crore", "12000000", "INR"), ("INR 40k", "40000", "INR"), ("$1,200.00", "1200", "USD"),
    ("-INR 500", "-500", "INR"), ("(INR 500)", "-500", "INR"), ("Rs.60000", "60000", "INR"), ("INR60,000", "60000", "INR"),
    ("60,000 INR", "60000", "INR"), ("INR 60,000/-", "60000", "INR"), ("INR 98,000 only", "98000", "INR"),
])
def test_money_is_parsed_exactly_with_its_currency(text, amount, currency):
    m = V.parse_money(text)
    assert (V.plain(m.amount), m.currency) == (amount, currency) and isinstance(m.amount, Decimal)
    assert "unrecognised_suffix" not in m.flags


def test_money_edge_cases():
    assert V.parse_money("98000").currency is None and "no_currency" in V.parse_money("98000").flags
    assert V.parse_money("POL-2201") is None                                  # an identifier is not an amount
    assert "unrecognised_suffix" in V.parse_money("INR 40abc").flags            # letters glued to the digits: unknown unit
    assert [V.plain(m.amount) for m in V.find_money("INR 98,000, of which INR 8,000 is an upgrade")] == ["98000", "8000"]
    assert V.whole_line_money("INR 110,000") and V.whole_line_money("INR 98,000, of which x") is None
    assert V.plain(D("1.50")) == "1.5" and V.plain(D("180000")) == "180000" and V.plain(D("0.00")) == "0"


@pytest.mark.parametrize("text", [
    "INR 45,000 to INR 60,000", "INR 45,000 - 60,000", "45,000-60,000", "INR 40k-60k", "between INR 40,000 and INR 60,000", "40 to 60",
])
def test_amount_ranges_are_detected(text):
    assert V.money_is_range(text)


@pytest.mark.parametrize("text", ["INR 98,000 (INR 8,000 excluded)", "INR 98,000, of which INR 8,000 is an upgrade", "INR 60,000 on 2026-05-05", "INR 45000 - not yet confirmed"])
def test_single_amounts_with_context_are_not_ranges(text):
    assert not V.money_is_range(text)


# ================================================================== logical lines and regions
def test_wrapped_lines_merge_but_paragraphs_do_not():
    def draw(page):
        for i, t in enumerate(["Findings: the joint was corroded and the", "pipe failed at the weld."]):
            page.insert_text((70, 100 + i * 12.5), t, fontsize=10.5)
        page.insert_text((70, 100 + 2 * 12.5 + 14), "Joint replaced on 16 March 2026.", fontsize=10.5)       # next paragraph: bigger gap

    texts = [ln.text for ln in pdf_lines(draw)]
    assert "Findings: the joint was corroded and the pipe failed at the weld." in texts
    assert "Joint replaced on 16 March 2026." in texts and len(texts) == 2


def test_label_and_right_aligned_amount_on_one_baseline_form_one_row():
    def draw(page):
        page.insert_text((64, 100), "Kitchen floor:", fontsize=10)
        page.insert_text((480, 100), " INR 110,000", fontsize=10)

    assert [ln.text for ln in pdf_lines(draw)] == ["Kitchen floor: INR 110,000"]


def test_tightly_spaced_label_value_lines_are_never_glued_together():
    def draw(page):
        for i, t in enumerate(["Claimant: R. Gupta", "Policy number: POL-2201", "Date of loss: 3 June 2026"]):
            page.insert_text((60, 100 + i * 10.5), t, fontsize=10)                 # leading barely larger than the font: tight

    assert [ln.text for ln in pdf_lines(draw)] == ["Claimant: R. Gupta", "Policy number: POL-2201", "Date of loss: 3 June 2026"]


def test_rotated_stamp_text_is_not_body_text():
    def draw(page):
        page.insert_text((70, 100), "Inspection date: 5 June 2026", fontsize=10)
        page.insert_text((300, 300), "RECEIVED 5 JUN 2026", fontsize=12, morph=(pymupdf.Point(300, 300), pymupdf.Matrix(15)))

    assert [ln.text for ln in pdf_lines(draw)] == ["Inspection date: 5 June 2026"]


@pytest.mark.parametrize("path", ["claims/CLM-0001/plumber_report.pdf", "claims/CLM-0001/estimate.pdf", "claims/CLM-0005/policy_schedule.pdf"])
def test_every_logical_line_is_verbatim_page_text_with_a_normalised_region(path):
    data = (SYN / path).read_bytes()
    d = load(data, path)
    page_text = normalise(" ".join(p.text for p in d.pages))
    lines = logical_lines(d.pages, kind="pdf", original=lambda: data, cfg=EXT)
    assert lines
    for ln in lines:
        assert normalise(ln.text) in page_text, ln.text
        b = ln.bbox
        assert 0 <= b.x0 < b.x1 <= 1 and 0 <= b.y0 < b.y1 <= 1


def test_pages_without_geometry_give_plain_lines_without_a_region():
    assert [(ln.text, ln.bbox) for ln in lines_of("A: 1\n\nB: 2\n")] == [("A: 1", None), ("B: 2", None)]
    assert all(ln.source == "ocr" for ln in lines_of("A: 1", source="ocr"))


# ================================================================== table extraction and the arithmetic self-check
def test_reconciled_table_with_tax_and_discount():
    t, fields = table_of("Plumbing: INR 50,000\nPainting: INR 20,000\nSubtotal: INR 70,000\nGST: INR 1,800\nDiscount: INR 5,000\nTotal: INR 66,800\n", True)
    assert t.status == "reconciled" and t.computed_total == D(66800) == t.stated_total and t.difference == 0
    assert (t.tax_total, t.discount_total, t.subtotal) == (D(1800), D(5000), D(70000)) and [i.description for i in t.items] == ["Plumbing", "Painting"]
    assert fields[0].name == "total_amount" and fields[0].normalised == "66800" and not fields[0].low_confidence


def test_mismatch_keeps_everything_records_the_difference_and_flags_low_confidence():
    t, fields = table_of("Plumbing: INR 30,000\nTiling: INR 12,000\nTotal: INR 45,000\n")
    assert t.status == "mismatch" and t.difference == D(3000) and t.computed_total == D(42000) and t.stated_total == D(45000)
    assert len(t.items) == 2 and all(i.low_confidence and i.confidence == EXT.table_mismatch_confidence for i in t.items)
    assert fields[0].low_confidence and "total_mismatch" in fields[0].flags and fields[0].normalised == "45000"


@pytest.mark.parametrize("text, status", [
    ("Plumbing: INR 15,000\nTiling: INR 25,000\nTotal: INR 40,000\nGrand total: INR 42,000\n", "ambiguous_total"),
    ("Plumbing: INR 15,000\nTiling: INR 25,000\n", "no_total"),
    ("Total: INR 40,000\n", "no_items"),
    ("Plumbing: INR 15,000\nImported: USD 200\nTotal: INR 20,000\n", "mixed_currency"),
])
def test_other_table_situations_have_their_own_status_and_never_a_made_up_total(text, status):
    t, fields = table_of(text)
    assert t.status == status
    if status == "ambiguous_total":
        assert t.stated_total is None and any(f.startswith("ambiguous_total:") for f in t.flags)
    if status == "mixed_currency":
        assert t.computed_total is None                                         # different currencies are never summed


def test_equal_repeated_totals_are_not_ambiguous_and_duplicates_are_flagged():
    t, _ = table_of("Plumbing: INR 15,000\nPlumbing: INR 15,000\nTotal: INR 30,000\nTotal: INR 30,000\n")
    assert t.status == "reconciled" and "duplicate_line_items" in t.flags


def test_subtotal_mismatch_and_missing_tax_line_are_flagged():
    t, _ = table_of("Plumbing: INR 50,000\nSubtotal: INR 60,000\nTotal: INR 50,000\n", expects_tax=True)
    assert "subtotal_mismatch" in t.flags and "no_tax_line" in t.flags


@pytest.mark.parametrize("label, kind", [
    ("Total", "total"), ("TOTAL", "total"), ("Grand Total", "total"), ("Total (incl. GST)", "total"), ("Total Cost", "total"),
    ("Total Amount Due", "total"), ("Total payable", "total"), ("Amount payable", "total"), ("Net payable", "total"),
    ("Total labour", "item"), ("Total materials", "item"), ("Labour (2 days)", "item"),
])
def test_total_labels_are_recognised_but_total_labour_stays_an_item(label, kind):
    t, fields = table_of(f"Repairs: INR 100\n{label}: INR 100\n")
    if kind == "total":
        assert [i.description for i in t.items] == ["Repairs"] and t.stated_total == D(100)
    else:
        assert label in [i.description for i in t.items]


def test_identifier_rows_and_bare_numbers_are_not_line_items():
    """Regression: 'Invoice number: 5521' used to be summed as an amount and broke the reconciliation."""
    t, _ = table_of("Invoice number: 5521\nLabour: INR 20,000\nTotal: INR 20,000\n")
    assert t.status == "reconciled" and [i.description for i in t.items] == ["Labour"]
    t2, _ = table_of("Labour: 20,000\nTotal: 20,000\n")                         # no currency at all: ignored AND reported
    assert t2.status == "no_items" and "amounts_without_currency_ignored:2" in t2.flags


def test_money_rows_that_are_not_line_items_are_skipped_by_label():
    t, _ = table_of("Repairs: INR 9,000\nDeposit: INR 1,000\nBalance due: INR 8,000\nEstimated loss: INR 12,000\nPremium: INR 500\nTotal: INR 9,000\n")
    assert t.status == "reconciled" and [i.description for i in t.items] == ["Repairs"]


def test_credit_rows_discount_prefix_and_colons_inside_labels():
    t, _ = table_of("Flooring: INR 1,25,000\nLess trade-in: -INR 25,000\nTotal: INR 1,00,000\n")
    assert t.status == "reconciled" and t.items[1].amount == D(-25000)
    t2, _ = table_of("Repairs: INR 5,000\nLess: Discount (10%): INR 500\nTotal: INR 4,500\n")
    assert t2.status == "reconciled" and t2.discount_total == D(500)
    t3, _ = table_of("Plumbing: labour: INR 5,000\nTotal: INR 5,000\n")
    assert t3.status == "reconciled" and t3.items[0].description == "Plumbing: labour"


def test_table_arithmetic_is_decimal_exact():
    t, _ = table_of("A: INR 0.10\nB: INR 0.20\nTotal: INR 0.30\n")
    assert t.status == "reconciled" and isinstance(t.computed_total, Decimal)      # 0.1 + 0.2 == 0.3 exactly (a float would not)


# ================================================================== fields
FNOL = "FIRST NOTICE OF LOSS\nClaimant: R. Gupta\nPolicy number: POL-2201\nDate of loss: 3 June 2026\nDescription: Pipe burst.\nEstimated loss: INR 45,000\n"


def test_fnol_fields_are_typed_located_and_scored():
    r = extract("fnol", FNOL)
    assert r.status == "ok" and r.missing_required == [] and r.low_confidence_fields == []
    assert r.values() == {"claimant_name": "R. Gupta", "policy_number": "POL-2201", "loss_date": "2026-06-03", "loss_description": "Pipe burst.",
                          "estimated_loss": "45000"}
    f = r.field("estimated_loss")
    assert (f.value, f.currency, f.label, f.confidence, f.method, f.page) == ("INR 45,000", "INR", "Estimated loss", EXT.label_confidence, "rule", 1)
    assert f.evidence_text == "Estimated loss: INR 45,000" and f.region is None and f.flags == []


def test_geometry_gives_a_region_and_the_verbatim_line():
    data = (SYN / "claims/CLM-0001/fnol.pdf").read_bytes()
    d = load(data, "f.pdf")
    r = RULES.extract(doc_type="fnol", kind="pdf", pages=d.pages, original=lambda: data)
    f = r.field("loss_date")
    assert f.evidence_text == "Date of loss: 14 March 2026" and f.region is not None and 0 < f.region.y0 < f.region.y1 < 1


def test_synonym_labels_lower_confidence_and_are_flagged_as_such():
    r = extract("fnol", "CLAIM FORM\nInsured name: S. K\nPolicy no: pol-4410\nDate of incident: 2026-08-02\n")
    assert r.values()["claimant_name"] == "S. K" and r.values()["policy_number"] == "POL-4410"
    f = r.field("loss_date")
    assert f.confidence == EXT.synonym_confidence and f.flags == ["synonym_label"] and not f.low_confidence


def test_labels_ignore_case_spacing_and_a_parenthetical_qualifier():
    r = extract("fnol", "CLAIMANT : M. Rao\nPOLICY NUMBER : POL-3310\nDate of loss (approx): 11 July 2026\n")
    assert r.values() == {"claimant_name": "M. Rao", "policy_number": "POL-3310", "loss_date": "2026-07-11"}


def test_required_fields_that_are_missing_are_reported_not_invented():
    r = extract("fnol", "FIRST NOTICE OF LOSS\nClaimant - R. Gupta\nDate of loss: 3 June 2026\n")
    assert sorted(r.missing_required) == ["claimant_name", "loss_description", "policy_number"]
    assert r.field("policy_number") is None and r.field("claimant_name") is None


@pytest.mark.parametrize("value, field", [
    ("last Tuesday", "loss_date"), ("31 February 2026", "loss_date"), ("3 June", "loss_date"), ("between 3 and 5 May 2026", "loss_date"),
    ("3-5 May 2026", "loss_date"),
])
def test_dates_that_cannot_be_read_as_one_date_are_kept_as_written_flagged_and_unnormalised(value, field):
    f = extract("fnol", FNOL.replace("3 June 2026", value)).field(field)
    assert f.value == value and f.normalised is None and f.low_confidence and "unparseable" in f.flags


@pytest.mark.parametrize("value", ["INR 45,000 to INR 60,000", "45,000-60,000", "INR 40k-60k", "nil", "INR 40abc"])
def test_amounts_that_cannot_be_read_as_one_amount_are_flagged_not_guessed(value):
    f = extract("fnol", FNOL.replace("INR 45,000", value)).field("estimated_loss")
    assert f.normalised is None and f.low_confidence and f.confidence == EXT.unparseable_confidence


def test_ambiguous_date_is_read_but_always_flagged_even_if_the_number_stays_high():
    f = extract("fnol", FNOL.replace("3 June 2026", "2/8/2026")).field("loss_date")
    assert f.normalised == "2026-08-02" and f.low_confidence and "ambiguous_day_month" in f.flags
    assert is_low(0.99, ["ambiguous_day_month"], EXT.low_confidence) and not is_low(0.99, [], EXT.low_confidence)
    assert {"unparseable", "conflicting_values", "range_not_single_value", "unrecognised_suffix", "two_digit_year"} <= ALWAYS_FLAG


def test_conflicting_values_keep_the_first_list_the_others_and_flag():
    f = extract("fnol", FNOL + "Date of loss: 5 August 2026\nDate of loss: 3 June 2026\n").field("loss_date")
    assert f.normalised == "2026-06-03" and f.alternatives == ["2026-08-05"] and f.low_confidence and "conflicting_values" in f.flags
    assert f.confidence == round(EXT.label_confidence * EXT.conflict_factor, 3)


def test_value_on_the_next_line_and_colons_inside_values():
    r = extract("fnol", "FIRST NOTICE OF LOSS\nClaimant: A\nPolicy number: POL-1001\nDate of loss: 3 June 2026\nDescription:\nWater came through.\n")
    f = r.field("loss_description")
    assert f.value == "Water came through." and "value_on_next_line" in f.flags and f.evidence_text == "Description: Water came through."
    assert extract("fnol", FNOL.replace("Pipe burst.", "Water damage: ceiling and wall.")).field("loss_description").value == "Water damage: ceiling and wall."


def test_two_policy_ids_in_one_value_take_the_first():
    assert extract("fnol", FNOL.replace("POL-2201", "POL-2201 (renewal of POL-1990)")).field("policy_number").normalised == "POL-2201"
    assert extract("fnol", FNOL.replace("POL-2201", "POL 2201")).field("policy_number").normalised is None


def test_policy_schedule_period_is_one_date_range_field():
    r = extract("policy_schedule", "POLICY SCHEDULE\nPolicy number: POL-2201\nPeriod of insurance: 1 January 2026 to 31 December 2026\n"
                "Building limit: INR 2,000,000\nStandard deductible: INR 10,000\nEndorsements attached: END-01\n")
    assert r.values()["period_of_insurance"] == "2026-01-01/2026-12-31" and r.values()["building_limit"] == "2000000"
    assert r.values()["endorsements_attached"] == "END-01"


def test_email_headers_become_fields():
    r = extract("email", "From: a@x.invalid\nTo: b@y.invalid\nDate: Tue, 10 Mar 2026 10:00:00 +0000\nSubject: Missing endorsement\n\nPlease resend.\n")
    assert r.values() == {"email_from": "a@x.invalid", "email_to": "b@y.invalid", "email_date": "2026-03-10", "email_subject": "Missing endorsement"}


# -- unlabelled body text
def test_body_text_is_taken_with_lower_confidence_and_flagged():
    r = extract("surveyor_report", "SURVEYOR'S REPORT\nSite visit: 6 June 2026\nMoisture readings were high in the ceiling.\nAssessed repair cost: INR 40,000\n")
    f = r.field("observations")
    assert f.value == "Moisture readings were high in the ceiling." and f.label == "" and f.confidence == EXT.body_confidence
    assert f.low_confidence and f.flags == ["unlabelled_body"] and "observations" in r.low_confidence_fields
    assert r.values()["assessed_repair_cost"] == "40000"                      # the labelled line is not also body text


def test_body_text_skips_boilerplate_fields_and_short_lines_and_keeps_other_passages_as_alternatives():
    text = ("SURVEY REPORT\nfor software testing only. Not a real document. All amounts are SAMPLE here.\nSite visit: 7 August 2026\nPage 1 of 1\n"
            "Short line.\nThe walls in the lounge show damp patches near the skirting.\nSigned: ______\n"
            "Photographs were taken of each affected wall in the lounge.\n")
    f = extract("surveyor_report", text).field("observations")
    assert f.value == "The walls in the lounge show damp patches near the skirting."
    assert f.alternatives == ["Photographs were taken of each affected wall in the lounge."]


def test_consecutive_body_lines_form_one_passage_with_a_union_region():
    data = (SYN / "claims/CLM-0001/surveyor_report.pdf").read_bytes()
    d = load(data, "s.pdf")
    f = RULES.extract(doc_type="surveyor_report", kind="pdf", pages=d.pages, original=lambda: data).field("observations")
    assert f.value.startswith("Moisture readings were elevated") and f.value.endswith("Cost of repair has not been assessed.")
    assert normalise(f.evidence_text) in normalise(d.pages[0].text) and f.region is not None


# -- provenance on the whole corpus
CORPUS_TYPES = {"fnol", "plumber_report", "surveyor_report", "estimate", "policy_schedule"}


def test_every_extracted_value_on_every_corpus_document_is_verbatim_and_located():
    checked = 0
    for sc in __import__("evals.ground_truth", fromlist=["x"]).load_scenarios().scenarios:
        for doc in sc.documents:
            if doc.doc_type not in CORPUS_TYPES:
                continue
            data = (SYN / "claims" / sc.claim_id / doc.filename).read_bytes()
            d = load(data, doc.filename)
            page = {p.number: normalise(p.text) for p in d.pages}
            r = RULES.extract(doc_type=doc.doc_type, kind="pdf", pages=d.pages, original=lambda data=data: data)
            for f in r.fields:
                assert normalise(f.evidence_text) in page[f.page], (doc.document_id, f.name)
                assert f.region is not None and 0 <= f.confidence <= 1
                if f.method == "rule" and f.label:
                    assert normalise(f.value) in normalise(f.evidence_text), (doc.document_id, f.name)
                checked += 1
            for it in (r.table.items if r.table else []):
                assert normalise(it.evidence_text) in page[it.page]
    assert checked >= 40


# ================================================================== skipped types
def test_unknown_or_unspecified_types_are_skipped_with_a_reason():
    r = RULES.extract(doc_type="unknown", kind="text", pages=[P(1, FNOL)])
    assert (r.status, r.fields, r.table) == ("skipped", [], None) and "classified" in r.skipped_reason
    for t in ("photo", "other", "policy_wording", "endorsement", "made_up"):
        r = RULES.extract(doc_type=t, kind="text", pages=[P(1, FNOL)])
        assert r.status == "skipped" and "no field specification" in r.skipped_reason
    assert set(RULES.extractable_types()) == {"fnol", "policy_schedule", "plumber_report", "surveyor_report", "estimate", "invoice", "email"}


def test_ocr_text_confidence_is_capped_and_flagged():
    r = RULES.extract(doc_type="fnol", kind="pdf", pages=[P(1, FNOL, source="ocr")])
    assert r.fields and all(f.confidence <= EXT.ocr_confidence_cap and "ocr" in f.flags for f in r.fields)


# ================================================================== optional LLM filler
class FakeLLM:
    name = "fake"

    def __init__(self, reply):
        self.reply, self.requests = reply, []

    def complete(self, req):
        self.requests.append(req)
        if isinstance(self.reply, Exception):
            raise self.reply
        return LLMResponse(text=json.dumps(self.reply), model="m", input_tokens=1, output_tokens=1)


def hybrid(reply):
    provider = FakeLLM(reply)
    llm = LLMClient(provider, text_model="t", vision_model="v")
    return HybridExtractor(RULES, LLMFieldFiller(llm, RULES, PACK)), provider


NO_POLICY = "FIRST NOTICE OF LOSS\nClaimant: R. Gupta\nCover reference - POL-2201\nDate of loss: 3 June 2026\nDescription: Leak.\n"


def run_hybrid(reply, text=NO_POLICY):
    clf, provider = hybrid(reply)
    return clf.extract(doc_type="fnol", kind="text", pages=[P(1, text)]), provider


def test_llm_fills_a_missing_required_field_only_with_verifiable_evidence():
    r, provider = run_hybrid({"fields": {"policy_number": {"value": "POL-2201", "evidence_snippet": "Cover reference - POL-2201", "confidence": 0.95}}})
    f = r.field("policy_number")
    assert (f.normalised, f.method, f.flags) == ("POL-2201", "llm", ["llm"]) and f.confidence == EXT.llm_confidence_cap
    assert f.evidence_text == "Cover reference - POL-2201" and f.page == 1 and r.missing_required == [] and len(provider.requests) == 1


@pytest.mark.parametrize("answer, why", [
    ({"value": "POL-2201", "evidence_snippet": "Policy reference POL-2201 appears here", "confidence": 0.9}, "not verbatim"),
    ({"value": "POL-2201", "evidence_snippet": "x", "confidence": 0.9}, "not verbatim"),
    ({"value": "not an id", "evidence_snippet": "Cover reference - POL-2201", "confidence": 0.9}, "does not parse"),
    ({"value": "POL-9999", "evidence_snippet": "Cover reference - POL-2201", "confidence": 0.9}, "does not parse|does not appear"),
    (None, "null"),
])
def test_llm_answers_that_do_not_check_out_leave_the_field_missing_with_a_warning(answer, why):
    r, _ = run_hybrid({"fields": {"policy_number": answer}})
    assert r.field("policy_number") is None and "policy_number" in r.missing_required
    assert any(re.search(why, w) for w in r.warnings), r.warnings


def test_llm_is_only_asked_when_required_fields_are_missing_and_failure_is_a_warning_not_a_crash():
    r, provider = run_hybrid({"fields": {}}, text=FNOL)
    assert r.missing_required == [] and provider.requests == []
    r, _ = run_hybrid(TimeoutError("down"))
    assert r.field("policy_number") is None and any("TimeoutError" in w for w in r.warnings)
    clf = HybridExtractor(RULES, LLMFieldFiller(LLMClient(FakeLLM({}), text_model="TBD", vision_model="TBD"), RULES, PACK))
    assert any("unavailable" in w for w in clf.extract(doc_type="fnol", kind="text", pages=[P(1, NO_POLICY)]).warnings)


def test_llm_money_values_must_appear_in_their_snippet():
    text = "FIRST NOTICE OF LOSS\nClaimant: R. Gupta\nPolicy number: POL-2201\nDescription: Leak.\nDamage was about 45,000 rupees.\n"
    r, _ = run_hybrid({"fields": {"loss_date": {"value": "3 June 2026", "evidence_snippet": "Damage was about 45,000 rupees", "confidence": 0.9}}}, text)
    assert r.field("loss_date") is None and "loss_date" in r.missing_required


# ================================================================== store integration
@pytest.fixture
def store(tmp_path):
    return DocumentStore(tmp_path / "s")


CLF = HybridClassifier(RuleClassifier(DT))
EXTRACTOR = HybridExtractor(RULES)


def test_extraction_is_persisted_with_the_document_and_survives_a_restart(tmp_path):
    st = DocumentStore(tmp_path / "s")
    data = (SYN / "claims/CLM-0001/estimate.pdf").read_bytes()
    st.ingest("CLM-T", "estimate.pdf", data, document_id="DOC-1", classifier=CLF, extractor=EXTRACTOR)
    r = DocumentStore(tmp_path / "s").stored("DOC-1").extraction
    assert r.document_type == "estimate" and r.table.status == "reconciled" and r.table.stated_total == D(180000)
    assert r.field("total_amount").region is not None and st.original("DOC-1") == data


def test_unknown_type_is_skipped_until_a_human_types_it_then_extraction_runs(store):
    store.ingest("CLM-T", "n.txt", FNOL.replace("FIRST NOTICE OF LOSS\n", "").encode(), document_id="DOC-H", classifier=CLF, extractor=EXTRACTOR)
    assert store.stored("DOC-H").extraction.status == "skipped"
    store.set_type("DOC-H", "fnol", by="adjuster", allowed=list(DT.types))
    r = store.extract("DOC-H", EXTRACTOR)
    assert r.status == "ok" and r.values()["policy_number"] == "POL-2201"


def test_email_attachments_are_classified_and_extracted_as_their_own_documents(store):
    m = EmailMessage()
    m["From"], m["To"], m["Subject"] = "p@x.example", "c@y.example", "Estimate"
    m.set_content("Attached.")
    m.add_attachment((SYN / "claims/CLM-0001/estimate.pdf").read_bytes(), maintype="application", subtype="pdf", filename="est.pdf")
    store.ingest("CLM-T", "mail.eml", bytes(m), document_id="DOC-M", classifier=CLF, extractor=EXTRACTOR)
    assert store.stored("DOC-M").extraction.values()["email_subject"] == "Estimate"
    kid = store.stored(store.children("DOC-M")[0].document_id).extraction
    assert kid.document_type == "estimate" and kid.table.status == "reconciled"


def test_extraction_after_ocr_uses_the_ocr_text_with_capped_confidence(store):
    class Ocr:
        name = "stub"

        def recognise(self, image, mime):
            return "SURVEYOR'S REPORT\nSite visit: 5 May 2026\nAssessed repair cost: INR 98,000\n"

    store.ingest("CLM-T", "scan.pdf", (SYN / "claims/CLM-0002/surveyor_report_scanned.pdf").read_bytes(), document_id="DOC-S", classifier=CLF, extractor=EXTRACTOR)
    assert store.stored("DOC-S").extraction.status == "skipped"                       # nothing to read yet
    store.run_ocr("DOC-S", Ocr())
    store.classify("DOC-S", CLF)
    r = store.extract("DOC-S", EXTRACTOR)
    assert r.values()["site_visit_date"] == "2026-05-05" and all(f.confidence <= EXT.ocr_confidence_cap for f in r.fields)


# ================================================================== config validation and label-file hygiene
def raw():
    return yaml.safe_load((ROOT / "packs/property_water_damage/document_types.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("mutate, match", [
    (lambda d: d["types"]["fnol"]["fields"]["loss_date"].update(labels=[]), "at least one label"),
    (lambda d: d["types"]["fnol"]["fields"]["policy_number"].update(pattern="(unclosed"), "bad field pattern"),
    (lambda d: d["types"]["fnol"]["fields"]["loss_date"].update(pattern="x"), "only for type: id"),
    (lambda d: d["extractor"]["table"]["total_labels"].append("(unclosed"), "bad table pattern"),
    (lambda d: d["extractor"].update(boilerplate_patterns=["(unclosed"]), "bad boilerplate pattern"),
    (lambda d: d["extractor"].update(wrap_gap_ratio=0), "wrap_gap_ratio"),
    (lambda d: d["extractor"].update(low_confidence=1.5), "low_confidence"),
    (lambda d: d["extractor"].pop("table"), "table"),
    (lambda d: d.pop("extractor"), "extractor"),
    (lambda d: d["types"]["fnol"]["fields"]["loss_date"].update(type="timestamp"), "type"),
])
def test_invalid_extractor_config_is_rejected_at_startup(mutate, match):
    d = raw()
    mutate(d)
    with pytest.raises(ValidationError, match=match):
        DocumentTypes.model_validate(d)


def test_field_and_table_specs_validate_directly():
    assert FieldSpec(labels=["Claimant"], type="text").labels == ["claimant"]
    assert FieldSpec(from_body=True).labels == []
    with pytest.raises(ValidationError):
        TableConfig(total_labels=["("], subtotal_labels=[], tax_labels=[], discount_labels=[], skip_labels=[])


def test_label_files_parse_to_exactly_what_was_written():
    """Guards the YAML trap that bit during development: an unquoted comma inside a {flow mapping} splits a value and adds a bogus key."""
    spec_fields = {t: set(s.fields) | ({"total_amount"} if s.table else set()) for t, s in DT.types.items()}
    corpus = yaml.safe_load((SYN / "extraction_set/corpus_labels.yaml").read_text(encoding="utf-8"))["documents"]
    items = yaml.safe_load((SYN / "extraction_set/items.yaml").read_text(encoding="utf-8"))["items"]
    entries = [(k, v["doc_type"], v) for k, v in corpus.items()] + [(i["id"], i["doc_type"], i) for i in items]
    assert len(entries) >= 80
    for key, dtype, e in entries:
        assert set(e) <= {"doc_type", "expected", "absent", "flagged", "table", "id", "split", "note", "text", "missing_required"}, key
        for name, val in (e.get("expected") or {}).items():
            assert name in spec_fields[dtype], (key, name)
            assert val is None or isinstance(val, str), (key, name, val)
        for name in [*e.get("absent", []), *e.get("flagged", []), *e.get("missing_required", [])]:
            assert name in spec_fields[dtype], (key, name)
        if "text" in e:
            assert e["text"].count("\n") >= 2, f"{key}: line breaks were lost"
        for row in (e.get("table") or {}).get("items", []):
            assert len(row) == 2 and all(isinstance(x, str) for x in row), (key, row)


# ================================================================== eval gate
def test_extractors_meet_the_proposed_targets_and_do_not_regress_the_recorded_baseline():
    from evals.metrics.extraction import all_cases, evaluate
    from evals.runners.eval_extractor import BASELINES, signals_hash

    reports = evaluate(EXTRACTOR, all_cases(), list(DT.types))
    assert {"corpus", "dev", "holdout", "fresh", "fresh2", "fresh3", "fresh4"} <= set(reports)
    for name, r in reports.items():                                          # never a silent wrong value; every required miss is reported
        assert r.silent_wrong == [], (name, [(o.case, o.field, o.got) for o in r.silent_wrong])
        assert not [o for o in r._k("required_wrong")], name
        assert not r._k("spurious"), name
    assert reports["corpus"].field_score == 1.0 and reports["corpus"].table_score() == 1.0
    assert min(r.field_score for r in reports.values() if r.field_score is not None) >= 0.95

    entries = json.loads(BASELINES.read_text(encoding="utf-8"))["entries"]
    latest = entries[-1]
    assert latest["signals_sha"] == signals_hash(), "extractor config/specs changed: re-run evals/runners/eval_extractor.py --write-baseline"
    for name, r in reports.items():
        old = latest["splits"][name]
        assert r.field_score >= old["field_score"] - 1e-9, f"{name}: field score regressed"
        assert r.summary()["silent_wrong"] <= old["silent_wrong"], f"{name}: more silent wrong values than the baseline"


def test_the_unseen_first_measurements_are_on_record_and_show_the_honest_story():
    from evals.runners.eval_extractor import BASELINES

    entries = json.loads(BASELINES.read_text(encoding="utf-8"))["entries"]
    first = {e["label"].split(" ")[0]: e for e in entries if "FIRST MEASUREMENT" in e["label"]}
    assert set(first) == {"v2", "v3", "v4"}
    assert first["v2"]["splits"]["fresh"]["silent_wrong"] == 2 and first["v3"]["splits"]["fresh3"]["silent_wrong"] == 2
    assert first["v4"]["splits"]["fresh4"]["silent_wrong"] == 0
    assert first["v3"]["splits"]["fresh3"]["field_score"] < 0.7               # the worst unseen round, kept visible
