"""Write the mock PAS database, prior claims/payments and invoice registry (T1.4). ALL SYNTHETIC.

    python scripts/seed_pas_mock.py           # (re)write data/synthetic/{pas_mock_db,prior_claims,invoice_registry}.json
    python scripts/seed_pas_mock.py --check   # fail if the files on disk differ

Every record is built through the typed models in claimtower.integrations.mock.data, so the JSON cannot contain anything invalid.
Policies POL-1001/1003/1004/1005 serve scenarios S-01..S-05 (kept consistent with `pas_expectation` in scenarios.yaml by a test).
The others exist for the structural cases T1.4 requires and for later scenarios (S-10, S-11, S-16).
"""
import json
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claimtower.integrations.mock.data import (  # noqa: E402
    CLAIMS_FILE, INVOICES_FILE, PAS_FILE, EndorsementEntry, InvoiceRecord, InvoiceRegistry, PasCase, PasDatabase, PasPolicy,
    PolicyVersion, PriorClaim, PriorClaimsFile, PriorPayment, StatusSegment,
)

D = date.fromisoformat
SCHEDULE = {"building_limit": "2000000", "standard_deductible": "10000"}      # SAMPLE


def end(eid: str, frm: str, to: str | None = None) -> EndorsementEntry:
    return EndorsementEntry(endorsement_id=eid, effective_from=D(frm), effective_to=D(to) if to else None)


def ver(v: str, frm: str, to: str | None, ends: list[EndorsementEntry], **extra: str) -> PolicyVersion:
    return PolicyVersion(version=v, effective_from=D(frm), effective_to=D(to) if to else None, base_wording_id="BASE-HOME-1",
                         endorsements=ends, schedule={**SCHEDULE, **extra})


def seg(status: str, frm: str, to: str | None = None, note: str = "") -> StatusSegment:
    return StatusSegment(status=status, effective_from=D(frm), effective_to=D(to) if to else None, note=note)


def simple_2026(no: str, holder: str, note: str, ends=("END-01",)) -> PasPolicy:
    return PasPolicy(policy_no=no, holder_name=holder, note=note, versions=[ver("v1", "2026-01-01", "2026-12-31", [end(e, "2026-01-01") for e in ends])],
                     status_history=[seg("in_force", "2026-01-01", "2026-12-31")])


POLICIES = [
    simple_2026("POL-1001", "SYNTHETIC Household 1001", "S-01 and S-02. In force all of 2026 with END-01."),
    # S-03: lapsed three days before the loss. Lapse is effective 2026-02-10, so 2026-02-09 is the last day in force.
    PasPolicy(policy_no="POL-1003", holder_name="SYNTHETIC Household 1003", note="S-03: lapsed on 2026-02-10, loss on 2026-02-13.",
              versions=[ver("v1", "2025-02-11", "2026-02-10", [])],
              status_history=[seg("in_force", "2025-02-11", "2026-02-09"), seg("lapsed", "2026-02-10", None, "premium not renewed")]),
    # S-04: END-02 is on the schedule but only takes effect on 2026-02-01, after the 2026-01-20 loss.
    PasPolicy(policy_no="POL-1004", holder_name="SYNTHETIC Household 1004", note="S-04: two endorsements with different effective dates.",
              versions=[ver("v1", "2025-12-01", "2026-11-30", [end("END-01", "2025-12-01"), end("END-02", "2026-02-01")])],
              status_history=[seg("in_force", "2025-12-01", "2026-11-30")]),
    # S-05: the schedule lists END-01, but the endorsement wording was never provided (a policy-store gap, not a PAS gap).
    simple_2026("POL-1005", "SYNTHETIC Household 1005", "S-05: schedule lists END-01 whose wording is not available to the claim."),
    simple_2026("POL-1006", "SYNTHETIC Household 1006", "Future S-10 (duplicate FNOL): same claimant files twice."),
    simple_2026("POL-1007", "SYNTHETIC Household 1007", "Future S-11 (duplicate invoice number)."),
    simple_2026("POL-1008", "SYNTHETIC Household 1008", "Future S-16: 1,850,000 already paid, so the remaining limit is 150,000."),
    simple_2026("POL-1009", "SYNTHETIC Household 1009", "Unrelated policy with an old fire claim (noise for duplicate detection).", ends=()),
    simple_2026("POL-1013", "SYNTHETIC Household 1013", "Sample claims under samples/ (claim_sample_1): in force all of 2026 with END-01."),
    # LAPSE BOUNDARY: in force through 2026-04-30 inclusive, lapsed from 2026-05-01.
    PasPolicy(policy_no="POL-1010", holder_name="SYNTHETIC Household 1010", note="Lapse boundary: loss on 04-30 is covered, on 05-01 is not.",
              versions=[ver("v1", "2025-05-01", "2026-04-30", [end("END-01", "2025-05-01")])],
              status_history=[seg("in_force", "2025-05-01", "2026-04-30"), seg("lapsed", "2026-05-01", None, "not renewed")]),
    # CANCELLED THEN REINSTATED: no cover in the cancellation gap, cover again from reinstatement. One continuous policy period.
    PasPolicy(policy_no="POL-1011", holder_name="SYNTHETIC Household 1011",
              note="Cancelled 2026-02-01, reinstated 2026-03-01; losses inside the gap are not covered.",
              versions=[ver("v1", "2025-06-01", "2026-05-31", [end("END-01", "2025-06-01")])],
              status_history=[seg("in_force", "2025-06-01", "2026-01-31"),
                              seg("cancelled", "2026-02-01", "2026-02-28", "premium not received"),
                              seg("reinstated", "2026-03-01", "2026-05-31", "premium received; no cover for the gap")]),
    # TWO VERSIONS STRADDLING THE LOSS DATE: a mid-term change on 2026-01-15 adds END-03 (higher water deductible).
    PasPolicy(policy_no="POL-1012", holder_name="SYNTHETIC Household 1012",
              note="Mid-term change: v1 until 2026-01-14, v2 from 2026-01-15 (adds END-03, water deductible 25,000 SAMPLE).",
              versions=[ver("v1", "2025-07-01", "2026-01-14", [end("END-01", "2025-07-01")]),
                        ver("v2", "2026-01-15", "2026-06-30", [end("END-01", "2025-07-01"), end("END-03", "2026-01-15")], water_deductible="25000")],
              status_history=[seg("in_force", "2025-07-01", "2026-06-30")]),
]


def case(no, as_of, status, version, ends, why) -> PasCase:
    return PasCase(policy_no=no, as_of=D(as_of), expected_status=status, expected_version=version, expected_endorsements=list(ends), why=why)


CASES = [
    case("POL-1001", "2026-03-14", "in_force", "v1", ["END-01"], "S-01 loss date"),
    case("POL-1001", "2026-05-02", "in_force", "v1", ["END-01"], "S-02 loss date"),
    case("POL-1003", "2026-02-13", "lapsed", "v1", [], "S-03: three days after the lapse; informational version only"),
    case("POL-1004", "2026-01-20", "in_force", "v1", ["END-01", "END-02"], "S-04: END-02 is listed but not yet effective on this date"),
    case("POL-1005", "2026-06-10", "in_force", "v1", ["END-01"], "S-05: END-01 listed in the schedule"),
    case("POL-1010", "2026-04-30", "in_force", "v1", ["END-01"], "lapse boundary: last day in force"),
    case("POL-1010", "2026-05-01", "lapsed", "v1", ["END-01"], "lapse boundary: first day lapsed"),
    case("POL-1010", "2025-04-30", "not_found", None, [], "day before inception: no cover record"),
    case("POL-1011", "2026-01-31", "in_force", "v1", ["END-01"], "last day before cancellation"),
    case("POL-1011", "2026-02-01", "cancelled", "v1", ["END-01"], "first day of cancellation"),
    case("POL-1011", "2026-02-15", "cancelled", "v1", ["END-01"], "inside the cancellation gap: not covered"),
    case("POL-1011", "2026-03-01", "reinstated", "v1", ["END-01"], "reinstatement day: covered again"),
    case("POL-1011", "2026-03-10", "reinstated", "v1", ["END-01"], "after reinstatement"),
    case("POL-1012", "2026-01-14", "in_force", "v1", ["END-01"], "straddle: last day of version v1"),
    case("POL-1012", "2026-01-15", "in_force", "v2", ["END-01", "END-03"], "straddle: first day of version v2"),
    case("POL-1012", "2026-03-01", "in_force", "v2", ["END-01", "END-03"], "inside version v2"),
    case("POL-9999", "2026-03-01", "not_found", None, [], "unknown policy number"),
    case("pol-1001", "2026-03-14", "not_found", None, [], "no fuzzy matching: a lower-case number is not an exact match, route to an operator"),
]

CLAIMS = [
    PriorClaim(claim_id="CLM-P-0010", policy_no="POL-1006", claimant_name="SYNTHETIC Claimant J", loss_date=D("2026-04-08"), loss_type="water_damage",
               channel="portal", description="On 8 April 2026 the bathroom pipe burst and flooded the floor.", amount_claimed=Decimal(90000),
               status="open", relation="exact_duplicate", relation_to="S-10",
               why="Same claimant, policy, loss date and description as the S-10 FNOL, filed earlier through the portal (S-10 arrives via the call centre)."),
    PriorClaim(claim_id="CLM-P-0020", policy_no="POL-1006", claimant_name="SYNTHETIC Claimant J", loss_date=D("2026-05-20"), loss_type="water_damage",
               channel="call_centre", description="On 20 May 2026 the kitchen sink overflowed and wet the cabinets.", amount_claimed=Decimal(35000),
               status="paid", relation="benign_near_match", relation_to="S-10",
               why="Same claimant and policy but a different date and location: a genuine second incident. Flag for review at most; never merge or reject."),
    PriorClaim(claim_id="CLM-P-0011", policy_no="POL-1007", claimant_name="SYNTHETIC Claimant K", loss_date=D("2026-03-01"), loss_type="water_damage",
               channel="email", description="On 1 March 2026 a laundry pipe leaked into the hallway.", amount_claimed=Decimal(85000), status="paid",
               relation="unrelated", relation_to="S-11",
               why="Not a duplicate claim. Holds invoice INV-2026-0042, whose number is reused by the S-11 claim (invoice-level duplicate check)."),
    PriorClaim(claim_id="CLM-P-0030", policy_no="POL-1008", claimant_name="SYNTHETIC Claimant L", loss_date=D("2026-01-15"), loss_type="water_damage",
               channel="portal", description="On 15 January 2026 a roof leak damaged several rooms.", amount_claimed=Decimal(1900000), status="paid",
               relation="unrelated", relation_to="S-16", why="Consumes most of the building limit so S-16 meets a nearly exhausted limit."),
    PriorClaim(claim_id="CLM-P-0050", policy_no="POL-1009", claimant_name="SYNTHETIC Claimant M", loss_date=D("2025-08-02"), loss_type="fire",
               channel="agent", description="Kitchen fire on 2 August 2025.", amount_claimed=Decimal(30000), status="closed", relation="unrelated",
               why="Noise: a different policy and peril."),
]

PAYMENTS = [
    PriorPayment(payment_id="PAY-P-0011", claim_id="CLM-P-0011", policy_no="POL-1007", amount=Decimal(85000), paid_on=D("2026-03-28"),
                 idempotency_key="CLM-P-0011:pay:1"),
    PriorPayment(payment_id="PAY-P-0030", claim_id="CLM-P-0030", policy_no="POL-1008", amount=Decimal(1850000), paid_on=D("2026-03-05"),
                 idempotency_key="CLM-P-0030:pay:1"),
    PriorPayment(payment_id="PAY-P-0050", claim_id="CLM-P-0050", policy_no="POL-1009", amount=Decimal(30000), paid_on=D("2025-09-01"),
                 idempotency_key="CLM-P-0050:pay:1"),
]

INVOICES = [
    InvoiceRecord(invoice_no="INV-2026-0042", vendor_name="SYNTHETIC Plumbing Works", vendor_ref="VND-0007", invoice_date=D("2026-03-10"),
                  amount=Decimal(85000), claim_id="CLM-P-0011", status="paid", relation="same_invoice_revised", relation_to="S-11",
                  why="The S-11 invoice reuses this number and vendor with a revised amount and later date: possible legitimate revision. Needs review, not a fraud label."),
    InvoiceRecord(invoice_no="INV-7788", vendor_name="SYNTHETIC Tiles Pvt Ltd", vendor_ref="VND-0011", invoice_date=D("2026-02-20"),
                  amount=Decimal(22000), claim_id="CLM-P-0050", status="paid", relation="same_number_other_vendor", relation_to="S-11",
                  why="Same number as a different vendor's invoice would be a coincidence: numbers are only unique per vendor."),
    InvoiceRecord(invoice_no="INV-2026-0043", vendor_name="SYNTHETIC Plumbing Works", vendor_ref="VND-0007", invoice_date=D("2026-03-12"),
                  amount=Decimal(12500), claim_id="CLM-P-0011", status="paid", relation="same_vendor_other_invoice",
                  why="Same vendor, different invoice number and amount: a normal second invoice."),
    InvoiceRecord(invoice_no="INV-3001", vendor_name="SYNTHETIC Electricals", vendor_ref="VND-0021", invoice_date=D("2025-08-20"),
                  amount=Decimal(30000), claim_id="CLM-P-0050", status="paid", relation="unrelated", why="Noise."),
    InvoiceRecord(invoice_no="INV-2026-0050", vendor_name="SYNTHETIC Plumbing Works", vendor_ref="VND-0007", invoice_date=D("2026-04-18"),
                  amount=Decimal(90000), claim_id="CLM-P-0010", status="submitted", relation="unrelated", why="Invoice of the S-10 prior claim."),
]


def render() -> dict[Path, str]:
    models = {PAS_FILE: PasDatabase(policies=POLICIES, cases=CASES), CLAIMS_FILE: PriorClaimsFile(claims=CLAIMS, payments=PAYMENTS),
              INVOICES_FILE: InvoiceRegistry(invoices=INVOICES)}
    return {p: json.dumps(m.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n" for p, m in models.items()}


def main() -> int:
    files = render()
    if "--check" in sys.argv:
        stale = [p.name for p, t in files.items() if not p.exists() or p.read_text(encoding="utf-8") != t]
        print("Mock data out of date - run scripts/seed_pas_mock.py: " + ", ".join(stale) if stale else "Mock data up to date.")
        return 1 if stale else 0
    for p, text in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {', '.join(p.name for p in files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
