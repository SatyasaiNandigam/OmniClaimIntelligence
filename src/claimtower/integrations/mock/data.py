"""Typed mock data for the PAS, the duplicate checks and the payment history (T1.4). ALL SYNTHETIC.

Files (written by scripts/seed_pas_mock.py, loaded here):
  data/synthetic/pas_mock_db.json        policies, versions, endorsements, status history, plus labelled lookup `cases`
  data/synthetic/prior_claims.json       earlier claims and payments (duplicate-FNOL and payment-idempotency tests, remaining limit)
  data/synthetic/invoice_registry.json   known invoices (duplicate-invoice tests)

`PasDatabase.lookup` is the read path of the mock PAS: policy number + date -> PolicyVersionResponse (frozen contract). The mock adapter
(T6.5) wraps it with latency / failure injection. Dates are inclusive on both ends: a status or version covers
effective_from .. effective_to (None = open-ended).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

from claimtower.core.schemas import EndorsementRef, PolicyVersionResponse
from claimtower.core.schemas.common import Contract, Money

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data" / "synthetic"
PAS_FILE, CLAIMS_FILE, INVOICES_FILE = DATA / "pas_mock_db.json", DATA / "prior_claims.json", DATA / "invoice_registry.json"
PAS_EXTRA_FILE = DATA.parent / "local" / "pas_policies.json"       # your own policies for testing with real documents (not synthetic data; env CLAIMTOWER_PAS_EXTRA overrides the path)

PolicyStatus = Literal["in_force", "lapsed", "cancelled", "reinstated"]


# ------------------------------------------------------------------------------------------- PAS
class StatusSegment(Contract):
    status: PolicyStatus
    effective_from: date
    effective_to: date | None = None
    note: str = ""


class EndorsementEntry(Contract):
    endorsement_id: str
    effective_from: date
    effective_to: date | None = None


class PolicyVersion(Contract):
    version: str
    effective_from: date
    effective_to: date | None = None
    base_wording_id: str
    endorsements: list[EndorsementEntry] = []     # as listed in the schedule, whether or not the wording document is available
    schedule: dict[str, str] = {}                 # SAMPLE schedule values (building limit, standard deductible)


class PasPolicy(Contract):
    policy_no: str
    holder_name: str
    versions: list[PolicyVersion]
    status_history: list[StatusSegment]
    note: str = ""


class PasCase(Contract):
    """A labelled lookup: what PasDatabase.lookup must return. Used by tests and later by the T3.3 snapshot tests."""

    policy_no: str
    as_of: date
    expected_status: Literal["in_force", "lapsed", "cancelled", "reinstated", "not_found"]
    expected_version: str | None = None
    expected_endorsements: list[str] = []
    why: str


def _covers(start: date, end: date | None, d: date) -> bool:
    return start <= d and (end is None or d <= end)


class PasDatabase(Contract):
    synthetic: Literal[True] = True
    note: str = "SYNTHETIC mock PAS - not a real policy administration system"
    policies: list[PasPolicy]
    cases: list[PasCase] = []

    def policy(self, policy_no: str) -> PasPolicy | None:
        return next((p for p in self.policies if p.policy_no == policy_no), None)

    def lookup(self, policy_no: str, as_of: date) -> PolicyVersionResponse:
        base = dict(policy_no=policy_no, as_of=as_of, mock=True, source_system="PAS-MOCK", response_ref=f"PAS-MOCK/{policy_no}/{as_of}")
        pol = self.policy(policy_no)
        if pol is None:
            return PolicyVersionResponse(status="not_found", **base)
        seg = next((s for s in pol.status_history if _covers(s.effective_from, s.effective_to, as_of)), None)
        if seg is None:                                                     # before inception, or in a gap with no record
            return PolicyVersionResponse(status="not_found", **base)
        # the version in force on that date; for a lapsed/cancelled policy fall back to the latest earlier version (informational)
        ver = next((v for v in pol.versions if _covers(v.effective_from, v.effective_to, as_of)), None)
        if ver is None:
            earlier = [v for v in pol.versions if v.effective_from <= as_of]
            ver = max(earlier, key=lambda v: v.effective_from) if earlier else None
        return PolicyVersionResponse(
            status=seg.status, status_effective_from=seg.effective_from, version=ver.version if ver else None,
            effective_from=ver.effective_from if ver else None, effective_to=ver.effective_to if ver else None,
            base_wording_id=ver.base_wording_id if ver else None,
            endorsements=[EndorsementRef(**e.model_dump()) for e in ver.endorsements] if ver else [], **base)


# ------------------------------------------------------------------------------------------- prior claims, payments, invoices
class PriorClaim(Contract):
    claim_id: str
    policy_no: str
    claimant_name: str
    loss_date: date
    loss_type: str
    channel: Literal["portal", "call_centre", "email", "agent", "other"]
    description: str
    amount_claimed: Money
    status: Literal["open", "paid", "closed", "rejected"]
    relation: Literal["exact_duplicate", "benign_near_match", "unrelated"]    # label for duplicate-detection evaluation
    relation_to: str | None = None                                             # scenario the label refers to (S-10 ...)
    why: str


class PriorPayment(Contract):
    payment_id: str
    claim_id: str
    policy_no: str
    amount: Money
    paid_on: date
    idempotency_key: str


class PriorClaimsFile(Contract):
    synthetic: Literal[True] = True
    claims: list[PriorClaim]
    payments: list[PriorPayment] = []

    def paid_against_policy(self, policy_no: str) -> Money:
        return sum((p.amount for p in self.payments if p.policy_no == policy_no), start=Money(0))


class InvoiceRecord(Contract):
    invoice_no: str
    vendor_name: str
    vendor_ref: str
    invoice_date: date
    amount: Money
    currency: str = "INR"
    claim_id: str
    status: Literal["submitted", "paid", "rejected"]
    relation: Literal["same_invoice_revised", "same_number_other_vendor", "same_vendor_other_invoice", "unrelated"]
    relation_to: str | None = None
    why: str


class InvoiceRegistry(Contract):
    synthetic: Literal[True] = True
    invoices: list[InvoiceRecord]


def _load(model, path: Path):
    return model.model_validate(json.loads(path.read_text(encoding="utf-8")))


def load_pas_db(path: Path = PAS_FILE, extra: Path | None = None) -> PasDatabase:
    """The mock PAS. Policies in the optional extra file ({"policies": [...]}, same layout) are added, and replace a synthetic policy of the same number."""
    db = _load(PasDatabase, path)
    if extra is not None and extra.is_file():
        added = [PasPolicy.model_validate(p) for p in json.loads(extra.read_text(encoding="utf-8")).get("policies", [])]
        db.policies = [p for p in db.policies if p.policy_no not in {a.policy_no for a in added}] + added
    return db


def load_prior_claims(path: Path = CLAIMS_FILE) -> PriorClaimsFile:
    return _load(PriorClaimsFile, path)


def load_invoice_registry(path: Path = INVOICES_FILE) -> InvoiceRegistry:
    return _load(InvoiceRegistry, path)
