"""T1.4 - mock PAS, prior claims/payments and invoice registry: structurally valid and consistent with the scenarios and policy corpus."""
import json
import subprocess
import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from claimtower.core.schemas import PolicyVersionResponse
from claimtower.integrations.mock.data import load_invoice_registry, load_pas_db, load_prior_claims
from evals.ground_truth import load_scenarios

ROOT = Path(__file__).resolve().parents[2]
PAS, CLAIMS, INVOICES = load_pas_db(), load_prior_claims(), load_invoice_registry()
SC = load_scenarios()
CORPUS = json.loads((ROOT / "data/synthetic/policies/clause_index.json").read_text(encoding="utf-8"))
CORPUS_DOCS = {d["doc_id"] for d in CORPUS["documents"]}


def ends(resp):
    return [e.endorsement_id for e in resp.endorsements]


def test_seed_script_output_is_up_to_date():
    r = subprocess.run([sys.executable, "scripts/seed_pas_mock.py", "--check"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("case", PAS.cases, ids=lambda c: f"{c.policy_no}@{c.as_of}")
def test_every_labelled_lookup_returns_what_it_says(case):
    r = PAS.lookup(case.policy_no, case.as_of)
    assert isinstance(r, PolicyVersionResponse) and r.mock is True and r.source_system == "PAS-MOCK"
    assert (r.status, r.version, ends(r)) == (case.expected_status, case.expected_version, case.expected_endorsements), case.why
    assert r.response_ref == f"PAS-MOCK/{case.policy_no}/{case.as_of}"


# ------------------------------------------------------------------ the three structural cases T1.4 requires
def test_lapse_boundary():
    last, first = PAS.lookup("POL-1010", date(2026, 4, 30)), PAS.lookup("POL-1010", date(2026, 5, 1))
    assert last.status == "in_force" and first.status == "lapsed" and first.status_effective_from == date(2026, 5, 1)
    assert PAS.lookup("POL-1010", date(2025, 4, 30)).status == "not_found"                    # and the day before inception


def test_cancelled_then_reinstated_policy_has_a_gap_with_no_cover():
    days = (date(2026, 1, 31), date(2026, 2, 1), date(2026, 2, 28), date(2026, 3, 1))
    assert [PAS.lookup("POL-1011", d).status for d in days] == ["in_force", "cancelled", "cancelled", "reinstated"]
    assert PAS.lookup("POL-1011", days[3]).version == PAS.lookup("POL-1011", days[0]).version == "v1"   # one continuous policy period


def test_two_versions_straddle_the_loss_date_with_different_endorsements():
    before, after = PAS.lookup("POL-1012", date(2026, 1, 14)), PAS.lookup("POL-1012", date(2026, 1, 15))
    assert (before.version, ends(before)) == ("v1", ["END-01"]) and (after.version, ends(after)) == ("v2", ["END-01", "END-03"])
    assert before.status == after.status == "in_force"
    assert PAS.policy("POL-1012").versions[1].schedule["water_deductible"] == "25000"           # SAMPLE


# ------------------------------------------------------------------ consistency with the scenarios and the policy corpus
@pytest.mark.parametrize("sc", SC.scenarios, ids=lambda s: s.id)
def test_pas_agrees_with_each_scenarios_expectation(sc):
    exp, loss = sc.pas_expectation, sc.header().loss_date
    r = PAS.lookup(exp["policy"], loss)
    assert r.status == exp["status"]
    if exp["status"] == "in_force":
        assert [str(r.effective_from), str(r.effective_to)] == exp["period"]
    assert ends(r) == [e if isinstance(e, str) else e["id"] for e in exp["endorsements"]]
    for e in exp["endorsements"]:
        if isinstance(e, dict):
            assert next(x for x in r.endorsements if x.endorsement_id == e["id"]).effective_from == date.fromisoformat(e["effective_from"])
    if exp["status"] == "lapsed":
        assert r.status_effective_from == date.fromisoformat(exp["lapsed_on"])


def test_s04_pas_lists_both_endorsements_but_only_one_is_effective_on_the_loss_date():
    loss = date(2026, 1, 20)
    eff = {e.endorsement_id: e.effective_from <= loss for e in PAS.lookup("POL-1004", loss).endorsements}
    assert eff == {"END-01": True, "END-02": False}


def test_exact_match_only_no_guessing():
    for bad in ("pol-1001", "POL-1001 ", "POL1001", "POL-1002", ""):
        assert PAS.lookup(bad, date(2026, 3, 14)).status == "not_found", bad


def test_endorsements_and_wordings_exist_in_the_policy_corpus():
    for pol in PAS.policies:
        for v in pol.versions:
            assert v.base_wording_id in CORPUS_DOCS
            assert {e.endorsement_id for e in v.endorsements} <= CORPUS_DOCS, pol.policy_no
    end02 = next(e for e in PAS.policy("POL-1004").versions[0].endorsements if e.endorsement_id == "END-02")
    assert str(end02.effective_from) == next(d for d in CORPUS["documents"] if d["doc_id"] == "END-02")["effective_from"]


# ------------------------------------------------------------------ data hygiene
def test_status_segments_and_versions_do_not_overlap_and_are_ordered():
    for pol in PAS.policies:
        for group in (pol.status_history, pol.versions):
            for a, b in zip(group, group[1:]):
                assert a.effective_to is not None and a.effective_to < b.effective_from, pol.policy_no
        assert all(s.effective_to is None or s.effective_from <= s.effective_to for s in pol.status_history)


def test_versions_of_a_policy_are_continuous_with_no_day_uncovered():
    for pol in PAS.policies:
        for a, b in zip(pol.versions, pol.versions[1:]):
            assert a.effective_to + timedelta(days=1) == b.effective_from, pol.policy_no


def test_prior_claims_cover_duplicate_and_benign_near_match_and_reference_real_policies():
    assert {c.relation for c in CLAIMS.claims} >= {"exact_duplicate", "benign_near_match", "unrelated"}
    assert all(c.policy_no in {p.policy_no for p in PAS.policies} for c in CLAIMS.claims)
    dup = next(c for c in CLAIMS.claims if c.relation == "exact_duplicate")
    near = next(c for c in CLAIMS.claims if c.relation == "benign_near_match")
    assert (dup.policy_no, dup.claimant_name) == (near.policy_no, near.claimant_name) and dup.loss_date != near.loss_date
    assert len({c.claim_id for c in CLAIMS.claims}) == len(CLAIMS.claims)


def test_payments_are_consistent_and_leave_the_expected_remaining_limit():
    assert all(p.claim_id in {c.claim_id for c in CLAIMS.claims} for p in CLAIMS.payments)
    keys = [p.idempotency_key for p in CLAIMS.payments]
    assert len(keys) == len(set(keys)) and len({p.payment_id for p in CLAIMS.payments}) == len(keys)
    limit = Decimal(PAS.policy("POL-1008").versions[0].schedule["building_limit"])
    assert limit - CLAIMS.paid_against_policy("POL-1008") == Decimal(150000)                  # S-16: nearly exhausted limit
    for no in ("POL-1001", "POL-1003", "POL-1004", "POL-1005"):
        assert CLAIMS.paid_against_policy(no) == 0, f"{no}: S-01..S-05 expect an untouched limit"


def test_invoice_registry_supports_duplicate_invoice_and_near_match_cases():
    kinds = {"same_invoice_revised", "same_number_other_vendor", "same_vendor_other_invoice", "unrelated"}
    assert {i.relation for i in INVOICES.invoices} >= kinds
    assert all(i.claim_id in {c.claim_id for c in CLAIMS.claims} for i in INVOICES.invoices)
    keys = [(i.invoice_no, i.vendor_ref) for i in INVOICES.invoices]
    assert len(keys) == len(set(keys)), "an invoice number is unique per vendor"
    revised = next(i for i in INVOICES.invoices if i.relation == "same_invoice_revised")
    other = next(i for i in INVOICES.invoices if i.relation == "same_number_other_vendor")
    assert (revised.invoice_no, revised.amount) == ("INV-2026-0042", Decimal(85000)) and other.vendor_ref != revised.vendor_ref


def test_all_records_are_labelled_synthetic():
    for f in ("pas_mock_db", "prior_claims", "invoice_registry"):
        assert json.loads((ROOT / f"data/synthetic/{f}.json").read_text(encoding="utf-8"))["synthetic"] is True
    assert all(p.holder_name.startswith("SYNTHETIC") for p in PAS.policies)
    assert all(c.claimant_name.startswith("SYNTHETIC") for c in CLAIMS.claims)
