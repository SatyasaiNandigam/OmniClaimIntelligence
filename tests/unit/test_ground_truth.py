"""Consistency checks on the synthetic scenarios and their ground-truth labels (T1.2)."""
from decimal import Decimal

import pytest

from claimtower.core.config import load_settings
from claimtower.core.schemas import SettlementInputs
from claimtower.settlement.engine import calculate
from claimtower.settlement.rules_loader import load_rules
from evals.ground_truth import load_all_ground_truth, load_scenarios

SC = load_scenarios()
GT = load_all_ground_truth()
IDS = ["S-01", "S-02", "S-03", "S-04", "S-05"]
CLAUSES = {c.clause_id for c in SC.clause_catalogue}
RULES = load_rules(load_settings().pack.calculation["settlement_v1"])


def test_five_scenarios_with_ground_truth():
    assert [s.id for s in SC.scenarios] == IDS
    assert sorted(GT) == IDS


def test_events_are_time_ordered_and_reference_real_documents():
    for s in SC.scenarios:
        dates = [e["on"] for e in s.events]
        assert dates == sorted(dates), s.id
        doc_ids = {d.document_id for d in s.documents}
        assert {e["document_id"] for e in s.events if "document_id" in e} <= doc_ids, s.id
        assert s.header().loss_date is not None


@pytest.mark.parametrize("sid", IDS)
def test_facts_have_verbatim_provenance(sid):
    s = next(x for x in SC.scenarios if x.id == sid)
    gt = GT[sid]
    assert gt.facts, sid
    for f in gt.facts:
        assert f.claim_id == s.claim_id
        assert f.evidence_text in s.page_text(f.source_document_id, f.page), f.fact_id


@pytest.mark.parametrize("sid", IDS)
def test_cross_references_resolve(sid):
    gt = GT[sid]
    fact_ids = {f.fact_id for f in gt.facts}
    assert len(fact_ids) == len(gt.facts)
    qids = {q.question_id for q in gt.questions}
    for f in gt.facts:
        assert set(f.related_question_ids) <= qids, f.fact_id
    for q in gt.questions:
        assert set(q.linked_fact_ids) <= fact_ids and set(q.linked_clause_ids) <= CLAUSES, q.question_id
    for c in gt.contradictions:
        assert set(c.fact_ids_a + c.fact_ids_b) <= fact_ids and c.question_id in qids
    for m in gt.missing_evidence:
        assert m.question_id in qids and set(m.resolves_question_ids) <= qids
    issue_ids = {i.issue_id for i in gt.issues}
    for i in gt.issues:
        assert set(i.clause_ids) <= CLAUSES and set(i.fact_ids) <= fact_ids, i.issue_id
    for c in gt.counterfactuals:
        assert set(c.affected_issue_ids) <= issue_ids
    snap = gt.snapshot
    assert set(snap.effective_clause_ids + snap.not_effective_clause_ids) <= CLAUSES


@pytest.mark.parametrize("sid", IDS)
def test_missing_evidence_requests_map_to_question_criteria(sid):
    gt = GT[sid]
    criteria = {c for q in gt.questions for c in q.establishing_criteria}
    for m in gt.missing_evidence:
        assert set(m.minimum_evidence_request.maps_to_criteria) <= criteria, m.question_id


@pytest.mark.parametrize("sid", IDS)
def test_settlement_labels_agree_with_the_engine(sid):
    st = GT[sid].settlement
    if st is None or not st.engine_check:
        pytest.skip("no engine-checked settlement for this scenario")
    out = calculate(RULES, SettlementInputs(**{k: Decimal(v) for k, v in st.inputs.items()}, unavailable_inputs=st.unavailable_inputs))
    assert out.status == st.status
    if st.status == "calculated":
        assert out.payable == Decimal(st.payable)
    else:
        assert out.missing_inputs == sorted(st.missing_inputs)


def test_scenario_specific_expectations():
    s1, s3, s4, s5 = GT["S-01"], GT["S-03"], GT["S-04"], GT["S-05"]
    assert {m.gap_type for m in s1.missing_evidence} == {"contradictory", "missing", "insufficient"}
    assert len(s1.contradictions) == 1 and {c.scenario_id for c in s1.counterfactuals} == {"A", "B", "C"}
    assert s1.routing.tier == "complex" and GT["S-02"].routing.tier == "simple"
    assert s3.snapshot.status == "not_in_force" and s3.settlement is None
    assert "CL-SUB-WATER-E2" in s4.snapshot.not_effective_clause_ids and s4.settlement.payable == "100000"
    assert s5.snapshot.missing_documents == ["END-01"]
    assert next(i for i in s5.issues if i.issue_id == "ISS-SUBLIMIT").status == "abstain"


def test_no_outcome_language_in_labels():
    banned = ("fraud", "will be approved", "will be denied", "claim approved", "claim denied", "covered / not covered")
    for gt in GT.values():
        blob = gt.model_dump_json().lower()
        assert not [b for b in banned if b in blob], gt.scenario_id
