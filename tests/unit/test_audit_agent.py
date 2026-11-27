"""T8.5 - the Quality & Audit agent: seeded defects (fabricated citation, mismatched total, outcome wording ...) are caught, a failing report blocks the release to the gate."""
import json
from decimal import Decimal
from pathlib import Path

import pytest

from claimtower.agents.audit import AuditAgent
from claimtower.audit.checks import QaConfig, run_checks
from claimtower.composition import build_registry
from claimtower.core.config import load_settings
from claimtower.core.schemas import ClaimState, ClauseCitation, Draft, ProposedAction
from claimtower.core.state import OwnershipViolation, StatePatch, apply_patch
from claimtower.orchestration.graph import ClaimRunner

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()
CFG = QaConfig.load(SETTINGS.pack)


def good(sid="S-02") -> ClaimState:
    """A fixture state completed by the REAL settlement agent (the fixture has no explanation): a state that passes."""
    s = ClaimState(**json.loads((ROOT / f"data/fixtures/{sid}/claim_state.json").read_text(encoding="utf-8")))
    return apply_patch(s, build_registry(SETTINGS)["settlement"].agent.run(s))


def failures(state):
    r = run_checks(state, CFG)
    return {f.check for f in r.failures}


@pytest.mark.parametrize("sid", ["S-02", "S-05"])
def test_a_consistent_assessment_passes(sid):
    r = run_checks(good(sid), CFG)
    assert r.passed and r.failures == []


# ---------------------------------------------------------------------------------------------------------------- seeded defects
def test_a_fabricated_citation_is_caught():
    s = good()
    fake = ClauseCitation(clause_id="CL-INVENTED", document="BASE-HOME-1", text="We always pay.")
    bad = s.model_copy(update={"coverage_issues": [s.coverage_issues[1].model_copy(update={"clause_citations": [fake]}), *s.coverage_issues[2:]]})
    r = run_checks(bad, CFG)
    assert not r.passed and any(f.check == "citation_unknown_clause" and f.ref == "CL-INVENTED" for f in r.failures)


def test_an_altered_quote_of_a_real_clause_is_caught():
    s = good()
    c = s.coverage_issues[1].clause_citations[0]
    bad = s.model_copy(update={"coverage_issues": [s.coverage_issues[1].model_copy(update={"clause_citations": [c.model_copy(update={"text": c.text + " In every case."})]})]})
    assert "citation_text_mismatch" in failures(bad)


def test_a_mismatched_total_is_caught_whether_the_payable_or_the_text_was_changed():
    s = good()
    assert "settlement_explanation" in failures(s.model_copy(update={"settlement": s.settlement.model_copy(update={"payable": Decimal(90000)})}))              # payable changed, text not
    assert "settlement_explanation" in failures(s.model_copy(update={"settlement": s.settlement.model_copy(update={"explanation": s.settlement.explanation.replace("INR 80,000", "INR 85,000", 1)})}))
    broken = s.settlement.model_copy(update={"steps": [s.settlement.steps[0], s.settlement.steps[1].model_copy(update={"amount_before": Decimal(5)}), *s.settlement.steps[2:]]})
    assert "settlement_trace" in failures(s.model_copy(update={"settlement": broken}))
    assert "settlement_explanation" in failures(s.model_copy(update={"settlement": s.settlement.model_copy(update={"explanation": None})}))


def test_a_payable_shown_while_the_settlement_cannot_be_calculated_is_caught():
    s = good("S-05")
    assert "settlement_forecast" in failures(s.model_copy(update={"settlement": s.settlement.model_copy(update={"payable": Decimal(100000)})}))


@pytest.mark.parametrize("text", ["Fraud confirmed on this claim.", "Your claim has been approved.", "The claim is denied.", "The loss is covered in full.", "You will be paid next week.",
                                  "The water damage is not covered."])
def test_wording_that_states_an_outcome_is_caught_in_any_text_written_for_a_person(text):
    s = good()
    draft = Draft(draft_id="D1", kind="status_message", addressee_role="claimant", subject="s", body=text)
    assert "forbidden_language" in failures(s.model_copy(update={"drafts": [draft]}))
    assert "forbidden_language" in failures(s.model_copy(update={"coverage_issues": [s.coverage_issues[0].model_copy(update={"note": text})]}))


def test_neutral_wording_is_not_flagged():
    s = good()
    draft = Draft(draft_id="D1", kind="status_message", addressee_role="claimant", subject="Claim", body="A claims colleague reviews every claim before anything is decided.")
    assert run_checks(s.model_copy(update={"drafts": [draft]}), CFG).passed


def test_an_unknown_fact_or_a_fact_without_evidence_or_a_question_without_support_is_caught():
    s = good()
    issue = s.coverage_issues[1].model_copy(update={"fact_ids": ["F-NOPE"]})
    assert "citation_unknown_fact" in failures(s.model_copy(update={"coverage_issues": [issue]}))
    f0 = s.facts[0].model_copy(update={"evidence_text": "  "})
    assert "fact_provenance" in failures(s.model_copy(update={"facts": [f0, *s.facts[1:]]}))
    assert "fact_provenance" in failures(s.model_copy(update={"facts": [s.facts[0].model_copy(update={"source_document_id": "DOC-GHOST"}), *s.facts[1:]]}))
    q = next(q for q in s.open_questions if q.question_id == "Q-SUDDEN").model_copy(update={"linked_fact_ids": [], "linked_clause_ids": [], "status": "established"})
    assert "established_without_evidence" in failures(s.model_copy(update={"open_questions": [q]}))


def test_a_coverage_row_without_a_citation_is_caught_except_an_abstention():
    s = good()
    none = s.coverage_issues[1].model_copy(update={"clause_citations": [], "status": "established"})
    assert "citation_missing" in failures(s.model_copy(update={"coverage_issues": [none]}))
    assert run_checks(s.model_copy(update={"coverage_issues": [none.model_copy(update={"status": "abstain"})]}), CFG).passed


def test_nothing_may_be_carried_out_or_sent_before_an_approval_and_a_human_decision_stays_required():
    s = good()
    done = ProposedAction(action_id="ACT-1", action_type="create_task", responsible_party="x", execution_mode="controlled_api", status="executed", idempotency_key="k")
    assert "action_before_approval" in failures(s.model_copy(update={"proposed_actions": [done]}))
    sent = Draft(draft_id="D1", kind="status_message", addressee_role="claimant", subject="s", body="b", status="sent")
    assert "action_before_approval" in failures(s.model_copy(update={"drafts": [sent]}))
    assert "action_before_approval" not in failures(s.model_copy(update={"proposed_actions": [done], "resolution_status": "approved"}))
    assert "approval_required" in failures(s.model_copy(update={"human_approval_required": False}))


# ---------------------------------------------------------------------------------------------------------------- the agent and the block
def test_the_agent_owns_the_report_and_the_reference_only_and_explains_a_failure():
    a = AuditAgent(CFG)
    assert (a.agent_id, set(a.owns)) == ("audit", {"qa_report", "audit_ref"})
    ok = a.run(good())
    assert ok.updates["qa_report"].passed and ok.updates["audit_ref"] == "AUDIT-CLM-0002" and ok.rationale.startswith("quality checks passed")
    s = good()
    bad = a.run(s.model_copy(update={"settlement": s.settlement.model_copy(update={"payable": Decimal(1)})}))
    assert not bad.updates["qa_report"].passed and bad.rationale.startswith("QUALITY CHECKS FAILED") and "settlement_explanation" in bad.rationale
    with pytest.raises(OwnershipViolation):
        apply_patch(good(), StatePatch(agent_id="audit", updates={"facts": []}))


def test_a_failing_report_blocks_the_release_to_the_gate_with_the_reasons():
    reg = build_registry(SETTINGS, force_mock=("intake", "gap", "coverage", "settlement", "action", "routing"))      # fixtures: the fixture settlement has no explanation
    st = ClaimRunner(reg).start(ClaimState(claim_id="CLM-0002"))
    assert st.phase == "halted" and st.paused_at == "audit" and "quality checks failed, the claim is not released to the gate" in st.halted_reason
    assert "settlement_explanation" in st.halted_reason and "routing" not in st.nodes_run and st.gate is None


def test_a_passing_report_goes_on_to_routing_and_the_gate():
    reg = build_registry(SETTINGS, force_mock=("intake", "gap", "coverage", "action", "routing"))                    # the real settlement agent writes the explanation
    r = ClaimRunner(reg)
    st = r.start(ClaimState(claim_id="CLM-0002"))
    assert st.phase == "awaiting_human_decision" and r.state("CLM-0002").qa_report.passed and r.state("CLM-0002").audit_ref == "AUDIT-CLM-0002"
