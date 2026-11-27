"""Quality checks on a finished assessment (T8.5): pure functions of the claim state, no model. A safety net must not depend on the thing it checks.

    report = run_checks(state, QaConfig.load(pack))          # QAReport(passed, failures[])

Each check returns `QAFailure(check, detail, ref)` for every defect it finds:
  citation_missing          a coverage issue (other than an abstention or the not-in-force row) cites no clause
  citation_unknown_clause   a cited clause is not in the policy snapshot (a fabricated citation)
  citation_text_mismatch    a cited clause's text differs from the snapshot's (the quote was altered)
  citation_unknown_fact     a fact cited by a coverage issue, a question, a request or a contradiction does not exist
  fact_provenance           a fact has no verbatim evidence, or names a document that is not in the claim
  established_without_evidence   a question is `established` with no fact and no clause behind it
  settlement_trace          the calculation trace does not chain, or its last amount is not the payable
  settlement_explanation    the explanation fails its verification against the engine's result (a displayed total that is not the payable)
  settlement_forecast       a payable is shown while the settlement cannot be calculated
  forbidden_language        wording that states an outcome (fraud confirmed, claim approved, "is covered" ...) in anything written for a person
  approval_required         the claim does not require a human decision (guardrail 2)
  action_before_approval    an action was executed, or a draft sent, before any human approval
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal

import yaml

from claimtower.core.config import Pack
from claimtower.core.schemas import ClaimState, QAFailure, QAReport
from claimtower.settlement.explain import verify

NOT_IN_FORCE_ISSUE = "ISS-INFORCE"


@dataclass
class QaConfig:
    forbidden: list[re.Pattern[str]] = field(default_factory=list)
    exempt_statuses: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, pack: Pack) -> QaConfig:
        raw = yaml.safe_load((pack.root / "qa_checks.yaml").read_text(encoding="utf-8"))
        return cls([re.compile(p, re.IGNORECASE) for p in raw["forbidden_patterns"]], set(raw.get("citation_exempt_statuses", [])))


def _norm(s: str) -> str:
    return " ".join(s.split())


def check_citations(state: ClaimState, cfg: QaConfig) -> list[QAFailure]:
    out: list[QAFailure] = []
    clauses = {c.clause_id: c for c in state.policy_snapshot.clauses} if state.policy_snapshot else {}
    facts = {f.fact_id for f in state.facts}
    for i in state.coverage_issues:
        if not i.clause_citations and i.status not in cfg.exempt_statuses and not (i.issue_id == NOT_IN_FORCE_ISSUE and state.policy_snapshot is None):
            out.append(QAFailure(check="citation_missing", detail=f"{i.issue_id} is '{i.status}' but cites no clause", ref=i.issue_id))
        for c in i.clause_citations:
            known = clauses.get(c.clause_id)
            if known is None:
                out.append(QAFailure(check="citation_unknown_clause", detail=f"{i.issue_id} cites {c.clause_id}, which is not in the policy snapshot", ref=c.clause_id))
            elif _norm(known.text) != _norm(c.text):
                out.append(QAFailure(check="citation_text_mismatch", detail=f"{i.issue_id}: the quoted text of {c.clause_id} differs from the policy snapshot", ref=c.clause_id))
        out += [QAFailure(check="citation_unknown_fact", detail=f"{i.issue_id} cites fact {f}, which does not exist", ref=f) for f in i.fact_ids if f not in facts]
    for q in state.open_questions:
        out += [QAFailure(check="citation_unknown_fact", detail=f"question {q.question_id} links fact {f}, which does not exist", ref=f) for f in q.linked_fact_ids if f not in facts]
        if clauses:
            out += [QAFailure(check="citation_unknown_clause", detail=f"question {q.question_id} links clause {c}, which is not in the policy snapshot", ref=c)
                    for c in q.linked_clause_ids if c not in clauses]
        if q.status == "established" and not q.linked_fact_ids and not q.linked_clause_ids:
            out.append(QAFailure(check="established_without_evidence", detail=f"question {q.question_id} is established with no fact or clause behind it", ref=q.question_id))
    for c in state.contradictions:
        out += [QAFailure(check="citation_unknown_fact", detail=f"contradiction {c.contradiction_id} cites fact {f}, which does not exist", ref=f)
                for f in [*c.fact_ids_a, *c.fact_ids_b] if f not in facts]
    return out


def check_provenance(state: ClaimState) -> list[QAFailure]:
    docs = {d.document_id for d in state.documents}
    out: list[QAFailure] = []
    for f in state.facts:
        if not f.evidence_text.strip():
            out.append(QAFailure(check="fact_provenance", detail=f"fact {f.fact_id} has no verbatim evidence", ref=f.fact_id))
        if state.documents and f.source_document_id not in docs:
            out.append(QAFailure(check="fact_provenance", detail=f"fact {f.fact_id} names document {f.source_document_id}, which is not in the claim", ref=f.fact_id))
    return out


def check_settlement(state: ClaimState) -> list[QAFailure]:
    s = state.settlement
    if s is None:
        return []
    out: list[QAFailure] = []
    if s.status == "cannot_calculate" and s.payable is not None:
        out.append(QAFailure(check="settlement_forecast", detail="a payable amount is set although the settlement cannot be calculated", ref="settlement"))
    if s.status == "calculated":
        if s.payable is None:
            out.append(QAFailure(check="settlement_trace", detail="the settlement is calculated but has no payable", ref="settlement"))
        trace = [t for t in s.steps if t.rule_id != "rounding"]
        for a, b in zip(trace, trace[1:]):
            if a.amount_after != b.amount_before:
                out.append(QAFailure(check="settlement_trace", detail=f"the trace does not chain: {a.rule_id} ends at {a.amount_after} but {b.rule_id} starts at {b.amount_before}", ref=b.rule_id))
        if s.steps and s.payable is not None and s.steps[-1].amount_after != s.payable:
            out.append(QAFailure(check="settlement_trace", detail=f"the last step ends at {s.steps[-1].amount_after} but the payable is {s.payable}", ref="settlement"))
        if s.payable is not None and s.payable < Decimal(0):
            out.append(QAFailure(check="settlement_trace", detail="the payable is negative", ref="settlement"))
    if s.explanation is None:
        out.append(QAFailure(check="settlement_explanation", detail="the settlement has no explanation", ref="settlement"))
    else:
        out += [QAFailure(check="settlement_explanation", detail=p, ref="settlement") for p in verify(s.explanation, s)]
    return out


def check_language(state: ClaimState, cfg: QaConfig) -> list[QAFailure]:
    texts: list[tuple[str, str]] = []
    texts += [(f"coverage {i.issue_id}", i.note or "") for i in state.coverage_issues]
    if state.settlement and state.settlement.explanation:
        texts.append(("settlement explanation", state.settlement.explanation))
    for d in state.drafts:
        texts += [(f"draft {d.draft_id} subject", d.subject), (f"draft {d.draft_id} body", d.body)]
    for c in state.counterfactuals:
        texts += [(f"scenario {c.scenario_id}", f"{c.implication_text} {c.next_step}")]
    for m in state.missing_evidence:
        texts += [(f"request {m.question_id}", " ".join(m.minimum_evidence_request.items))]
    out: list[QAFailure] = []
    for where, text in texts:
        for rx in cfg.forbidden:
            m = rx.search(text)
            if m:
                out.append(QAFailure(check="forbidden_language", detail=f"{where} states an outcome: '{m.group(0)}'", ref=where))
    return out


def check_authority(state: ClaimState) -> list[QAFailure]:
    out: list[QAFailure] = []
    if not state.human_approval_required:
        out.append(QAFailure(check="approval_required", detail="the claim does not require a human decision", ref="human_approval_required"))
    if state.resolution_status != "approved":                         # before any approval nothing may have been carried out or sent
        out += [QAFailure(check="action_before_approval", detail=f"action {a.action_id} ({a.action_type}) is '{a.status}' with no approval", ref=a.action_id)
                for a in state.proposed_actions if a.status == "executed"]
        out += [QAFailure(check="action_before_approval", detail=f"draft {d.draft_id} is '{d.status}' with no approval", ref=d.draft_id) for d in state.drafts if d.status == "sent"]
    return out


def run_checks(state: ClaimState, cfg: QaConfig) -> QAReport:
    failures = [*check_citations(state, cfg), *check_provenance(state), *check_settlement(state), *check_language(state, cfg), *check_authority(state)]
    return QAReport(passed=not failures, failures=failures)


__all__ = ["QaConfig", "check_authority", "check_citations", "check_language", "check_provenance", "check_settlement", "run_checks"]
