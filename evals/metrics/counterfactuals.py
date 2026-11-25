"""Metrics for the counterfactual engine (T4.6).

Cases (data/synthetic/counterfactual_eval/cases.yaml): the sufficiency state of the cause question, the coverage issues, and the issue ids a
careful adjuster would say each scenario must re-assess (none when the question is established or no open issue is affected).
  scenario completeness   every question with an expectation gets exactly A, B and C; a question with none gets no scenario
  affected-issue match    the scenario's affected issues equal the expected ones (exact set)
  wording lint            every scenario text passes the linter (no wording that predicts or recommends a decision); must be 100%
  rewrite acceptance      share of scenarios whose text is the model's rewrite (the rest keep the rule text because the rewrite was refused)
There is no label for how GOOD the wording is: that is for an adjuster (see export_counterfactual_review).
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from claimtower.core.schemas import ClaimHeader, CoverageIssue, MinimumEvidenceRequest, OpenQuestion, PolicySnapshot
from claimtower.core.schemas.decision import ClauseCitation
from claimtower.core.schemas.evidence import MissingEvidence
from claimtower.evidence.counterfactual import CounterfactualEngine
from claimtower.evidence.questions import QuestionRegister
from claimtower.evidence.sufficiency import CriterionAssessment, QuestionAssessment, SufficiencyResult

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "data" / "synthetic" / "counterfactual_eval" / "cases.yaml"


@dataclass
class Case:
    id: str
    split: str
    questions: list[OpenQuestion]
    sufficiency: SufficiencyResult
    issues: list[CoverageIssue]
    requests: list[MissingEvidence]
    facts: list
    links: list
    expect: dict[str, list[str]]
    clauses: list


def load_cases(register: QuestionRegister) -> list[Case]:
    from claimtower.core.schemas import Fact
    from claimtower.evidence.linker import Link

    raw = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    snap = PolicySnapshot(**json.loads((ROOT / "data/fixtures/S-01/policy.json").read_text(encoding="utf-8"))["policy_snapshot"])
    questions = register.build(ClaimHeader(claim_id="X"), snap).questions
    by_clause = {c.clause_id: c for c in snap.clauses}
    out = []
    for c in raw["cases"]:
        crit = tuple(CriterionAssessment(i, t, "unmet", "model", "x") for i, t in enumerate(next(q for q in questions if q.question_id == "Q-SUDDEN").establishing_criteria))
        st = c["status"]
        a = QuestionAssessment("Q-SUDDEN", "established" if st == "established" else "unverified" if st == "abstained" else "requires_validation",
                               None if st in ("established", "abstained") else "contradictory", "labelled", crit, (), (), st == "abstained", "labelled")
        issues = [CoverageIssue(issue_id=i["id"], issue=i["title"], status=i["status"],
                                clause_citations=[ClauseCitation(clause_id=cid, document=by_clause[cid].document, text=by_clause[cid].text) for cid in i["clauses"]])
                  for i in c["issues"]]
        reqs = [MissingEvidence(question_id="Q-SUDDEN", gap_type="contradictory", resolves_question_ids=["Q-SUDDEN"],
                                minimum_evidence_request=MinimumEvidenceRequest(addressee_role=r, items=["x"])) for r in c["requests"]]
        facts = [Fact(fact_id=f"{c['id']}/f{n}", claim_id=c["id"], proposition=t, source_document_id="D", page=1, evidence_text=t, evidence_type="neutral",
                      related_question_ids=["Q-SUDDEN"], extraction_confidence=0.9) for n, t in enumerate(c["facts"])]
        links = [Link(f.fact_id, "Q-SUDDEN", "neutral", "labelled", "template") for f in facts]
        out.append(Case(c["id"], c["split"], questions, SufficiencyResult([a]), issues, reqs, facts, links, c["expect"], snap.clauses))
    return out


@dataclass
class Tally:
    cases: int = 0
    expected_questions: int = 0
    complete: int = 0
    wrongly_produced: int = 0
    affected_right: int = 0
    scenarios: int = 0
    lint_failures: int = 0
    model_text: int = 0
    refused: Counter = field(default_factory=Counter)
    problems: int = 0
    wrong: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        def pr(a, b):
            return a / b if b else None

        return {"cases": self.cases, "questions_with_scenarios_expected": self.expected_questions, "complete_a_b_c": pr(self.complete, self.expected_questions),
                "scenarios_where_none_expected": self.wrongly_produced, "affected_issues_exact": pr(self.affected_right, self.expected_questions),
                "scenarios": self.scenarios, "lint_failures": self.lint_failures, "self_check_problems": self.problems,
                "model_text_share": pr(self.model_text, self.scenarios), "refused_rewrites": dict(self.refused), "wrong": self.wrong}


def reason_key(reason: str) -> str:
    for k in ("predicts", "not conditional", "longer", "ids", "does not mention", "keep the effect", "not a scenario"):
        if k in reason:
            return k
    return "other"


def evaluate(engine: CounterfactualEngine, cases: list[Case]) -> dict[str, Tally]:
    out: dict[str, Tally] = {}
    for c in cases:
        t = out.setdefault(c.split, Tally())
        t.cases += 1
        r = engine.generate(c.questions, c.sufficiency, c.issues, c.clauses, c.requests, c.facts, c.links)
        t.problems += len(r.problems)
        for x in r.refused:
            t.refused[reason_key(x.reason)] += 1
        for s in r.scenarios:
            t.scenarios += 1
            t.lint_failures += bool(engine.lint_implication(s.implication_text) or engine.lint(s.next_step))
            t.model_text += r.source[(s.question_id, s.scenario_id)] == "model"
        for qid, want in c.expect.items():
            t.expected_questions += 1
            got = r.for_question(qid)
            t.complete += [s.scenario_id for s in got] == ["A", "B", "C"]
            ok = bool(got) and all(set(s.affected_issue_ids) == set(want) for s in got)
            t.affected_right += ok
            if not ok:
                t.wrong.append(f"{c.id} {qid}: expected {want}, got {[s.affected_issue_ids for s in got][:1]}")
        if not c.expect and r.scenarios:
            t.wrongly_produced += len(r.scenarios)
            t.wrong.append(f"{c.id}: produced {len(r.scenarios)} scenario(s) where none were expected")
    return out
