"""Metrics for the coverage matrix (T5.1) and the abstention policy (T5.2): does the reading of the facts under the clause match what an adjuster would say, and are the guarantees kept.

Cases (data/synthetic/coverage_eval/cases.yaml): facts for the cause question plus the labelled reading for the cover and the exclusion issue.
  status accuracy       the row's status equals the status the label maps to (supports -> established, against -> unresolved / potentially_relevant, unclear -> abstain)
  established vs label  an `established` row where the label said against / unclear (must be 0: the unacceptable error)
  abstained when unclear  unclear labels that abstain (the right answer when it is genuinely unclear)
  over-abstained        supports/against labels where the matrix abstained (costly but safe)
  lint / problems       wording lint failures and self-check problems; must be 0
  T5.2 abstention       label `ambiguous` (cases with policy: BASE-HOME-2, the "resulting damage" exception): the row must abstain with reason AMBIGUOUS_CLAUSE
                        (missed otherwise); a clear or merely thin-facts label must NOT get that reason (false ambiguity). Abstentions are counted by reason code.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from claimtower.core.schemas import ClaimState, Fact, PolicyClause
from claimtower.coverage.issue_matrix import IssueMatrixBuilder

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "data" / "synthetic" / "coverage_eval" / "cases.yaml"
WANT = {("ISS-COVER-WATER", "supports"): "established", ("ISS-COVER-WATER", "against"): "unresolved", ("ISS-COVER-WATER", "unclear"): "abstain",
        ("ISS-EXC-GRADUAL", "supports"): "established", ("ISS-EXC-GRADUAL", "against"): "potentially_relevant", ("ISS-EXC-GRADUAL", "unclear"): "abstain",
        ("ISS-EXC-GRADUAL", "ambiguous"): "abstain", ("ISS-COVER-WATER", "ambiguous"): "abstain"}


@dataclass
class Case:
    id: str
    split: str
    state: ClaimState
    labels: dict[str, str]


def load_cases() -> list[Case]:
    base = ClaimState(**json.loads((ROOT / "data/fixtures/S-02/claim_state.json").read_text(encoding="utf-8")))
    index = json.loads((ROOT / "data/synthetic/policies/clause_index.json").read_text(encoding="utf-8"))["clauses"]
    out = []
    for c in yaml.safe_load(CASES.read_text(encoding="utf-8"))["cases"]:
        facts = [Fact(fact_id=f"{c['id']}/f{n}", claim_id=c["id"], proposition=t, source_document_id="DOC", page=1, evidence_text=t, evidence_type="neutral",
                      related_question_ids=["Q-SUDDEN", "Q-START"], extraction_confidence=0.9) for n, t in enumerate(c["facts"])]
        ids = [f.fact_id for f in facts]
        qs = [q.model_copy(update={"status": "established", "linked_fact_ids": ids}) if q.question_id in ("Q-SUDDEN", "Q-START") else q for q in base.open_questions]
        update = {"facts": facts, "open_questions": qs}
        if c.get("policy"):                                                  # a different wording: its clauses replace the snapshot's
            clauses = [PolicyClause(clause_id=x["clause_id"], clause_type=x["clause_type"], heading=x.get("heading"), text=x["text"], document=x["doc_id"],
                                    page=x.get("page"), span=tuple(x["span"]) if x.get("span") else None) for x in index if x["doc_id"] == c["policy"]]
            update["policy_snapshot"] = base.policy_snapshot.model_copy(update={"clauses": clauses, "relations": []})
        labels = {k: c[v] for k, v in (("ISS-COVER-WATER", "cover"), ("ISS-EXC-GRADUAL", "exclusion")) if v in c}
        out.append(Case(c["id"], c["split"], base.model_copy(update=update), labels))
    return out


@dataclass
class Tally:
    rows: int = 0
    right: int = 0
    wrong_established: int = 0
    abstained_right: int = 0
    over_abstained: int = 0
    refused: int = 0
    ambiguous_expected: int = 0
    ambiguous_found: int = 0
    false_ambiguous: int = 0
    abstentions: Counter = field(default_factory=Counter)
    lint_failures: int = 0
    problems: int = 0
    wrong: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {"rows": self.rows, "status_accuracy": self.right / self.rows if self.rows else None, "established_against_the_label": self.wrong_established,
                "abstained_when_unclear": self.abstained_right, "over_abstained": self.over_abstained, "refused_readings": self.refused,
                "ambiguity_expected": self.ambiguous_expected, "ambiguity_found": self.ambiguous_found, "false_ambiguity": self.false_ambiguous,
                "abstentions_by_reason": dict(sorted(self.abstentions.items())), "lint_failures": self.lint_failures, "self_check_problems": self.problems, "wrong": self.wrong}


def evaluate(builder: IssueMatrixBuilder, cases: list[Case]) -> dict[str, Tally]:
    out: dict[str, Tally] = {}
    for c in cases:
        t = out.setdefault(c.split, Tally())
        s = c.state
        r = builder.build(s.policy_snapshot, False, s.facts, s.open_questions, s.contradictions)
        t.problems += len(r.problems)
        t.refused += len(r.refused)
        for row in r.issues:
            if row.issue_id not in c.labels:
                continue
            label = c.labels[row.issue_id]
            want = WANT[(row.issue_id, label)]
            t.rows += 1
            reason = r.details[row.issue_id].reason
            t.abstentions[reason] += bool(reason)
            if label == "ambiguous":
                t.ambiguous_expected += 1
                t.ambiguous_found += reason == "AMBIGUOUS_CLAUSE"
            else:
                t.false_ambiguous += reason == "AMBIGUOUS_CLAUSE"
            t.lint_failures += bool(builder.lint(row.note))
            if row.status == want:
                t.right += 1
                t.abstained_right += want == "abstain"
            else:
                t.wrong.append(f"{c.id} {row.issue_id}: label {label} -> expected {want}, got {row.status}")
                t.wrong_established += row.status == "established" and label != "supports"
                t.over_abstained += row.status == "abstain" and label != "unclear"
    return out
