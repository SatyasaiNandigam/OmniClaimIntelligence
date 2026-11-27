"""Metrics for the settlement input extraction (T5.5): does the Settlement agent take the right amounts out of the facts, and never a wrong one.

Cases (data/synthetic/settlement_eval/cases.yaml): the money facts of a claim and the labelled assessed amount / non-covered total.
  assessed / non-covered correct   the extracted value equals the label (a missing label, `null`, must come out as no value; a labelled conflict as `unavailable`)
  wrong number                     a value that differs from the label (must be 0: a wrong number is worse than a missing one)
  missing / over-missing           label has a value but none was extracted (safe: the settlement says cannot_calculate)
  model-only                       values the model found that the rules did not (flagged for review)
  refused                          model amount readings refused by the checks
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

import yaml

from claimtower.agents.settlement import SettlementAgent
from claimtower.core.schemas import ClaimHeader, ClaimState, Fact, OpenQuestion

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "data" / "synthetic" / "settlement_eval" / "cases.yaml"


@dataclass
class Case:
    id: str
    split: str
    state: ClaimState
    assessed: Decimal | None
    non_covered: Decimal | None
    conflict: bool


def _d(v):
    return None if v is None else Decimal(str(v))


def load_cases() -> list[Case]:
    out = []
    for c in yaml.safe_load(CASES.read_text(encoding="utf-8"))["cases"]:
        facts = [Fact(fact_id=f"{c['id']}/f{n}", claim_id=c["id"], proposition=f["p"], source_document_id="DOC", page=1, evidence_text=f["e"], evidence_type="neutral",
                      related_question_ids=["Q-AMOUNT"], extraction_confidence=0.9) for n, f in enumerate(c["facts"])]
        q = OpenQuestion(question_id="Q-AMOUNT", text="What is the amount of the loss?", category="amount", status="unverified", linked_fact_ids=[f.fact_id for f in facts])
        state = ClaimState(claim_id=c["id"], claim_header=ClaimHeader(claim_id=c["id"], policy_ref="POL-1001", loss_date=date(2026, 5, 2)), facts=facts, open_questions=[q])
        out.append(Case(c["id"], c["split"], state, _d(c.get("assessed")), _d(c.get("non_covered")), bool(c.get("assessed_conflict"))))
    return out


@dataclass
class Tally:
    cases: int = 0
    assessed_right: int = 0
    non_covered_right: int = 0
    wrong_number: int = 0
    missing: int = 0
    model_only: int = 0
    refused: int = 0
    wrong: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        n = self.cases or 1
        return {"cases": self.cases, "assessed_correct": self.assessed_right / n, "non_covered_correct": self.non_covered_right / n, "wrong_numbers": self.wrong_number,
                "missing_when_labelled": self.missing, "model_only_values": self.model_only, "refused_readings": self.refused, "wrong": self.wrong}


def evaluate(agent: SettlementAgent, cases: list[Case]) -> dict[str, Tally]:
    out: dict[str, Tally] = {}
    for c in cases:
        t = out.setdefault(c.split, Tally())
        t.cases += 1
        inputs, parts, notes = agent.extract(c.state)
        t.refused += sum("refused" in n for n in notes)
        t.model_only += sum(1 for p in parts.values() if p.how == "model")
        for name, label, got in (("assessed", c.assessed, inputs.assessed_covered_loss), ("non_covered", c.non_covered, inputs.non_covered_items_total)):
            ok = (got is None and label is None) or (got is not None and label is not None and got == label)
            if name == "assessed" and c.conflict:
                ok = got is None and "assessed_covered_loss" in inputs.unavailable_inputs
            if ok:
                t.assessed_right += name == "assessed"
                t.non_covered_right += name == "non_covered"
            else:
                t.wrong.append(f"{c.id} {name}: label {label}, got {got}")
                if got is not None and got != label:
                    t.wrong_number += 1
                elif got is None and label is not None:
                    t.missing += 1
    return out
