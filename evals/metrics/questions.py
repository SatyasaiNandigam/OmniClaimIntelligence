"""Metrics for the claim-specific questions a model may add to the register (T4.1).

For every labelled case (data/synthetic/question_eval/cases.yaml) the register is built with the model and compared with the label:
  needed group   hit if an ACCEPTED model question cites a clause of the group (recall of what a careful adjuster would add)
  none_needed    any accepted model question is SPURIOUS (the standard questions already cover the claim)
  extras         accepted questions in a needed case that cite no needed clause: not necessarily wrong, listed for a person to read
Grounding (ids exist, verbatim quote) is enforced by the checks, so it is not a metric; what the checks refused is counted by reason.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from claimtower.core.schemas import ClaimHeader, Fact, PolicyClause, PolicySnapshot
from claimtower.evidence.questions import QuestionRegister, RegisterResult

CASES = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "question_eval" / "cases.yaml"


@dataclass
class Case:
    id: str
    split: str
    header: ClaimHeader
    snapshot: PolicySnapshot
    facts: list[Fact]
    needed: list[set[str]]
    none_needed: bool


def load_cases(path: Path = CASES) -> list[Case]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    lib = raw["clauses"]
    cases = []
    for c in raw["cases"]:
        clauses = [PolicyClause(clause_id=f"CL-{k}", clause_type=lib[k]["type"], heading=lib[k]["heading"], text=lib[k]["text"],
                                document="BASE-HOME-1", page=1) for k in c["clauses"]]
        snap = PolicySnapshot(policy_id="POL-EVAL", version="2025-01", effective_from="2026-01-01", effective_to="2026-12-31", clauses=clauses)
        facts = [Fact(fact_id=f"{c['id']}/{f['id']}", claim_id=c["id"], proposition=f["text"], source_document_id="DOC-1", page=1,
                      evidence_text=f["text"], evidence_type="neutral", extraction_confidence=0.9) for f in c["facts"]]
        cases.append(Case(c["id"], c["split"], ClaimHeader(claim_id=c["id"], **c["header"]), snap, facts,
                          [{f"CL-{k}" for k in g} for g in c.get("needed", [])], bool(c.get("none_needed"))))
    return cases


@dataclass
class Tally:
    cases: int = 0
    needed_groups: int = 0
    hits: int = 0
    none_needed_cases: int = 0
    spurious_cases: int = 0
    accepted: int = 0
    refused: Counter = field(default_factory=Counter)
    misses: list[str] = field(default_factory=list)
    spurious: list[str] = field(default_factory=list)
    extras: list[str] = field(default_factory=list)
    unavailable: int = 0

    def summary(self) -> dict:
        return {"cases": self.cases, "needed_groups": self.needed_groups, "hits": self.hits,
                "recall": self.hits / self.needed_groups if self.needed_groups else None,
                "none_needed_cases": self.none_needed_cases, "spurious_cases": self.spurious_cases,
                "accepted_total": self.accepted, "accepted_per_case": round(self.accepted / self.cases, 2) if self.cases else None,
                "refused_by_reason": dict(self.refused), "unavailable": self.unavailable,
                "misses": self.misses, "spurious": self.spurious, "extras": self.extras}


REASONS = ("repeats standard question", "repeats an existing", "verbatim", "do not exist", "pre-judges", "not phrased as a question", "criteria",
           "below", "allowed category", "cites no clause", "over the limit", "could not be checked", "no usable verdict")


def reason_key(reason: str) -> str:
    return next((k for k in REASONS if k in reason), "other")


def score(case: Case, result: RegisterResult, tally: Tally) -> None:
    tally.cases += 1
    model = list(result.proposed.values())
    tally.accepted += len(model)
    tally.unavailable += any("unavailable" in w for w in result.warnings)
    for r in result.rejected:
        tally.refused[reason_key(r.reason)] += 1
    cited = [set(p.clause_ids) for p in model]
    for group in case.needed:
        tally.needed_groups += 1
        if any(c & group for c in cited):
            tally.hits += 1
        else:
            tally.misses.append(f"{case.id}: nothing about {sorted(group)}")
    if case.none_needed:
        tally.none_needed_cases += 1
        if model:
            tally.spurious_cases += 1
            tally.spurious += [f"{case.id}: {p.text}" for p in model]
    else:
        wanted = set().union(*case.needed) if case.needed else set()
        tally.extras += [f"{case.id}: {p.text}" for p in model if not (set(p.clause_ids) & wanted)]


def evaluate(register: QuestionRegister, cases: list[Case]) -> dict[str, Tally]:
    out: dict[str, Tally] = {}
    for c in cases:
        r = register.build(c.header, c.snapshot, c.facts)
        score(c, r, out.setdefault(c.split, Tally()))
    return out
