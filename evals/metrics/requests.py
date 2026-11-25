"""Metrics for the minimum-evidence request generator (T4.5).

Cases (data/synthetic/request_eval/cases.yaml): what T4.4 hands over (gaps and criterion verdicts per question) and what a careful adjuster would ask
(`expect`: addressee -> the questions that one request must resolve). Dev = S-01 and S-05 rebuilt from the ground truth + synthetic dev cases;
fresh = unseen cases.
  addressee precision / recall   over the set of people asked
  question coverage              over (addressee, question) pairs: is each expected question resolved by the right person's request, and none extra
  invariants (must be 0)         self-check problems; an item asking for a document type already on file; an internal-only need sent to an external role;
                                 a request addressed to someone not expected
  rule-written items             items the model did not cover (the fallback); a high share means the model's items miss needs
  "sufficient and minimal"       rated by a MODEL against the needs (see the prompt min_evidence_review): a self-assessment, NOT a domain reviewer
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from claimtower.core.config import Pack
from claimtower.core.schemas import ClaimHeader, Contradiction, Fact, OpenQuestion, PolicySnapshot
from claimtower.evidence.linker import Link
from claimtower.evidence.min_evidence import MinimumEvidenceGenerator, RequestResult
from claimtower.evidence.questions import QuestionRegister
from claimtower.evidence.sufficiency import CriterionAssessment, QuestionAssessment, SufficiencyResult

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "data" / "synthetic" / "request_eval" / "cases.yaml"
STATUS = {"contradictory": "requires_validation", "missing": "missing", "insufficient": "unverified"}


@dataclass
class Case:
    id: str
    split: str
    questions: list[OpenQuestion]
    sufficiency: SufficiencyResult
    facts: list[Fact]
    links: list[Link]
    contradictions: list[Contradiction]
    received: tuple[str, ...]
    expect: dict[str, set[str]]


def load_cases(register: QuestionRegister) -> list[Case]:
    raw = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    snap = PolicySnapshot(**__import__("json").loads((ROOT / "data/fixtures/S-01/policy.json").read_text(encoding="utf-8"))["policy_snapshot"])
    questions = register.build(ClaimHeader(claim_id="X"), snap).questions
    qmap = {q.question_id: q for q in questions}
    out = []
    for c in raw["cases"]:
        facts = [Fact(fact_id=f["id"], claim_id=c["id"], proposition=f["text"], source_document_id="DOC-1", page=1, evidence_text=f["text"],
                      evidence_type="neutral", related_question_ids=f["tags"], extraction_confidence=0.9) for f in c["facts"]]
        links = [Link(f["id"], q, "neutral", "labelled", "template") for f in c["facts"] for q in f["tags"]]
        contras = [Contradiction(contradiction_id=f"C-{c['id']}-{n}", question_id=x["question"], fact_ids_a=x["a"], fact_ids_b=x["b"], kind="cause",
                                 severity="high", explanation=x["explanation"]) for n, x in enumerate(c.get("contradictions", []))]
        assessments = []
        for qid, spec in (c.get("questions") or {}).items():
            gap = spec.get("gap")
            crit = tuple(CriterionAssessment(i, qmap[qid].establishing_criteria[i], v, "model", "labelled") for i, v in enumerate(spec["verdicts"]))
            cids = tuple(x.contradiction_id for x in contras if x.question_id == qid) if gap == "contradictory" else ()
            abstained = bool(spec.get("abstained"))
            assessments.append(QuestionAssessment(qid, "unverified" if abstained else STATUS[gap], None if abstained else gap, "labelled", crit, cids, (),
                                                  abstained, "labelled"))
        out.append(Case(c["id"], c["split"], questions, SufficiencyResult(assessments), facts, links, contras, tuple(c["received"]),
                        {role: set(qs) for role, qs in (c.get("expect") or {}).items()}))
    return out


class Rating(BaseModel):
    sufficient: bool
    minimal: bool
    reason: str = ""


class RequestRater:
    """Rates a generated request against its needs with a model: a self-assessment, not a domain reviewer."""

    def __init__(self, llm: Any, pack: Pack) -> None:
        self.llm, self.pack = llm, pack

    def rate(self, req: Any, qmap: dict[str, OpenQuestion]) -> Rating:
        from claimtower.core.llm import load_prompt

        needs = "\n".join(f"- {n.question_id} ({qmap[n.question_id].text}): " + (f"conflict to settle: {n.explanation}" if n.kind == "conflict" else str(n.criterion))
                          for n in req.needs)
        items = "\n".join(f"- {i.text}" for i in req.items)
        return self.llm.structured(Rating, load_prompt("min_evidence_review", self.pack, role_label=req.label, needs_block=needs, items_block=items))


@dataclass
class Tally:
    cases: int = 0
    expected_addressees: int = 0
    predicted_addressees: int = 0
    matched_addressees: int = 0
    expected_pairs: int = 0
    matched_pairs: int = 0
    extra_pairs: int = 0
    problems: int = 0
    duplicates_of_received: int = 0
    internal_leaks: int = 0
    unexpected_requests: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    model_items: int = 0
    rule_items: int = 0
    refused: int = 0
    rated: int = 0
    sufficient: int = 0
    minimal: int = 0
    both: int = 0
    low_rated: list[str] = field(default_factory=list)
    review_wrong: int = 0

    def summary(self) -> dict:
        def pr(a, b):
            return a / b if b else None

        items = self.model_items + self.rule_items
        return {"cases": self.cases, "addressee_precision": pr(self.matched_addressees, self.predicted_addressees),
                "addressee_recall": pr(self.matched_addressees, self.expected_addressees),
                "question_recall": pr(self.matched_pairs, self.expected_pairs), "extra_question_pairs": self.extra_pairs,
                "self_check_problems": self.problems, "duplicates_of_received": self.duplicates_of_received, "internal_needs_sent_externally": self.internal_leaks,
                "rule_written_item_share": pr(self.rule_items, items), "refused_items": self.refused,
                "rated_requests": self.rated, "rated_sufficient": pr(self.sufficient, self.rated), "rated_minimal": pr(self.minimal, self.rated),
                "rated_both": pr(self.both, self.rated), "abstained_wrongly_asked": self.review_wrong,
                "unexpected_requests": self.unexpected_requests, "missed": self.missed, "low_rated": self.low_rated}


def evaluate(generator: MinimumEvidenceGenerator, cases: list[Case], rater: RequestRater | None = None) -> dict[str, Tally]:
    out: dict[str, Tally] = {}
    internal = {r for r, cfg in generator.cfg.roles.items() if cfg.internal}
    for c in cases:
        t = out.setdefault(c.split, Tally())
        t.cases += 1
        r: RequestResult = generator.generate(c.questions, c.sufficiency, c.facts, c.links, c.contradictions, c.received)
        t.problems += len(r.problems)
        t.refused += len(r.refused)
        got = {d.role: set(d.question_ids) for d in r.details}
        t.expected_addressees += len(c.expect)
        t.predicted_addressees += len(got)
        t.matched_addressees += len(set(got) & set(c.expect))
        t.unexpected_requests += [f"{c.id}: {role} {sorted(qs)}" for role, qs in got.items() if role not in c.expect]
        t.missed += [f"{c.id}: {role} {sorted(qs)}" for role, qs in c.expect.items() if role not in got]
        for role, qs in c.expect.items():
            t.expected_pairs += len(qs)
            t.matched_pairs += len(qs & got.get(role, set()))
            t.extra_pairs += len(got.get(role, set()) - qs)
        abstained = {a.question_id for a in c.sufficiency.assessments if a.abstained}
        t.review_wrong += sum(1 for d in r.details if abstained & {n.question_id for n in d.needs})
        qmap = {q.question_id: q for q in c.questions}
        for d in r.details:
            for it in d.items:
                t.model_items += it.source == "model"
                t.rule_items += it.source == "rule"
                t.duplicates_of_received += it.asks_for == "new_document" and it.document_type in c.received and it.document_type != "other"
            t.internal_leaks += d.role not in internal and any(n.criterion and generator._role_for(n.question_id, n.criterion) in internal for n in d.needs)
            if rater is not None:
                rating = rater.rate(d, qmap)
                t.rated += 1
                t.sufficient += rating.sufficient
                t.minimal += rating.minimal
                t.both += rating.sufficient and rating.minimal
                if not (rating.sufficient and rating.minimal):
                    t.low_rated.append(f"{c.id} {d.role}: sufficient={rating.sufficient} minimal={rating.minimal}: {rating.reason}")
    return out
