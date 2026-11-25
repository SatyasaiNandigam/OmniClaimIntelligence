"""Metrics for the sufficiency classifier (T4.4).

Dev = the five scenario ground-truth files (question status and gap type per question) plus the dev cases of
data/synthetic/sufficiency_eval/cases.yaml; fresh = the unseen cases of that file. Per scored question:
  status accuracy      predicted status == labelled status
  gap-type accuracy    predicted gap (missing / contradictory / insufficient / none) == labelled gap
  over-claims          predicted `established` where the label is anything else: the dangerous error (a gap hidden from the adjuster)
  abstained            the classifier declined to decide (counted as a wrong status when a status was labelled; reported separately)
Gap labels of the ground-truth files come from `missing_evidence`: the entry for the question itself, else the entry whose request resolves it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from claimtower.core.schemas import ClaimHeader, Contradiction, Fact, NotInForce, OpenQuestion, PolicySnapshot
from claimtower.evidence.linker import Link
from claimtower.evidence.questions import QuestionRegister
from claimtower.evidence.sufficiency import SufficiencyClassifier

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "data" / "synthetic" / "sufficiency_eval" / "cases.yaml"


@dataclass
class Case:
    id: str
    split: str
    questions: list[OpenQuestion]
    facts: list[Fact]
    links: list[Link]
    contradictions: list[Contradiction]
    snapshot: PolicySnapshot | NotInForce | None
    expect: dict[str, tuple[str, str]]          # question id -> (status, gap)
    documents: dict[str, str] = field(default_factory=dict)       # document id -> type, as the Evidence agent records it


def _scenario_cases(register: QuestionRegister) -> list[Case]:
    out = []
    for s in ["S-01", "S-02", "S-03", "S-04", "S-05"]:
        gt = json.loads((ROOT / "data" / "ground_truth" / f"{s}.json").read_text(encoding="utf-8"))
        d = ROOT / "data" / "fixtures" / s
        header = ClaimHeader(**json.loads((d / "intake.json").read_text(encoding="utf-8"))["claim_header"])
        pol = json.loads((d / "policy.json").read_text(encoding="utf-8")).get("policy_snapshot")
        snap = PolicySnapshot(**pol) if pol else NotInForce(policy_id="?", loss_date=header.loss_date or "2026-01-01", reason="not in force")
        gaps: dict[str, str] = {}
        for m in gt["missing_evidence"]:
            for q in m["resolves_question_ids"]:
                gaps.setdefault(q, m["gap_type"])
        for m in gt["missing_evidence"]:
            gaps[m["question_id"]] = m["gap_type"]
        facts = [Fact(**{**f, "evidence_type": "neutral"}) for f in gt["facts"]]
        links = [Link(f["fact_id"], q, f["evidence_type"], "labelled", "template") for f in gt["facts"] for q in f["related_question_ids"]]
        contradictions = [Contradiction(**{k: c[k] for k in ("contradiction_id", "question_id", "fact_ids_a", "fact_ids_b", "kind", "severity", "explanation")})
                          for c in gt["contradictions"]]
        expect = {q["question_id"]: (q["status"], gaps.get(q["question_id"], "none") if q["status"] != "established" else "none") for q in gt["questions"]}
        docs = {d_["document_id"]: d_["detected_type"] for d_ in json.loads((d / "evidence.json").read_text(encoding="utf-8"))["documents"]}
        out.append(Case(s, "dev", register.build(header, snap).questions, facts, links, contradictions, snap, expect, docs))
    return out


def _synthetic_cases(register: QuestionRegister) -> list[Case]:
    raw = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    fixture = PolicySnapshot(**json.loads((ROOT / "data/fixtures/S-01/policy.json").read_text(encoding="utf-8"))["policy_snapshot"])
    no_sublimit = PolicySnapshot(**json.loads((ROOT / "data/fixtures/S-05/policy.json").read_text(encoding="utf-8"))["policy_snapshot"])
    out = []
    for c in raw["cases"]:
        kind = c.get("policy", "in_force")
        snap = {"in_force": fixture, "no_sublimit": no_sublimit, "none": None,
                "not_in_force": NotInForce(policy_id="POL-1", loss_date="2026-04-12", reason="lapsed")}[kind]
        header = ClaimHeader(claim_id=c["id"])
        fid = lambda i, cid=c["id"]: f"{cid}/{i}"  # noqa: E731
        facts = [Fact(fact_id=fid(f["id"]), claim_id=c["id"], proposition=f["text"], source_document_id="DOC-1", page=1, evidence_text=f["text"],
                      evidence_type="neutral", related_question_ids=f["tags"], extraction_confidence=0.9) for f in c["facts"]]
        links = [Link(fid(f["id"]), q, f["dir"], "labelled", "template") for f in c["facts"] for q in f["tags"]]
        contradictions = [Contradiction(contradiction_id=f"C-{c['id']}-{n}", question_id=x["question"], fact_ids_a=[fid(i) for i in x["a"]],
                                        fact_ids_b=[fid(i) for i in x["b"]], kind="cause", severity=x["severity"], explanation="labelled")
                          for n, x in enumerate(c.get("contradictions", []))]
        expect = {q: (v["status"], v["gap"]) for q, v in c["expect"].items()}
        out.append(Case(c["id"], c["split"], register.build(header, snap).questions, facts, links, contradictions, snap, expect))
    return out


def load_cases(register: QuestionRegister) -> list[Case]:
    return _scenario_cases(register) + _synthetic_cases(register)


@dataclass
class Tally:
    cases: int = 0
    scored: int = 0
    status_right: int = 0
    gap_right: int = 0
    abstained: int = 0
    overclaims: list[str] = field(default_factory=list)
    wrong: list[str] = field(default_factory=list)
    refused: int = 0

    def summary(self) -> dict:
        return {"cases": self.cases, "questions_scored": self.scored,
                "status_accuracy": self.status_right / self.scored if self.scored else None,
                "gap_accuracy": self.gap_right / self.scored if self.scored else None,
                "abstained": self.abstained, "over_claims": len(self.overclaims), "refused_judgements": self.refused,
                "overclaims": self.overclaims, "wrong": self.wrong}


def evaluate(classifier: SufficiencyClassifier, cases: list[Case], modes: dict[str, str] | None = None) -> dict[str, Tally]:
    out: dict[str, Tally] = {}
    for c in cases:
        t = out.setdefault(c.split, Tally())
        t.cases += 1
        r = classifier.classify(c.questions, c.facts, c.links, c.contradictions, c.snapshot, c.documents, modes)
        t.refused += len(r.refused)
        for qid, (want_status, want_gap) in c.expect.items():
            if qid not in {q.question_id for q in c.questions}:
                continue
            a = r.get(qid)
            got_gap = a.gap_type or "none"
            t.scored += 1
            t.abstained += a.abstained
            t.status_right += a.status == want_status
            t.gap_right += got_gap == want_gap
            if a.status == "established" and want_status != "established":
                t.overclaims.append(f"{c.id} {qid}: labelled {want_status}, got established")
            if a.status != want_status or got_gap != want_gap:
                t.wrong.append(f"{c.id} {qid}: labelled {want_status}/{want_gap}, got {a.status}/{got_gap} ({a.row_id}"
                               + (", abstained: " + "; ".join(f"[{x.index}] {x.rationale}" for x in a.criteria if x.verdict == "unclear") if a.abstained else "") + ")")
    return out
