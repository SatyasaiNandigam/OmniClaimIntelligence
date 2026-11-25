"""Metrics for the fact <-> question linker (T4.2).

Items: `dev` = the five scenario ground-truth files (labelled links = each fact's related_question_ids, direction = its evidence_type: one
direction per fact, a coarser label than the linker's per-link answer); `fresh` = data/synthetic/link_eval/cases.yaml (per-link labels,
imperfect tags, written before any result was seen).
  link precision / recall   over (fact, question) pairs
  direction accuracy        over the pairs that are both predicted and labelled
  critical errors           a labelled-contradicting link predicted supporting (the error that hides a conflict from a person)
`tags` = "given": facts keep the tags the fact templates gave; "none": tags removed, the model must find the links itself.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from claimtower.core.schemas import ClaimHeader, Fact, NotInForce, PolicySnapshot
from claimtower.evidence.linker import FactQuestionLinker
from claimtower.evidence.questions import QuestionRegister

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "data" / "synthetic" / "link_eval" / "cases.yaml"
DIRMAP = {"supporting": "supporting", "contradicting": "contradicting", "neutral": "neutral"}


@dataclass
class Item:
    id: str
    split: str
    facts: list[Fact]
    labels: dict[tuple[str, str], str]
    questions: list


def _facts_items(register: QuestionRegister) -> list[Item]:
    items = []
    for s in ["S-01", "S-02", "S-03", "S-04", "S-05"]:
        gt = json.loads((ROOT / "data" / "ground_truth" / f"{s}.json").read_text(encoding="utf-8"))
        d = ROOT / "data" / "fixtures" / s
        header = ClaimHeader(**json.loads((d / "intake.json").read_text(encoding="utf-8"))["claim_header"])
        pol = json.loads((d / "policy.json").read_text(encoding="utf-8")).get("policy_snapshot")
        snap = PolicySnapshot(**pol) if pol else NotInForce(policy_id="?", loss_date=header.loss_date or "2026-01-01", reason="not in force")
        questions = register.build(header, snap).questions
        facts, labels = [], {}
        for f in gt["facts"]:
            labels |= {(f["fact_id"], q): f["evidence_type"] for q in f["related_question_ids"]}
            facts.append(Fact(**{**f, "evidence_type": "neutral"}))              # the linker must decide the direction itself
        items.append(Item(s, "dev", facts, labels, questions))
    return items


def _fresh_items(register: QuestionRegister) -> list[Item]:
    raw = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    questions = register.build(ClaimHeader(claim_id="X"), PolicySnapshot(policy_id="P", version="1", effective_from="2026-01-01")).questions
    out = []
    for c in raw["cases"]:
        facts, labels = [], {}
        for f in c["facts"]:
            fid = f"{c['id']}/{f['id']}"
            labels |= {(fid, q): d for q, d in f["links"].items()}
            facts.append(Fact(fact_id=fid, claim_id=c["id"], proposition=f["text"], source_document_id="DOC-1", page=1, evidence_text=f["text"],
                              evidence_type="neutral", related_question_ids=f["tags"], extraction_confidence=0.9))
        out.append(Item(c["id"], "fresh", facts, labels, questions))
    return out


def load_items(register: QuestionRegister) -> list[Item]:
    return _facts_items(register) + _fresh_items(register)


@dataclass
class Tally:
    items: int = 0
    labelled: int = 0
    predicted: int = 0
    correct_pairs: int = 0
    direction_right: int = 0
    critical: list[str] = field(default_factory=list)
    wrong_direction: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    refused: int = 0
    dropped: int = 0

    def summary(self) -> dict:
        p = self.correct_pairs / self.predicted if self.predicted else None
        r = self.correct_pairs / self.labelled if self.labelled else None
        return {"items": self.items, "labelled_links": self.labelled, "predicted_links": self.predicted, "correct_links": self.correct_pairs,
                "link_precision": p, "link_recall": r,
                "direction_accuracy": self.direction_right / self.correct_pairs if self.correct_pairs else None,
                "critical_errors": len(self.critical), "refused_judgements": self.refused, "dropped_links": self.dropped,
                "critical": self.critical, "wrong_direction": self.wrong_direction, "missed": self.missed, "extra": self.extra}


def evaluate(linker: FactQuestionLinker, items: list[Item], tags: str = "given") -> dict[str, Tally]:
    out: dict[str, Tally] = {}
    for it in items:
        facts = it.facts if tags == "given" else [f.model_copy(update={"related_question_ids": []}) for f in it.facts]
        r = linker.link(facts, it.questions)
        t = out.setdefault(it.split, Tally())
        t.items += 1
        t.labelled += len(it.labels)
        t.predicted += len(r.links)
        t.refused += len(r.refused)
        t.dropped += len(r.dropped)
        pred = {(x.fact_id, x.question_id): x for x in r.links}
        for pair, x in pred.items():
            if pair in it.labels:
                t.correct_pairs += 1
                want = it.labels[pair]
                if x.direction == want:
                    t.direction_right += 1
                else:
                    line = f"{it.id} {pair[0]}->{pair[1]}: labelled {want}, got {x.direction}"
                    t.wrong_direction.append(line)
                    if want == "contradicting" and x.direction == "supporting":
                        t.critical.append(line)
            else:
                t.extra.append(f"{it.id} {pair[0]}->{pair[1]} ({x.direction})")
        t.missed += [f"{it.id} {a}->{b}" for (a, b) in it.labels if (a, b) not in pred]
    return out
