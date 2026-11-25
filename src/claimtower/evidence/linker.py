"""Fact <-> question linker (T4.2): which facts bear on which open questions, and which way each one points.

    linker = FactQuestionLinker(pack.question_templates.fact_links, judge=LLMLinkJudge(llm, pack))      # judge optional
    result = linker.link(facts, questions)          # facts: from the Evidence agent (tagged by the fact templates); questions: T4.1 register
    result.links                                    # list[Link] with direction, rationale, source and a verbatim quote
    apply_to_questions(questions, result.links)     # -> questions with linked_fact_ids filled in

Two sources, one link table:
  * TAGGED links come from the fact templates (T2.4: `Fact.related_question_ids`). They are the floor: a tagged link is never lost silently.
    Without a model they are kept with direction `neutral` ("tagged by the fact template; direction not judged").
  * MODEL judgements (llm_linking.py) set the DIRECTION of a link (supporting = consistent with the claim as presented, contradicting = points
    away from it, neutral), may ADD a link the tags missed, and may mark a tagged link UNRELATED. A dropped link is listed in `result.dropped`
    with the model's rationale, so nothing disappears without a trace.
Every judgement is checked (real ids, verbatim quote, rationale, confidence), see llm_linking.py; an unusable one leaves the tagged link neutral.
A model that is down never breaks the linker: the tags are kept and a warning says directions were not judged.

The direction is relative to the claim as presented, as in the labelled data: a plumber finding no corrosion SUPPORTS "was it sudden?".
Detecting that two facts conflict with each other is T4.3; deciding whether the evidence is enough is T4.4.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from claimtower.core.config import FactLinkConfig
from claimtower.core.schemas import Fact, OpenQuestion

Direction = Literal["supporting", "contradicting", "neutral"]
DIRECTIONS: tuple[str, ...] = ("supporting", "contradicting", "neutral")
Verdict = Literal["supporting", "contradicting", "neutral", "unrelated"]


@dataclass(frozen=True)
class Link:
    fact_id: str
    question_id: str
    direction: Direction
    rationale: str
    source: Literal["template", "template+model", "model"]
    confidence: float | None = None            # the model's (capped) confidence; None when no model judged the pair
    quote: str = ""                            # verbatim words of the fact that decided the direction


@dataclass(frozen=True)
class LinkVerdict:
    """A model judgement that passed the checks."""

    fact_id: str
    question_id: str
    direction: Verdict
    rationale: str
    confidence: float
    quote: str = ""


@dataclass(frozen=True)
class RefusedJudgement:
    fact_id: str
    question_id: str
    reason: str


@dataclass
class JudgeOutcome:
    verdicts: list[LinkVerdict] = field(default_factory=list)
    refused: list[RefusedJudgement] = field(default_factory=list)
    unavailable: bool = False
    detail: str = ""


class LinkJudge(Protocol):
    def judge(self, facts: Sequence[Fact], questions: Sequence[OpenQuestion], tagged: dict[str, list[str]]) -> JudgeOutcome: ...


@dataclass
class LinkResult:
    links: list[Link]
    dropped: list[Link] = field(default_factory=list)                 # tagged links the model judged unrelated, kept for audit
    refused: list[RefusedJudgement] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def for_question(self, question_id: str) -> list[Link]:
        return [x for x in self.links if x.question_id == question_id]

    def for_fact(self, fact_id: str) -> list[Link]:
        return [x for x in self.links if x.fact_id == fact_id]

    def pairs(self) -> set[tuple[str, str]]:
        return {(x.fact_id, x.question_id) for x in self.links}


class FactQuestionLinker:
    def __init__(self, cfg: FactLinkConfig, judge: LinkJudge | None = None) -> None:
        self.cfg = cfg
        self.judge = judge if cfg.enabled else None

    def link(self, facts: Sequence[Fact], questions: Sequence[OpenQuestion]) -> LinkResult:
        usable = [f for f in facts if f.verification_status != "rejected"]          # a fact a person rejected is not evidence
        qids = [q.question_id for q in questions]
        tagged = {f.fact_id: [q for q in f.related_question_ids if q in set(qids)] for f in usable}
        result = LinkResult(links=[])
        outcome = self.judge.judge(usable, questions, tagged) if self.judge is not None and usable and questions else JudgeOutcome()
        result.refused = list(outcome.refused)
        verdicts = {(v.fact_id, v.question_id): v for v in outcome.verdicts}
        if outcome.unavailable:
            result.warnings.append(f"link directions were not judged: the model was unavailable ({outcome.detail or 'no detail'}); "
                                   "tagged links are kept as neutral")
        order = {q: i for i, q in enumerate(qids)}
        for f in usable:
            pairs = list(tagged[f.fact_id])
            pairs += [q for (fid, q), v in verdicts.items() if fid == f.fact_id and q not in pairs and v.direction != "unrelated"]
            for q in sorted(pairs, key=order.__getitem__):
                v = verdicts.get((f.fact_id, q))
                is_tagged = q in tagged[f.fact_id]
                if v is None:
                    result.links.append(Link(f.fact_id, q, "neutral", "tagged by the fact template; direction not judged", "template"))
                elif v.direction == "unrelated":
                    result.dropped.append(Link(f.fact_id, q, "neutral", v.rationale, "template+model", v.confidence))
                else:
                    result.links.append(Link(f.fact_id, q, v.direction, v.rationale, "template+model" if is_tagged else "model", v.confidence, v.quote))
        return result


def apply_to_questions(questions: Sequence[OpenQuestion], links: Sequence[Link]) -> list[OpenQuestion]:
    """The questions with `linked_fact_ids` set from the links (facts of the questions that were linked before are kept, in order)."""
    out = []
    for q in questions:
        ids = list(q.linked_fact_ids)
        ids += [x.fact_id for x in links if x.question_id == q.question_id and x.fact_id not in ids]
        out.append(q.model_copy(update={"linked_fact_ids": ids}))
    return out


def fact_direction(links: Sequence[Link], fact_id: str) -> Direction:
    """One direction for a fact that bears on several questions: contradicting wins (it is the one a person must not miss), then supporting."""
    dirs = {x.direction for x in links if x.fact_id == fact_id}
    return "contradicting" if "contradicting" in dirs else "supporting" if "supporting" in dirs else "neutral"
