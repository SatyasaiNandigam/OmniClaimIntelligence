"""Minimum-evidence request generator (T4.5): instead of "upload more documents", the smallest, most specific request that would resolve
what is still open, grouped so each person is asked once.

    generator = MinimumEvidenceGenerator(pack.question_templates, writer=LLMRequestWriter(llm, pack))      # writer optional
    result = generator.generate(questions, sufficiency, facts, links, contradictions, received_types={"fnol", "plumber_report"})
    result.requests          # list[MissingEvidence] (contract): addressee, items, the questions it must answer, the criteria it maps to
    result.review_questions  # abstained questions: they go to a person, not to an external request

How a request is built:
  1. NEEDS come from the sufficiency result (T4.4): every unmet or partly met criterion of a question that is not established (for a question
     whose criteria are alternatives, only when none is met), plus one conflict need for a question blocked by a contradiction. An abstained
     question makes no external request: it is listed for review.
  2. WHO is asked is decided by the pack (`requests.criteria_roles`, `conflict_roles`): needs for the same person are grouped into ONE request.
     An internal role (policy administration) is asked by the system; a claimant is never asked for policy wording. A THIRD PARTY (plumber,
     surveyor, contractor) is never contacted by the system: its request has delivery `via_claimant_or_adjuster` (drafted, passed on by a person).
  3. The WRITER (model) words the items from the needs and the facts already known. Each item is CHECKED: it serves at least one need of this
     request, is not empty or too long, does not pre-judge the decision, is not a repeat of another item, and asks for no document type that is
     already on file (it must ask for the missing information instead).
  4. COVERAGE: a need no accepted item serves gets a rule-written item, so no unresolved criterion is dropped (and a model that is down still
     yields a valid request). `self_check` re-verifies every request: each item maps to a need and a pack criterion, each need is covered.
"""
from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

from claimtower.core.config import QuestionTemplates, RequestRole
from claimtower.core.schemas import Contradiction, Fact, MinimumEvidenceRequest, OpenQuestion
from claimtower.core.schemas.evidence import MissingEvidence
from claimtower.evidence.questions import overlap
from claimtower.evidence.sufficiency import QuestionAssessment, SufficiencyResult

GAP_ORDER: tuple[str, ...] = ("contradictory", "missing", "insufficient")
ASKS_FOR: tuple[str, ...] = ("new_document", "statement", "information")


@dataclass(frozen=True)
class Need:
    index: int                                    # position within its addressee's request
    question_id: str
    kind: Literal["criterion", "conflict"]
    gap_type: str
    criterion: str | None = None
    verdict: str = ""                             # unmet | partly | conflict
    rationale: str = ""
    contradiction_id: str | None = None
    explanation: str = ""                         # of the contradiction, for a conflict need


@dataclass(frozen=True)
class DraftItem:
    """What the writer proposes, before the checks."""

    need_ids: tuple[int, ...]
    text: str
    asks_for: str = "information"
    document_type: str = ""


@dataclass(frozen=True)
class RequestItem:
    text: str
    need_ids: tuple[int, ...]
    criteria: tuple[str, ...]
    asks_for: str
    document_type: str
    source: Literal["model", "rule"]


@dataclass(frozen=True)
class RefusedItem:
    role: str
    text: str
    reason: str


@dataclass
class WriterContext:
    role: str
    role_cfg: RequestRole
    needs: Sequence[Need]
    questions: Mapping[str, OpenQuestion]
    known: Mapping[int, list[str]]                # need index -> facts already known on its question (text)
    received_types: frozenset[str]
    max_items: int
    max_item_chars: int
    feedback: Sequence[str] = ()                  # on a repair round: why the previous items were refused
    uncovered: Sequence[int] = ()                 # on a repair round: the needs still without an item


@dataclass
class WriterOutcome:
    drafts: list[DraftItem] = field(default_factory=list)
    unavailable: bool = False
    detail: str = ""


class RequestWriter(Protocol):
    def write(self, ctx: WriterContext) -> WriterOutcome: ...


@dataclass
class GeneratedRequest:
    role: str
    label: str
    internal: bool
    delivery: str                                 # direct | via_claimant_or_adjuster | internal: how it may reach the addressee
    needs: list[Need]
    items: list[RequestItem]
    question_ids: list[str]                       # the questions this request resolves, in register order

    def to_contract(self, gap_of: Mapping[str, str]) -> MissingEvidence:
        first = min(self.needs, key=lambda n: (GAP_ORDER.index(n.gap_type), self.question_ids.index(n.question_id)))
        criteria = list(dict.fromkeys(c for i in self.items for c in i.criteria))
        return MissingEvidence(
            question_id=first.question_id, gap_type=first.gap_type,  # type: ignore[arg-type]
            minimum_evidence_request=MinimumEvidenceRequest(addressee_role=self.role, items=[i.text for i in self.items],
                                                            must_answer_sub_questions=list(self.question_ids), maps_to_criteria=criteria),
            resolves_question_ids=list(self.question_ids))


@dataclass
class RequestResult:
    requests: list[MissingEvidence]
    details: list[GeneratedRequest] = field(default_factory=list)
    review_questions: list[str] = field(default_factory=list)
    refused: list[RefusedItem] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)                      # self-check failures: must be empty
    warnings: list[str] = field(default_factory=list)

    def for_role(self, role: str) -> GeneratedRequest | None:
        return next((d for d in self.details if d.role == role), None)


class MinimumEvidenceGenerator:
    def __init__(self, templates: QuestionTemplates, writer: RequestWriter | None = None) -> None:
        if templates.requests is None:
            raise ValueError("the pack has no `requests` section in question_templates.yaml")
        self.templates, self.cfg = templates, templates.requests
        self.writer = writer if self.cfg.enabled else None
        self._forbidden = [re.compile(p, re.IGNORECASE) for p in self.cfg.forbidden_patterns]
        self._crit_role = {(c.question, c.criterion): c.role for c in self.cfg.criteria_roles}
        self._mode = {q.id: q.criteria_mode for q in templates.questions}

    # ------------------------------------------------------------------ needs
    def _role_for(self, question_id: str, criterion: str) -> str:
        return self._crit_role.get((question_id, criterion), self.cfg.default_role)

    def needs_for(self, a: QuestionAssessment, contradictions: Sequence[Contradiction]) -> list[tuple[str, Need]]:
        """(role, need) pairs for one assessed question; empty when established or abstained."""
        if a.status == "established" or a.abstained or a.gap_type is None:
            return []
        out: list[tuple[str, Need]] = []
        gap = a.gap_type
        if a.contradiction_ids:
            c = next((x for x in contradictions if x.contradiction_id in a.contradiction_ids), None)
            role = self.cfg.conflict_roles.get(a.question_id, self.cfg.default_role)
            out.append((role, Need(0, a.question_id, "conflict", gap, verdict="conflict",
                                   contradiction_id=c.contradiction_id if c else None, explanation=c.explanation if c else "")))
        wanted = [c for c in a.criteria if c.verdict in ("unmet", "partly")]
        if self._mode.get(a.question_id, "all") == "any" and any(c.verdict == "met" for c in a.criteria):
            wanted = []                                                    # alternatives: one is met, no more evidence is needed
        for c in wanted:
            out.append((self._role_for(a.question_id, c.criterion), Need(0, a.question_id, "criterion", gap, c.criterion, c.verdict, c.rationale)))
        return out

    # ------------------------------------------------------------------ generate
    def plan(self, sufficiency: SufficiencyResult, contradictions: Sequence[Contradiction]) -> dict[str, list[Need]]:
        """The needs of every addressee (role -> needs), without writing anything: what an incremental run compares to see which requests changed."""
        by_role: dict[str, list[Need]] = {}
        for a in sufficiency.assessments:
            for role, need in self.needs_for(a, contradictions):
                by_role.setdefault(role, []).append(need)
        return by_role

    @staticmethod
    def fingerprint(needs: Sequence[Need]) -> str:
        """A stable summary of what an addressee is still asked for: when it is unchanged, the earlier request is still the right one."""
        return "|".join(sorted(f"{n.question_id}/{n.kind}/{n.criterion or ''}/{n.verdict}/{n.gap_type}/{n.contradiction_id or ''}" for n in needs))

    def generate(self, questions: Sequence[OpenQuestion], sufficiency: SufficiencyResult, facts: Sequence[Fact], links: Sequence[object],
                 contradictions: Sequence[Contradiction], received_types: Collection[str] = (),
                 documents: Mapping[str, str] | None = None, only_roles: Collection[str] | None = None) -> RequestResult:
        """`only_roles`: write the requests of these addressees only (an incremental run keeps the others as they were)."""
        qmap = {q.question_id: q for q in questions}
        order = {q.question_id: i for i, q in enumerate(questions)}
        by_role: dict[str, list[Need]] = {}
        result = RequestResult(requests=[])
        for a in sufficiency.assessments:
            if a.abstained:
                result.review_questions.append(a.question_id)
            for role, need in self.needs_for(a, contradictions):
                by_role.setdefault(role, []).append(need)
        fact_by_id = {f.fact_id: f for f in facts if f.verification_status != "rejected"}
        received = frozenset(received_types)
        roles = sorted((r for r in by_role if only_roles is None or r in only_roles),
                       key=lambda r: (min(GAP_ORDER.index(n.gap_type) for n in by_role[r]), min(order[n.question_id] for n in by_role[r])))
        for role in roles:
            needs = [Need(i, n.question_id, n.kind, n.gap_type, n.criterion, n.verdict, n.rationale, n.contradiction_id, n.explanation)
                     for i, n in enumerate(sorted(by_role[role], key=lambda n: (order[n.question_id], n.kind != "conflict")))]
            role_cfg = self.cfg.roles[role]
            known = {n.index: self._known(n.question_id, fact_by_id, links, documents) for n in needs}
            ctx = WriterContext(role, role_cfg, needs, qmap, known, received, self.cfg.max_items_per_request, self.cfg.max_item_chars)
            items = self._items(ctx, qmap, result)
            qids = sorted({n.question_id for n in needs}, key=order.__getitem__)
            gen = GeneratedRequest(role, role_cfg.label, role_cfg.internal, role_cfg.delivery, needs, items, qids)
            result.details.append(gen)
            result.requests.append(gen.to_contract({}))
            result.problems += self.self_check(gen, qmap)
        return result

    def _known(self, question_id: str, facts: Mapping[str, Fact], links: Sequence[object], documents: Mapping[str, str] | None) -> list[str]:
        out = []
        for x in links:
            if getattr(x, "question_id", None) != question_id or getattr(x, "fact_id", None) not in facts:
                continue
            f = facts[x.fact_id]  # type: ignore[attr-defined]
            src = f" ({documents[f.source_document_id]})" if documents and f.source_document_id in documents else ""
            out.append(" ".join(f.proposition.split())[:self.cfg.max_fact_chars] + src)
        return out[:self.cfg.max_known_facts]

    # ------------------------------------------------------------------ items
    def _items(self, ctx: WriterContext, qmap: Mapping[str, OpenQuestion], result: RequestResult) -> list[RequestItem]:
        items: list[RequestItem] = []
        covered: set[int] = set()
        if self.writer is not None:
            feedback: list[str] = []
            for attempt in range(self.cfg.repair_rounds + 1):
                round_ctx = ctx if attempt == 0 else replace(ctx, feedback=feedback, uncovered=[n.index for n in ctx.needs if n.index not in covered])
                outcome = self.writer.write(round_ctx)
                if outcome.unavailable:
                    result.warnings.append(f"{ctx.role}: the model was unavailable ({outcome.detail or 'no detail'}); the request uses rule-written items")
                    break
                feedback = []
                for d in outcome.drafts:
                    why = self._problem(d, ctx, [i.text for i in items])
                    if why:
                        result.refused.append(RefusedItem(ctx.role, d.text, why))
                        feedback.append(f"refused ({why}): {d.text}")
                    elif len(items) >= ctx.max_items:
                        result.refused.append(RefusedItem(ctx.role, d.text, f"over the limit of {ctx.max_items} items"))
                    else:
                        ids = tuple(dict.fromkeys(d.need_ids))
                        items.append(RequestItem(" ".join(d.text.split()), ids, self._criteria(ids, ctx), d.asks_for, d.document_type, "model"))
                        covered |= set(ids)
                if not feedback or all(n.index in covered for n in ctx.needs):
                    break                                                  # nothing refused, or every need already has an item: no repair needed
        for n in ctx.needs:                                                # coverage: nothing unresolved is dropped
            if n.index not in covered:
                items.append(RequestItem(self._fallback_text(n, qmap), (n.index,), self._criteria((n.index,), ctx), "information", "", "rule"))
        return items

    def _problem(self, d: DraftItem, ctx: WriterContext, existing: Sequence[str]) -> str | None:
        text = " ".join(d.text.split())
        if len(text) < 8:
            return "the item is empty or too short"
        if len(text) > ctx.max_item_chars:
            return f"the item is longer than {ctx.max_item_chars} characters"
        if not d.need_ids or any(i < 0 or i >= len(ctx.needs) for i in d.need_ids):
            return "the item does not serve a need of this request"
        if any(rx.search(text) for rx in self._forbidden):
            return "the item pre-judges the decision instead of asking for information"
        if d.asks_for not in ASKS_FOR:
            return f"'{d.asks_for}' is not an allowed kind of ask"
        if d.asks_for == "new_document":
            if d.document_type not in ctx.role_cfg.document_types:
                return f"'{d.document_type}' is not a document this addressee can supply"
            if d.document_type in ctx.received_types and d.document_type != "other":
                return f"a {d.document_type} is already on file: ask for the missing information, not the document"
        if any(overlap(text, t) >= self.cfg.duplicate_overlap and overlap(t, text) >= self.cfg.duplicate_overlap for t in existing):
            return "the item repeats another item of the request"
        return None

    def _criteria(self, need_ids: Sequence[int], ctx: WriterContext) -> tuple[str, ...]:
        """The pack criteria an item answers. A conflict need maps to the question's criteria this addressee supplies (else all of them)."""
        out: list[str] = []
        for i in need_ids:
            n = ctx.needs[i]
            if n.kind == "criterion" and n.criterion:
                out.append(n.criterion)
            else:
                allc = self.templates_criteria(n.question_id)
                mine = [c for c in allc if self._role_for(n.question_id, c) == ctx.role]
                out += mine or allc
        return tuple(dict.fromkeys(out))

    def templates_criteria(self, question_id: str) -> list[str]:
        return next((q.establishing_criteria for q in self.templates.questions if q.id == question_id), [])

    @staticmethod
    def _fallback_text(n: Need, qmap: Mapping[str, OpenQuestion]) -> str:
        if n.kind == "conflict":
            return f"Clarify the conflict between statements on file: {n.explanation or qmap[n.question_id].text}"
        return f"Please provide: {n.criterion} (so that we can answer: {qmap[n.question_id].text})"

    # ------------------------------------------------------------------ self-check
    def self_check(self, req: GeneratedRequest, qmap: Mapping[str, OpenQuestion]) -> list[str]:
        """Does this request actually resolve what it claims to? Every item maps to a need and to a pack criterion; every need is covered;
        an internal request is never addressed to the claimant."""
        problems: list[str] = []
        valid = {c for q in self.templates.questions for c in q.establishing_criteria}
        served = {i for it in req.items for i in it.need_ids}
        for it in req.items:
            if not it.need_ids or not it.criteria:
                problems.append(f"{req.role}: item {it.text!r} serves no need")
            elif any(c not in valid for c in it.criteria):
                problems.append(f"{req.role}: item {it.text!r} maps to a criterion that is not in the pack")
        for n in req.needs:
            if n.index not in served:
                problems.append(f"{req.role}: need {n.question_id}/{n.criterion or 'conflict'} is not covered by any item")
        if req.internal and req.role == self.cfg.default_role:
            problems.append(f"{req.role}: an internal request is addressed to the default (external) role")
        return problems
