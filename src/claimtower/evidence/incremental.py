"""Incremental reassessment helpers (T4.7): which open questions does new evidence touch, and what changed between two state versions.

    new = new_or_changed_facts(prev_state, state)                         # facts added, edited, re-verified or removed since prev_state
    ids = affected_questions(prev_state, new, links=links, state=state)    # the questions to re-run; everything else keeps its previous result
    diff = diff_gap(prev_state, state)                                     # what moved: statuses, contradictions, requests, scenarios

No model is called here: this is the bookkeeping that lets the Gap agent re-run only what a new document touches (the rest of the pipeline's output
for the other questions is carried over unchanged), and lets the supervisor show "state v1 -> v2: Q-SUDDEN requires_validation -> established".

A fact touches a question when
  * the fact templates tagged it with that question (`related_question_ids`), or
  * a link says so (the model found a link the tag missed: pass `links`), or
  * the fact was already linked to the question in `prev_state` (it was edited, re-verified or REMOVED: its old questions must be re-judged), or
  * the policy changed (`state.policy_snapshot` differs): the questions that depend on the policy (`policy_questions`) and those whose linked clauses changed.
A fact with no tag and no link touches nothing: it can only matter if a link is found for it, which is why the Gap agent links new facts first.
"""
from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

from claimtower.core.schemas import ClaimState, Fact


def _same_fact(a: Fact, b: Fact) -> bool:
    return (a.proposition, a.evidence_text, a.verification_status, a.source_document_id, sorted(a.related_question_ids)) == \
           (b.proposition, b.evidence_text, b.verification_status, b.source_document_id, sorted(b.related_question_ids))


def new_or_changed_facts(prev_state: ClaimState, state: ClaimState) -> list[Fact]:
    """Facts of `state` that are new or differ from `prev_state`, plus the facts that disappeared (as they were in `prev_state`)."""
    before = {f.fact_id: f for f in prev_state.facts}
    now = {f.fact_id: f for f in state.facts}
    changed = [f for f in state.facts if f.fact_id not in before or not _same_fact(before[f.fact_id], f)]
    return changed + [f for i, f in before.items() if i not in now]


def snapshot_changed(prev_state: ClaimState, state: ClaimState) -> bool:
    """Did the policy in force change? Compared without the retrieval timestamp (re-fetching an identical snapshot is not a change)."""
    a, b = prev_state.policy_snapshot, state.policy_snapshot
    if a is None or b is None:
        return a is not b
    return a.model_copy(update={"retrieved_at": None}) != b.model_copy(update={"retrieved_at": None})


def affected_questions(prev_state: ClaimState, new_facts: Sequence[Fact], *, links: Sequence[object] | None = None, state: ClaimState | None = None,
                       policy_questions: Collection[str] = ()) -> set[str]:
    """The ids of the open questions that `new_facts` (and, when `state` is given, a changed policy) touch. Only questions that exist are returned."""
    known = {q.question_id for q in prev_state.open_questions} | ({q.question_id for q in state.open_questions} if state is not None else set())
    out: set[str] = set()
    for f in new_facts:
        out |= set(f.related_question_ids)
        out |= {q.question_id for q in prev_state.open_questions if f.fact_id in q.linked_fact_ids}
        if links is not None:
            out |= {x.question_id for x in links if getattr(x, "fact_id", None) == f.fact_id}  # type: ignore[attr-defined]
    if state is not None and snapshot_changed(prev_state, state):
        out |= set(policy_questions)
        before = {q.question_id: q.linked_clause_ids for q in prev_state.open_questions}
        out |= {q.question_id for q in state.open_questions if q.question_id in before and q.linked_clause_ids != before[q.question_id]}
    return out & known


# ---------------------------------------------------------------------------------------------------- what changed between two versions
@dataclass(frozen=True)
class StatusChange:
    question_id: str
    before: str | None                            # None: the question is new
    after: str | None                             # None: the question is gone


@dataclass
class GapDiff:
    status_changes: list[StatusChange] = field(default_factory=list)
    contradictions_added: list[str] = field(default_factory=list)
    contradictions_removed: list[str] = field(default_factory=list)
    requests_changed: list[str] = field(default_factory=list)                 # addressee roles whose request is new, gone or different
    scenarios_changed: list[str] = field(default_factory=list)                # question ids whose counterfactual scenarios differ
    questions_unchanged: list[str] = field(default_factory=list)              # questions whose OpenQuestion is identical

    @property
    def empty(self) -> bool:
        return not (self.status_changes or self.contradictions_added or self.contradictions_removed or self.requests_changed or self.scenarios_changed)

    def describe(self) -> str:
        parts = [f"{c.question_id} {c.before or '(new)'} -> {c.after or '(gone)'}" for c in self.status_changes]
        if self.contradictions_added:
            parts.append("contradictions added: " + ", ".join(self.contradictions_added))
        if self.contradictions_removed:
            parts.append("contradictions removed: " + ", ".join(self.contradictions_removed))
        if self.requests_changed:
            parts.append("requests changed: " + ", ".join(self.requests_changed))
        if self.scenarios_changed:
            parts.append("scenarios changed: " + ", ".join(self.scenarios_changed))
        return "; ".join(parts) or "no change"


def diff_gap(prev_state: ClaimState, state: ClaimState) -> GapDiff:
    """The Gap agent's outputs compared between two versions of the state (statuses, contradictions, requests, scenarios)."""
    d = GapDiff()
    before = {q.question_id: q for q in prev_state.open_questions}
    after = {q.question_id: q for q in state.open_questions}
    for qid in [*before, *[q for q in after if q not in before]]:
        b, a = before.get(qid), after.get(qid)
        if b is None or a is None or b.status != a.status:
            d.status_changes.append(StatusChange(qid, b.status if b else None, a.status if a else None))
        if b is not None and a is not None and b == a:
            d.questions_unchanged.append(qid)
    cb, ca = {c.contradiction_id for c in prev_state.contradictions}, {c.contradiction_id for c in state.contradictions}
    d.contradictions_added, d.contradictions_removed = sorted(ca - cb), sorted(cb - ca)
    rb = {m.minimum_evidence_request.addressee_role: m for m in prev_state.missing_evidence}
    ra = {m.minimum_evidence_request.addressee_role: m for m in state.missing_evidence}
    d.requests_changed = sorted(r for r in {*rb, *ra} if rb.get(r) != ra.get(r))
    sb = {q: [s for s in prev_state.counterfactuals if s.question_id == q] for q in {s.question_id for s in prev_state.counterfactuals}}
    sa = {q: [s for s in state.counterfactuals if s.question_id == q] for q in {s.question_id for s in state.counterfactuals}}
    d.scenarios_changed = sorted(q for q in {*sb, *sa} if sb.get(q) != sa.get(q))
    return d
