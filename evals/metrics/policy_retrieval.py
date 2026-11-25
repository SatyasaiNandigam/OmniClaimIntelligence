"""Retrieval metrics (T3.4): does a question bring back the clause that answers it - with the endorsements that modify it?

For every query of `data/synthetic/policy_eval/retrieval_set.yaml` the clauses IN FORCE on the snapshot's loss date are searched, and
  * recall@k, hit@1 / hit@3, MRR  measure whether the relevant clauses come back (negative queries are excluded);
  * modifier completeness          measures whether, for a retrieved clause, every labelled modifying endorsement is in its `modified_by`
                                   AND is returned in the result list (the failure mode of plain vector search);
  * no-answer rate                 measures whether a question nothing answers returns nothing.
A relevant clause that is not in the snapshot at all (e.g. the decomposer left it unclassified) cannot be retrieved; it is counted as
`unreachable` and left out of recall, so a decomposition gap is not blamed on retrieval.
Splits: `dev` (corpus snapshots; parameters may be tuned on them) and `fresh` (unseen policies; never tuned on).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import yaml

from claimtower.core.schemas import EndorsementRef, PolicyVersionResponse
from claimtower.core.schemas.policy import PolicySnapshot
from claimtower.policy.decompose import PolicyDecomposer
from claimtower.policy.relations import RelationExtractor
from claimtower.policy.retrieval import ClauseRetriever
from claimtower.policy.snapshot import PolicyLibrary, SnapshotBuilder
from evals.metrics.policy_clauses import FRESH
from evals.metrics.policy_relations import all_pools, decompose_pool
from evals.metrics.policy_snapshot import MockPas

SET = FRESH / "retrieval_set.yaml"
FIXED_NOW = datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)
K = 5


class FakePas:
    """A tiny policy system for the unseen policies: one version, the endorsements listed, always in force."""

    def __init__(self, specs: dict[str, dict]) -> None:
        self._specs = specs

    def get_policy_version(self, policy_no: str, as_of: date) -> PolicyVersionResponse:
        s = self._specs[policy_no]
        return PolicyVersionResponse(
            policy_no=policy_no, as_of=as_of, status="in_force", version="v1", effective_from=date(2026, 1, 1), effective_to=date(2026, 12, 31),
            base_wording_id=s["base"], mock=True, source_system="PAS-MOCK", response_ref=f"PAS-MOCK/{policy_no}/{as_of}",
            endorsements=[EndorsementRef(endorsement_id=e["id"], effective_from=date.fromisoformat(e["from"])) for e in s["endorsements"]])


class Pas:
    def __init__(self, fake: dict[str, dict]) -> None:
        self._mock, self._fake = MockPas(), FakePas(fake)
        self._fake_ids = set(fake)

    def get_policy_version(self, policy_no: str, as_of: date) -> PolicyVersionResponse:
        return (self._fake if policy_no in self._fake_ids else self._mock).get_policy_version(policy_no, as_of)


@dataclass
class World:
    snapshots: dict[str, PolicySnapshot]
    to_label: dict[str, str]                          # snapshot clause id -> labelled id
    to_id: dict[str, str]                             # labelled id -> snapshot clause id
    queries: list[dict]


def build_world(decomposer: PolicyDecomposer, extractor: RelationExtractor) -> World:
    spec = yaml.safe_load(SET.read_text(encoding="utf-8"))
    docs, to_label = [], {}
    for pool in all_pools():
        outs, label_of = decompose_pool(decomposer, pool)
        docs += [o for o in outs if all(o.doc_id != d.doc_id for d in docs)]
        to_label.update(label_of)
    fake = {v["fake"]["policy"]: v["fake"] for v in spec["snapshots"].values() if "fake" in v}
    builder = SnapshotBuilder(Pas(fake), PolicyLibrary(docs), extractor, clock=lambda: FIXED_NOW)
    snaps = {}
    for sid, v in spec["snapshots"].items():
        policy, as_of = (v["fake"]["policy"], v["fake"]["as_of"]) if "fake" in v else (v["policy"], v["as_of"])
        snap = builder.get_snapshot(policy, date.fromisoformat(as_of))
        assert isinstance(snap, PolicySnapshot), f"snapshot {sid} is not in force"
        snaps[sid] = snap
    return World(snaps, to_label, {v: k for k, v in to_label.items()}, spec["queries"])


@dataclass
class Tally:
    queries: int = 0
    answerable: int = 0
    unreachable: int = 0
    hit1: int = 0
    hit3: int = 0
    hit5: int = 0
    recall_sum: float = 0.0
    rr_sum: float = 0.0
    negatives: int = 0
    negatives_empty: int = 0
    mod_checked: int = 0
    mod_ok: int = 0
    mod_not_retrieved: int = 0
    degraded: int = 0
    misses: list[str] = field(default_factory=list)
    mod_failures: list[str] = field(default_factory=list)
    false_answers: list[str] = field(default_factory=list)

    def rate(self, num: float, den: float) -> float | None:
        return None if not den else num / den

    def summary(self) -> dict:
        r = lambda x: None if x is None else round(x, 4)  # noqa: E731
        return {"queries": self.queries, "answerable": self.answerable, "unreachable": self.unreachable,
                "hit@1": r(self.rate(self.hit1, self.answerable)), "hit@3": r(self.rate(self.hit3, self.answerable)),
                "hit@5": r(self.rate(self.hit5, self.answerable)), "recall@5": r(self.rate(self.recall_sum, self.answerable)),
                "mrr": r(self.rate(self.rr_sum, self.answerable)), "no_answer_rate": r(self.rate(self.negatives_empty, self.negatives)),
                "modifier_completeness": r(self.rate(self.mod_ok, self.mod_checked)), "modifier_checks": self.mod_checked,
                "clause_not_retrieved_for_modifier_check": self.mod_not_retrieved, "degraded_queries": self.degraded,
                "misses": self.misses, "modifier_failures": self.mod_failures, "false_answers": self.false_answers}


def evaluate(retriever: ClauseRetriever, world: World, k: int = K) -> dict[str, Tally]:
    out: dict[str, Tally] = {}
    for q in world.queries:
        t = out.setdefault(q["split"], Tally())
        t.queries += 1
        snap = world.snapshots[q["snapshot"]]
        in_snapshot = {c.clause_id for c in snap.clauses}
        res = retriever.search(q["query"], snap, k)
        t.degraded += bool(res.warnings)
        got = [world.to_label.get(c.clause_id, c.clause_id) for c in res.hits]
        if q["kind"] == "negative":
            t.negatives += 1
            if not res.hits:
                t.negatives_empty += 1
            else:
                t.false_answers.append(f"{q['id']} '{q['query']}' -> {got[:3]}")
            continue
        relevant = [r for r in q["relevant"] if world.to_id.get(r) in in_snapshot]
        t.unreachable += len(q["relevant"]) - len(relevant)
        if relevant:
            t.answerable += 1
            ranks = [got.index(r) + 1 for r in relevant if r in got]
            t.recall_sum += len(ranks) / len(relevant)
            if ranks:
                t.rr_sum += 1 / min(ranks)
                t.hit1 += min(ranks) <= 1
                t.hit3 += min(ranks) <= 3
                t.hit5 += min(ranks) <= 5
            else:
                t.misses.append(f"{q['id']} '{q['query']}' -> {got[:3]} (wanted {relevant})")
        shown = {c.clause_id: c for c in res.citations}
        for clause, mods in (q.get("modifiers") or {}).items():
            cid = world.to_id.get(clause)
            hit = next((c for c in res.hits if c.clause_id == cid), None)
            if hit is None:
                t.mod_not_retrieved += 1
                continue
            for m in mods:
                t.mod_checked += 1
                mid = world.to_id.get(m)
                if mid in hit.modified_by and mid in shown:
                    t.mod_ok += 1
                else:
                    t.mod_failures.append(f"{q['id']}: {clause} lacks modifier {m}")
    return out
