"""Policy snapshot as of the date of loss (T3.3): the contract in force on that day, or an explicit `NotInForce`.

    builder = SnapshotBuilder(pas, library, RelationExtractor(pack.policy_structure))
    result = builder.get_snapshot("POL-1004", date(2026, 1, 20))     # PolicySnapshot | NotInForce  (the PolicyService contract)
    outcome = builder.build("POL-1004", date(2026, 1, 20))           # the same, plus the effective set (missing documents, abstentions)

What the snapshot is made of
  * the policy system (PAS) answers three questions for the loss date: is the policy covered (status), which policy version applies, and
    which endorsements are listed in its schedule. Nothing in the documents can override that: a status of `lapsed`, `cancelled` or
    `not_found` ends here with `NotInForce(reason)`; a `reinstated` policy is covered again from its reinstatement day.
  * the wording of that version, plus the endorsements the PAS lists, is decomposed (T3.1), linked (T3.2) and filtered by effective date
    (T3.2): only clauses in force on the loss date reach the snapshot, and what was left out is described in `precedence_rules`.
  * the PAS response reference and timestamp are stored on the snapshot, so a later reader can see which answer it rests on.

Never guessed
  * No wording for the base document: `PolicyDocumentMissing` (the contract cannot be rebuilt without it; a human must supply it).
  * An endorsement listed in the schedule with no wording: the snapshot is built WITHOUT it and says so in `precedence_rules`;
    `Outcome.effective` lists it as missing and names the clauses to abstain on (when its title is known).
  * Unstated precedence between two endorsements: both stay in the snapshot, the conflict is written into `precedence_rules`, and the clause
    is listed for abstention. The snapshot never silently picks one.

`PolicySnapshot.version` is the PAS POLICY version in force on the loss date (v1, v2 ...), because a mid-term change is what makes
two losses in one policy year see different contracts. `clause_id`s are those the decomposer produced from the document's own numbering.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Protocol

from claimtower.core.schemas import PolicyVersionResponse
from claimtower.core.schemas.policy import ClauseRelation, Endorsement, NotInForce, PolicyClause, PolicySnapshot
from claimtower.policy.decompose import Decomposition, Unclassified
from claimtower.policy.effective import EffectiveSet, EffectiveSetBuilder, EndorsementListing
from claimtower.policy.relations import PolicyGraph, RelationExtractor
from claimtower.policy.store import PolicyStore

COVERED = ("in_force", "reinstated")


class PolicyDocumentMissing(Exception):
    """The base wording of the policy version is not in the document library: a snapshot cannot be built."""


class PasLike(Protocol):
    """What the snapshot needs from the policy administration system (the PASClient contract in integrations/base.py)."""

    def get_policy_version(self, policy_no: str, as_of: date) -> PolicyVersionResponse: ...


class PolicyLibrary:
    """The policy documents we have, already decomposed, by document id."""

    def __init__(self, documents: Iterable[Decomposition]) -> None:
        self._docs = {d.doc_id: d for d in documents}

    @classmethod
    def from_store(cls, store: PolicyStore) -> "PolicyLibrary":
        return cls(Decomposition(d["doc_id"], d["title"], d["kind"], clauses=(cs := store.clauses(d["doc_id"])),
                                 info={c.clause_id: store.info(c.clause_id) for c in cs},
                                 unclassified=list(store.unclassified(d["doc_id"])))
                   for d in store.documents())

    def get(self, doc_id: str) -> Decomposition | None:
        return self._docs.get(doc_id)

    def without(self, *doc_ids: str) -> "PolicyLibrary":
        """The same library minus some documents: what a claim that lacks them would see."""
        return PolicyLibrary(d for i, d in self._docs.items() if i not in doc_ids)

    def ids(self) -> list[str]:
        return sorted(self._docs)


@dataclass
class Outcome:
    result: PolicySnapshot | NotInForce
    response: PolicyVersionResponse
    graph: PolicyGraph | None = None
    effective: EffectiveSet | None = None
    model_typed: list[str] = field(default_factory=list)                  # clauses in the snapshot whose TYPE came from the optional model
    unclassified: list[Unclassified] = field(default_factory=list)        # clauses of the documents in force that are NOT in the snapshot

    @property
    def in_force(self) -> bool:
        return isinstance(self.result, PolicySnapshot)


class SnapshotBuilder:
    def __init__(self, pas: PasLike, library: PolicyLibrary, extractor: RelationExtractor,
                 clock: Callable[[], datetime] | None = None) -> None:
        self._pas, self._library, self._extractor = pas, library, extractor
        self._engine = EffectiveSetBuilder(extractor)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def set_library(self, library: PolicyLibrary) -> None:
        """Use another library from now on (a policy document was added or the folder was re-read): the next snapshot sees it."""
        self._library = library

    # ------------------------------------------------------------------------------------------------ the PolicyService contract
    def get_snapshot(self, policy_no: str, loss_date: date,
                     endorsement_titles: Mapping[str, str] | None = None) -> PolicySnapshot | NotInForce:
        return self.build(policy_no, loss_date, endorsement_titles).result

    def build(self, policy_no: str, loss_date: date, endorsement_titles: Mapping[str, str] | None = None) -> Outcome:
        """`endorsement_titles`: titles as the policy schedule names them. Only used to say which clauses an endorsement WITHOUT wording
        may affect; they never change what is in force."""
        resp = self._pas.get_policy_version(policy_no, loss_date)
        if resp.status == "not_found":
            reason = f"PAS status 'not_found' for {policy_no} on {loss_date}: no policy by that exact number, or no cover on that date"
            return Outcome(NotInForce(policy_id=policy_no, loss_date=loss_date, reason=reason), resp)
        if resp.status not in COVERED:
            since = f" since {resp.status_effective_from}" if resp.status_effective_from else ""
            return Outcome(NotInForce(policy_id=policy_no, loss_date=loss_date,
                                      reason=f"PAS status '{resp.status}'{since}; loss on {loss_date}"), resp)
        if resp.version is None or resp.base_wording_id is None or resp.effective_from is None:
            raise PolicyDocumentMissing(f"the policy system returned status '{resp.status}' for {policy_no} on {loss_date} "
                                        "but no policy version or wording reference")
        if not (resp.effective_from <= loss_date and (resp.effective_to is None or loss_date <= resp.effective_to)):
            reason = f"policy version {resp.version} ran {resp.effective_from} to {resp.effective_to}; no version applies on {loss_date}"
            return Outcome(NotInForce(policy_id=policy_no, loss_date=loss_date, reason=reason), resp)

        base = self._library.get(resp.base_wording_id)
        if base is None or not base.clauses:
            raise PolicyDocumentMissing(f"the wording {resp.base_wording_id} of {policy_no} {resp.version} is not available")
        titles = dict(endorsement_titles or {})
        docs = [base]
        for e in resp.endorsements:
            d = self._library.get(e.endorsement_id)
            if d is not None and d.clauses:
                docs.append(d)
                titles.setdefault(e.endorsement_id, d.title or e.endorsement_id)
        graph = self._extractor.build(f"{policy_no}:{resp.version}", docs)
        listings = [EndorsementListing(e.endorsement_id, e.effective_from, e.effective_to, titles.get(e.endorsement_id))
                    for e in resp.endorsements]
        eff = self._engine.build(graph, base.doc_id, listings, loss_date)
        in_force_docs = {c.document for c in eff.clauses}
        in_force_ids = set(eff.clause_ids)
        model_typed = sorted(i for d in docs for i, info in d.info.items() if info.type_source == "llm" and i in in_force_ids)
        unclassified = [u for d in docs if d.doc_id in in_force_docs for u in d.unclassified]
        notes = self._review_notes(docs, model_typed, unclassified)
        return Outcome(self._snapshot(policy_no, resp, graph, eff, titles, notes), resp, graph, eff, model_typed, unclassified)

    # ------------------------------------------------------------------------------------------------ assembly
    @staticmethod
    def _review_notes(docs: list[Decomposition], model_typed: list[str], unclassified: list[Unclassified]) -> list[str]:
        """What a person must know before relying on the snapshot: clauses whose type is a model's reading, and clauses that are missing."""
        notes = []
        if model_typed:
            conf = {i: info.confidence for d in docs for i, info in d.info.items() if info.confidence is not None}
            notes.append("REVIEW: the type of these clauses was decided by a language model, not by rules: "
                         + ", ".join(f"{i} ({conf[i]:.2f})" if i in conf else i for i in model_typed) + ".")
        if unclassified:
            notes.append("NOT IN THIS SNAPSHOT: no type could be decided for "
                         + "; ".join(f"{u.clause_id} '{u.heading}'" for u in unclassified)
                         + ". A person must classify them; they may change what is covered or paid.")
        return notes

    def _snapshot(self, policy_no: str, resp: PolicyVersionResponse, graph: PolicyGraph, eff: EffectiveSet,
                  titles: Mapping[str, str], notes: list[str] | None = None) -> PolicySnapshot:
        in_force: set[str] = set(eff.clause_ids)
        by_doc: dict[str, list[PolicyClause]] = {}
        for c in eff.clauses:
            by_doc.setdefault(c.document, []).append(c)
        endorsements = [Endorsement(endorsement_id=e.endorsement_id, title=titles.get(e.endorsement_id, e.endorsement_id),
                                    effective_from=e.effective_from, effective_to=e.effective_to,
                                    clause_ids=[c.clause_id for c in by_doc[e.endorsement_id]])
                        for e in resp.endorsements if by_doc.get(e.endorsement_id) and graph.kinds.get(e.endorsement_id) == "endorsement"]
        relations: list[ClauseRelation] = [r.as_contract() for r in graph.relations if r.source in in_force and r.target in in_force]
        # A replaced endorsement leaves a gap: END-02 modifies END-01's sublimit, which modified the water cover. Once END-01's clause is
        # no longer in force the snapshot would lose the link from the cover to END-02, so the governing modifier is linked directly.
        present = {(r.source_clause_id, r.target_clause_id) for r in relations}
        for target, mods in sorted(eff.modified_by.items()):
            relations += [ClauseRelation(source_clause_id=m, target_clause_id=target, relation="modifies")
                          for m in mods if (m, target) not in present]
        return PolicySnapshot(
            policy_id=policy_no, version=resp.version or "", effective_from=resp.effective_from, effective_to=resp.effective_to,
            pas_status=resp.status if resp.status in COVERED else "in_force", clauses=list(eff.clauses), endorsements=endorsements,
            relations=relations, precedence_rules=[*eff.precedence_rules, *self._conflict_notes(eff), *(notes or [])],
            source_system_response_ref=resp.response_ref, retrieved_at=resp.retrieved_at or self._clock(), mock=resp.mock)

    @staticmethod
    def _conflict_notes(eff: EffectiveSet) -> list[str]:
        notes = [f"UNRESOLVED ({c.kind}): {c.detail}." for c in eff.conflicts]
        if eff.abstain_clause_ids:
            notes.append("Clauses to abstain on until this is resolved: " + ", ".join(eff.abstain_clause_ids) + ".")
        return notes
