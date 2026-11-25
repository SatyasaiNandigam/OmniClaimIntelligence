"""The effective clause set (T3.2): which clauses of a policy are in force on a given date, and what is NOT known.

    eff = EffectiveSetBuilder(extractor).build(graph, "BASE-HOME-1", listings, as_of=date(2026, 1, 20))
    eff.clauses              # the wording plus the endorsement clauses in force on that date, minus clauses a later endorsement replaced
    eff.excluded             # what was left out and WHY (not yet effective, expired, superseded, wording unavailable): cite these too
    eff.modified_by          # clause id -> the endorsement clauses in force that modify it (for citations)
    eff.missing_documents    # endorsements the policy lists but whose wording we do not have
    eff.abstain_clause_ids   # clauses whose effective wording cannot be settled: the coverage step must abstain on them
    eff.conflicts            # why: unstated precedence, equal dates, a stated date that disagrees with the policy system

The endorsement LISTINGS (which endorsements, effective from / to) come from the policy administration system (PAS) or the schedule, never
from the endorsement text: the text is only compared with them, and a disagreement is reported, not resolved silently.

Rules, in order
  1. An endorsement applies on the loss date only if `effective_from <= as_of` and (`effective_to` is empty or `as_of <= effective_to`).
  2. If an endorsement in force modifies a clause of another endorsement in force, one replaces the other ONLY if the precedence is stated:
     by the wording ('the later endorsement applies from its effective date') together with a later effective date, or by the endorsement
     itself ('this endorsement prevails over Endorsement X'). Otherwise both stay in the set, the conflict is recorded, and the wording
     clause
     they both bear on is abstained on: the engine never picks a winner on its own.
  3. A listed endorsement with no wording available is reported as missing. If its title is known, the wording clauses it probably affects
     (from the title alone) are abstained on. It never changes the clauses in force.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from claimtower.core.schemas.policy import PolicyClause
from claimtower.policy.relations import PolicyGraph, RelationExtractor

ExclusionReason = Literal["not_yet_effective", "expired", "superseded", "wording_unavailable"]
ConflictKind = Literal["precedence_unstated", "same_effective_date", "date_mismatch", "modified_clause_not_in_force", "references_later"]


@dataclass(frozen=True)
class EndorsementListing:
    endorsement_id: str
    effective_from: date
    effective_to: date | None = None
    title: str | None = None            # as the policy system names it: lets us say what a missing endorsement may affect

    def in_force(self, as_of: date) -> bool:
        return self.effective_from <= as_of and (self.effective_to is None or as_of <= self.effective_to)


@dataclass(frozen=True)
class Excluded:
    document: str
    reason: ExclusionReason
    detail: str
    clause_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Conflict:
    kind: ConflictKind
    documents: tuple[str, ...]
    clause_ids: tuple[str, ...]
    detail: str


@dataclass
class EffectiveSet:
    as_of: date
    base_document: str
    clauses: list[PolicyClause] = field(default_factory=list)
    excluded: list[Excluded] = field(default_factory=list)
    modified_by: dict[str, list[str]] = field(default_factory=dict)
    missing_documents: list[str] = field(default_factory=list)
    abstain_clause_ids: list[str] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    precedence_rules: list[str] = field(default_factory=list)

    @property
    def clause_ids(self) -> list[str]:
        return [c.clause_id for c in self.clauses]


class EffectiveSetBuilder:
    def __init__(self, extractor: RelationExtractor) -> None:
        self._extractor = extractor

    def build(self, graph: PolicyGraph, base_document: str, listings: list[EndorsementListing], as_of: date) -> EffectiveSet:
        out = EffectiveSet(as_of, base_document)
        out.precedence_rules.append(self._statement(graph))
        active = self._active(graph, listings, as_of, out)
        superseded = self._resolve(graph, active, out)
        in_force = [c for c in graph.of_document(base_document)]
        for doc in sorted(active):
            in_force += [c for c in graph.of_document(doc) if c.clause_id not in superseded]
        out.clauses = in_force
        effective = {c.clause_id for c in in_force}
        out.modified_by = self._modified_by(graph, effective, superseded)
        out.abstain_clause_ids = sorted(set(out.abstain_clause_ids))
        out.missing_documents = sorted(set(out.missing_documents))
        return out

    # ------------------------------------------------------------------------------------------------ which listed endorsements apply
    def _active(self, graph: PolicyGraph, listings: list[EndorsementListing], as_of: date,
                out: EffectiveSet) -> dict[str, EndorsementListing]:
        active: dict[str, EndorsementListing] = {}
        for lst in sorted(listings, key=lambda x: (x.effective_from, x.endorsement_id)):
            doc = lst.endorsement_id
            if not lst.in_force(as_of):
                late = as_of < lst.effective_from
                out.excluded.append(Excluded(doc, "not_yet_effective" if late else "expired",
                                             f"effective from {lst.effective_from}" if late else f"ended {lst.effective_to}",
                                             tuple(c.clause_id for c in graph.of_document(doc))))
                out.precedence_rules.append(
                    f"Endorsement {doc} is listed but "
                    + (f"not effective until {lst.effective_from}" if late else f"ended on {lst.effective_to}")
                    + f"; it is not applied to a loss on {as_of}.")
                continue
            if graph.kinds.get(doc) != "endorsement" or not graph.of_document(doc):
                out.missing_documents.append(doc)
                out.excluded.append(Excluded(doc, "wording_unavailable", "listed in the policy schedule but its wording is not available"))
                hint = self._extractor.hint_targets(graph, lst.title) if lst.title else []
                out.abstain_clause_ids += hint
                out.precedence_rules.append(
                    f"Endorsement {doc} is listed in the schedule but its wording is not available"
                    + (f"; it may affect {', '.join(hint)}." if hint else "; which clauses it affects is unknown."))
                continue
            active[doc] = lst
            stated = graph.stated_dates.get(doc)
            if stated and stated != lst.effective_from:
                targets = self._wording_targets(graph, doc)
                out.conflicts.append(Conflict("date_mismatch", (doc,), tuple(targets),
                                              f"the endorsement states {stated} but the policy system says {lst.effective_from}; "
                                              "the policy system is used"))
                out.abstain_clause_ids += targets
        return active

    # ------------------------------------------------------------------------------------------------ endorsement against endorsement
    def _resolve(self, graph: PolicyGraph, active: dict[str, EndorsementListing], out: EffectiveSet) -> set[str]:
        pairs: dict[tuple[str, str], list[str]] = {}
        for r in graph.relations:
            if r.relation != "modifies":
                continue
            e, x = graph.clauses[r.source].document, graph.clauses[r.target].document
            if e != x and graph.kinds[e] == "endorsement" and graph.kinds[x] == "endorsement":
                pairs.setdefault((e, x), []).append(r.target)
        superseded: set[str] = set()
        rule_later = graph.precedence.kind == "later_effective_date_prevails"
        for (e, x), targets in sorted(pairs.items()):
            if e not in active:
                continue                                                  # the modifying endorsement is not in force: nothing to resolve
            if x not in active:
                out.conflicts.append(Conflict("modified_clause_not_in_force", (e, x), tuple(targets),
                                              f"{e} modifies a clause of {x}, which is not in force on {out.as_of}"))
                continue
            ed, xd = active[e].effective_from, active[x].effective_from
            explicit = any(p.document == e and p.over == x for p in graph.prevails_over)
            heading = {graph.clauses[t].heading for t in targets}
            if ed < xd:
                out.conflicts.append(Conflict("references_later", (e, x), tuple(targets),
                                              f"{e} (from {ed}) modifies {x}, which starts later ({xd})"))
                out.abstain_clause_ids += self._wording_targets(graph, x)
            elif explicit or (rule_later and ed > xd):
                superseded |= set(targets)
                how = (f"{e} states that it prevails over {x}" if explicit
                       else "the policy wording says the later endorsement applies from its effective date")
                out.precedence_rules.append(f"{e} (from {ed}) replaces {x} for {', '.join(sorted(h for h in heading if h))}: {how}.")
            elif ed == xd:
                out.conflicts.append(Conflict("same_effective_date", (e, x), tuple(targets),
                                              f"{e} and {x} both take effect on {ed} and neither states that it prevails"))
                out.abstain_clause_ids += self._wording_targets(graph, x)
            else:
                out.conflicts.append(Conflict("precedence_unstated", (e, x), tuple(targets),
                                              f"{e} modifies {x} but neither the wording nor the endorsement says which prevails"))
                out.abstain_clause_ids += self._wording_targets(graph, x)
        for t in sorted(superseded):
            doc = graph.clauses[t].document
            out.excluded.append(Excluded(doc, "superseded", "replaced by a later endorsement", (t,)))
        return superseded

    def _wording_targets(self, graph: PolicyGraph, doc: str) -> list[str]:
        """The wording clauses an endorsement acts on (what is uncertain when the endorsement is), following a chain of endorsements that
        modify one another until it reaches the wording."""
        found: set[str] = set()
        seen: set[str] = set()
        todo = [doc]
        while todo:
            d = todo.pop()
            if d in seen:
                continue
            seen.add(d)
            for r in graph.relations:
                if graph.clauses[r.source].document != d:
                    continue
                target = graph.clauses[r.target]
                if graph.kinds[target.document] == "policy_wording":
                    found.add(r.target)
                else:
                    todo.append(target.document)
        return sorted(found)

    # ------------------------------------------------------------------------------------------------ citations
    def _modified_by(self, graph: PolicyGraph, effective: set[str], superseded: set[str]) -> dict[str, list[str]]:
        """clause -> endorsement clauses in force that act on it. A superseded modifier is replaced by whatever now modifies IT, so the
        wording clause still points at the endorsement that governs today (END-02 for the sublimit once it replaces END-01)."""
        by_target: dict[str, list[str]] = {}
        for r in graph.relations:
            if graph.kinds[graph.clauses[r.source].document] == "endorsement":
                by_target.setdefault(r.target, []).append(r.source)

        def governing(clause_id: str, seen: frozenset[str]) -> list[str]:
            found: list[str] = []
            for src in by_target.get(clause_id, []):
                if src in seen:
                    continue
                if src in effective:
                    found.append(src)
                elif src in superseded:
                    found += governing(src, seen | {src})
            return found

        result = {}
        for cid in sorted(effective):
            mods = sorted(set(governing(cid, frozenset({cid}))))
            if mods:
                result[cid] = mods
        return result

    @staticmethod
    def _statement(graph: PolicyGraph) -> str:
        p = graph.precedence
        if p.clause_id is None:
            return p.statement
        doc = graph.clauses[p.clause_id].document
        return f"{doc} {graph.clauses[p.clause_id].heading}: {p.statement}"
