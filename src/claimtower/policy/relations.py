"""Relations and precedence (T3.2): how the clauses of ONE policy (its wording plus the endorsements attached to it) act on each other.

    graph = RelationExtractor(pack.policy_structure).build("POL-1004", [wording, end01, end02])
    graph.relations      # END-02 modifies END-01's sublimit, END-01's sublimit modifies the escape-of-water cover, ...
    graph.precedence     # the rule the wording states for conflicting endorsements - or 'unstated'
    graph.ambiguities    # clauses where no single target could be named: a human decides, nothing is guessed

Relation types (the contract's): `modifies` (an endorsement changes a clause), `excepts` (an exclusion carves out a cover, or an
exception carves out an exclusion), `defines` (a definition governs a term used by a clause), `conditions` (a condition limits a cover).

How a target is found, in order of strength
  reference   the clause names another endorsement ('the sublimit in Endorsement END-01')
  exception   an exclusion that starts 'However ...' carves out the exclusion just before it
  term        the clause uses the term a definition defines
  overlap     shared words with the target's heading (weighted) and text; the best target must be UNIQUE and score at least `min_score`
  unique      only one clause of an allowed type exists in the policy
No evidence, or a tie = no relation, and the clause is listed as ambiguous. A wrong 'modifies' would apply the wrong limit to a payout.

Precedence is never assumed. It is read from the wording ('the later endorsement applies from its effective date') or from the
endorsement itself ('this endorsement prevails over Endorsement END-01'). If neither says so, `precedence.kind == "unstated"` and the
effective-set engine abstains on conflicts instead of picking a winner.

Scope guard: an endorsement may act on the wording freely, but on ANOTHER endorsement only through an explicit reference to it (an earlier
endorsement cannot modify a later one, and similar-looking endorsements are not linked by word overlap alone).
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

from claimtower.core.config import PolicyStructure, RelationRule
from claimtower.core.schemas.policy import ClauseRelation, PolicyClause, RelationType
from claimtower.policy.decompose import Decomposition, PolicyDecomposer

Basis = Literal["reference", "exception", "term", "overlap", "unique", "precedence"]
PrecedenceKind = Literal["later_effective_date_prevails", "stated_other", "unstated"]
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_STEM_SUFFIXES = ("ing", "ed", "es", "s", "e")


def stem(word: str) -> str:
    """A crude stemmer, enough to match escape / escapes / escaping and pipe / pipes. Not linguistics: a heuristic for word overlap."""
    w = word.lower().strip("'-")
    for suf in _STEM_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


@dataclass(frozen=True)
class Relation:
    source: str
    target: str
    relation: RelationType
    basis: Basis
    evidence: str                       # why: the matched words, the quoted reference, the cue

    def as_contract(self) -> ClauseRelation:
        return ClauseRelation(source_clause_id=self.source, target_clause_id=self.target, relation=self.relation)


@dataclass(frozen=True)
class Ambiguity:
    source: str
    relation: RelationType
    candidates: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class PrecedenceRule:
    kind: PrecedenceKind
    statement: str                      # verbatim text of the clause, or an explicit note that none was found
    clause_id: str | None = None


@dataclass(frozen=True)
class PrevailsOver:
    document: str                       # the endorsement making the statement
    over: str                           # the endorsement it says it prevails over
    evidence: str


@dataclass
class PolicyGraph:
    pool: str
    clauses: dict[str, PolicyClause]
    kinds: dict[str, str]                                        # document id -> "endorsement" | "policy_wording"
    relations: list[Relation] = field(default_factory=list)
    ambiguities: list[Ambiguity] = field(default_factory=list)
    precedence: PrecedenceRule = PrecedenceRule("unstated", "No precedence rule is stated in the policy wording.")
    prevails_over: list[PrevailsOver] = field(default_factory=list)
    stated_dates: dict[str, date] = field(default_factory=dict)  # endorsement id -> date the endorsement states for itself

    def contracts(self) -> list[ClauseRelation]:
        return [r.as_contract() for r in self.relations]

    def documents(self, kind: str) -> list[str]:
        return [d for d, k in self.kinds.items() if k == kind]

    def of_document(self, doc_id: str) -> list[PolicyClause]:
        return [c for c in self.clauses.values() if c.document == doc_id]


class RelationExtractor:
    def __init__(self, structure: PolicyStructure) -> None:
        self.cfg = structure.relations
        self._decomposer = PolicyDecomposer(structure)
        flags = re.IGNORECASE
        self._stop = {stem(w) for w in self.cfg.stopwords}
        self._ref = [re.compile(p, flags) for p in self.cfg.explicit_reference_patterns]
        self._exception = [re.compile(p, flags) for p in self.cfg.exception_cues]
        self._prec_clause = [re.compile(p, flags | re.DOTALL) for p in self.cfg.precedence_clause_cues]
        self._later = [re.compile(p, flags) for p in self.cfg.later_prevails_cues]
        self._prevails = [re.compile(p, flags) for p in self.cfg.prevails_over_patterns]
        self._dates = [re.compile(p, flags) for p in self.cfg.effective_date_cues]

    # ------------------------------------------------------------------------------------------------ words
    def words(self, text: str) -> set[str]:
        return {s for w in _WORD.findall(text) if len(s := stem(w)) >= 3 and s not in self._stop}

    # ------------------------------------------------------------------------------------------------ build
    def build(self, pool: str, documents: Sequence[Decomposition]) -> PolicyGraph:
        clauses: dict[str, PolicyClause] = {}
        order: dict[str, list[str]] = {}
        for d in documents:
            order[d.doc_id] = [c.clause_id for c in d.clauses]
            clauses.update({c.clause_id: c for c in d.clauses})
        graph = PolicyGraph(pool, clauses, {d.doc_id: d.kind for d in documents})
        for d in documents:
            for c in d.clauses:
                self._link(graph, c, order)
        self._read_precedence(graph)
        self._link_precedence_clause(graph)
        self._read_endorsement_statements(graph)
        graph.relations.sort(key=lambda r: (r.source, r.target, r.relation))
        return graph

    # ------------------------------------------------------------------------------------------------ linking
    def _link(self, graph: PolicyGraph, c: PolicyClause, order: dict[str, list[str]]) -> None:
        kind = "endorsement" if graph.kinds[c.document] == "endorsement" else "wording"
        for rule in self.cfg.rules:
            if rule.source_kind != kind or c.clause_type not in rule.source_types:
                continue
            found = self._apply(graph, c, rule, order)
            if found:
                return
        # no rule produced a relation: say so when a rule applied but could not name a target (recorded by _apply as an ambiguity)

    def _apply(self, graph: PolicyGraph, c: PolicyClause, rule: RelationRule, order: dict[str, list[str]]) -> bool:
        if rule.method == "exception":
            if not any(rx.search(c.text) for rx in self._exception):
                return False
            earlier = [i for i in order[c.document] if i != c.clause_id and graph.clauses[i].clause_type in rule.target_types]
            before = [i for i in earlier if order[c.document].index(i) < order[c.document].index(c.clause_id)]
            if not before:
                return False
            graph.relations.append(Relation(c.clause_id, before[-1], "excepts", "exception", f"starts like an exception: '{c.text[:40]}'"))
            return True
        if rule.method == "term":
            return self._defines(graph, c, rule)
        return self._by_overlap(graph, c, rule)

    def _candidates(self, graph: PolicyGraph, c: PolicyClause, rule: RelationRule) -> tuple[list[PolicyClause], str | None]:
        """The clauses this clause may act on, plus the reference that narrowed them (if any)."""
        types = set(rule.target_types)
        if graph.kinds[c.document] != "endorsement":                       # wording: the same document only
            return [x for x in graph.of_document(c.document) if x.clause_type in types and x.clause_id != c.clause_id], None
        named = [m.group("id") for rx in self._ref for m in rx.finditer(c.text)]
        refs = [i for i in named if i in graph.kinds and i != c.document]
        if refs:                                                           # a named endorsement: only its clauses
            ref = refs[0]
            return [x for x in graph.of_document(ref) if x.clause_type in types], ref
        wording = graph.documents("policy_wording")
        return [x for x in graph.clauses.values() if x.document in wording and x.clause_type in types], None

    def _by_overlap(self, graph: PolicyGraph, c: PolicyClause, rule: RelationRule) -> bool:
        cands, ref = self._candidates(graph, c, rule)
        if not cands:
            return False
        if ref and len(cands) == 1:
            graph.relations.append(Relation(c.clause_id, cands[0].clause_id, rule.relation, "reference", f"names Endorsement {ref}"))
            return True
        scored = self._scores(c, cands)
        if len(cands) == 1 and not ref:
            graph.relations.append(Relation(c.clause_id, cands[0].clause_id, rule.relation, "unique",
                                            f"the only {cands[0].clause_type} clause in the policy"))
            return True
        best = max(s for s, _, _ in scored)
        top = [(s, x, w) for s, x, w in scored if s == best]
        if best < self.cfg.min_score:
            graph.ambiguities.append(Ambiguity(c.clause_id, rule.relation, tuple(x.clause_id for x in cands),
                                               "no shared words with any candidate target"))
            return False
        if len(top) > 1:
            graph.ambiguities.append(Ambiguity(c.clause_id, rule.relation, tuple(x.clause_id for _, x, _ in top),
                                               f"{len(top)} candidate targets fit equally well (score {best})"))
            return False
        _, target, shared = top[0]
        basis: Basis = "reference" if ref else "overlap"
        graph.relations.append(Relation(c.clause_id, target.clause_id, rule.relation, basis,
                                        f"names Endorsement {ref}" if ref else f"shared words: {', '.join(sorted(shared))}"))
        return True

    def _scores(self, c: PolicyClause, cands: list[PolicyClause]) -> list[tuple[int, PolicyClause, set[str]]]:
        source = self.words(f"{c.heading or ''} {c.text}")
        out = []
        for x in cands:
            heading = self.words(x.heading or "")
            body = self.words(x.text) - heading
            in_heading, in_body = source & heading, source & body
            out.append((self.cfg.heading_weight * len(in_heading) + self.cfg.text_weight * len(in_body), x, in_heading | in_body))
        return out

    def _defines(self, graph: PolicyGraph, c: PolicyClause, rule: RelationRule) -> bool:
        head, sep, _ = c.text.partition(" means ")
        if not sep or len(head.split()) > 8 or not _WORD.findall(head):
            graph.ambiguities.append(Ambiguity(c.clause_id, "defines", (), "the defined term could not be read"))
            return False
        term = self.words(head)
        if not term:                                                      # only function words: too generic to link anything
            return False
        hit = False
        for x in graph.of_document(c.document):
            if x.clause_id == c.clause_id or x.clause_type not in rule.target_types:
                continue
            if term <= self.words(f"{x.heading or ''} {x.text}"):
                graph.relations.append(Relation(c.clause_id, x.clause_id, "defines", "term", f"uses the defined term '{head.strip()}'"))
                hit = True
        return hit

    # ------------------------------------------------------------------------------------------------ precedence
    def _read_precedence(self, graph: PolicyGraph) -> None:
        for doc in graph.documents("policy_wording"):
            for c in graph.of_document(doc):
                if any(rx.search(c.text) for rx in self._prec_clause):
                    later = any(rx.search(c.text) for rx in self._later)
                    kind: PrecedenceKind = "later_effective_date_prevails" if later else "stated_other"
                    graph.precedence = PrecedenceRule(kind, c.text, c.clause_id)
                    return

    def _link_precedence_clause(self, graph: PolicyGraph) -> None:
        """The precedence clause conditions every endorsement clause that modifies a clause of ANOTHER endorsement (the conflicts
        it governs)."""
        pc = graph.precedence.clause_id
        if pc is None:
            return
        for r in list(graph.relations):
            target = graph.clauses[r.target]
            if r.relation == "modifies" and graph.kinds[graph.clauses[r.source].document] == "endorsement" \
                    and graph.kinds[target.document] == "endorsement":
                graph.relations.append(Relation(pc, r.source, "conditions", "precedence",
                                                f"{r.source} modifies {r.target} (another endorsement): governed by this clause"))

    def _read_endorsement_statements(self, graph: PolicyGraph) -> None:
        for doc in graph.documents("endorsement"):
            stated: date | None = None
            for c in graph.of_document(doc):
                for rx in self._prevails:
                    for m in rx.finditer(c.text):
                        if m.group("id") in graph.kinds and m.group("id") != doc:
                            graph.prevails_over.append(PrevailsOver(doc, m.group("id"), m.group(0)))
                if stated is None:
                    stated = self._stated_date(c.text)
            if stated:
                graph.stated_dates[doc] = stated

    def _stated_date(self, text: str) -> date | None:
        for rx in self._dates:
            if m := rx.search(text):
                try:
                    return datetime.strptime(m.group("date"), "%d %B %Y").date()
                except ValueError:
                    continue
        return None

    # ------------------------------------------------------------------------------------------------ hints for a missing document
    def hint_targets(self, graph: PolicyGraph, title: str) -> list[str]:
        """The wording clauses an endorsement is probably about, judged from its TITLE alone (its text is not available). Used only to say
        which clauses a missing endorsement may affect, so that they are abstained on; it never changes what is in force."""
        decided = self._decomposer.classify(heading=title, section=None, title=title, text="")
        if decided is None:
            return []
        stub = PolicyClause(clause_id="~title", clause_type=decided[0], heading=title, text=title, document="~missing")
        out: set[str] = set()
        for rule in self.cfg.rules:
            if rule.source_kind != "endorsement" or stub.clause_type not in rule.source_types or rule.method != "overlap":
                continue
            wording = set(graph.documents("policy_wording"))
            cands = [x for x in graph.clauses.values() if x.document in wording and x.clause_type in rule.target_types]
            if not cands:
                continue
            scored = self._scores(stub, cands)
            best = max(s for s, _, _ in scored)
            if best >= self.cfg.min_score:
                out |= {x.clause_id for s, x, _ in scored if s == best}
            elif len(cands) == 1:
                out.add(cands[0].clause_id)
            break
        return sorted(out)
