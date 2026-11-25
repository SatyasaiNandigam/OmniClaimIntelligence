"""Policy decomposition (T3.1): a policy document's pages -> typed, cited clauses.

    result = PolicyDecomposer(pack.policy_structure).decompose("BASE-HOME-1", pages)
    result.clauses        # PolicyClause: id, type, heading, VERBATIM text, document, page, span
    result.unclassified   # clauses whose type no rule could decide - for a human, never guessed

Deterministic and config-driven (packs/<pack>/policy_structure.yaml): no model is involved, so the same document always gives the same
clauses and every clause can be traced to the line and rule that produced it.

Guarantees
  * `clause.text` is exactly `normalise(page text)[span]` (whitespace collapsed, the same basis as the policy corpus labels). Checked again
    by `verify_clauses`, which needs only the clause and the pages, so a stored clause can be re-verified against the original at any time.
  * A clause ends at the end of its page. Text at the top of a page that continues the previous page's clause is NOT attached to it;
    a warning names it, so the gap is visible and not silent.
  * A type is decided by the first matching rule, looking at the clause heading, then its section, then the document title, then its text.
    No match = unclassified (not a guess). An abstention is cheaper than a wrong type: a wrong 'exclusion' or 'limit' reaches a payout.
  * Clause ids are `<doc_id>/<clause number>` (`BASE-HOME-1/2.2`): they come from the document's own numbering, so re-ingesting the same
    wording gives the same ids. Labelled ids (`CL-COV-WATER`) are ground truth; evals align by document, page and text.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

from claimtower.core.config import ClauseTypeRule, PolicyStructure
from claimtower.core.schemas.policy import ClauseType, PolicyClause

TypeSource = Literal["heading", "section", "title", "text", "llm"]
SOURCES: tuple[TypeSource, ...] = ("heading", "section", "title", "text")        # the rule evidence, in order; "llm" is never a rule
_ATTR = {"heading": "heading_keywords", "section": "section_keywords", "title": "title_keywords", "text": "text_cues"}
_NOT_A_HEADING_END = (".", ";", ",", ":")


class PageLike(Protocol):
    number: int
    text: str


def normalise(text: str) -> str:
    return " ".join(text.split())


@dataclass(frozen=True)
class ClauseInfo:
    """What the contract has no room for: how the clause was found and typed (audit, evals, UI)."""

    number: str
    section: str | None
    type_source: TypeSource
    confidence: float | None = None                  # only for type_source "llm" (capped); rules carry no confidence


@dataclass(frozen=True)
class Unclassified:
    document: str
    page: int
    number: str
    heading: str
    section: str | None
    text: str
    reason: str
    clause_id: str = ""                              # the id it will carry if it is typed later (ids do not depend on the type)
    span: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class TypeVerdict:
    """A checked answer. `clause_type` None = declined: the clause stays unclassified and `reason` tells a person why."""

    clause_type: ClauseType | None
    confidence: float = 0.0
    reason: str = ""
    unavailable: bool = False                        # model unreachable or unusable output (not a judgement on the clause)


class ClauseTyper(Protocol):
    """Optional second opinion for a clause the rules could not type (see policy/llm_typing.py). Only checked verdicts may be returned."""

    def type_clause(self, clause: Unclassified, document_title: str | None) -> TypeVerdict: ...


@dataclass
class Decomposition:
    doc_id: str
    title: str | None
    kind: Literal["endorsement", "policy_wording"]
    clauses: list[PolicyClause] = field(default_factory=list)
    info: dict[str, ClauseInfo] = field(default_factory=dict)
    unclassified: list[Unclassified] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Draft:
    page: int
    number: str
    heading: str
    title: str
    section: str | None
    lines: list[str] = field(default_factory=list)


class PolicyDecomposer:
    def __init__(self, structure: PolicyStructure, typer: ClauseTyper | None = None) -> None:
        self.cfg = structure
        self.typer = typer
        self._noise = [re.compile(p, re.IGNORECASE) for p in structure.noise_patterns]
        self._sections = [re.compile(p) for p in structure.section_patterns]
        self._clauses = [re.compile(p) for p in structure.clause_patterns]
        self._endorsement = [re.compile(p, re.IGNORECASE) for p in structure.endorsement_title_patterns]
        self._rules: list[tuple[ClauseTypeRule, dict[str, list[re.Pattern[str]]]]] = [
            (r, {src: [re.compile(p, re.IGNORECASE) for p in getattr(r, _ATTR[src])] for src in SOURCES}) for r in structure.types]

    # ------------------------------------------------------------------------------------------------ line classes
    def _heading_like(self, line: str) -> bool:
        return len(line) <= self.cfg.max_heading_chars and not line.endswith(_NOT_A_HEADING_END)

    def _section(self, line: str) -> str | None:
        if self._heading_like(line):
            for rx in self._sections:
                if m := rx.match(line):
                    return m.group("title").strip()
        return None

    def _clause_heading(self, line: str) -> tuple[str, str] | None:
        if self._heading_like(line):
            for rx in self._clauses:
                if m := rx.match(line):
                    return m.group("number"), m.group("title").strip()
        return None

    # ------------------------------------------------------------------------------------------------ typing
    def classify(self, *, heading: str, section: str | None, title: str | None, text: str) -> tuple[ClauseType, TypeSource] | None:
        evidence: dict[TypeSource, str | None] = {"heading": heading, "section": section, "title": title, "text": text}
        for src in SOURCES:
            value = evidence[src]
            if not value:
                continue
            for rule, patterns in self._rules:
                if any(p.search(value) for p in patterns[src]):
                    return rule.type, src
        return None

    # ------------------------------------------------------------------------------------------------ main
    def decompose(self, doc_id: str, pages: Sequence[PageLike]) -> Decomposition:
        out = Decomposition(doc_id=doc_id, title=None, kind="policy_wording")
        drafts: list[_Draft] = []
        section: str | None = None
        for index, page in enumerate(pages):
            current: _Draft | None = None
            seen_heading = False
            for raw in page.text.splitlines():
                line = raw.strip()
                if not line or any(rx.search(line) for rx in self._noise):
                    continue
                if (sec := self._section(line)) is not None:
                    section, current, seen_heading = sec, None, True
                    continue
                if (head := self._clause_heading(line)) is not None:
                    current = _Draft(page.number, head[0], line, head[1], section)
                    drafts.append(current)
                    seen_heading = True
                    continue
                if current is not None:
                    current.lines.append(line)
                elif out.title is None and index == 0 and not seen_heading:
                    out.title = line                                                # first line of a document is its title
                elif drafts and not seen_heading:
                    out.warnings.append(f"page {page.number}: text at the top of the page continues the previous page and was not "
                                        f"attached to clause {drafts[-1].number}: '{line[:60]}'")
                    seen_heading = True                                              # one warning per page is enough
        if out.title and any(rx.search(out.title) for rx in self._endorsement):
            out.kind = "endorsement"
        self._build(out, drafts, {p.number: normalise(p.text) for p in pages})
        return out

    def _build(self, out: Decomposition, drafts: list[_Draft], page_text: dict[int, str]) -> None:
        cursor: dict[int, int] = {}
        used: set[str] = set()
        for d in drafts:
            body = normalise(" ".join(d.lines))
            if not body:
                out.warnings.append(f"page {d.page}: heading '{d.heading}' has no text under it")
                continue
            full = page_text[d.page]
            at = full.find(normalise(d.heading), cursor.get(d.page, 0))
            start = full.find(body, at + len(normalise(d.heading))) if at >= 0 else -1
            if start < 0:
                out.warnings.append(f"page {d.page}: clause '{d.heading}' could not be located in the page text; left out")
                continue
            end = start + len(body)
            cursor[d.page] = end
            clause_id = f"{out.doc_id}/{d.number}"                      # the id never depends on how (or whether) the clause is typed
            if clause_id in used:
                clause_id = f"{clause_id}@p{d.page}"
            while clause_id in used:
                clause_id += "'"
            used.add(clause_id)
            decided = self.classify(heading=d.title, section=d.section, title=out.title, text=body)
            if decided is None:
                out.unclassified.append(Unclassified(out.doc_id, d.page, d.number, d.heading, d.section, body,
                                                     "no type rule matched its heading, section, document title or text",
                                                     clause_id, (start, end)))
                continue
            out.clauses.append(PolicyClause(clause_id=clause_id, clause_type=decided[0], heading=d.heading, text=body,
                                            document=out.doc_id, page=d.page, span=(start, end)))
            out.info[clause_id] = ClauseInfo(d.number, d.section, decided[1])
        if self.typer is not None:
            self._ask_typer(out)

    def _ask_typer(self, out: Decomposition) -> None:
        """Offer each unclassified clause to the optional typer. A clause is typed only on a checked verdict; otherwise it stays
        unclassified with the typer's reason appended. Rule-typed clauses are never revisited."""
        assert self.typer is not None
        remaining: list[Unclassified] = []
        unavailable = 0
        for u in out.unclassified:
            v = self.typer.type_clause(u, out.title)
            unavailable += v.unavailable
            if v.clause_type is None:
                remaining.append(replace(u, reason=f"{u.reason}; model typing declined: {v.reason}" if v.reason else u.reason))
                continue
            out.clauses.append(PolicyClause(clause_id=u.clause_id, clause_type=v.clause_type, heading=u.heading, text=u.text,
                                            document=u.document, page=u.page, span=u.span))
            out.info[u.clause_id] = ClauseInfo(u.number, u.section, "llm", v.confidence)
        out.unclassified = remaining
        if unavailable:
            out.warnings.append(f"model typing was unavailable for {unavailable} clause(s): they stay unclassified, NOT in a snapshot")
        out.clauses.sort(key=lambda c: (c.page or 0, c.span or (0, 0)))


def verify_clauses(clauses: Sequence[PolicyClause], pages: Sequence[PageLike]) -> list[str]:
    """Independent re-check against the original pages: every clause's text must be exactly its span of the normalised page text."""
    by_page = {p.number: normalise(p.text) for p in pages}
    problems: list[str] = []
    for c in clauses:
        if c.page not in by_page or c.span is None:
            problems.append(f"{c.clause_id}: no such page or no span")
            continue
        start, end = c.span
        if not (0 <= start < end <= len(by_page[c.page])) or by_page[c.page][start:end] != c.text:
            problems.append(f"{c.clause_id}: text is not the page text at span {c.span}")
    return problems
