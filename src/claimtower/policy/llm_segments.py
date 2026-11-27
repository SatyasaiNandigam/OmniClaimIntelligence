"""Where do the clauses begin? Read by the model, checked, never trusted (policy ingestion, for wordings the numbering patterns cannot cut).

    splitter = LLMSegmenter(llm, pack)
    marks = splitter.split(pages, title)          # Marks(clauses={line: number}, sections={line}) or None, with `splitter.last_problem`

The model is shown each page as numbered lines and names the lines that are clause headings (and the headings of groups of clauses). A heading is
accepted only if it is EXACTLY one of the lines of that page: the model cannot invent a heading or move a boundary, it can only choose among the
document's own lines. The text of every clause is then cut out of the page by the same code as for patterned documents, so it stays verbatim
and verifiable. Typing of the clauses is a separate, already checked step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

MAX_LINES = 400


class ClauseMark(BaseModel):
    heading: str
    number: str = ""


class SegmentAnswer(BaseModel):
    clauses: list[ClauseMark] = []
    sections: list[str] = []


@dataclass
class Marks:
    clauses: dict[str, str] = field(default_factory=dict)        # normalised heading line -> number as written ("" if none)
    sections: set[str] = field(default_factory=set)


def _norm(s: str) -> str:
    return " ".join(s.split())


class LLMSegmenter:
    def __init__(self, llm: Any, pack: Any) -> None:
        self._llm, self._pack = llm, pack
        self.last_problem: str | None = None

    def split(self, pages: list[Any], title: str | None = None) -> Marks | None:
        from claimtower.core.llm import load_prompt

        self.last_problem = None
        marks = Marks()
        for p in pages:
            lines = [_norm(x) for x in p.text.splitlines() if _norm(x)][:MAX_LINES]
            if not lines:
                continue
            try:
                ans = self._llm.structured(SegmentAnswer, load_prompt("policy_segments", self._pack, title=title or "(untitled)", page=str(p.number),
                                                                       lines="\n".join(lines)))
            except Exception as e:                                       # model down, bad JSON ...
                self.last_problem = f"the model could not be reached or answered badly ({type(e).__name__}: {e})"
                return None
            self.check(ans, lines, marks)
        if not marks.clauses:
            self.last_problem = self.last_problem or "the model found no clause headings in the document"
            return None
        return marks

    @staticmethod
    def check(ans: SegmentAnswer, lines: list[str], marks: Marks) -> None:
        """Keep only headings that are exactly a line of the page; a number is kept only if the line starts with it."""
        on_page = set(lines)
        for s in ans.sections:
            if _norm(s) in on_page:
                marks.sections.add(_norm(s))
        for c in ans.clauses:
            h = _norm(c.heading)
            if h in on_page and h not in marks.sections:
                i = lines.index(h)
                if i > 0 and (h[0].islower() or h[0] == "(") and lines[i - 1][-1] not in ".;:!?":        # a clause cannot begin in the middle of a sentence: a wrapped continuation line
                    continue
                n = _norm(c.number)
                marks.clauses[h] = n if n and h.startswith(n) else ""


__all__ = ["ClauseMark", "LLMSegmenter", "Marks", "SegmentAnswer"]
