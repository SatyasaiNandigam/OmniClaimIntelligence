"""Which document is this? Read by the model, checked, never trusted (policy ingestion).

    reader = LLMIdentityReader(llm, pack)
    ident = reader.read(pages)         # Identity(doc_id, version, evidence, source="model", details={...}) or None, with `reader.last_problem` saying why not

The model reads the first pages and names the identifier the document gives itself (a wording reference, a policy or endorsement number), the version, the title, the kind and, for information, the policy
number and period. An answer is accepted only if ALL hold: the identifier appears verbatim in the text of the pages (so it was read, not invented; the evidence shown is the model's quote when that is verbatim, otherwise the document's own words around the identifier); after making it safe for a file
name (slashes and spaces become '-') it is a usable id; the confidence reaches the minimum. Anything else is refused and the caller falls back to the fixed patterns, or asks for an explicit document id.
The policy number and period are only reported (each must also be verbatim text): they help whoever sets up the policy in the policy system, and decide nothing here.
"""
from __future__ import annotations


import re
from typing import Any

from pydantic import BaseModel, Field

from claimtower.policy.problem import ThreadLocalProblem

MIN_CONFIDENCE = 0.6
MAX_CHARS = 7000
SAFE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{1,63}$")


class IdentityAnswer(BaseModel):
    document_id: str = ""
    evidence_quote: str = ""
    version: str = ""
    title: str = ""
    kind: str = "other"
    policy_number: str = ""
    period: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    note: str = ""


def _norm(s: str) -> str:
    return " ".join(s.split())


def safe_id(raw: str) -> str:
    """The identifier as a file-name-safe id: runs of anything else become '-'."""
    return re.sub(r"-{2,}", "-", re.sub(r"[^A-Za-z0-9._-]+", "-", raw.strip())).strip("-._")


class LLMIdentityReader:
    last_problem = ThreadLocalProblem()                # why the last reading gave nothing; per thread, so documents read in parallel do not mix their reasons

    def __init__(self, llm: Any, pack: Any) -> None:
        self._llm, self._pack = llm, pack

    def read(self, pages: list[Any], n_pages: int = 3) -> tuple[str, str | None, str, str, dict[str, str]] | None:
        """(doc_id, version, evidence quote, kind, details) or None (see `last_problem`)."""
        from claimtower.core.llm import load_prompt

        self.last_problem = None
        text = _norm(" ".join(p.text for p in pages[:n_pages]))[:MAX_CHARS]
        try:
            ans = self._llm.structured(IdentityAnswer, load_prompt("policy_identity", self._pack, text=text))
        except Exception as e:                                         # model down, bad JSON ...
            self.last_problem = f"the model could not be reached or answered badly ({type(e).__name__}: {e})"
            return None
        return self.check(ans, text)

    def check(self, a: IdentityAnswer, text: str) -> tuple[str, str | None, str, str, dict[str, str]] | None:
        if not a.document_id.strip():
            self.last_problem = f"the model found no identifier in the document ({a.note or 'no reason given'})"
            return None
        raw = _norm(a.document_id)
        pos = text.lower().find(raw.lower())
        if pos < 0:                                                    # the identifier must be text of the document: it was read, not invented
            self.last_problem = f"the identifier '{raw}' does not appear in the text of the document"
            return None
        quote = _norm(a.evidence_quote)
        if len(quote) < 3 or quote not in text or raw.lower() not in quote.lower():
            quote = text[max(0, pos - 30): pos + len(raw) + 30].strip()          # the model's quote did not hold: show the document's own words around the identifier instead
        if a.confidence < MIN_CONFIDENCE:
            self.last_problem = f"the model is not sure enough of '{raw}' (confidence {a.confidence:.2f})"
            return None
        doc_id = safe_id(raw)
        if not SAFE_ID.match(doc_id):
            self.last_problem = f"'{raw}' cannot be used as a document id"
            return None
        details = {k: v for k, v in (("title", a.title), ("policy_number", a.policy_number), ("period", a.period), ("id_as_written", raw), ("model_kind", a.kind)) if v and _norm(v) and
                   (k in ("model_kind", "id_as_written") or _norm(v) in text)}                       # a reported value must be text of the document too
        version = a.version.strip() if a.version.strip() and _norm(a.version) in text else None
        return doc_id, version, quote, a.kind, details


__all__ = ["IdentityAnswer", "LLMIdentityReader", "safe_id"]
