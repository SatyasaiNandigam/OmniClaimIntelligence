"""Clause retrieval with citations (T3.4): a question -> the clauses IN FORCE that answer it, each with the endorsements that modify it.

    retriever = ClauseRetriever(pack.policy_structure.retrieval, embedder=llm)     # embedder optional: without it, keyword search only
    found = retriever.search("What is the water-damage sublimit?", snapshot)
    found.hits         # ranked ClauseCitation(clause_id, document, page, span, text, modified_by=[endorsement clause ids])
    found.modifiers    # the modifying endorsement clauses that are not hits themselves, so their text can be shown with the clause
    retriever.retrieve_clauses(query, snapshot)        # the PolicyService contract: hits followed by the modifiers

Why hybrid. Keyword search (BM25) finds the clause that uses the asker's words; embeddings find the clause that MEANS the same ("slow leak"
~ "gradual leakage"). Each ranks the clauses, and the two rankings are fused by reciprocal rank, so a clause both agree on rises to the top.

Why the endorsement comes back with the clause. Plain similarity search returns the base clause ("we cover escape of water") and misses the
endorsement that caps it, because the endorsement's own wording is about something else. Here the snapshot's relations say which endorsement
clauses act on a clause, so `modified_by` is filled from them, never from similarity, and the endorsement text is returned alongside.

Only clauses in the snapshot are searched: a clause an endorsement replaced, or an endorsement not yet effective, can never be returned.

No weak guesses. A keyword match needs a shared word; an embedding match needs `min_dense_similarity`. A question nothing in the policy
answers returns an empty list, and the caller treats that as 'not found', not as an answer.

If the embedding service is unavailable the search degrades to keyword only and `Retrieval.mode` says so.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from claimtower.core.config import RetrievalConfig
from claimtower.core.errors import ConfigError
from claimtower.core.llm import LLMError
from claimtower.core.schemas.policy import ClauseCitation, PolicyClause, PolicySnapshot
from claimtower.policy.relations import stem

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass
class Retrieval:
    query: str
    mode: str                                                       # "hybrid" | "keyword" | "keyword (embeddings unavailable)"
    hits: list[ClauseCitation] = field(default_factory=list)
    modifiers: list[ClauseCitation] = field(default_factory=list)
    keyword_rank: list[str] = field(default_factory=list)           # clause ids, best first (audit)
    dense_rank: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def citations(self) -> list[ClauseCitation]:
        return [*self.hits, *self.modifiers]


class ClauseRetriever:
    def __init__(self, cfg: RetrievalConfig, embedder: Embedder | None = None) -> None:
        self.cfg, self.embedder = cfg, embedder
        self._stop = {w.lower() for w in cfg.stopwords}
        self._vectors: dict[str, list[float]] = {}                  # text -> embedding (the cassette caches across runs)

    # ------------------------------------------------------------------------------------------------ public
    def retrieve_clauses(self, query: str, snapshot: PolicySnapshot) -> list[ClauseCitation]:
        return self.search(query, snapshot).citations

    def search(self, query: str, snapshot: PolicySnapshot, k: int | None = None) -> Retrieval:
        k = k or self.cfg.k
        clauses = list(snapshot.clauses)
        found = Retrieval(query, "keyword" if self.embedder is None else "hybrid")
        if not clauses or not query.strip():
            return found
        keyword = self._keyword(query, clauses)
        dense: list[tuple[str, float]] = []
        if self.embedder is not None:
            try:
                dense = self._dense(query, clauses)
            except (LLMError, ConfigError) as e:
                found.mode = "keyword (embeddings unavailable)"
                found.warnings.append(f"embeddings unavailable ({type(e).__name__}): keyword results only")
        found.keyword_rank, found.dense_rank = [c for c, _ in keyword], [c for c, _ in dense]
        by_id = {c.clause_id: c for c in clauses}
        kw_score = dict(keyword)
        fused: dict[str, float] = {}
        for weight, ranking in ((self.cfg.keyword_weight, found.keyword_rank), (self.cfg.dense_weight, found.dense_rank)):
            for rank, cid in enumerate(ranking, start=1):
                fused[cid] = fused.get(cid, 0.0) + weight / (self.cfg.rrf_k + rank)
        order = sorted(fused, key=lambda c: (-fused[c], -kw_score.get(c, 0.0), c))[:k]
        modifiers_of = self._modifiers(snapshot)
        found.hits = [self._citation(by_id[c], modifiers_of) for c in order]
        shown = set(order)
        for hit in list(found.hits):
            for m in hit.modified_by:
                if m not in shown and m in by_id:
                    shown.add(m)
                    found.modifiers.append(self._citation(by_id[m], modifiers_of))
        return found

    # ------------------------------------------------------------------------------------------------ keyword
    def _tokens(self, text: str) -> list[str]:
        return [s for w in _WORD.findall(text.lower()) if w not in self._stop and len(s := stem(w)) >= 2]

    def _keyword(self, query: str, clauses: list[PolicyClause]) -> list[tuple[str, float]]:
        """BM25 over `heading x heading_weight + text`. Only clauses that share at least one query word are ranked."""
        docs = {c.clause_id: self._tokens(((c.heading or "") + " ") * self.cfg.heading_weight + c.text) for c in clauses}
        q = set(self._tokens(query))
        n = len(docs)
        avg = sum(len(d) for d in docs.values()) / n or 1.0
        df = {t: sum(1 for d in docs.values() if t in d) for t in q}
        scores: dict[str, float] = {}
        for cid, toks in docs.items():
            s = 0.0
            for t in q:
                tf = toks.count(t)
                if tf:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    norm = 1 - self.cfg.bm25_b + self.cfg.bm25_b * len(toks) / avg
                    s += idf * tf * (self.cfg.bm25_k1 + 1) / (tf + self.cfg.bm25_k1 * norm)
            if s > 0:
                scores[cid] = s
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

    # ------------------------------------------------------------------------------------------------ embeddings
    def _dense(self, query: str, clauses: list[PolicyClause]) -> list[tuple[str, float]]:
        assert self.embedder is not None
        texts = {c.clause_id: f"{c.heading or ''}. {c.text}" for c in clauses}
        missing = sorted({t for t in [query, *texts.values()] if t not in self._vectors})
        if missing:
            self._vectors.update(zip(missing, self.embedder.embed(missing)))
        qv = self._vectors[query]
        sims = {cid: _cosine(qv, self._vectors[t]) for cid, t in texts.items()}
        return sorted(((cid, s) for cid, s in sims.items() if s >= self.cfg.min_dense_similarity), key=lambda kv: (-kv[1], kv[0]))

    # ------------------------------------------------------------------------------------------------ citations
    @staticmethod
    def _modifiers(snapshot: PolicySnapshot) -> dict[str, list[str]]:
        endorsement_clauses = {i for e in snapshot.endorsements for i in e.clause_ids}
        out: dict[str, list[str]] = {}
        for r in snapshot.relations:
            if r.source_clause_id in endorsement_clauses:
                out.setdefault(r.target_clause_id, []).append(r.source_clause_id)
        return {k: sorted(set(v)) for k, v in out.items()}

    @staticmethod
    def _citation(c: PolicyClause, modifiers_of: dict[str, list[str]]) -> ClauseCitation:
        return ClauseCitation(clause_id=c.clause_id, document=c.document, page=c.page, span=c.span, text=c.text,
                              modified_by=modifiers_of.get(c.clause_id, []))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
