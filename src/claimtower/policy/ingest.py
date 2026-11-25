"""Policy ingestion (T3.1 -> T3.3): a policy document's pages -> decomposed, verified and stored, ready for the snapshot builder.

    ingestor = PolicyIngestor(store, make_decomposer(pack, llm))     # llm: core.llm.LLMClient, or None for rules only
    result = ingestor.ingest("BASE-HOME-1", pages)                   # decompose + verify + save
    builder = SnapshotBuilder(pas, ingestor.library(), RelationExtractor(pack.policy_structure))

Nothing unverifiable is stored: every clause is re-checked against the pages it came from, and a document with a failing clause is refused.
With the model typing on (`llm_typing.enabled`), the clauses it types are stored with `type_source = "llm"` and their capped confidence, so
the snapshot can name them for review; clauses nobody could type are stored as unclassified and named as missing from the snapshot.
"""
from __future__ import annotations

from collections.abc import Sequence

from claimtower.policy.decompose import Decomposition, PageLike, PolicyDecomposer, verify_clauses
from claimtower.policy.snapshot import PolicyLibrary
from claimtower.policy.store import PolicyStore


class PolicyIngestError(Exception):
    """The document's clauses could not be verified against its pages: nothing was stored."""


class PolicyIngestor:
    def __init__(self, store: PolicyStore, decomposer: PolicyDecomposer) -> None:
        self.store, self.decomposer = store, decomposer

    def ingest(self, doc_id: str, pages: Sequence[PageLike]) -> Decomposition:
        out = self.decomposer.decompose(doc_id, pages)
        problems = verify_clauses(out.clauses, pages)
        if problems:
            raise PolicyIngestError(f"{doc_id}: " + "; ".join(problems))
        self.store.save(out, len(pages))
        return out

    def library(self) -> PolicyLibrary:
        return PolicyLibrary.from_store(self.store)
