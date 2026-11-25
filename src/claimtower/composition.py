"""Composition root: which agent runs in each slot of the graph.

    registry = build_registry(settings, policy_library=..., document_store=..., llm=...)

A slot gets its REAL agent when the dependencies it needs are given, and a MOCK ("YET TO IMPLEMENT") otherwise. Today:

    real if given           intake (T6.7)   mock     coverage (T5.1)  mock     action (T6.4)  mock
      evidence  needs a DocumentStore       policy  needs a PolicyLibrary (and a PAS: the mock adapter by default)       gap  always (rules; model if llm)
    mock, always            intake, coverage, settlement, action, audit, routing  (they are implemented in later phases: replace the line in `_MOCKS`)

Replacing a mock later is: write the agent, import it here, put it in `real`. The graph does not change.

The helpers below load SYNTHETIC data for the prototype and the demo (the claim folders and the policy PDFs under data/synthetic); the API will upload
documents into the store and ingest policy wordings instead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from claimtower.agents.evidence import EvidenceAgent
from claimtower.agents.gap import make_gap_agent
from claimtower.agents.policy import PolicyAgent
from claimtower.core.config import Settings
from claimtower.ingestion.classifier import HybridClassifier, RuleClassifier
from claimtower.ingestion.extractors import HybridExtractor, RuleExtractor
from claimtower.ingestion.store import DocumentStore
from claimtower.integrations.mock.pas_mock import MockPasAdapter
from claimtower.orchestration.mocks import FixtureSource, MockAgent
from claimtower.orchestration.registry import PIPELINE, AgentRegistry
from claimtower.policy.decompose import PolicyDecomposer
from claimtower.policy.llm_typing import make_decomposer
from claimtower.policy.relations import RelationExtractor
from claimtower.policy.snapshot import PolicyLibrary, SnapshotBuilder

REPO = Path(__file__).resolve().parents[2]
SYNTHETIC = REPO / "data" / "synthetic"
_MOCKS = ("intake", "coverage", "settlement", "action", "audit", "routing")      # not implemented yet: always mocks


def build_registry(settings: Settings, *, document_store: DocumentStore | None = None, policy_library: PolicyLibrary | None = None, pas: Any | None = None,
                   llm: Any | None = None, fixtures: FixtureSource | None = None, force_mock: tuple[str, ...] = ()) -> AgentRegistry:
    """The registry for a run. `force_mock` makes a node a mock even when it could be real (used to run the whole graph on fixtures)."""
    pack = settings.pack
    fx = fixtures or FixtureSource()
    real: dict[str, Any] = {"gap": make_gap_agent(pack, llm)}
    if document_store is not None:
        classifier = HybridClassifier(RuleClassifier(pack.document_types))
        real["evidence"] = EvidenceAgent(document_store, pack, classifier, HybridExtractor(RuleExtractor(pack.document_types)))
    if policy_library is not None:
        real["policy"] = PolicyAgent(SnapshotBuilder(pas or MockPasAdapter(), policy_library, RelationExtractor(pack.policy_structure)))
    slots: dict[str, tuple[Any, str]] = {}
    for node in PIPELINE:
        if node in real and node not in force_mock and node not in _MOCKS:
            slots[node] = (real[node], "implemented")
        else:
            slots[node] = (MockAgent(node, fx), "mock")
    return AgentRegistry(slots)


# ---------------------------------------------------------------------------------------------------- synthetic data loaders (prototype and demo)
@dataclass
class _Page:
    number: int
    text: str


def policy_library_from_pdfs(settings: Settings, directory: Path = SYNTHETIC / "policies", llm: Any | None = None) -> PolicyLibrary:
    """Every policy wording and endorsement PDF in `directory`, decomposed into clauses (rules, plus the model's typing when the pack enables it)."""
    decomposer = make_decomposer(settings.pack, llm) if llm is not None else PolicyDecomposer(settings.pack.policy_structure)
    docs = []
    for f in sorted(directory.glob("*.pdf")):
        with pymupdf.open(f) as pdf:
            pages = [_Page(i + 1, p.get_text()) for i, p in enumerate(pdf)]
        docs.append(decomposer.decompose(f.stem, pages))
    return PolicyLibrary(docs)


def ingest_claim_folder(store: DocumentStore, claim_dir: Path) -> list[str]:
    """Store the documents of one synthetic claim folder (data/synthetic/claims/<CLM>/manifest.json) under their manifest ids. Returns the document ids."""
    manifest = json.loads((claim_dir / "manifest.json").read_text(encoding="utf-8"))
    ids = []
    for d in manifest["documents"]:
        if not d.get("in_scenario", True):
            continue
        store.ingest(manifest["claim_id"], d["filename"], (claim_dir / d["filename"]).read_bytes(), document_id=d["document_id"],
                     uploaded_by=d.get("uploaded_by"))
        ids.append(d["document_id"])
    return ids
