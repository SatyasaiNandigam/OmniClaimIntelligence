"""Composition root: which agent runs in each slot of the graph.

    registry = build_registry(settings, policy_library=..., document_store=..., llm=...)

A slot gets its REAL agent when the dependencies it needs are given, and a MOCK ("YET TO IMPLEMENT") otherwise. Today:

      evidence  needs a DocumentStore (with a model: classifier tie-breaker, field filler and scan OCR are on)       policy  needs a PolicyLibrary (and a PAS: the mock adapter by default)       gap  always (rules; model if llm)
      coverage  always (the decision table; the model's reading too if llm)
    settlement  always (the deterministic engine; the model reads amounts and rewords the explanation if llm)
    routing     always (the routing rules; deterministic, no model)
    action      always (the planner and the pack templates; the model rewords drafts if llm)
    intake      always (the first notice of loss, the checklist, the policy lookup and the registration in the claims system)
    audit       always (the quality checks; deterministic, no model; a failing report blocks the release to the gate)
    (no agent is a mock by default any more)  (they are implemented in later phases: replace the line in `_MOCKS`)

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

from claimtower.actions.planner import ActionPlanner, ActionTemplates
from claimtower.agents.action import make_action_agent
from claimtower.agents.audit import AuditAgent
from claimtower.agents.coverage import make_coverage_agent
from claimtower.agents.evidence import EvidenceAgent
from claimtower.agents.gap import make_gap_agent
from claimtower.agents.intake import IntakeAgent
from claimtower.agents.policy import PolicyAgent
from claimtower.agents.routing import RoutingAgent
from claimtower.agents.settlement import make_settlement_agent
from claimtower.audit.checks import QaConfig
from claimtower.audit.guards import AutonomyGuard, GuardedRecon
from claimtower.core.config import Settings
from claimtower.ingestion.classifier import HybridClassifier, LLMClassifier, RuleClassifier
from claimtower.ingestion.extractors import HybridExtractor, LLMFieldFiller, RuleExtractor
from claimtower.ingestion.ocr import VisionLLMOcr
from claimtower.ingestion.store import DocumentStore
from claimtower.integrations import Adapters, build_adapters
from claimtower.integrations.dispatcher import Dispatcher
from claimtower.integrations.mock.claims_mock import MockPaymentsLedger
from claimtower.orchestration.commit import CommitService
from claimtower.orchestration.mocks import FixtureSource, MockAgent
from claimtower.orchestration.registry import PIPELINE, AgentRegistry
from claimtower.policy.decompose import PolicyDecomposer
from claimtower.policy.llm_typing import make_decomposer
from claimtower.policy.relations import RelationExtractor
from claimtower.policy.snapshot import PolicyLibrary, SnapshotBuilder
from claimtower.routing.tiering import TieringEngine
from claimtower.settlement.engine import SettlementCalculator
from claimtower.settlement.rules_loader import rules_by_version

REPO = Path(__file__).resolve().parents[2]
SYNTHETIC = REPO / "data" / "synthetic"
SETTLEMENT_RULES = "sample-1.0"           # settlement_v1.yaml (the guidelines PDF order); v2 is an open domain question (docs/assumptions.md Q-S1)
_MOCKS: tuple[str, ...] = ()      # every agent is implemented; a node becomes a mock only through `force_mock`


def build_registry(settings: Settings, *, document_store: DocumentStore | None = None, policy_library: PolicyLibrary | None = None, pas: Any | None = None,
                   adapters: Adapters | None = None, dispatcher: Dispatcher | None = None, overrides: Any | None = None, policy_manager: Any | None = None,
                   llm: Any | None = None, fixtures: FixtureSource | None = None, force_mock: tuple[str, ...] = ()) -> AgentRegistry:
    """The registry for a run. `force_mock` makes a node a mock even when it could be real (used to run the whole graph on fixtures)."""
    pack = settings.pack
    fx = fixtures or FixtureSource()
    real: dict[str, Any] = {"gap": make_gap_agent(pack, llm), "coverage": make_coverage_agent(pack, llm, settings.app.abstention["min_citation_count"])}
    classifier = extractor = None
    if document_store is not None:
        rule_classifier, rule_extractor = RuleClassifier(pack.document_types), RuleExtractor(pack.document_types)
        # with a model the document-intelligence fallbacks are on (classifier tie-breaker, required-field filler, scan transcription); rules always go first
        classifier = HybridClassifier(rule_classifier, LLMClassifier(llm, rule_classifier, pack) if llm is not None else None)
        extractor = HybridExtractor(rule_extractor, LLMFieldFiller(llm, rule_extractor, pack) if llm is not None else None)
        real["evidence"] = EvidenceAgent(document_store, pack, classifier, extractor, VisionLLMOcr(llm) if llm is not None else None)
    adapters = adapters or build_adapters(settings)           # mock or live adapters from the flags (a live one is not implemented yet and fails loudly)
    dispatcher = dispatcher or Dispatcher()
    pas_adapter = pas or adapters.pas
    real["intake"] = IntakeAgent(pack.intake_checklist, pas_adapter, adapters.claims, dispatcher, store=document_store, classifier=classifier, extractor=extractor)
    rules = rules_by_version(pack.calculation)
    real["settlement"] = make_settlement_agent(pack, SettlementCalculator(rules), SETTLEMENT_RULES, pas_adapter if hasattr(pas_adapter, "get_schedule") else None,
                                               MockPaymentsLedger(), llm, overrides)
    real["action"] = make_action_agent(pack, settings.autonomy, llm)
    amount_issues = [i.id for i in pack.coverage.issues if i.kind == "amount"] if pack.coverage else []
    real["audit"] = AuditAgent(QaConfig.load(pack))
    real["routing"] = RoutingAgent(TieringEngine(settings.routing), amount_issue_ids=amount_issues)
    if policy_manager is not None:                                  # the library can change while running: the builder follows the manager
        builder = SnapshotBuilder(pas_adapter, policy_manager.library, RelationExtractor(pack.policy_structure))
        policy_manager.attach(builder)
        real["policy"] = PolicyAgent(builder)
    elif policy_library is not None:
        real["policy"] = PolicyAgent(SnapshotBuilder(pas_adapter, policy_library, RelationExtractor(pack.policy_structure)))
    slots: dict[str, tuple[Any, str]] = {}
    for node in PIPELINE:
        if node in real and node not in force_mock and node not in _MOCKS:
            slots[node] = (real[node], "implemented")
        else:
            slots[node] = (MockAgent(node, fx), "mock")
    return AgentRegistry(slots)


def build_commit_service(settings: Settings, adapters: Adapters | None = None, dispatcher: Dispatcher | None = None) -> CommitService:
    """The commit step after an approval: the claims and recon adapters the flags select (mock by default), behind the idempotent, retrying, dead-lettering dispatcher."""
    adapters = adapters or build_adapters(settings)
    guard = AutonomyGuard(settings.autonomy, settings.rbac.approval_order)
    adapters = Adapters(adapters.pas, adapters.claims, GuardedRecon(adapters.recon, guard))        # nothing but an authorised person can instruct a payment (T8.4)
    return CommitService(adapters, dispatcher or Dispatcher(), ActionPlanner(settings.autonomy, ActionTemplates.load(settings.pack)), settings.autonomy, guard)


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


def build_policy_manager(settings: Settings, upload_dir: Path | None, builtin_dir: Path = SYNTHETIC / "policies", llm: Any | None = None):
    """The policy library that can be added to and reloaded while the server runs. Built-in wordings come from `builtin_dir`, uploaded ones live in `upload_dir`."""
    from claimtower.policy.manager import IdentityReader, PolicyDocumentManager

    decomposer = make_decomposer(settings.pack, llm) if llm is not None else PolicyDecomposer(settings.pack.policy_structure)
    return PolicyDocumentManager(decomposer, IdentityReader.load(settings.pack, llm), builtin_dir, upload_dir)


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
