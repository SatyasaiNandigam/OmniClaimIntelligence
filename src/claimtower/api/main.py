"""The default API application: real Evidence, Policy, Gap, Coverage, Settlement and Routing agents, mocks for the rest, SYNTHETIC data.

    uvicorn claimtower.api.main:build_default_app --factory --port 8000        # then open http://127.0.0.1:8000/docs
    python scripts/run_api.py                                                  # the same, with options

Environment (all optional):
    CLAIMTOWER_STORE   directory for the document store (default: a fresh temporary one, like the in-memory claims it goes with)
    CLAIMTOWER_POLICIES  directory for uploaded policy documents (default: <store>/policies; kept between runs only when CLAIMTOWER_STORE is set)
    CLAIMTOWER_LLM     1 = the Evidence, Gap, Coverage, Settlement and Action agents also use the model (needs LLM_API_KEY in .env; only synthetic data is sent)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI

from claimtower.api.app import create_app
from claimtower.composition import SYNTHETIC, build_commit_service, build_policy_manager, build_registry
from claimtower.audit.trail import AuditTrail
from claimtower.core.config import load_settings
from claimtower.integrations import build_adapters
from claimtower.integrations.dispatcher import Dispatcher
from claimtower.ingestion.store import DocumentStore
from claimtower.orchestration.graph import ClaimRunner
from claimtower.orchestration.overrides import OverrideStore
from claimtower.integrations.mock.schedule import ScheduleReader


def build_default_app() -> FastAPI:
    settings = load_settings()
    llm = None
    if os.environ.get("CLAIMTOWER_LLM") == "1":
        from dotenv import load_dotenv

        from claimtower.core.llm import LLMClient
        load_dotenv()
        llm = LLMClient.from_settings(settings)
    store = DocumentStore(Path(os.environ.get("CLAIMTOWER_STORE") or tempfile.mkdtemp(prefix="claimtower-store-")))
    adapters, dispatcher = build_adapters(settings), Dispatcher()                         # one claims system and one retrying dispatcher for intake and the commit step
    overrides = OverrideStore()                                                         # what a reviewer edits at the gate is read by the Settlement agent
    policies = build_policy_manager(settings, Path(os.environ.get("CLAIMTOWER_POLICIES") or store.root / "policies"), llm=llm)        # add or reload policy documents while running
    registry = build_registry(settings, document_store=store, policy_manager=policies, adapters=adapters, dispatcher=dispatcher, overrides=overrides, llm=llm)
    trail = AuditTrail(Path(os.environ.get("CLAIMTOWER_AUDIT") or store.root / "audit-trail.jsonl"))         # persistent when CLAIMTOWER_STORE is set (the default store is temporary)
    runner = ClaimRunner(registry, commit_service=build_commit_service(settings, adapters, dispatcher), overrides=overrides, settings=settings, trail=trail, llm=llm)
    schedule_reader = ScheduleReader(llm, settings.pack)                                  # reads a policy schedule for the mock PAS (the model when there is one)
    return create_app(runner, store, synthetic_claims=SYNTHETIC / "claims", policies=policies, pas=adapters.pas, schedule_reader=schedule_reader)
