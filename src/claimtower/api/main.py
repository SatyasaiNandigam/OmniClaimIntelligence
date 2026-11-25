"""The default API application: real Evidence, Policy and Gap agents, mocks for the rest, SYNTHETIC data.

    uvicorn claimtower.api.main:build_default_app --factory --port 8000        # then open http://127.0.0.1:8000/docs
    python scripts/run_api.py                                                  # the same, with options

Environment (all optional):
    CLAIMTOWER_STORE   directory for the document store (default: a fresh temporary one, like the in-memory claims it goes with)
    CLAIMTOWER_LLM     1 = the Gap agent also uses the model (needs LLM_API_KEY in .env; only synthetic data is sent)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI

from claimtower.api.app import create_app
from claimtower.composition import SYNTHETIC, build_registry, policy_library_from_pdfs
from claimtower.core.config import load_settings
from claimtower.ingestion.store import DocumentStore
from claimtower.orchestration.graph import ClaimRunner


def build_default_app() -> FastAPI:
    settings = load_settings()
    llm = None
    if os.environ.get("CLAIMTOWER_LLM") == "1":
        from dotenv import load_dotenv

        from claimtower.core.llm import LLMClient
        load_dotenv()
        llm = LLMClient.from_settings(settings)
    store = DocumentStore(Path(os.environ.get("CLAIMTOWER_STORE") or tempfile.mkdtemp(prefix="claimtower-store-")))
    registry = build_registry(settings, document_store=store, policy_library=policy_library_from_pdfs(settings), llm=llm)
    return create_app(ClaimRunner(registry), store, synthetic_claims=SYNTHETIC / "claims")
