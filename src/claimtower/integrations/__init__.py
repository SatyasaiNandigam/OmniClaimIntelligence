"""integrations

OWNER   : A (Lead: orchestration + integrations)
TASKS   : T6.5-T6.6
PURPOSE : PAS / Claims / Recon protocols + mocks. mock -> live is a config flag.

    pas, claims, recon = build_adapters(settings)      # the mock or the live adapter of each system, from the flags in configs/app.yaml (USE_MOCK_PAS ... in the environment)

The mocks are in-process objects (no server): they keep the production contract (idempotent writes, every response labelled `mock`) and can be made to fail on purpose
(`integrations.mock.faults`). The live adapters are stubs that raise `AdapterNotConfigured`; docs/REPLACING_MOCK_ADAPTERS.md says how to replace a mock.
"""
from __future__ import annotations

from typing import Any, NamedTuple

from claimtower.core.config import Settings
from claimtower.core.registry import ADAPTERS, resolve_adapter
from claimtower.integrations.live import not_configured
from claimtower.integrations.mock.claims_mock import MockClaimsAdapter
from claimtower.integrations.mock.pas_mock import MockPasAdapter
from claimtower.integrations.mock.recon_mock import MockReconAdapter


class Adapters(NamedTuple):
    pas: Any
    claims: Any
    recon: Any


def register_adapters() -> None:
    """Register the mock and the (stub) live adapter of each system. Safe to call more than once."""
    for name, obj in {"pas:mock": MockPasAdapter, "claims:mock": MockClaimsAdapter, "recon:mock": MockReconAdapter, "pas:live": not_configured("pas"),
                      "claims:live": not_configured("claims"), "recon:live": not_configured("recon")}.items():
        if name not in ADAPTERS.names():
            ADAPTERS.register(name, obj)


def build_adapters(settings: Settings) -> Adapters:
    """A new adapter of each system, mock or live as the flags say."""
    register_adapters()
    return Adapters(*(resolve_adapter(kind, settings)() for kind in ("pas", "claims", "recon")))


__all__ = ["Adapters", "build_adapters", "register_adapters"]
