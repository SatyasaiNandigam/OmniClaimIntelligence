"""Plug-in registries (T0.5): agents, adapters, packs are looked up by name, never imported directly.

    @AGENTS.register("gap")
    class GapAgent: ...

    ADAPTERS.register("pas:mock", MockPAS)       # integrations/mock/pas_mock.py
    pas = resolve_adapter("pas", settings)       # picks "pas:mock" or "pas:live" from the flags
"""
from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

from claimtower.core.config import Pack, Settings, available_packs, load_pack
from claimtower.core.errors import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, obj: T | None = None) -> Any:
        """Register `obj`, or use as a decorator when `obj` is omitted."""
        def _add(o: T) -> T:
            if name in self._items:
                raise RegistryError(f"{self.kind} '{name}' is already registered")
            self._items[name] = o
            return o
        return _add if obj is None else _add(obj)

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise RegistryError(f"unknown {self.kind} '{name}'; registered: {self.names()}") from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def clear(self) -> None:           # for tests
        self._items.clear()


AGENTS: Registry[Callable[..., Any]] = Registry("agent")
ADAPTERS: Registry[Callable[..., Any]] = Registry("adapter")

_MOCK_FLAG = {"pas": "use_mock_pas", "claims": "use_mock_claims", "recon": "use_mock_recon"}


def resolve_adapter(kind: str, settings: Settings) -> Callable[..., Any]:
    """Mock -> live is a config flag (guardrail 10): returns the adapter factory for pas|claims|recon."""
    if kind not in _MOCK_FLAG:
        raise RegistryError(f"unknown adapter kind '{kind}'; expected one of {sorted(_MOCK_FLAG)}")
    variant = "mock" if getattr(settings.app.flags, _MOCK_FLAG[kind]) else "live"
    return ADAPTERS.get(f"{kind}:{variant}")


def get_pack(name: str, settings: Settings | None = None) -> Pack:
    """Active pack is already loaded in settings; others are loaded on demand from packs/<name>/."""
    if settings and settings.pack.meta.name == name:
        return settings.pack
    root = settings.root if settings else None
    pack = load_pack(name, root)
    assert pack is not None
    return pack


def list_packs(settings: Settings | None = None) -> list[str]:
    return available_packs(settings.root if settings else None)
