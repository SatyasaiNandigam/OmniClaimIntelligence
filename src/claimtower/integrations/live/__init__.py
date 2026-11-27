"""Live adapters (T6.5): NOT implemented. Selecting them (a `use_mock_*` flag off) fails loudly instead of quietly falling back to a mock.

To implement one, write a class with the methods of the protocol in `claimtower.integrations.base` that talks to the real system, return responses with `mock=False`,
keep every write idempotent on its key, and register it in `claimtower.integrations.register_adapters` in place of the stub. docs/REPLACING_MOCK_ADAPTERS.md has the steps.
"""
from __future__ import annotations

from typing import Any

from claimtower.integrations.base import AdapterNotConfigured


def not_configured(kind: str) -> Any:
    def factory(*args: Any, **kwargs: Any) -> Any:
        raise AdapterNotConfigured(f"the live {kind.upper()} adapter is not implemented (the use_mock flag for '{kind}' is off). "
                                   "Turn the flag back on, or implement the adapter: docs/REPLACING_MOCK_ADAPTERS.md")
    return factory
