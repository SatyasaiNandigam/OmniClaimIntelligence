"""Checkpointer factory (T6.1). The prototype keeps checkpoints in memory (ADR-0001: a persistent saver is a separate package, to be added and pinned
when the API needs claims to survive a restart). Everything that needs a checkpointer takes one as an argument, so this is the only place to change.

Our contract models travel through the checkpoint, so they are REGISTERED with the serializer (`allowed_msgpack_modules`): LangGraph will block
deserialising a type it was not told about, and an unregistered contract class in a state would otherwise fail at the first resume, not at the first save.
"""
from __future__ import annotations

import inspect
import sys
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from claimtower.core import schemas


def contract_types() -> list[tuple[str, str]]:
    """(module, class) of every contract model a ClaimState can hold: all the models of the modules that `claimtower.core.schemas` exports from."""
    modules = {obj.__module__ for obj in vars(schemas).values() if inspect.isclass(obj) and issubclass(obj, BaseModel)}
    out = {(m, name) for m in modules for name, obj in vars(sys.modules[m]).items()
           if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj.__module__ == m}
    return sorted(out)


def make_checkpointer(kind: str = "memory") -> Any:
    if kind == "memory":
        return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=contract_types()))
    raise ValueError(f"unknown checkpointer '{kind}': only 'memory' exists in the prototype (a persistent one is a later task)")
