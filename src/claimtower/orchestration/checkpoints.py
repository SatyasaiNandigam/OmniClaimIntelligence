"""Checkpointer factory (T6.1). The prototype keeps checkpoints in memory (ADR-0001); `file` keeps the same tables on disk so claims survive a restart. Everything that needs a checkpointer takes one as an argument, so this is the only place to change.

Our contract models travel through the checkpoint, so they are REGISTERED with the serializer (`allowed_msgpack_modules`): LangGraph will block
deserialising a type it was not told about, and an unregistered contract class in a state would otherwise fail at the first resume, not at the first save.
"""
from __future__ import annotations

import base64
import builtins
import inspect
import json
import sys
import threading
from collections import defaultdict
from pathlib import Path
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


def _enc(x: Any) -> Any:
    """The checkpoint tables as JSON: bytes, tuples and dicts with any key type keep their type (tagged), so what is read back is exactly what was written."""
    if isinstance(x, (bytes, bytearray)):
        return {"$b": base64.b64encode(bytes(x)).decode("ascii")}
    if isinstance(x, tuple):
        return {"$t": [_enc(i) for i in x]}
    if isinstance(x, dict):
        return {"$d": [[_enc(k), _enc(v)] for k, v in x.items()]}
    if isinstance(x, builtins.list):
        return [_enc(i) for i in x]
    return x


def _dec(x: Any) -> Any:
    if isinstance(x, builtins.list):
        return [_dec(i) for i in x]
    if isinstance(x, dict):
        if "$b" in x:
            return base64.b64decode(x["$b"])
        if "$t" in x:
            return tuple(_dec(i) for i in x["$t"])
        if "$d" in x:
            return {_dec(k): _dec(v) for k, v in x["$d"]}
    return x


class FileCheckpointSaver(InMemorySaver):
    """The in-memory checkpointer, kept on disk: one file per claim (`<directory>/<claim id>.json`), read back when the server starts with the same folder. The graph works on the in-memory
    tables as before; `save(claim_id)` writes that claim's slice (the runner calls it when a run, a decision or a re-assessment ends), so a claim that is waiting at the human gate is still
    waiting after a restart. A lock makes reading a claim safe while another claim is being assessed."""

    def __init__(self, directory: str | Path, serde: Any | None = None) -> None:
        super().__init__(serde=serde or JsonPlusSerializer(allowed_msgpack_modules=contract_types()))
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._guard = threading.RLock()
        self._load()

    # ------------------------------------------------------------------ the file
    def thread_ids(self) -> builtins.list[str]:
        with self._guard:
            return sorted(self.storage)

    def save(self, thread_id: str) -> None:
        with self._guard:
            doc = {"storage": _enc({ns: dict(cps) for ns, cps in self.storage.get(thread_id, {}).items()}),
                   "writes": _enc({k: v for k, v in self.writes.items() if k[0] == thread_id}),
                   "blobs": _enc({k: v for k, v in self.blobs.items() if k[0] == thread_id})}
            path = self.directory / f"{thread_id}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8", newline="\n")
            tmp.replace(path)                                                    # the file is never half-written

    def _load(self) -> None:
        for f in sorted(self.directory.glob("*.json")):
            try:
                doc = json.loads(f.read_text(encoding="utf-8"))
                storage, writes, blobs = _dec(doc["storage"]), _dec(doc["writes"]), _dec(doc["blobs"])
            except (OSError, ValueError, KeyError):
                continue                                                         # an unreadable file is skipped: that claim is simply not restored
            self.storage[f.stem] = defaultdict(dict, storage)
            self.writes.update(writes)
            self.blobs.update(blobs)

    # ------------------------------------------------------------------ the same tables, under a lock
    def put(self, *a: Any, **k: Any) -> Any:
        with self._guard:
            return super().put(*a, **k)

    def put_writes(self, *a: Any, **k: Any) -> Any:
        with self._guard:
            return super().put_writes(*a, **k)

    def get_tuple(self, *a: Any, **k: Any) -> Any:
        with self._guard:
            return super().get_tuple(*a, **k)

    def list(self, *a: Any, **k: Any) -> Any:
        with self._guard:
            return iter(builtins.list(super().list(*a, **k)))


def make_checkpointer(kind: str = "memory", directory: str | Path | None = None) -> Any:
    if kind == "memory":
        return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=contract_types()))
    if kind == "file":
        if directory is None:
            raise ValueError("the file checkpointer needs a directory")
        return FileCheckpointSaver(directory)
    raise ValueError(f"unknown checkpointer '{kind}': 'memory' or 'file'")
