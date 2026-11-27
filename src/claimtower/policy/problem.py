"""`last_problem`: why the last model reading gave nothing, kept per THREAD so parallel reads never mix their reasons."""
from __future__ import annotations

import threading
from typing import Any


class ThreadLocalProblem:
    """A class attribute that behaves like `self.last_problem: str | None`, but holds a separate value in each thread."""

    def __set_name__(self, owner: type, name: str) -> None:
        self._key = f"_thread_local_{name}"

    def _local(self, obj: Any) -> threading.local:
        loc = obj.__dict__.get(self._key)
        return loc if loc is not None else obj.__dict__.setdefault(self._key, threading.local())

    def __get__(self, obj: Any, owner: type | None = None) -> Any:
        return self if obj is None else getattr(self._local(obj), "value", None)

    def __set__(self, obj: Any, value: str | None) -> None:
        self._local(obj).value = value
