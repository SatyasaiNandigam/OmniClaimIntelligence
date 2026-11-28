"""Human overrides of the settlement inputs (T6.3): what a reviewer changed at the gate with the "edit" decision, who changed it and why.

    store = OverrideStore()
    store.set("CLM-1", {"non_covered_items_total": Decimal(5000)}, by="rev-9", reason="the surveyor confirmed the upgrade")
    store.get("CLM-1")            # {"non_covered_items_total": Decimal("5000")}

The Settlement agent applies the overrides after it has extracted its inputs (an overridden input is labelled `human`, cited as the reviewer, and flagged for review), and the
engine still computes: a person changes an INPUT, never the payable. An override does not give anyone authority to release a payment (that is the commit step's rule).
Later edits replace earlier ones for the same input; every edit stays in `history` for the audit trail. In memory, or kept in a file when given a path.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OverrideRecord:
    claim_id: str
    values: dict[str, Decimal]
    by: str
    reason: str
    at: datetime = field(default_factory=lambda: datetime.now(UTC))


class OverrideStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self._history: list[OverrideRecord] = []
        self._lock = threading.RLock()
        self.path = Path(path) if path else None                        # with a path, the edits are kept on disk and read back at the next start
        if self.path is not None and self.path.is_file():
            for d in json.loads(self.path.read_text(encoding="utf-8")):
                self._history.append(OverrideRecord(d["claim_id"], {k: Decimal(v) for k, v in d["values"].items()}, d["by"], d["reason"], datetime.fromisoformat(d["at"])))

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"claim_id": r.claim_id, "values": {k: str(v) for k, v in r.values.items()}, "by": r.by, "reason": r.reason, "at": r.at.isoformat()} for r in self._history]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + chr(10), encoding="utf-8", newline=chr(10))
        tmp.replace(self.path)

    def set(self, claim_id: str, values: dict[str, Decimal], *, by: str, reason: str) -> None:
        if not by.strip() or not reason.strip():
            raise ValueError("an override needs the person who made it and a reason")
        with self._lock:
            last = self.last(claim_id)
            if last is not None and last.values == values and last.by == by.strip() and last.reason == reason.strip():
                return                                                    # the same edit replayed (the gate node is re-run on resume): recorded once
            self._history.append(OverrideRecord(claim_id, dict(values), by.strip(), reason.strip()))
            self._save()

    def get(self, claim_id: str) -> dict[str, Decimal]:
        with self._lock:
            out: dict[str, Decimal] = {}
            for r in self._history:
                if r.claim_id == claim_id:
                    out.update(r.values)
            return out

    def last(self, claim_id: str) -> OverrideRecord | None:
        with self._lock:
            return next((r for r in reversed(self._history) if r.claim_id == claim_id), None)

    def history(self, claim_id: str) -> list[OverrideRecord]:
        with self._lock:
            return [r for r in self._history if r.claim_id == claim_id]

    def fingerprint(self, claim_id: str) -> dict[str, Any]:
        """What the node's input digest must include: the overrides in force."""
        return {"human_overrides": {k: str(v) for k, v in sorted(self.get(claim_id).items())}}


__all__ = ["OverrideRecord", "OverrideStore"]
