"""The audit trail (T8.3): an append-only, hash-chained record of everything that changed a claim, enough to reconstruct any decision from the trail alone.

    trail = AuditTrail("var/audit-trail.jsonl")          # one JSON line per record; reloaded (and verified) on start
    trail.append(claim_id, "state_patch", {...})          # the only write: records are never changed or removed
    trail.verify()                                       # [] when the chain is intact; otherwise what was tampered with
    trail.reconstruct(claim_id)                          # the ClaimState rebuilt from the trail ONLY (base state + every patch, in order)

Record kinds (all written by the supervisor and the runner):
  claim_created   the claim state at the start of its first assessment (the base for a replay): the evidence it starts with
  state_patch     one change of the state: who made it (`agent_id`), the values written (`updates`, the agent's full output), the version it produced, the rationale, whether the agent was a
                  MOCK, the digest of its inputs, and the model calls made while it ran (model, prompt name and version, tokens, replayed or live)
  event           every run event (human decisions with the reviewer, role, reason and edits; commits and their calls; denied attempts; skipped nodes; failures)
Each record carries the hash of the previous record and its own hash over its content, so removing or editing any record breaks the chain at that point. That detects tampering inside the
file; protecting the file itself (write-once storage, signing) is a deployment matter, listed in docs/REPLACING_MOCK_ADAPTERS.md's neighbour, docs/orchestration.md.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic_core import to_jsonable_python

from claimtower.core.schemas import ClaimState

GENESIS = "0" * 64


@dataclass(frozen=True)
class TrailRecord:
    seq: int
    ts: str
    claim_id: str
    kind: str
    data: dict[str, Any]
    prev_hash: str
    hash: str


def _digest(seq: int, ts: str, claim_id: str, kind: str, data: Any, prev_hash: str) -> str:
    body = json.dumps({"seq": seq, "ts": ts, "claim_id": claim_id, "kind": kind, "data": data, "prev": prev_hash}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class AuditTrail:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._records: list[TrailRecord] = []
        self._lock = threading.RLock()
        if self.path is not None and self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    self._records.append(TrailRecord(**d))
        elif self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ write (append only)
    def append(self, claim_id: str, kind: str, data: dict[str, Any]) -> TrailRecord:
        with self._lock:
            prev = self._records[-1].hash if self._records else GENESIS
            seq, ts = len(self._records) + 1, datetime.now(UTC).isoformat()
            clean = to_jsonable_python(data)
            rec = TrailRecord(seq, ts, claim_id, kind, clean, prev, _digest(seq, ts, claim_id, kind, clean, prev))
            self._records.append(rec)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(json.dumps(rec.__dict__, ensure_ascii=False, sort_keys=True) + "\n")
            return rec

    # ------------------------------------------------------------------ read
    def records(self, claim_id: str | None = None, kind: str | None = None) -> list[TrailRecord]:
        with self._lock:
            return [r for r in self._records if (claim_id is None or r.claim_id == claim_id) and (kind is None or r.kind == kind)]

    def __len__(self) -> int:
        return len(self._records)

    @property
    def head(self) -> str:
        return self._records[-1].hash if self._records else GENESIS

    def verify(self, records: list[TrailRecord] | None = None) -> list[str]:
        """Problems with the chain ([] = intact): a record that was edited, removed, reordered or inserted."""
        recs = self._records if records is None else records
        problems: list[str] = []
        prev = GENESIS
        for i, r in enumerate(recs, start=1):
            if r.seq != i:
                problems.append(f"record {i}: sequence number is {r.seq} (a record was removed or inserted)")
            if r.prev_hash != prev:
                problems.append(f"record {r.seq}: does not follow the record before it (the chain is broken here)")
            if _digest(r.seq, r.ts, r.claim_id, r.kind, r.data, r.prev_hash) != r.hash:
                problems.append(f"record {r.seq} ({r.kind}): its content does not match its hash (it was edited)")
            prev = r.hash
        return problems

    # ------------------------------------------------------------------ reconstruction from the trail alone
    def reconstruct(self, claim_id: str, version: int | None = None) -> ClaimState:
        """The claim state as of `version` (default: the latest), rebuilt from the base state and the patches in the trail."""
        recs = self.records(claim_id)
        base = next((r for r in recs if r.kind == "claim_created"), None)
        if base is None:
            raise KeyError(f"the trail has no claim_created record for {claim_id}")
        data = dict(base.data["state"])
        for r in recs:
            if r.kind != "state_patch":
                continue
            if version is not None and r.data["version"] > version:
                break
            data.update(r.data["updates"])
            data["state_version"] = r.data["version"]
        return ClaimState.model_validate(data)

    def versions(self, claim_id: str) -> list[int]:
        return [r.data["version"] for r in self.records(claim_id, "state_patch")]


__all__ = ["AuditTrail", "GENESIS", "TrailRecord"]
