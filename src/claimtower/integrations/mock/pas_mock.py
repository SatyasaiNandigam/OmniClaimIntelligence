"""Mock PAS adapter, read only for the Policy agent (T6.5; policies can be registered into it for demos): the policy system the Policy agent asks "which policy version applies on this date?" and "what is its status?".

It answers from the synthetic database (data/synthetic/pas_mock_db.json, SYNTHETIC) and is the only PAS the prototype has: every answer is a
MOCK (guardrail 10). The real adapter replaces it behind the same methods (docs/REPLACING_MOCK_ADAPTERS.md).
"""
from __future__ import annotations

import json
import os
import threading
from datetime import date
from pathlib import Path
from typing import Any

from claimtower.core.schemas import PolicyStatusResponse, PolicyVersionResponse
from claimtower.integrations.mock.data import PAS_EXTRA_FILE, PAS_FILE, PasPolicy, load_pas_db


class PasRegisterError(ValueError):
    """The policy cannot be registered as given."""


class PasConflict(PasRegisterError):
    """The policy number already exists and `replace` was not asked for."""


class MockPasAdapter:
    mock = True

    def __init__(self, path: Path = PAS_FILE, extra: Path | None = None) -> None:
        if extra is None and path == PAS_FILE:                       # the default database also reads your own policies, when there are any
            extra = Path(os.environ.get("CLAIMTOWER_PAS_EXTRA") or PAS_EXTRA_FILE)
        self._path, self._extra = path, extra
        self._lock = threading.RLock()
        self._db = load_pas_db(path, extra)

    # ------------------------------------------------------------------ registering a policy (the mock only: a real PAS is read-only for us)
    def can_register(self) -> bool:
        return self._extra is not None

    def policies(self) -> list[dict[str, Any]]:
        """Every policy the mock knows: number, holder, period of the latest version, endorsements, and where it came from (synthetic data or registered through the API)."""
        with self._lock:
            registered = {p["policy_no"] for p in self._read_extra()["policies"]}
            out = []
            for p in self._db.policies:
                v = max(p.versions, key=lambda x: x.effective_from) if p.versions else None
                out.append({"policy_no": p.policy_no, "holder_name": p.holder_name, "versions": len(p.versions), "effective_from": v.effective_from if v else None,
                            "effective_to": v.effective_to if v else None, "base_wording_id": v.base_wording_id if v else "",
                            "endorsements": [e.endorsement_id for e in v.endorsements] if v else [], "schedule": dict(v.schedule) if v else {},
                            "source": "registered" if p.policy_no in registered else "synthetic", "note": p.note})
            return sorted(out, key=lambda r: r["policy_no"])

    def register_policy(self, policy: PasPolicy, replace: bool = False) -> str:
        """Add the policy to the file of registered policies and use it at once. `added` or `replaced`; a number that already exists (synthetic or registered) needs `replace`."""
        if self._extra is None:
            raise PasRegisterError("this mock PAS has no file for registered policies")
        if not policy.versions or not all(v.base_wording_id.strip() for v in policy.versions):
            raise PasRegisterError("every version needs the id of its base wording")
        with self._lock:
            existed = self._db.policy(policy.policy_no) is not None
            if existed and not replace:
                raise PasConflict(f"policy {policy.policy_no} already exists in the mock PAS; register it with replace=true to supersede it")
            raw = self._read_extra()
            raw["policies"] = [p for p in raw["policies"] if p.get("policy_no") != policy.policy_no] + [policy.model_dump(mode="json")]
            self._extra.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._extra.with_suffix(".tmp")
            tmp.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
            tmp.replace(self._extra)                                                         # the file is never half-written
            self._db = load_pas_db(self._path, self._extra)
            return "replaced" if existed else "added"

    def _read_extra(self) -> dict[str, Any]:
        if self._extra is not None and self._extra.is_file():
            return json.loads(self._extra.read_text(encoding="utf-8"))
        return {"policies": []}

    def get_policy_version(self, policy_no: str, as_of: date) -> PolicyVersionResponse:
        return self._db.lookup(policy_no, as_of)

    def get_status(self, policy_no: str, as_of: date) -> PolicyStatusResponse:
        r = self._db.lookup(policy_no, as_of)
        return PolicyStatusResponse(policy_no=r.policy_no, as_of=r.as_of, status=r.status, status_effective_from=r.status_effective_from, mock=True,
                                    source_system=r.source_system)

    def get_schedule(self, policy_no: str, as_of: date) -> dict[str, str] | None:
        """The schedule values (building limit, standard deductible) of the policy version in force on `as_of`, or None when there is none. SAMPLE values."""
        policy = self._db.policy(policy_no)
        if policy is None:
            return None
        v = next((v for v in policy.versions if v.effective_from <= as_of and (v.effective_to is None or as_of <= v.effective_to)), None)
        return dict(v.schedule) if v is not None else None
