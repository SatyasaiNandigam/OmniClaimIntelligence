"""Mock PAS adapter, read only (T6.5): the policy system the Policy agent asks "which policy version applies on this date?" and "what is its status?".

It answers from the synthetic database (data/synthetic/pas_mock_db.json, SYNTHETIC) and is the only PAS the prototype has: every answer is a
MOCK (guardrail 10). The real adapter replaces it behind the same methods (docs/REPLACING_MOCK_ADAPTERS.md).
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from claimtower.core.schemas import PolicyStatusResponse, PolicyVersionResponse
from claimtower.integrations.mock.data import PAS_EXTRA_FILE, PAS_FILE, load_pas_db


class MockPasAdapter:
    mock = True

    def __init__(self, path: Path = PAS_FILE, extra: Path | None = None) -> None:
        if extra is None and path == PAS_FILE:                       # the default database also reads your own policies, when there are any
            extra = Path(os.environ.get("CLAIMTOWER_PAS_EXTRA") or PAS_EXTRA_FILE)
        self._db = load_pas_db(path, extra)

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
