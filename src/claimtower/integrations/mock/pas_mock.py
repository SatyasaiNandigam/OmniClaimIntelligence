"""Mock PAS adapter, read path only (T6.5, first cut): the policy system the Policy agent asks "which policy version applies on this date?".

It answers from the synthetic database (data/synthetic/pas_mock_db.json, SYNTHETIC) and is the only PAS the prototype has: every answer is a
MOCK (guardrail 10). The real adapter replaces it behind the same method, `get_policy_version(policy_no, as_of)`;
the write adapters (Claims, Recon) are T6.5-T6.6.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from claimtower.core.schemas import PolicyVersionResponse
from claimtower.integrations.mock.data import PAS_FILE, load_pas_db


class MockPasAdapter:
    mock = True

    def __init__(self, path: Path = PAS_FILE) -> None:
        self._db = load_pas_db(path)

    def get_policy_version(self, policy_no: str, as_of: date) -> PolicyVersionResponse:
        return self._db.lookup(policy_no, as_of)
