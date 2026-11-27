"""Mock claims system (T6.5): create_claim, update_status, create_task, get_claim, idempotent on the key. SYNTHETIC; in memory; every response says `mock=True`.

    claims = MockClaimsAdapter()
    ack = claims.create_claim(ClaimCreateRequest(claim_header=h, idempotency_key="CLM-1:create"))      # a replay returns the SAME claim with created=False

Also `MockPaymentsLedger`: the payments already made, read from the synthetic prior-claims file (the Settlement agent's `prior_payments_total` and remaining limit).
The real adapter replaces both behind the same methods (docs/REPLACING_MOCK_ADAPTERS.md).
"""
from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from claimtower.core.schemas import ClaimAck, ClaimCreateRequest, ClaimHeader, ClaimRecord, ClaimStatusUpdate, TaskAck, TaskRequest
from claimtower.core.schemas.common import ResolutionStatus
from claimtower.integrations.base import AdapterError
from claimtower.integrations.mock.data import CLAIMS_FILE, load_prior_claims

SYSTEM = "CLAIMS-MOCK"


def _now() -> datetime:
    return datetime.now(UTC)


class MockClaimsAdapter:
    mock = True

    def __init__(self) -> None:
        self._claims: dict[str, tuple[ResolutionStatus, ClaimHeader]] = {}
        self._by_key: dict[str, object] = {}                       # idempotency key -> the first answer
        self._tasks: dict[str, TaskAck] = {}
        self.calls: list[tuple[str, str]] = []                      # (operation, idempotency key): what the system was asked, replays included

    def _base(self, correlation_id: str = "") -> dict:
        return {"mock": True, "source_system": SYSTEM, "retrieved_at": _now(), "correlation_id": correlation_id}

    def create_claim(self, request: ClaimCreateRequest) -> ClaimAck:
        self.calls.append(("create_claim", request.idempotency_key))
        first = self._by_key.get(request.idempotency_key)
        if first is not None:
            assert isinstance(first, ClaimAck)
            return first.model_copy(update={"created": False, **self._base(request.correlation_id)})
        cid = request.claim_header.claim_id
        if cid in self._claims:                                     # the same claim under a different key: a second claim is never created
            return ClaimAck(claim_id=cid, status=self._claims[cid][0], created=False, **self._base(request.correlation_id))
        self._claims[cid] = ("registered", request.claim_header)
        ack = ClaimAck(claim_id=cid, status="registered", created=True, **self._base(request.correlation_id))
        self._by_key[request.idempotency_key] = ack
        return ack

    def update_status(self, update: ClaimStatusUpdate) -> ClaimAck:
        self.calls.append(("update_status", update.idempotency_key))
        first = self._by_key.get(update.idempotency_key)
        if first is not None:
            assert isinstance(first, ClaimAck)
            return first.model_copy(update={"created": False, **self._base(update.correlation_id)})
        if update.claim_id not in self._claims:
            raise AdapterError(f"{SYSTEM}: unknown claim '{update.claim_id}'")
        self._claims[update.claim_id] = (update.status, self._claims[update.claim_id][1])
        ack = ClaimAck(claim_id=update.claim_id, status=update.status, created=True, **self._base(update.correlation_id))
        self._by_key[update.idempotency_key] = ack
        return ack

    def create_task(self, request: TaskRequest) -> TaskAck:
        self.calls.append(("create_task", request.idempotency_key))
        first = self._tasks.get(request.idempotency_key)
        if first is not None:
            return first.model_copy(update={"created": False, **self._base(request.correlation_id)})
        if request.claim_id not in self._claims:
            raise AdapterError(f"{SYSTEM}: unknown claim '{request.claim_id}'")
        ack = TaskAck(task_id=f"TASK-{len(self._tasks) + 1:04d}", claim_id=request.claim_id, created=True, **self._base(request.correlation_id))
        self._tasks[request.idempotency_key] = ack
        return ack

    def get_claim(self, claim_id: str) -> ClaimRecord:
        if claim_id not in self._claims:
            raise AdapterError(f"{SYSTEM}: unknown claim '{claim_id}'")
        status, header = self._claims[claim_id]
        return ClaimRecord(claim_id=claim_id, status=status, claim_header=header, **self._base())

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    @property
    def claim_count(self) -> int:
        return len(self._claims)


class MockPaymentsLedger:
    mock = True

    def __init__(self, path: Path = CLAIMS_FILE) -> None:
        self._file = load_prior_claims(path)

    def paid_on_claim(self, claim_id: str) -> Decimal:
        return sum((p.amount for p in self._file.payments if p.claim_id == claim_id), start=Decimal(0))

    def paid_against_policy(self, policy_no: str) -> Decimal:
        return Decimal(self._file.paid_against_policy(policy_no))
