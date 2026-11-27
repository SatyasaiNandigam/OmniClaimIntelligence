"""Mock reconciliation system (T6.5): payment instructions, match status, breaks. Idempotent on the key. SYNTHETIC; in memory; every response says `mock=True`.

    recon = MockReconAdapter()
    ack = recon.submit_payment_instruction(instruction, instruction.idempotency_key)        # replayed 5 times: still ONE record, later answers `duplicate_ignored`

An instruction is accepted only with a recorded approver and approval reference (there is no path to payment without a human decision); anything else is `rejected`.
A payment for the same claim and amount under a different key is not created twice either: it is reported as a break for a person to look at, never merged silently.
"""
from __future__ import annotations

from datetime import UTC, datetime

from claimtower.core.schemas import MatchStatus, PaymentInstruction, ReconAck, ReconBreak
from claimtower.integrations.base import AdapterError

SYSTEM = "RECON-MOCK"


class MockReconAdapter:
    mock = True

    def __init__(self) -> None:
        self._by_key: dict[str, ReconAck] = {}
        self._instructions: dict[str, PaymentInstruction] = {}       # recon id -> the instruction
        self._breaks: list[ReconBreak] = []
        self.calls: list[str] = []                                   # the keys the system was asked about, replays included

    def _base(self, correlation_id: str = "") -> dict:
        return {"mock": True, "source_system": SYSTEM, "retrieved_at": datetime.now(UTC), "correlation_id": correlation_id}

    def submit_payment_instruction(self, request: PaymentInstruction, idempotency_key: str) -> ReconAck:
        self.calls.append(idempotency_key)
        if idempotency_key != request.idempotency_key:
            raise AdapterError(f"{SYSTEM}: the idempotency key of the call does not match the one in the instruction")
        first = self._by_key.get(idempotency_key)
        if first is not None:
            return first.model_copy(update={"status": "duplicate_ignored", **self._base(request.correlation_id)})
        if not request.approved_by.strip() or not request.approval_ref.strip():
            ack = ReconAck(recon_id="", status="rejected", idempotency_key=idempotency_key, **self._base(request.correlation_id))
            self._by_key[idempotency_key] = ack
            return ack
        recon_id = f"RECON-{len(self._instructions) + 1:04d}"
        twin = next((i for i in self._instructions.values() if i.claim_id == request.claim_id and i.amount == request.amount), None)
        self._instructions[recon_id] = request
        if twin is not None:                                         # the same payment under another key: recorded, but flagged for a person
            self._breaks.append(ReconBreak(recon_id=recon_id, claim_id=request.claim_id, reason="possible duplicate payment: same claim and amount under another key",
                                           amount=request.amount, **self._base(request.correlation_id)))
        ack = ReconAck(recon_id=recon_id, status="accepted", idempotency_key=idempotency_key, **self._base(request.correlation_id))
        self._by_key[idempotency_key] = ack
        return ack

    def get_match_status(self, recon_id: str) -> MatchStatus:
        if recon_id not in self._instructions:
            raise AdapterError(f"{SYSTEM}: unknown reconciliation '{recon_id}'")
        broken = any(b.recon_id == recon_id for b in self._breaks)
        return MatchStatus(recon_id=recon_id, state="unmatched" if broken else "pending", detail="a break is open" if broken else "awaiting the bank statement", **self._base())

    def list_breaks(self, claim_id: str | None = None) -> list[ReconBreak]:
        return [b for b in self._breaks if claim_id is None or b.claim_id == claim_id]

    @property
    def instruction_count(self) -> int:
        return len(self._instructions)
