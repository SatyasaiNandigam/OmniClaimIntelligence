"""Interface signatures between workstreams (BUILD_PLAN section 11). CONTRACT - frozen at M0.

Producers implement these (returning fixture data first, real logic later); consumers depend only on these signatures.
Changing a signature after M0 requires Tech-lead sign-off (see docs/contracts/CHANGELOG.md and scripts/freeze_contracts.py).
"""
from __future__ import annotations

from datetime import date
from typing import Protocol

from claimtower.core.schemas import (
    AuditEvent, ClaimState, ClauseCitation, DocumentRecord, Fact, ImageFinding, NotInForce, PolicySnapshot,
    RoutingDecision, SettlementInputs, SettlementResult,
)


class FactExtractor(Protocol):
    """WS2 -> WS4. Every returned Fact has verbatim, page-anchored provenance."""

    def extract_facts(self, documents: list[DocumentRecord]) -> list[Fact]: ...


class ImageAnalyser(Protocol):
    """WS2 -> WS4/WS5. Visible observations only; never cause, date or coverage."""

    def analyse_images(self, images: list[DocumentRecord]) -> list[ImageFinding]: ...


class PolicyService(Protocol):
    """WS3 -> WS4/WS5."""

    def get_snapshot(self, policy_no: str, loss_date: date) -> PolicySnapshot | NotInForce: ...

    def retrieve_clauses(self, query: str, snapshot: PolicySnapshot) -> list[ClauseCitation]: ...


class SettlementEngine(Protocol):
    """WS5 -> WS6. Deterministic; no LLM. Returns cannot_calculate (with missing_inputs) instead of guessing."""

    def compute_settlement(self, inputs: SettlementInputs, rules_version: str) -> SettlementResult: ...


class Router(Protocol):
    """WS5 -> WS6. Rules-driven (configs/routing_rules.yaml); extraction confidence is not an input."""

    def route(self, state: ClaimState) -> RoutingDecision: ...


class AuditLog(Protocol):
    """All -> WS8. Append-only."""

    def record(self, event: AuditEvent) -> None: ...
