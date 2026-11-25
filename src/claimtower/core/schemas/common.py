"""Shared base model, enums and small value types used across all contracts."""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    """Base for every contract model: unknown fields are rejected so drift is caught early."""

    model_config = ConfigDict(extra="forbid")


Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
Money = Decimal  # never float (guardrail 1); serialised as string in JSON

# --- shared enums (Literals keep the JSON Schema readable) ---
QuestionStatus = Literal[
    "unverified", "possible", "missing", "partially_established", "established", "requires_validation"
]
QuestionCategory = Literal["coverage_condition", "cause", "timing", "extent", "amount"]
GapType = Literal["missing", "contradictory", "insufficient"]
Severity = Literal["none", "low", "medium", "high"]
IssueStatus = Literal[
    "established", "identified", "unresolved", "potentially_relevant", "incomplete", "abstain"
]
Tier = Literal["simple", "medium", "complex"]
ExecutionMode = Literal["draft", "controlled_api"]
ActorType = Literal["agent", "human", "system"]
ResolutionStatus = Literal[
    "registered", "assessing", "awaiting_evidence", "awaiting_human", "approved", "paid", "closed"
]


class BBox(Contract):
    """Region on a page/image, normalised 0..1 in page coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float


__all__ = [
    "Contract", "Confidence", "Money", "BBox", "date", "datetime", "Decimal",
    "QuestionStatus", "QuestionCategory", "GapType", "Severity", "IssueStatus", "Tier",
    "ExecutionMode", "ActorType", "ResolutionStatus",
]
