"""Money handling for the settlement engine: Decimal only, floats rejected at the boundary (guardrail 1)."""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext

PRECISION = 60          # digits; far beyond any realistic amount, so intermediate arithmetic is exact


class InvalidAmount(ValueError):
    """An amount is a float, non-numeric, non-finite or negative. Never guessed or coerced."""


def to_money(value: object, name: str = "amount") -> Decimal:
    """int, str or Decimal -> Decimal. Floats (and bools) are refused because they carry binary rounding error."""
    if isinstance(value, bool) or isinstance(value, float):
        raise InvalidAmount(f"{name}: {type(value).__name__} is not allowed for money; pass a Decimal, int or str")
    try:
        d = value if isinstance(value, Decimal) else Decimal(value)  # type: ignore[arg-type]
    except (InvalidOperation, TypeError, ValueError):
        raise InvalidAmount(f"{name}: {value!r} is not a valid amount") from None
    if not d.is_finite():
        raise InvalidAmount(f"{name}: {value!r} is not finite")
    return d


def non_negative(value: object, name: str = "amount") -> Decimal:
    d = to_money(value, name)
    if d < 0:
        raise InvalidAmount(f"{name}: {d} is negative")
    return d


def quantize(amount: Decimal, places: int) -> Decimal:
    """Round half-up to `places` decimal places."""
    with localcontext() as ctx:
        ctx.prec = PRECISION
        return amount.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
