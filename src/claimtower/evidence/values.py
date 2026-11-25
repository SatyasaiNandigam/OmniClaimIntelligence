"""Reading dates and amounts out of the wording of a fact (T4.3). Small on purpose: anything it cannot read returns None and the fact is
simply not compared by a rule (a rule never guesses a value). Day-first for numeric dates (assumption Q-D5), Indian grouping for amounts."""
from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation

_MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                                       "november", "december"], 1)}
_MONTHS |= {k[:3]: v for k, v in list(_MONTHS.items())} | {"sept": 9}
_DMY = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\.?,?\s+(\d{4})$")
_MDY = re.compile(r"^([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})$")
_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_NUM = re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$")
_AMOUNT = re.compile(r"(?:INR|Rs\.?|₹)?\s*(\d[\d,]*(?:\.\d+)?)\s*(lakhs?|lacs?|crores?)?", re.IGNORECASE)
_UNIT = {"lakh": Decimal(100_000), "lac": Decimal(100_000), "crore": Decimal(10_000_000)}


def parse_date(text: str) -> date | None:
    t = " ".join(text.strip().rstrip(".").split())
    try:
        if m := _ISO.match(t):
            return date(int(m[1]), int(m[2]), int(m[3]))
        if m := _NUM.match(t):
            return date(int(m[3]), int(m[2]), int(m[1]))
        if m := _DMY.match(t):
            return date(int(m[3]), _MONTHS[m[2].lower()], int(m[1]))
        if m := _MDY.match(t):
            return date(int(m[3]), _MONTHS[m[1].lower()], int(m[2]))
    except (ValueError, KeyError):
        return None
    return None


def parse_amount(text: str) -> Decimal | None:
    m = _AMOUNT.search(text)
    if not m:
        return None
    try:
        value = Decimal(m[1].replace(",", ""))
    except InvalidOperation:
        return None
    if m[2]:
        value *= _UNIT[m[2].lower().rstrip("s")]
    return value
