"""Typed value parsers (T2.3): dates, money, identifiers. Pure functions, no model involved, Decimal not float.

Every parser returns the MATCHED TEXT (so the value can be shown next to its evidence) and a normalised value, plus flags when the
reading involved a judgement (e.g. '03/04/2026' could be 3 April or 4 March: day-first is assumed and flagged ambiguous).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation

_MONTH = (r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
          r"nov(?:ember)?|dec(?:ember)?)")
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_DMY_TEXT = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?[\s\-/.,]*" + _MONTH + r"\.?[\s\-/.,]*(\d{4})\b", re.IGNORECASE)
_MDY_TEXT = re.compile(r"\b" + _MONTH + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.IGNORECASE)
_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_NUMERIC = re.compile(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})\b")


@dataclass(frozen=True)
class ParsedDate:
    value: date
    text: str                       # the matched text, e.g. '14 March 2026'
    start: int
    end: int
    ambiguous: bool = False
    flags: tuple[str, ...] = ()

    @property
    def iso(self) -> str:
        return self.value.isoformat()


def _mk(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def find_dates(text: str) -> list[ParsedDate]:
    """All dates in `text`, in reading order, without overlaps. Invalid calendar dates (31 February) are ignored, never repaired."""
    found: list[ParsedDate] = []
    for m in _ISO.finditer(text):
        v = _mk(int(m[1]), int(m[2]), int(m[3]))
        if v:
            found.append(ParsedDate(v, m[0], m.start(), m.end()))
    for m in _DMY_TEXT.finditer(text):
        v = _mk(int(m[3]), _MONTHS[m[2][:3].lower()], int(m[1]))
        if v:
            found.append(ParsedDate(v, m[0], m.start(), m.end()))
    for m in _MDY_TEXT.finditer(text):
        v = _mk(int(m[3]), _MONTHS[m[1][:3].lower()], int(m[2]))
        if v:
            found.append(ParsedDate(v, m[0], m.start(), m.end()))
    for m in _NUMERIC.finditer(text):
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        flags: list[str] = []
        if y < 100:
            y += 2000
            flags.append("two_digit_year")
        day, month = a, b                                  # day-first is the convention assumed (India)
        if a <= 12 < b:                                    # 03/25/2026 can only be month-first
            day, month = b, a
            flags.append("assumed_month_first")
        v = _mk(y, month, day)
        if v:
            ambiguous = bool(flags) or (a <= 12 and b <= 12 and a != b)
            if a <= 12 and b <= 12 and a != b:
                flags.append("ambiguous_day_month")
            found.append(ParsedDate(v, m[0], m.start(), m.end(), ambiguous, tuple(flags)))
    found.sort(key=lambda d: (d.start, -(d.end - d.start)))
    out: list[ParsedDate] = []
    for d in found:
        if not out or d.start >= out[-1].end:
            out.append(d)
    return out


_DASHES = "-" + chr(0x2212) + chr(0x2013) + chr(0x2014)
_RANGE_WORDS = {"to", "-", chr(0x2013), chr(0x2014), "and", "or", "until", "till", "through", "thru", "~", "/"}
_NUMBER_RANGE = re.compile(   # '45,000-60,000', '40k-60k', '40 to 60': two numbers (the first may carry a unit) joined by a range word
    r"\d[\d,]*(?:\.\d+)?\s*[A-Za-z]{0,7}\s*(?:-|" + chr(0x2013) + r"|to|or|and|~)\s*[^\d\s]{0,4}\s*\d[\d,]*", re.IGNORECASE)
# a day (with an optional month) joined to a second day+month: '3-5 May', '3 May to 5 May', 'night of 3/4 May'
_JOIN = r"\s*(?:-|" + chr(0x2013) + r"|to|until|till|through|and|or|/)\s*"
_DAY_TO_DAY = re.compile(
    r"\b\d{1,2}(?:st|nd|rd|th)?(?:\s+" + _MONTH + r")?" + _JOIN + r"\d{1,2}(?:st|nd|rd|th)?\s+" + _MONTH, re.IGNORECASE)


def date_is_range(text: str) -> bool:
    """'between 3 and 5 May 2026', '3-5 May 2026', 'from 3 May to 5 May 2026': there is no single date, so none must be picked."""
    if _DAY_TO_DAY.search(text):
        return True
    dates = find_dates(text)
    return any(text[a.end:b.start].strip().lower() in _RANGE_WORDS for a, b in zip(dates, dates[1:]))


def parse_date(text: str) -> ParsedDate | None:
    dates = find_dates(text)
    return dates[0] if dates else None


def parse_date_range(text: str) -> tuple[ParsedDate, ParsedDate, str] | None:
    """'1 January 2026 to 31 December 2026' -> (start, end, matched text). Needs two dates joined by to / until / - / through."""
    dates = find_dates(text)
    for a, b in zip(dates, dates[1:]):
        between = text[a.end:b.start].strip().lower()
        if re.fullmatch(r"(to|until|till|through|thru|-|–|—|and)", between):
            return a, b, text[a.start:b.end]
    return None


# ---------------------------------------------------------------------------------------------------------------- money
_CURRENCY = {"inr": "INR", "rs": "INR", "rs.": "INR", "₹": "INR", "usd": "USD", "us$": "USD", "$": "USD", "eur": "EUR", "€": "EUR",
             "gbp": "GBP", "£": "GBP"}
_CUR = r"(?:INR|Rs\.?|₹|USD|US\$|\$|EUR|€|GBP|£)"
_MONEY = re.compile(
    r"(?P<cur1>" + _CUR + r")?\s*(?P<neg>[-−(])?\s*(?P<cur2>" + _CUR + r")?\s*"
    r"(?P<num>\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)\)?\s*"
    r"(?P<unit>lakhs?|lacs?|crores?|cr\b|k\b|mn\b|m\b|million\b)?\s*(?P<cur3>(?:INR|USD|EUR|GBP)\b)?", re.IGNORECASE)
_UNIT = {"lakh": Decimal(100000), "lac": Decimal(100000), "crore": Decimal(10000000), "cr": Decimal(10000000), "k": Decimal(1000),
         "m": Decimal(1000000), "mn": Decimal(1000000), "million": Decimal(1000000)}


@dataclass(frozen=True)
class ParsedMoney:
    amount: Decimal
    currency: str | None
    text: str
    start: int
    end: int
    flags: tuple[str, ...] = field(default=())


def plain(amount: Decimal) -> str:
    """Decimal -> plain string without exponent or trailing zeros: 98000, 1250.5."""
    s = format(amount, "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


def find_money(text: str) -> list[ParsedMoney]:
    out = []
    for m in _MONEY.finditer(text):
        num = m["num"]
        cur_raw = (m["cur1"] or m["cur2"] or m["cur3"] or "").lower()          # a currency code may also FOLLOW the amount: '60,000 INR'
        # a bare number glued to letters or part of an identifier is not money
        before, after = text[: m.start("num")], text[m.end():]
        if not cur_raw and (re.search(r"[A-Za-z0-9/]-?$", before) or re.match(r"[A-Za-z]", after)):
            continue
        try:
            amount = Decimal(num.replace(",", ""))
        except InvalidOperation:
            continue
        flags: list[str] = []
        unit = (m["unit"] or "").lower().rstrip("s")
        if unit:
            amount *= _UNIT[unit]
            flags.append("unit_scaled")
        elif text[m.end("num"): m.end("num") + 1].isalpha():   # '40abc': letters glued to the digits are an unknown unit, never ignored
            flags.append("unrecognised_suffix")
        if m["neg"]:
            amount = -amount
        currency = _CURRENCY.get(cur_raw) if cur_raw else None
        if currency is None:
            flags.append("no_currency")
        out.append(ParsedMoney(amount, currency, m.group(0).strip(), m.start(), m.end(), tuple(flags)))
    return out


def money_is_range(text: str) -> bool:
    """'INR 45,000 to INR 60,000', 'INR 45,000 - 60,000', '45,000-60,000': two amounts joined by a range word are not one amount."""
    found = find_money(text)
    if any(text[a.end:b.start].strip().lower() in _RANGE_WORDS or (text[a.end:b.start].strip() == "" and b.text[:1] in _DASHES)
           for a, b in zip(found, found[1:])):
        return True
    masked = text
    for d in find_dates(text):                                   # '2026-05-05' contains a dash but is a date, not an amount range
        masked = masked[: d.start] + " " * (d.end - d.start) + masked[d.end:]
    return bool(_NUMBER_RANGE.search(masked))


def parse_money(text: str) -> ParsedMoney | None:
    found = find_money(text)
    return found[0] if found else None


def whole_line_money(text: str) -> ParsedMoney | None:
    """The text is ONLY an amount (e.g. 'INR 110,000'): used to recognise table rows. Extra words mean it is prose, not a row."""
    m = parse_money(text.strip())
    return m if m and m.start == 0 and m.end >= len(text.strip()) else None


# ---------------------------------------------------------------------------------------------------------------- identifiers / text
def parse_id(text: str, pattern: re.Pattern[str]) -> str | None:
    m = pattern.search(text)
    return m.group(0) if m else None


def collapse(text: str) -> str:
    return " ".join(text.split())
