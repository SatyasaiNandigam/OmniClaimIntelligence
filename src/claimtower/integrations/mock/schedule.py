"""Policy schedule -> a policy record for the policy administration system (the mock PAS). Read by the model, checked, never trusted.

    reader = ScheduleReader(llm, pack)                # llm may be None: a fixed-pattern reading of the usual layout is used instead
    proposal = reader.read(pdf_bytes, "schedule.pdf") # ScheduleProposal: the record (or why there is none), a check per value, warnings

Every value the model reports must be text of the document: the policy number, each endorsement id, each amount (its digits), the period wording and each date (its year, day and month appear in that
wording). A value that is not found is DROPPED and listed as a problem or warning, never kept. Nothing is stored here: the proposal is shown to a person, who registers it (or edits it first).
A schedule does not state the policy's status, so the record says it is in force for the period and the note says that was assumed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pymupdf
from pydantic import BaseModel, Field

from claimtower.integrations.mock.data import EndorsementEntry, PasPolicy, PolicyVersion, StatusSegment

MIN_CONFIDENCE = 0.6
MAX_CHARS = 9000
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]


class ScheduleEndorsement(BaseModel):
    id: str = ""
    effective_from: str = ""


class ScheduleAmount(BaseModel):
    name: str = ""
    value: str = ""
    quote: str = ""


class ScheduleAnswer(BaseModel):
    policy_number: str = ""
    holder_name: str = ""
    period_text: str = ""
    period_start: str = ""
    period_end: str = ""
    base_wording_id: str = ""
    endorsements: list[ScheduleEndorsement] = []
    amounts: list[ScheduleAmount] = []
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    note: str = ""


@dataclass
class Check:
    field: str
    value: str
    found: bool
    note: str = ""


@dataclass
class ScheduleProposal:
    policy: PasPolicy | None
    checks: list[Check] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)          # why there is no record
    warnings: list[str] = field(default_factory=list)          # what to look at before registering
    source: str = "model"                                       # model | rules


def _norm(s: str) -> str:
    return " ".join(s.split())


def is_sum(s: str) -> bool:
    """A sum of money ('INR 3,200,000', '10,000', '₹4500000'), as opposed to words that happen to contain a number ('as per wording clause 5.1')."""
    return re.fullmatch(r"(?:INR|Rs\.?|₹)?\s*[0-9][0-9,]*(?:\.[0-9]+)?", s.strip(), re.I) is not None


def _digits(s: str) -> str:
    return re.sub(r"[^0-9]", "", s)


def parse_date(s: str) -> date | None:
    """'2026-01-31', '31 January 2026' or 'January 31, 2026' -> date, else None."""
    s = _norm(s).strip(" .,")
    try:
        return date.fromisoformat(s)
    except ValueError:
        pass
    day = mon = year = None
    m = re.match(r"^(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\.?,?\s+(\d{4})$", s)
    if m:
        day, mon, year = m.group(1), m.group(2), m.group(3)
    else:
        m = re.match(r"^([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})$", s)
        if m:
            mon, day, year = m.group(1), m.group(2), m.group(3)
    if not (day and mon and year):
        return None
    i = next((n for n, full in enumerate(MONTHS) if full == mon.lower() or full[:3] == mon.lower()[:3] and len(mon) <= 4), None)
    try:
        return date(int(year), i + 1, int(day)) if i is not None else None
    except ValueError:
        return None


def date_in_text(d: date, text: str) -> bool:
    """The date is written in the text: its year and day as numbers and its month by name or short name, or in a numeric form."""
    t = text.lower()
    if d.isoformat() in t or f"{d.day:02d}/{d.month:02d}/{d.year}" in t or f"{d.day}/{d.month}/{d.year}" in t:
        return True
    month = MONTHS[d.month - 1]
    return re.search(rf"\b0?{d.day}(?:st|nd|rd|th)?\b", t) is not None and str(d.year) in t and (month in t or month[:3] in t)


def _text_of(data: bytes) -> str:
    try:
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            return "\n".join(p.get_text() for p in pdf)
    except Exception as e:
        raise ValueError(f"this is not a readable PDF ({type(e).__name__}: {e})") from None


class ScheduleReader:
    def __init__(self, llm: Any | None, pack: Any | None = None) -> None:
        self._llm, self._pack = llm, pack

    def read(self, data: bytes, filename: str = "schedule.pdf") -> ScheduleProposal:
        text = _text_of(data)
        if not text.strip():
            return ScheduleProposal(None, problems=["the document has no text layer (a scan?): it cannot be read"])
        if self._llm is not None and self._pack is not None:
            from claimtower.core.llm import load_prompt

            try:
                ans = self._llm.structured(ScheduleAnswer, load_prompt("policy_schedule", self._pack, text=_norm(text)[:MAX_CHARS]))
            except Exception as e:
                return ScheduleProposal(None, problems=[f"the model could not be reached or answered badly ({type(e).__name__}: {e})"])
            return self.check(ans, text, filename, "model")
        return self.check(self._by_rules(text), text, filename, "rules")

    # ------------------------------------------------------------------ a fixed-pattern reading of the usual layout (used when there is no model)
    @staticmethod
    def _by_rules(text: str) -> ScheduleAnswer:
        flat = _norm(text)
        num = re.search(r"Policy (?:number|no\.?)\s*[:\-]?\s*([A-Z0-9][A-Z0-9/_-]{3,})", flat, re.I)
        per = re.search(r"Period of insurance\s*[:\-]?\s*(.+?)\s+to\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})", flat, re.I)
        holder = re.search(r"(?:Policyholder|Insured|Name of insured)\s*[:\-]\s*(.+?)(?=\s+(?:Policy|Period|Endorsements?|Building|Contents|Standard)\b|$)", flat, re.I)
        wording = re.search(r"(?:Policy wording|Base wording|Wording)\s*(?:reference|ref|id)?\s*[:\-]\s*([A-Z0-9][A-Z0-9._/-]{2,})", flat, re.I)
        ends = re.search(r"Endorsements? (?:attached|applying)\s*[:\-]\s*(.+?)(?=\s+(?:Building|Contents|Standard|Sum|Deductible)\b|$)", flat, re.I)
        amounts = [ScheduleAmount(name=re.sub(r"\W+", "_", n.strip().lower()).strip("_"), value=_digits(v), quote=f"{n.strip()}: INR {v}")
                   for n, v in re.findall(r"\b([A-Za-z]+ (?:limit|deductible)|sum insured)\s*[:\-]\s*(?:INR|Rs\.?)?\s*([\d,]+)", flat, re.I)]
        start = parse_date(per.group(1)) if per else None
        end = parse_date(per.group(2)) if per else None
        return ScheduleAnswer(policy_number=num.group(1) if num else "", holder_name=holder.group(1).strip() if holder else "",
                              period_text=f"{per.group(1)} to {per.group(2)}" if per else "", period_start=start.isoformat() if start else "", period_end=end.isoformat() if end else "",
                              base_wording_id=wording.group(1) if wording else "",
                              endorsements=[ScheduleEndorsement(id=e) for e in re.findall(r"\b[A-Z]{1,6}-?\d{1,3}\b", ends.group(1))] if ends else [], amounts=amounts, confidence=0.7)

    # ------------------------------------------------------------------ the checks: nothing the model says is kept unless the document says it too
    def check(self, a: ScheduleAnswer, text: str, filename: str, source: str) -> ScheduleProposal:
        flat = _norm(text)
        low = flat.lower()
        out = ScheduleProposal(None, source=source)

        def grounded(label: str, value: str) -> bool:
            ok = bool(value.strip()) and _norm(value).lower() in low
            out.checks.append(Check(label, value, ok, "" if ok else "not found in the document"))
            return ok

        if a.confidence < MIN_CONFIDENCE:
            out.problems.append(f"the reading is not reliable enough (confidence {a.confidence:.2f}): {a.note or 'no reason given'}")
        number = _norm(a.policy_number)
        if not grounded("policy number", number):
            out.problems.append("the policy number was not found in the document")
        start, end = parse_date(a.period_start), parse_date(a.period_end)
        period_ok = bool(a.period_text.strip()) and _norm(a.period_text).lower() in low
        out.checks.append(Check("period", a.period_text, period_ok, "" if period_ok else "the period wording is not in the document"))
        if not period_ok or start is None or end is None or not (date_in_text(start, a.period_text) and date_in_text(end, a.period_text)):
            out.problems.append("the period of insurance could not be confirmed from the document")
        elif end < start:
            out.problems.append("the period ends before it starts")
        holder = _norm(a.holder_name) if grounded("holder", _norm(a.holder_name)) else ""
        if not holder:
            out.warnings.append("the policyholder is not stated in the document")
        wording = _norm(a.base_wording_id) if _norm(a.base_wording_id) and grounded("base wording", _norm(a.base_wording_id)) else ""
        if not wording:
            out.warnings.append("the schedule does not name the base wording: enter its id before registering")
        ends: list[EndorsementEntry] = []
        for e in a.endorsements:
            if not grounded("endorsement", _norm(e.id)):
                out.warnings.append(f"endorsement '{e.id}' was reported but is not in the document: left out")
                continue
            eff = parse_date(e.effective_from) if e.effective_from else None
            if e.effective_from and (eff is None or not date_in_text(eff, flat)):
                out.warnings.append(f"the date reported for {e.id} is not in the document: the start of the period is used")
                eff = None
            ends.append(EndorsementEntry(endorsement_id=_norm(e.id), effective_from=eff or start or date.min))
        schedule: dict[str, str] = {}
        for m in a.amounts:
            name, raw = re.sub(r"\W+", "_", m.name.strip().lower()).strip("_"), _norm(m.value)
            if not name or not raw:
                continue
            if is_sum(raw):                                      # a sum: kept as digits, and the digits must be in the document
                digits = _digits(raw)
                ok = any(_digits(n) == digits for n in re.findall(r"\d[\d,]*", flat))
                value, shown = digits, digits
            else:                                                # words ("as per wording clause 5.1"): kept as written, and the words must be in the document
                ok = raw.lower() in low
                value, shown = raw, raw
            out.checks.append(Check(name, shown, ok, "" if ok else "not found in the document"))
            if ok:
                schedule[name] = value
            else:
                out.warnings.append(f"'{m.name} = {m.value}' was reported but is not in the document: left out")
        if "standard_deductible" not in schedule:
            out.warnings.append("the schedule states no standard deductible: the calculation will look for it in the wording")
        elif not is_sum(schedule["standard_deductible"]):
            out.warnings.append(f"the schedule gives the deductible in words ('{schedule['standard_deductible']}'): the calculation will read it from the wording")
        if out.problems or start is None or end is None:
            return out
        out.policy = PasPolicy(policy_no=number, holder_name=holder or "(not stated)",
                               versions=[PolicyVersion(version="v1", effective_from=start, effective_to=end, base_wording_id=wording, endorsements=ends, schedule=schedule)],
                               status_history=[StatusSegment(status="in_force", effective_from=start, effective_to=end)],
                               note=f"Read from {filename}. The status (in force for the period) is assumed: a schedule does not state it.")
        return out


__all__ = ["Check", "ScheduleAnswer", "ScheduleProposal", "ScheduleReader", "date_in_text", "parse_date"]
