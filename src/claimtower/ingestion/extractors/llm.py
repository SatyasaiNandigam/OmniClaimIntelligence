"""Optional LLM tie-breaker for REQUIRED fields the rules could not find (T2.3). Checked, never trusted.

An answer is accepted only if ALL of these hold:
  * the field is one the pack declares for this document type, and was actually missing
  * its `evidence_snippet` is verbatim text from a line of the document (so it can be located and shown)
  * for typed fields (date / money / id / date range) the value parses with the same deterministic parsers, and the parsed text
    appears inside the snippet
Accepted values are capped at `llm_confidence_cap`, flagged `llm`, and carry the page / region / verbatim line of the snippet.
If the model is unavailable or the answer is invalid, the field stays missing and a warning is recorded: never a guess.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, Field

from claimtower.core.config import Pack
from claimtower.core.llm import LLMClient, load_prompt
from claimtower.ingestion.extractors.fields import RuleExtractor, is_low
from claimtower.ingestion.extractors.values import plain
from claimtower.ingestion.extractors.lines import Line, PageLike, logical_lines
from claimtower.ingestion.extractors.models import ExtractedField, ExtractionResult, LineItem, TableResult
from claimtower.ingestion.loaders import normalise

LLM_TEXT_CHARS = 5000


class LLMAnswer(BaseModel):
    value: str
    evidence_snippet: str
    confidence: float = Field(ge=0.0, le=1.0)


class LLMFields(BaseModel):
    fields: dict[str, LLMAnswer | None]


class TableLine(BaseModel):
    line: str
    description: str = ""
    amount: str


class TableAnswer(BaseModel):
    items: list[TableLine] = []
    total: TableLine | None = None


class LLMFieldFiller:
    def __init__(self, llm: LLMClient, rules: RuleExtractor, pack: Pack) -> None:
        self._llm, self._rules, self._pack = llm, rules, pack

    def fill(self, result: ExtractionResult, *, kind: str | None, pages: Sequence[PageLike],
             original: Callable[[], bytes] | None) -> ExtractionResult:
        cfg, spec = self._rules.cfg, self._rules.types[result.document_type]
        wanted = [n for n, fs in spec.fields.items() if result.field(n) is None and not fs.from_body and (n in result.missing_required or fs.type in ("money", "date", "text", "id"))]   # any declared field the rules did not find, not only the required ones
        if not wanted:
            return result
        lines = logical_lines(pages, kind=kind, original=original, cfg=cfg)
        text = "\n".join(ln.text for ln in lines)[:LLM_TEXT_CHARS]
        described = "\n".join(f"- {n}: {spec.fields[n].type}; usually labelled " + ", ".join(spec.fields[n].labels[:3]) for n in wanted)
        try:
            prompt = load_prompt("field_extraction", self._pack, doc_type=result.document_type, fields=described, text=text)
            answer = self._llm.structured(LLMFields, prompt)
        except Exception as e:                                          # model down, bad JSON, TBD model name ...
            return result.model_copy(update={"warnings": [*result.warnings, f"LLM field filler unavailable ({type(e).__name__}: {e})"]})
        fields, still_missing, warnings = list(result.fields), [], list(result.warnings)
        for name in wanted:
            got = answer.fields.get(name)
            built, why = self._accept(result.document_type, name, got, lines) if got else (None, "the model returned null")
            if built:
                fields.append(built)
            else:
                still_missing.append(name)
                warnings.append(f"LLM answer for '{name}' not used: {why}")
        order = list(spec.fields)
        fields.sort(key=lambda f: order.index(f.name) if f.name in order else len(order))
        low = [*result.low_confidence_fields, *[f.name for f in fields if f.low_confidence and f.name not in result.low_confidence_fields]]
        missing = [m for m in result.missing_required if m in still_missing or m not in wanted]
        return result.model_copy(update={"fields": fields, "missing_required": missing, "low_confidence_fields": low, "warnings": warnings})

    def _accept(self, doc_type: str, name: str, a: LLMAnswer, lines: list[Line]) -> tuple[ExtractedField | None, str]:
        fs = self._rules.types[doc_type].fields[name]
        snippet = normalise(a.evidence_snippet)
        line = next((ln for ln in lines if len(snippet) >= 4 and snippet in normalise(ln.text)), None)
        if line is None:
            return None, "its evidence snippet is not verbatim text from the document"
        flags: list[str] = ["llm"]
        value, normalised, currency, ambiguous, ok = self._rules._typed(doc_type, name, fs, a.value, flags)
        if not ok:
            return None, f"value '{a.value}' does not parse as {fs.type}"
        if fs.type not in ("text", "narrative") and normalise(value) not in snippet:
            return None, "the parsed value does not appear inside its own evidence snippet"
        confidence = min(a.confidence, self._rules.cfg.llm_confidence_cap)
        if ambiguous:
            confidence *= self._rules.cfg.ambiguous_factor
        confidence = round(confidence, 3)
        low = is_low(confidence, flags, self._rules.cfg.low_confidence)
        return ExtractedField(name=name, value=value, normalised=normalised, currency=currency, label="", page=line.page, region=line.bbox,
                              evidence_text=line.text, confidence=confidence, low_confidence=low, method="llm", flags=flags), ""


def _amount(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace(",", "").replace("INR", "").replace("Rs.", "").replace("₹", "").strip())
    except InvalidOperation:
        return None


def fill_table(filler: LLMFieldFiller, result: ExtractionResult, *, kind: str | None, pages: Sequence[PageLike],
               original: Callable[[], bytes] | None) -> ExtractionResult:
    """The rows and the stated total of an estimate or invoice the rules could not read, by the model. Every row must be a verbatim line of the document and its amount must
    appear in that row; the arithmetic check (sum of rows against the stated total) is done here in code. Anything else is dropped and the table stays as the rules left it."""
    cfg = filler._rules.cfg
    lines = logical_lines(pages, kind=kind, original=original, cfg=cfg)
    text = "\n".join(ln.text for ln in lines)[:LLM_TEXT_CHARS * 2]
    try:
        answer = filler._llm.structured(TableAnswer, load_prompt("table_extraction", filler._pack, doc_type=result.document_type, text=text))
    except Exception as e:
        return result.model_copy(update={"warnings": [*result.warnings, f"LLM table reader unavailable ({type(e).__name__}: {e})"]})
    by_text = {normalise(ln.text): ln for ln in lines}
    currency = "INR" if any(w in text for w in ("INR", "Rs.", "₹")) else None
    items: list[LineItem] = []
    used: set[str] = set()
    warnings = list(result.warnings)

    def locate(row: TableLine) -> tuple[Line | None, Decimal | None]:
        key = normalise(row.line)
        ln = by_text.get(key) or next((v for k, v in by_text.items() if len(key) >= 8 and key in k), None)
        amount = _amount(row.amount)
        if ln is None or amount is None or normalise(row.amount) not in normalise(ln.text):
            return None, None
        return ln, amount

    for row in answer.items:
        ln, amount = locate(row)
        if ln is None or amount is None or ln.text in used:
            warnings.append(f"LLM table row not used: '{row.line[:50]}' is not a verbatim row of the document with that amount")
            continue
        used.add(ln.text)
        items.append(LineItem(description=normalise(row.description) or ln.text, amount=amount, currency=currency, page=ln.page, region=ln.bbox, evidence_text=ln.text,
                              confidence=0, low_confidence=False))
    if not items:
        return result.model_copy(update={"warnings": warnings})
    stated, total_line = None, None
    if answer.total is not None:
        total_line, stated = locate(answer.total)
        if total_line is None:
            warnings.append("LLM table total not used: not a verbatim row of the document with that amount")
    computed = sum((i.amount for i in items), Decimal(0))
    if stated is None:
        status = "no_total"
    else:
        status = "reconciled" if abs(stated - computed) <= cfg.table_tolerance else "mismatch"
    ok = status == "reconciled"
    confidence = round(min(cfg.llm_confidence_cap, cfg.label_confidence if ok else cfg.table_mismatch_confidence), 3)
    low = confidence < cfg.low_confidence
    final = [i.model_copy(update={"confidence": confidence, "low_confidence": low}) for i in items]
    fields = [f for f in result.fields if f.name != "total_amount"]
    if total_line is not None and stated is not None and answer.total is not None:
        fields.append(ExtractedField(name="total_amount", value=normalise(answer.total.amount), normalised=plain(stated), currency=currency, label="total", page=total_line.page,
                                     region=total_line.bbox, evidence_text=total_line.text, confidence=confidence, low_confidence=low, method="llm",
                                     flags=["llm"] + (["total_mismatch"] if status == "mismatch" else [])))
    table = TableResult(items=final, stated_total=stated, computed_total=computed, difference=(stated - computed) if stated is not None else None, currency=currency, status=status,
                        flags=["llm"] + (["no_total_row"] if status == "no_total" else []))
    return result.model_copy(update={"fields": fields, "table": table, "warnings": warnings, "low_confidence_fields": [*result.low_confidence_fields, *(["table"] if low else [])]})


class Finding(BaseModel):
    text: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class LLMEverything(BaseModel):
    fields: dict[str, LLMAnswer | None] = {}
    findings: dict[str, list[Finding]] = {}


MAX_DOC_CHARS = 9000


def _locate(snippet: str, lines: list[Line], page_norm: dict[int, str]) -> tuple[int, object, str] | None:
    """Where the model's quote is in the document: a single logical line (with its region), else the whitespace-collapsed page text. None = not verbatim anywhere."""
    s = normalise(snippet)
    if len(s) < 4:
        return None
    ln = next((x for x in lines if s in normalise(x.text)), None)
    if ln is not None:
        return ln.page, ln.bbox, ln.text
    for number, text in page_norm.items():
        if s in text:
            return number, None, s
    return None


def extract_all(filler: "LLMFieldFiller", result: ExtractionResult, *, kind: str | None, pages: Sequence[PageLike],
                original: Callable[[], bytes] | None) -> ExtractionResult:
    """Model-first extraction: the model reads the whole document and proposes every declared field and every finding; each answer is checked (verbatim quote, a value that
    parses and sits inside its quote, a capped confidence). The rules' reading is kept only for a field the model did not give (and as the whole answer when the model is down)."""
    cfg, spec = filler._rules.cfg, filler._rules.types[result.document_type]
    if not spec.fields:
        return result
    scalars = {n: fs for n, fs in spec.fields.items() if fs.type != "narrative"}
    narratives = {n: fs for n, fs in spec.fields.items() if fs.type == "narrative"}
    lines = logical_lines(pages, kind=kind, original=original, cfg=cfg)
    marked: list[str] = []
    for p in pages:                                                      # the page text itself (what a quote is checked against); table cells come one per line, in reading order
        marked.append(f"--- page {p.number} ---")
        marked.extend(x.strip() for x in p.text.splitlines() if x.strip())
    described = "\n".join(f"- {n}: {fs.type}" + (f"; {fs.description}" if fs.description else "") + ("; usually labelled " + ", ".join(fs.labels[:3]) if fs.labels else "") for n, fs in scalars.items()) or "(none)"
    wanted = "\n".join(f"- {n}: {fs.description or n.replace('_', ' ')}" for n, fs in narratives.items()) or "(none)"
    try:
        llm = filler._llm.derive(reasoning_effort="low") if hasattr(filler._llm, "derive") else filler._llm          # reading a whole document is worth a little more effort than the default
        answer = llm.structured(LLMEverything, load_prompt("document_extraction", filler._pack, doc_type=result.document_type, fields=described, findings=wanted,
                                                                  text="\n".join(marked)[:MAX_DOC_CHARS]))
    except Exception as e:                                              # model down, bad JSON ...: the rules' reading stands
        return result.model_copy(update={"warnings": [*result.warnings, f"LLM extraction unavailable ({type(e).__name__}: {e})"]})
    page_norm = {p.number: normalise(p.text) for p in pages}
    got: dict[str, ExtractedField] = {}
    warnings = list(result.warnings)
    for name in scalars:
        a = answer.fields.get(name)
        if a is None:
            if scalars[name].required and result.field(name) is None:
                warnings.append(f"LLM answer for '{name}' not used: the model returned null")
            continue
        built, why = _accept_anywhere(filler, result.document_type, name, a, lines, page_norm)
        if built is not None:
            got[name] = built
        else:
            warnings.append(f"LLM answer for '{name}' not used: {why}")
    for name in narratives:
        texts: list[str] = []
        confs: list[float] = []
        first_page = None
        for item in answer.findings.get(name) or []:
            t = normalise(item.text)
            page = next((n for n, pt in page_norm.items() if t in pt), None) if len(t) >= 8 else None
            if page is None or t in texts:
                if page is None and t:
                    warnings.append(f"LLM finding for '{name}' not used: not verbatim text of the document")
                continue
            texts.append(t)
            confs.append(item.confidence)
            first_page = first_page or page
        if texts:
            c = round(min(cfg.llm_confidence_cap, min(confs)), 3)
            got[name] = ExtractedField(name=name, value="\n".join(texts), normalised="\n".join(texts), label="", page=first_page or 1, region=None, evidence_text=texts[0],
                                       confidence=c, low_confidence=is_low(c, ["llm"], cfg.low_confidence), method="llm", flags=["llm", "findings_list"])
    fields = {**{f.name: f for f in result.fields}, **got}                  # the model's reading wins; the rules' reading stays for what the model did not give
    order = list(spec.fields)
    ordered = sorted(fields.values(), key=lambda f: order.index(f.name) if f.name in order else len(order))
    missing = [n for n, fs in spec.fields.items() if fs.required and n not in fields]
    low = sorted({*[f.name for f in ordered if f.low_confidence], *[n for n in result.low_confidence_fields if n not in got]})
    return result.model_copy(update={"fields": ordered, "missing_required": missing, "low_confidence_fields": low, "warnings": warnings})


def _accept_anywhere(filler: "LLMFieldFiller", doc_type: str, name: str, a: LLMAnswer, lines: list[Line], page_norm: dict[int, str]) -> tuple[ExtractedField | None, str]:
    rules, cfg = filler._rules, filler._rules.cfg
    fs = rules.types[doc_type].fields[name]
    found = _locate(a.evidence_snippet, lines, page_norm)
    if found is None:
        return None, "its evidence snippet is not verbatim text from the document"
    page, region, evidence = found
    snippet = normalise(a.evidence_snippet)
    flags: list[str] = ["llm"]
    value, normalised, currency, ambiguous, ok = rules._typed(doc_type, name, fs, a.value, flags)
    if not ok:
        return None, f"value '{a.value}' does not parse as {fs.type}"
    if fs.type not in ("text", "narrative") and normalise(value) not in snippet:
        return None, "the parsed value does not appear inside its own evidence snippet"
    confidence = min(a.confidence, cfg.llm_confidence_cap)
    if ambiguous:
        confidence *= cfg.ambiguous_factor
    confidence = round(confidence, 3)
    return ExtractedField(name=name, value=value, normalised=normalised, currency=currency, label="", page=page, region=region, evidence_text=evidence, confidence=confidence,
                          low_confidence=is_low(confidence, flags, cfg.low_confidence), method="llm", flags=flags), ""
