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

from pydantic import BaseModel, Field

from claimtower.core.config import Pack
from claimtower.core.llm import LLMClient, load_prompt
from claimtower.ingestion.extractors.fields import RuleExtractor, is_low
from claimtower.ingestion.extractors.lines import Line, PageLike, logical_lines
from claimtower.ingestion.extractors.models import ExtractedField, ExtractionResult
from claimtower.ingestion.loaders import normalise

LLM_TEXT_CHARS = 5000


class LLMAnswer(BaseModel):
    value: str
    evidence_snippet: str
    confidence: float = Field(ge=0.0, le=1.0)


class LLMFields(BaseModel):
    fields: dict[str, LLMAnswer | None]


class LLMFieldFiller:
    def __init__(self, llm: LLMClient, rules: RuleExtractor, pack: Pack) -> None:
        self._llm, self._rules, self._pack = llm, rules, pack

    def fill(self, result: ExtractionResult, *, kind: str | None, pages: Sequence[PageLike],
             original: Callable[[], bytes] | None) -> ExtractionResult:
        cfg, spec = self._rules.cfg, self._rules.types[result.document_type]
        wanted = [n for n in result.missing_required if n in spec.fields]
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
