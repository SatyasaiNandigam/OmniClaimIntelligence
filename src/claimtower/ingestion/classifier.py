"""Document type classifier (T2.2): what kind of document is this, why, and when it must go to a human.

    clf = HybridClassifier(RuleClassifier(pack.document_types), llm=None)
    result = clf.classify(kind="pdf", pages=store.pages(doc_id), original=lambda: store.original(doc_id))
    result.type, result.confidence, result.evidence_snippet, result.needs_human

Principles (BUILD_PLAN T2.2, guardrails 5 and 9)
  * JUSTIFIED: every answer carries the verbatim line it was based on (`evidence_snippet`, with its page) and the per-type scores.
  * NEVER GUESSED: when the score is low, two types are too close, the page has no text yet (scan awaiting OCR), or the file is unreadable,
    the answer is `unknown` and `needs_human` is True. The classifier never outputs "other": a human assigns that.
  * Signals are data (packs/*/document_types.yaml), thresholds are SAMPLE values; nothing document-specific is hard-coded here.
  * Confidence is a heuristic over score and margin - NOT a probability, and NOT a coverage probability.
        confidence = score * (0.7 + 0.3 * min(1, margin / 0.4)),  capped for OCR text and LLM verdicts.
  * E-mail is recognised from the container (headers), photos from the image itself. An optional LLM tie-breaker may answer only when the
    rules abstained, and is accepted only if its quoted evidence is verifiably in the document and its type is allowed.
"""
from __future__ import annotations

import io
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from PIL import Image, ImageStat
from pydantic import BaseModel, Field

from claimtower.core.config import DocTypeSpec, DocumentTypes, Pack
from claimtower.core.llm import LLMClient, LLMError
from claimtower.ingestion.loaders import normalise

UNKNOWN = "unknown"
MIN_TEXT_CHARS = 20
LLM_TEXT_CHARS = 4000
Method = Literal["rules", "container", "visual", "llm", "human"]
_LABEL_LINE = re.compile(r"^[^:]{1,60}:\s*\S")             # "Label: value" lines are fields, not headings


class PageLike(Protocol):
    number: int
    text: str
    source: str
    needs_ocr: bool


@dataclass(frozen=True)
class Classification:
    type: str                                   # a pack type, or "unknown" when abstaining
    confidence: float                           # 0 when unknown
    evidence_snippet: str | None                # verbatim line from the document (None for photos / unknown)
    page: int | None
    method: Method
    code: str          # ok | needs_ocr | no_text | unreadable | low_score | ambiguous | no_heading | uncorroborated | llm_*
    reason: str                                 # human-readable justification or abstention reason
    scores: dict[str, float] = field(default_factory=dict)
    runner_up: str | None = None

    @property
    def abstained(self) -> bool:
        return self.type == UNKNOWN

    @property
    def needs_human(self) -> bool:
        return self.abstained


def _unknown(code: str, reason: str, scores: dict[str, float] | None = None, runner_up: str | None = None) -> Classification:
    return Classification(UNKNOWN, 0.0, None, None, "rules", code, reason, scores or {}, runner_up)


# ---------------------------------------------------------------------------------------------------- visual check for images
def image_page_metrics(data: bytes) -> dict[str, float]:
    """Cheap tonal statistics of the first frame: how much is paper-white, how much is ink-dark, how colourful."""
    img = Image.open(io.BytesIO(data))
    img.seek(0)
    img = img.convert("RGB")
    img.thumbnail((160, 160))
    gray = img.convert("L")
    hist, n = gray.histogram(), gray.width * gray.height
    return {"bright": sum(hist[216:]) / n, "dark": sum(hist[:100]) / n,
            "saturation": ImageStat.Stat(img.convert("HSV").split()[1]).mean[0] / 255}


def looks_like_page(m: dict[str, float]) -> bool:
    """A scanned page: mostly paper-white, a little dark ink, hardly any colour. Photographs fail at least one of these."""
    return m["bright"] >= 0.55 and 0.002 <= m["dark"] <= 0.30 and m["saturation"] <= 0.15


# ---------------------------------------------------------------------------------------------------- rule classifier
class RuleClassifier:
    def __init__(self, document_types: DocumentTypes) -> None:
        self.cfg = document_types.classifier
        self.types = document_types.types
        self._title = {t: [re.compile(p, re.IGNORECASE) for p in s.title_patterns] for t, s in self.types.items()}
        self._structure = {t: [re.compile(p, re.IGNORECASE) for p in s.structure_patterns] for t, s in self.types.items()}
        self._phrase = {t: [(p, self._phrase_re(p)) for p in s.phrases] for t, s in self.types.items()}
        self._negative = {t: [self._phrase_re(p) for p in s.negative_phrases] for t, s in self.types.items()}

    @staticmethod
    def _phrase_re(phrase: str) -> re.Pattern[str]:
        return re.compile(r"(?<![A-Za-z0-9])" + re.escape(phrase) + r"(?![A-Za-z0-9])", re.IGNORECASE)

    def text_types(self) -> list[str]:
        """Types that have text signals (the ones a text classifier, rule or LLM, may answer)."""
        return [t for t, s in self.types.items() if s.title_patterns or s.phrases or s.structure_patterns]

    def classify(self, *, kind: str | None, pages: Sequence[PageLike], original: Callable[[], bytes] | None = None,
                 load_error: str | None = None) -> Classification:
        if load_error:
            return _unknown("unreadable", f"file could not be read ({load_error})")
        if kind == "email":
            return self._email(pages)
        text_pages = [p for p in pages if len(re.sub(r"\s", "", p.text)) >= MIN_TEXT_CHARS][: self.cfg.pages_scanned]
        if kind == "image" and not text_pages:
            return self._image(original)
        if not text_pages:
            if any(p.needs_ocr for p in pages):
                return _unknown("needs_ocr", "no text layer: the page is a scan or image and needs OCR before it can be classified")
            return _unknown("no_text", "the document contains no readable text")
        return self._by_text(text_pages)

    # -- container / visual
    def _email(self, pages: Sequence[PageLike]) -> Classification:
        if "email" not in self.types:
            return _unknown("no_type", "e-mail is not an allowed type in this pack")
        first = next((ln.strip() for ln in (pages[0].text if pages else "").splitlines() if ln.lower().startswith("subject:")), None)
        return Classification("email", self.cfg.container_confidence, first, 1, "container", "ok", "message headers identify an e-mail")

    def _image(self, original: Callable[[], bytes] | None) -> Classification:
        if original is None:
            return _unknown("needs_ocr", "image without text and no original available to inspect")
        m = image_page_metrics(original())
        if looks_like_page(m):
            return _unknown("needs_ocr", f"image looks like a scanned page (paper-white {m['bright']:.0%}, "
                                         f"ink {m['dark']:.1%}); run OCR first")
        if "photo" not in self.types:
            return _unknown("no_type", "photo is not an allowed type in this pack")
        return Classification("photo", self.cfg.photo_confidence, None, 1, "visual", "ok",
                              f"no text and not page-like (paper-white {m['bright']:.0%}, colour {m['saturation']:.0%}): a photograph")

    # -- text
    def _heading_lines(self, text: str) -> list[str]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()][: self.cfg.title_zone_lines]
        return [ln for ln in lines if len(ln) <= self.cfg.title_max_chars and not _LABEL_LINE.match(ln)]

    def _by_text(self, pages: Sequence[PageLike]) -> Classification:
        cfg = self.cfg
        headings = self._heading_lines(pages[0].text)
        body = [(p.number, [ln.strip() for ln in p.text.splitlines() if ln.strip()]) for p in pages]
        top_lines = [ln for ln in (ln.strip() for ln in pages[0].text.splitlines()) if ln][: cfg.title_zone_lines]
        scores: dict[str, float] = {}
        evidence: dict[str, tuple[str, int]] = {}
        has_heading: dict[str, bool] = {}
        phrase_hits: dict[str, int] = {}
        for t in self.text_types():
            spec: DocTypeSpec = self.types[t]
            title_line = next((h for h in headings if any(r.search(h) for r in self._title[t])), None)
            structure = [next((ln for ln in top_lines if rx.search(ln)), None) for rx in self._structure[t]]
            structure_hits = [ln for ln in structure if ln]
            if not title_line and self._structure[t] and len(structure_hits) >= spec.structure_needed:
                title_line = structure_hits[0]                          # a header block (e.g. From/To/Subject) is heading-level evidence
            has_heading[t] = title_line is not None
            hits = []
            for phrase, rx in self._phrase[t]:
                found = next(((ln, num) for num, lines in body for ln in lines if rx.search(ln)), None)
                if found:
                    hits.append(found)
            phrase_hits[t] = len(hits)
            phrase_score = min(1.0, len(hits) / spec.phrases_needed) if spec.phrases else 0.0
            negative = any(rx.search(ln) for rx in self._negative[t] for _, lines in body for ln in lines)
            scores[t] = round(max(0.0, cfg.title_weight * (1.0 if title_line else 0.0) + cfg.phrase_weight * phrase_score
                                  - (cfg.negative_penalty if negative else 0.0)), 4)
            if title_line:
                evidence[t] = (title_line, pages[0].number)
            elif hits:
                evidence[t] = hits[0]
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        (best, top), (second, runner) = ranked[0], (ranked[1] if len(ranked) > 1 else (None, 0.0))
        if top < cfg.min_score:
            return _unknown("low_score", f"no type scored high enough (best {best} {top:.2f} < {cfg.min_score:.2f})",
                            scores, best)
        if top - runner < cfg.min_margin:
            return _unknown("ambiguous", f"ambiguous: {best} {top:.2f} vs {second} {runner:.2f} "
                                         f"(margin < {cfg.min_margin:.2f})", scores, best)
        if not has_heading[best] and not cfg.answer_without_heading:
            return _unknown("no_heading", f"only supporting phrases for {best} (score {top:.2f}) and no heading or header evidence: "
                                          "vocabulary alone is a guess", scores, best)
        if has_heading[best] and phrase_hits[best] < cfg.min_phrase_hits and self.types[best].phrases:
            snippet, _ = evidence[best]
            return _unknown("uncorroborated", f"heading '{snippet}' suggests {best} but nothing else in the text supports it "
                                              f"({phrase_hits[best]} of {cfg.min_phrase_hits} required supporting phrases)", scores, best)
        confidence = top * (0.7 + 0.3 * min(1.0, (top - runner) / 0.4))
        if any(p.source == "ocr" for p in pages):
            confidence = min(confidence, cfg.ocr_confidence_cap)
        snippet, page = evidence[best]
        how = "heading or header block" if has_heading[best] else "supporting phrase"
        return Classification(best, round(min(confidence, 0.99), 3), snippet, page, "rules", "ok",
                              f"{how} on page {page}: '{snippet}' (score {top:.2f}, runner-up {second} {runner:.2f})", scores, second)


# ---------------------------------------------------------------------------------------------------- optional LLM tie-breaker
class LLMVerdict(BaseModel):
    document_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_snippet: str = ""


class LLMClassifier:
    """Asked only when the rules abstain. Its answer is checked, not trusted: allowed type, verbatim evidence, enough confidence."""

    def __init__(self, llm: LLMClient, rules: RuleClassifier, pack: Pack) -> None:
        self._llm, self._rules, self._pack = llm, rules, pack

    def classify_document(self, pages: Sequence[PageLike]) -> Classification:
        """The model alone decides the type of a text document (no rule answer is shown to it). Its answer is checked: an allowed type, a verbatim evidence line, enough confidence.
        code `llm_unavailable` means the model could not answer at all (the caller may then fall back to the rules); any other abstention is the model's own judgement."""
        from claimtower.core.llm import load_prompt

        cfg = self._rules.cfg
        allowed = [t for t in self._rules.types if t not in ("other", "email")]
        described = "\n".join(f"- {t}: {self._rules.types[t].title}" for t in allowed)
        text_pages = [p for p in pages if p.text.strip()][: cfg.pages_scanned]
        full = normalise(" ".join(p.text for p in text_pages))
        try:
            verdict = self._llm.structured(LLMVerdict, load_prompt("classifier", self._pack, types=described, text=full[:LLM_TEXT_CHARS]))
        except Exception as e:                                            # model down, bad JSON, TBD model name ...
            return _unknown("llm_unavailable", f"the model could not answer ({type(e).__name__}: {e})")
        t, snippet = verdict.document_type.strip(), normalise(verdict.evidence_snippet)
        if snippet not in full:                                           # a quote padded over several lines: keep its longest line that IS the document's own text
            lines = sorted((normalise(x) for x in verdict.evidence_snippet.splitlines()), key=len, reverse=True)
            snippet = next((x for x in lines if len(x) >= 8 and x in full), snippet)
        problem = None
        if t == UNKNOWN:
            problem = "the model answered unknown"
        elif t not in allowed:
            problem = f"type '{t}' is not allowed"
        elif len(snippet) < 8 or snippet not in full:
            problem = "its evidence snippet is not verbatim text from the document"
        elif verdict.confidence < cfg.llm_min_confidence:
            problem = f"confidence {verdict.confidence:.2f} below {cfg.llm_min_confidence:.2f}"
        if problem:
            return Classification(UNKNOWN, 0.0, None, None, "llm", "llm_rejected", f"model verdict not used: {problem}")
        page = next((p.number for p in text_pages if snippet in normalise(p.text)), text_pages[0].number)
        return Classification(t, round(min(verdict.confidence, cfg.llm_confidence_cap), 3), snippet, page, "llm", "ok",
                              f"the model chose {t} citing '{snippet}'")

    def classify(self, pages: Sequence[PageLike], rules_result: Classification) -> Classification:
        from claimtower.core.llm import load_prompt                      # local: avoids loading the prompt machinery unless needed
        cfg, allowed = self._rules.cfg, [t for t in self._rules.text_types() if t != "other"]
        text_pages = [p for p in pages if p.text.strip()][: cfg.pages_scanned]
        full = normalise(" ".join(p.text for p in text_pages))
        try:
            prompt = load_prompt("classifier", self._pack, types=", ".join(allowed), text=full[:LLM_TEXT_CHARS])
            verdict = self._llm.structured(LLMVerdict, prompt)
        except Exception as e:                                            # model down, bad JSON, TBD model name ...
            return _unknown("llm_unavailable", f"{rules_result.reason}; LLM tie-breaker unavailable ({type(e).__name__}: {e})",
                            rules_result.scores, rules_result.runner_up)
        t, snippet = verdict.document_type.strip(), normalise(verdict.evidence_snippet)
        problem = None
        if t == UNKNOWN:
            problem = "the model also answered unknown"
        elif t not in allowed:
            problem = f"type '{t}' is not allowed"
        elif len(snippet) < 8 or snippet not in full:
            problem = "its evidence snippet is not verbatim text from the document"
        elif verdict.confidence < cfg.llm_min_confidence:
            problem = f"confidence {verdict.confidence:.2f} below {cfg.llm_min_confidence:.2f}"
        if problem:
            return _unknown("llm_rejected", f"{rules_result.reason}; LLM verdict rejected: {problem}",
                            rules_result.scores, rules_result.runner_up)
        page = next((p.number for p in text_pages if snippet in normalise(p.text)), text_pages[0].number)
        confidence = round(min(verdict.confidence, cfg.llm_confidence_cap), 3)
        return Classification(t, confidence, verdict.evidence_snippet.strip(), page, "llm", "ok",
                              f"rules abstained ({rules_result.code}); LLM chose {t} citing '{snippet}'",
                              rules_result.scores, rules_result.runner_up)


class HybridClassifier:
    """With a model: the MODEL decides the type of every text document, on its own (no rule answer or score is shown to it); the rules answer only for what is not a content
    judgement (an e-mail by its container, a photo by its image, a scan awaiting OCR, an unreadable file) and as the fallback when the model cannot be reached.
    Without a model: the rules alone, abstaining when unsure."""

    ASK_LLM_FOR = {"low_score", "ambiguous", "no_heading", "uncorroborated"}

    def __init__(self, rules: RuleClassifier, llm: LLMClassifier | None = None) -> None:
        self.rules, self.llm = rules, llm

    def classify(self, *, kind: str | None, pages: Sequence[PageLike], original: Callable[[], bytes] | None = None,
                 load_error: str | None = None) -> Classification:
        result = self.rules.classify(kind=kind, pages=pages, original=original, load_error=load_error)
        if self.llm is None:
            return result
        if result.method in ("container", "visual") or result.code in ("needs_ocr", "no_text", "unreadable", "no_type"):
            return result                                                          # not a content judgement: the file format or the image decides
        verdict = self.llm.classify_document(pages)
        return result if verdict.code == "llm_unavailable" else verdict


__all__ = ["Classification", "HybridClassifier", "LLMClassifier", "LLMError", "LLMVerdict", "RuleClassifier", "UNKNOWN",
           "image_page_metrics", "looks_like_page"]
