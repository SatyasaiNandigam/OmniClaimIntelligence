"""Model-first extraction: the model reads the whole document and proposes every field and finding; every answer is checked; the rules only fill what it did not give."""
import json

from claimtower.core.config import load_settings
from claimtower.core.llm import LLMClient, LLMResponse
from claimtower.ingestion.extractors import HybridExtractor, LLMFieldFiller, RuleExtractor
from claimtower.ingestion.provenance import split_sentences

PACK = load_settings().pack
RULES = RuleExtractor(PACK.document_types)

PLUMBER = ("Plumbing Failure Inspection Report\nReport number\nSSP/1\nDate of attendance\n14 October 2026\n"
           "2. Observations at site\n1 The concealed pipe was found separated at a threaded elbow joint.\n"
           "2 Corrosion observed near the joint: heavy scaling and pitting on the pipe threads\n  and the adjoining 0.3 m of pipe.\n")


class Page:
    def __init__(self, n, text):
        self.number, self.text, self.source, self.needs_ocr, self.width, self.height = n, text, "text_layer", False, None, None


class FakeLLM:
    name = "fake"

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def complete(self, req):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return LLMResponse(text=json.dumps(self.reply), model="m", input_tokens=1, output_tokens=1)


def run(reply, text=PLUMBER, doc_type="plumber_report"):
    provider = FakeLLM(reply)
    hx = HybridExtractor(RULES, LLMFieldFiller(LLMClient(provider, text_model="t", vision_model="v"), RULES, PACK))
    return hx.extract(doc_type=doc_type, kind="text", pages=[Page(1, text)]), provider


GOOD = {"fields": {"inspection_date": {"value": "14 October 2026", "evidence_snippet": "14 October 2026", "confidence": 0.9}},
        "findings": {"findings": [{"text": "1 The concealed pipe was found separated at a threaded elbow joint.", "confidence": 0.9},
                                  {"text": "2 Corrosion observed near the joint: heavy scaling and pitting on the pipe threads and the adjoining 0.3 m of pipe.", "confidence": 0.9}]}}


def test_findings_that_wrap_over_lines_are_accepted_when_verbatim_in_the_page_text_and_become_one_fact_each():
    r, provider = run(GOOD)
    f = r.field("findings")
    assert f is not None and f.method == "llm" and "findings_list" in f.flags and provider.calls >= 1
    items = [s for piece in f.value.split("\n") for s in split_sentences(piece)]
    assert len(items) == 2 and "Corrosion observed near the joint" in items[1]
    assert r.field("inspection_date").normalised == "2026-10-14" and r.missing_required == []


def test_a_finding_the_document_never_said_is_dropped_with_a_warning():
    bad = {"fields": {}, "findings": {"findings": [{"text": "The pipe failed because of frost damage last winter, according to the plumber.", "confidence": 0.9}]}}
    r, _ = run(bad)
    assert r.field("findings") is None and "findings" in r.missing_required
    assert any("not verbatim" in w for w in r.warnings)


def test_a_value_must_parse_and_sit_inside_its_quote():
    reply = {"fields": {"inspection_date": {"value": "15 October 2026", "evidence_snippet": "14 October 2026", "confidence": 0.9}}, "findings": {}}
    r, _ = run(reply)
    assert r.field("inspection_date") is None or r.field("inspection_date").normalised != "2026-10-15"


def test_when_the_model_is_down_the_rules_reading_stands_with_a_warning():
    r, _ = run(TimeoutError("down"))
    assert any("unavailable" in w for w in r.warnings)
    assert r.status == "ok"
