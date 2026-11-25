"""Live check of the configured model against our synthetic documents (needs LLM_API_KEY in .env).

    python evals/runners/live_llm_check.py            # run every experiment; new calls are recorded in the cassette
    python evals/runners/live_llm_check.py --only E1  # one experiment (E1 classifier, E2 field filler, E3 OCR, E4 facts)

Mode is "auto": a request already in data/cassettes/live_check.json is replayed (free, offline); a new one is sent and recorded.
The key is read by python-dotenv and never printed. Only synthetic documents are sent.
"""
import argparse
import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import pymupdf  # noqa: E402

from claimtower.core.config import load_settings  # noqa: E402
from claimtower.core.llm import Cassette, LLMClient  # noqa: E402
from claimtower.ingestion.classifier import HybridClassifier, LLMClassifier, RuleClassifier  # noqa: E402
from claimtower.ingestion.extractors import HybridExtractor, LLMFieldFiller, RuleExtractor  # noqa: E402
from claimtower.ingestion.ocr import VisionLLMOcr  # noqa: E402
from claimtower.ingestion.store import DocumentStore  # noqa: E402
from evals.metrics import classification as mc  # noqa: E402
from evals.metrics import extraction as me  # noqa: E402
from evals.metrics.provenance import evaluate_facts  # noqa: E402

CASSETTE = ROOT / "data" / "cassettes" / "live_check.json"


def norm(s: str) -> str:
    return " ".join(s.split())


def usage(llm: LLMClient, since: int) -> str:
    calls = llm.calls[since:]
    live = [c for c in calls if not c.replayed]
    return (f"{len(calls)} call(s), {len(calls) - len(live)} replayed; live tokens in/out "
            f"{sum(c.input_tokens or 0 for c in live)}/{sum(c.output_tokens or 0 for c in live)}, "
            f"live time {sum(c.latency_s for c in live):.0f}s")


def e1(settings, llm, rules):
    print("\n== E1 classifier: rules only vs rules + model tie-breaker ==")
    base = HybridClassifier(rules)
    hyb = HybridClassifier(rules, LLMClassifier(llm, rules, settings.pack))
    n0 = len(llm.calls)
    rb, rh = mc.evaluate(base, mc.all_cases()), mc.evaluate(hyb, mc.all_cases())
    for split in sorted(rb):
        a, b = rb[split], rh[split]
        print(f"{split:10s} n={a.n:3d}  rules: acc {a.accuracy:.1%} wrong {len(a.wrong_answers)} over-abstain {len(a.over_abstained)}"
              f"   +model: acc {b.accuracy:.1%} wrong {len(b.wrong_answers)} over-abstain {len(b.over_abstained)}")
        for o in b.wrong_answers:
            print(f"    WRONG   {o.case.id}: expected {o.case.expected}, got {o.got} ({o.confidence:.2f}) - {o.reason[:160]}")
        for o in b.over_abstained:
            print(f"    ABSTAIN {o.case.id}: expected {o.case.expected} [{o.code}] - {o.reason[-160:]}")
    print("usage:", usage(llm, n0))


def e2(settings, llm, rules_ex):
    print("\n== E2 field filler: rules only vs rules + model (required fields only) ==")
    dt = settings.pack.document_types
    base = HybridExtractor(RuleExtractor(dt))
    hyb = HybridExtractor(RuleExtractor(dt), LLMFieldFiller(llm, RuleExtractor(dt), settings.pack))
    n0 = len(llm.calls)
    rb, rh = me.evaluate(base, me.all_cases(), list(dt.types)), me.evaluate(hyb, me.all_cases(), list(dt.types))
    for split in sorted(rb):
        a, b = rb[split].summary(), rh[split].summary()
        print(f"{split:10s} fields {a['expected_fields']:3d}  rules: score {rb[split].field_score:.1%} wrong {len(a['wrong_values'])} "
              f"silent {a['silent_wrong']} missed {len(a['missed'])} spurious {len(a['spurious'])}"
              f"   +model: score {rh[split].field_score:.1%} wrong {len(b['wrong_values'])} silent {b['silent_wrong']} "
              f"missed {len(b['missed'])} spurious {len(b['spurious'])}")
        for key in ("wrong_values", "spurious"):
            for line in b[key]:
                if line not in a[key]:
                    print(f"    NEW {key.upper():12s} {line}")
    print("usage:", usage(llm, n0))


def e3(settings, llm, rules):
    print("\n== E3 vision OCR on the scanned, rotated surveyor report (CLM-0002) ==")
    base = ROOT / "data/synthetic/claims/CLM-0002"
    m = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    v = m["variants"][0]
    data = (base / v["filename"]).read_bytes()
    assert pymupdf.open(stream=data, filetype="pdf")[0].get_text().strip() == "", "scan must have no text layer"
    n0 = len(llm.calls)
    store = DocumentStore(Path(__import__("tempfile").mkdtemp()))
    dt = settings.pack.document_types
    classifier = HybridClassifier(rules)
    extractor = HybridExtractor(RuleExtractor(dt))
    rec = store.ingest("CLM-0002", v["filename"], data, document_id="DOC-LIVE-OCR", ocr=VisionLLMOcr(llm), classifier=classifier,
                       extractor=extractor)
    page = store.pages("DOC-LIVE-OCR")[0]
    expected = norm(" ".join(v["expected_text"]))
    got = norm(page.text)
    ratio = difflib.SequenceMatcher(None, expected, got).ratio()
    lines = [norm(x) for x in " ".join(v["expected_text"]).split("  ")]
    print(f"page source={page.source} chars expected {len(expected)} / got {len(got)}; similarity {ratio:.1%}")
    exp_words, got_words = set(expected.split()), set(got.split())
    print(f"words recovered {len(exp_words & got_words)}/{len(exp_words)}; invented words {len(got_words - exp_words)}: "
          f"{sorted(got_words - exp_words)[:12]}")
    print(f"then classified as {rec.detected_type} ({rec.type_confidence}) via {store.stored('DOC-LIVE-OCR').type_method}")
    print("(extra words are the page's own header/footer stamps, which the scenario source text does not contain)")
    ex = store.stored("DOC-LIVE-OCR").extraction
    if ex:
        for f in ex.fields:
            print(f"    {f.name:22s} {f.value!r:40} conf {f.confidence} flags {f.flags}")
        print("    missing required:", ex.missing_required)
    del lines
    print("usage:", usage(llm, n0))


def e4(settings, llm, rules):
    print("\n== E4 facts with model-assisted classifier and extractor: are citations still all verifiable? ==")
    pack = settings.pack
    dt = pack.document_types
    clf = HybridClassifier(rules, LLMClassifier(llm, rules, pack))
    ext = HybridExtractor(RuleExtractor(dt), LLMFieldFiller(llm, RuleExtractor(dt), pack))
    n0 = len(llm.calls)
    ev = evaluate_facts(pack, clf, ext)[0]
    c = ev.citation
    print(f"facts verified {c.total}: verbatim {c.verbatim_ok}/{c.total}, "
          f"region {c.region_ok}/{c.region_checked}, problems {len(c.problems)}")
    print(f"ground-truth coverage {ev.gt_covered}/{ev.gt_total}; "
          f"question links {ev.links_covered}/{ev.links_total}; refused {len(ev.refused)}")
    print("usage:", usage(llm, n0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["E0", "E1", "E2", "E3", "E4"])
    a = ap.parse_args()
    settings = load_settings(ROOT)
    llm = LLMClient.from_settings(settings)
    llm.mode, llm.cassette = "auto", Cassette(CASSETTE)
    cfg = settings.app.llm
    print(f"model {cfg.provider}/{cfg.text_model}, temperature {cfg.temperature}, reasoning_effort {cfg.reasoning_effort}")
    rules = RuleClassifier(settings.pack.document_types)
    for name, fn in (("E1", e1), ("E2", e2), ("E3", e3), ("E4", e4)):
        if a.only in (None, name):
            fn(settings, llm, rules)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
