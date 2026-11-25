"""Evaluate facts-with-provenance (T2.4): citation correctness and ground-truth coverage; optionally record a baseline.

    python evals/runners/eval_facts.py                          # print the report
    python evals/runners/eval_facts.py --details                # also list uncovered ground-truth facts and missing question links
    python evals/runners/eval_facts.py --write-baseline "label"

Rules only: no model is called. Baselines go to evals/baselines/facts.json.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from claimtower.core.config import load_settings  # noqa: E402
from claimtower.ingestion.classifier import HybridClassifier, RuleClassifier  # noqa: E402
from claimtower.ingestion.extractors import HybridExtractor, RuleExtractor  # noqa: E402
from evals.metrics.provenance import evaluate_facts  # noqa: E402
from evals.runners.eval_extractor import signals_hash as extractor_hash  # noqa: E402

BASELINES = ROOT / "evals" / "baselines" / "facts.json"


def run():
    pack = load_settings(ROOT).pack
    classifier = HybridClassifier(RuleClassifier(pack.document_types))
    extractor = HybridExtractor(RuleExtractor(pack.document_types))
    return evaluate_facts(pack, classifier, extractor)[0]


def pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", action="store_true")
    ap.add_argument("--write-baseline", metavar="LABEL")
    a = ap.parse_args()
    ev = run()
    c = ev.citation
    print(f"facts: {ev.scenario_facts} from the 5 scenarios + {ev.text_facts} from labelled text documents = {c.total} verified")
    print(f"citation correctness : verbatim {c.verbatim_ok}/{c.total} ({pct(c.verbatim_rate)})  "
          f"region {c.region_ok}/{c.region_checked} ({pct(c.region_rate)})"
          f"  ocr-basis {c.ocr_basis}  problems {len(c.problems)}")
    print(f"ground-truth coverage: {ev.gt_covered}/{ev.gt_total} facts ({pct(ev.gt_recall)}); "
          f"question links {ev.links_covered}/{ev.links_total} ({pct(ev.link_recall)})")
    print(f"low-confidence facts : {ev.low_confidence_facts} (kept, flagged by their confidence)   "
          f"refused: {len(ev.refused)}   all neutral: {ev.neutral_only}")
    if a.details:
        for p in c.problems:
            print("   PROBLEM  ", p.fact_id, p.problem)
        for u in ev.gt_uncovered:
            print("   UNCOVERED", u)
        for m in ev.links_missing:
            print("   NO LINK  ", m)
        for r in ev.refused:
            print("   REFUSED  ", r)
    if a.write_baseline:
        data = json.loads(BASELINES.read_text(encoding="utf-8")) if BASELINES.exists() else {"entries": []}
        data["entries"].append({"label": a.write_baseline, "date": date.today().isoformat(), "extractor_sha": extractor_hash(),
                                "result": ev.summary()})
        BASELINES.parent.mkdir(parents=True, exist_ok=True)
        BASELINES.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"baseline '{a.write_baseline}' recorded in {BASELINES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
