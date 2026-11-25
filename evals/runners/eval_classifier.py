"""Evaluate the document-type classifier and (optionally) record a baseline.

    python evals/runners/eval_classifier.py                       # print the report
    python evals/runners/eval_classifier.py --details             # also list every wrong / over-abstained case
    python evals/runners/eval_classifier.py --write-baseline "v1 initial signals"

Baselines are appended to evals/baselines/classification.json together with a hash of document_types.yaml, so a number can always be traced
back to the exact signals and thresholds that produced it.
"""
import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from claimtower.core.config import load_settings  # noqa: E402
from claimtower.ingestion.classifier import HybridClassifier, RuleClassifier  # noqa: E402
from evals.metrics.classification import all_cases, evaluate  # noqa: E402

BASELINES = ROOT / "evals" / "baselines" / "classification.json"


def signals_hash() -> str:
    """Hash of what the CLASSIFIER uses (its thresholds and each type's title/phrase/structure signals). Extractor field specs in the same
    YAML do not change a classification result, so they are deliberately excluded."""
    dt = load_settings(ROOT).pack.document_types
    keep = {"title_patterns", "phrases", "phrases_needed", "negative_phrases", "structure_patterns", "structure_needed"}
    payload = {"classifier": dt.classifier.model_dump(mode="json"),
               "types": {t: spec.model_dump(include=keep, mode="json") for t, spec in sorted(dt.types.items())}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def run():
    settings = load_settings(ROOT)
    classifier = HybridClassifier(RuleClassifier(settings.pack.document_types))       # rules only: no model is called
    return evaluate(classifier, all_cases())


def fmt(x):
    return "  n/a" if x is None else f"{x:5.1%}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", action="store_true")
    ap.add_argument("--write-baseline", metavar="LABEL")
    a = ap.parse_args()
    reports = run()
    print(f"{'split':10s} {'n':>4s} {'accuracy':>9s} {'ans.prec':>9s} {'abst.rec':>9s} {'wrong':>6s} {'over-abst':>9s}")
    for name in sorted(reports):
        r = reports[name]
        print(f"{name:10s} {r.n:4d} {fmt(r.accuracy):>9s} {fmt(r.answered_precision):>9s} {fmt(r.abstain_recall):>9s} "
              f"{len(r.wrong_answers):6d} {len(r.over_abstained):9d}")
        if a.details:
            for o in r.wrong_answers:
                print(f"    WRONG  {o.case.id}: expected {o.case.expected}, got {o.got} ({o.confidence:.2f}) - {o.reason}")
            for o in r.over_abstained:
                print(f"    ABSTAIN {o.case.id}: expected {o.case.expected} [{o.code}] - {o.reason}")
    if a.write_baseline:
        data = json.loads(BASELINES.read_text(encoding="utf-8")) if BASELINES.exists() else {"entries": []}
        data["entries"].append({"label": a.write_baseline, "date": date.today().isoformat(), "signals_sha": signals_hash(),
                                "splits": {k: v.summary() for k, v in sorted(reports.items())}})
        BASELINES.parent.mkdir(parents=True, exist_ok=True)
        BASELINES.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"baseline '{a.write_baseline}' recorded in {BASELINES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
