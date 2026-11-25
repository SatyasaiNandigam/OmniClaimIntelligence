"""Evaluate the field/table extractors and (optionally) record a baseline.

    python evals/runners/eval_extractor.py                         # print the scores per split
    python evals/runners/eval_extractor.py --details               # also list every wrong / missed / flagged-miss
    python evals/runners/eval_extractor.py --write-baseline "v1 first run"

Baselines are appended to evals/baselines/extraction.json with a hash of the extractor config and field specs, so a number can always be
traced back to the parameters that produced it. Rules only: no model is called.
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
from claimtower.ingestion.extractors import HybridExtractor, RuleExtractor  # noqa: E402
from evals.metrics.extraction import all_cases, evaluate  # noqa: E402

BASELINES = ROOT / "evals" / "baselines" / "extraction.json"


def signals_hash() -> str:
    """Hash of the extractor parameters and every type's field / table specification."""
    dt = load_settings(ROOT).pack.document_types
    keep = {"fields", "table", "table_expects_tax"}
    types = {t: spec.model_dump(include=keep, mode="json") for t, spec in sorted(dt.types.items()) if spec.fields or spec.table}
    payload = {"extractor": dt.extractor.model_dump(mode="json"), "types": types}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def run():
    dt = load_settings(ROOT).pack.document_types
    return evaluate(HybridExtractor(RuleExtractor(dt)), all_cases(), list(dt.types))


def pct(x):
    return "  n/a" if x is None else f"{x:5.1%}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", action="store_true")
    ap.add_argument("--write-baseline", metavar="LABEL")
    a = ap.parse_args()
    reports = run()
    print(f"{'split':10s} {'docs':>4s} {'fields':>7s} {'score':>7s} {'wrong':>6s} {'SILENT':>7s} {'missed':>7s} {'spur.':>6s} "
          f"{'flag-rec':>9s} {'tables':>7s}")
    for name in sorted(reports):
        r, s = reports[name], reports[name].summary()
        print(f"{name:10s} {r.docs:4d} {s['expected_fields']:7d} {pct(r.field_score):>7s} {len(s['wrong_values']):6d} "
              f"{s['silent_wrong']:7d} {len(s['missed']):7d} {len(s['spurious']):6d} {pct(r.flag_recall):>9s} {pct(r.table_score()):>7s}")
        if a.details:
            for key in ("wrong_values", "missed", "spurious", "flag_missed", "table_failures", "required_failures"):
                for line in s[key]:
                    print(f"    {key.upper():18s} {line}")
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
