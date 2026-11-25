"""Evaluate policy relations (T3.2) and (optionally) record a baseline.

    python evals/runners/eval_policy_relations.py                  # print the report
    python evals/runners/eval_policy_relations.py --details        # also list missed, audited and unlabelled relations, ambiguities
    python evals/runners/eval_policy_relations.py --write-baseline "label"

Rules only: no model is called. Baselines go to evals/baselines/policy_relations.json with a hash of policy_structure.yaml.
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
from claimtower.policy.decompose import PolicyDecomposer  # noqa: E402
from claimtower.policy.relations import RelationExtractor  # noqa: E402
from evals.metrics.policy_relations import evaluate  # noqa: E402
from evals.runners.eval_policy_clauses import full_hash  # noqa: E402

BASELINES = ROOT / "evals" / "baselines" / "policy_relations.json"


def run():
    ps = load_settings(ROOT).pack.policy_structure
    return evaluate(PolicyDecomposer(ps), RelationExtractor(ps))


def pct(x):
    return "  n/a" if x is None else f"{x:5.1%}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", action="store_true")
    ap.add_argument("--write-baseline", metavar="LABEL")
    a = ap.parse_args()
    reports = run()
    print(f"{'split':8s} {'labelled':>8s} {'found':>6s} {'recall':>7s} {'precision':>10s} {'+audited':>9s} {'unlabelled':>10s} {'ambiguous':>9s}")
    for name in sorted(reports):
        r = reports[name]
        print(f"{name:8s} {r.labelled:8d} {r.matched:6d} {pct(r.recall):>7s} {pct(r.precision):>10s} {pct(r.lenient_precision):>9s} "
              f"{len(r.unlabelled):10d} {len(r.ambiguous):9d}")
        for rel, t in sorted(r.by_type.items()):
            print(f"           {rel:10s} labelled {t['labelled']:2d}  found {t['matched']:2d}  predicted {t['predicted']:2d}")
        if a.details:
            for key, items in (("MISSED", r.missed), ("AUDITED", r.audited_valid), ("UNLABELLED", r.unlabelled), ("AMBIGUOUS", r.ambiguous)):
                for line in items:
                    print(f"    {key:10s} {line}")
    if a.write_baseline:
        data = json.loads(BASELINES.read_text(encoding="utf-8")) if BASELINES.exists() else {"entries": []}
        data["entries"].append({"label": a.write_baseline, "date": date.today().isoformat(), "signals_sha": full_hash(),
                                "splits": {k: v.summary() for k, v in sorted(reports.items())}})
        BASELINES.parent.mkdir(parents=True, exist_ok=True)
        BASELINES.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"baseline '{a.write_baseline}' recorded in {BASELINES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
