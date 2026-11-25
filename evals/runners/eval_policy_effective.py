"""Evaluate the effective clause set (T3.2): scenario ground truth S-01/02/04/05 plus the dated cases, and (optionally) record a baseline.

    python evals/runners/eval_policy_effective.py                  # print pass / fail per case
    python evals/runners/eval_policy_effective.py --write-baseline "label"

Rules only: no model is called. Baselines go to evals/baselines/policy_effective.json with a hash of policy_structure.yaml.
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
from evals.metrics.policy_effective import run_cases, run_scenarios  # noqa: E402
from evals.runners.eval_policy_clauses import full_hash  # noqa: E402

BASELINES = ROOT / "evals" / "baselines" / "policy_effective.json"


def run():
    ps = load_settings(ROOT).pack.policy_structure
    d, e = PolicyDecomposer(ps), RelationExtractor(ps)
    return {"scenarios": run_scenarios(d, e), "cases": run_cases(d, e)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-baseline", metavar="LABEL")
    a = ap.parse_args()
    results = run()
    for name, rs in results.items():
        print(f"{name:10s} {sum(r.passed for r in rs)}/{len(rs)} passed")
        for r in rs:
            print(f"    {'ok  ' if r.passed else 'FAIL'} {r.case_id}  {r.why}")
            for f in r.failures:
                print(f"           {f}")
    if a.write_baseline:
        data = json.loads(BASELINES.read_text(encoding="utf-8")) if BASELINES.exists() else {"entries": []}
        data["entries"].append({"label": a.write_baseline, "date": date.today().isoformat(), "signals_sha": full_hash(),
                                "results": {n: {"passed": sum(r.passed for r in rs), "total": len(rs), "failed": [r.case_id for r in rs if not r.passed]}
                                            for n, rs in results.items()}})
        BASELINES.parent.mkdir(parents=True, exist_ok=True)
        BASELINES.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"baseline '{a.write_baseline}' recorded in {BASELINES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
