"""`make eval`: the scorecard (T8.1). Runs every end-to-end case offline, scores them against the labelled scenarios, adds the component baselines, and writes evals/reports/scorecard.md.

    python evals/runners/run_eval.py                         # print the scorecard, write evals/reports/scorecard.md and scorecard.json
    python evals/runners/run_eval.py --write-baseline "label"   # also commit these numbers as the scorecard baseline (evals/baselines/scorecard.json)
    python evals/runners/run_eval.py --check                 # exit 1 if any case or metric is worse than the committed baseline

Nothing here calls a live model; to (re)record the model's answers for a case use scripts/replay_scenario.py <case> --record.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evals.scorecard import SCORECARD_BASELINE, build, regressions, run_cases, to_markdown  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-baseline", metavar="LABEL")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    results = run_cases()
    card = build(results)
    md = to_markdown(card, results)
    out = ROOT / "evals" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "scorecard.md").write_text(md, encoding="utf-8", newline="\n")
    (out / "scorecard.json").write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    if not a.quiet:
        print(md)
    if a.write_baseline:
        data = json.loads(SCORECARD_BASELINE.read_text(encoding="utf-8")) if SCORECARD_BASELINE.exists() else {"entries": []}
        data["entries"].append({"label": a.write_baseline, "date": card["date"], "cases": {n: {"passed": c["passed"]} for n, c in card["cases"].items()}, "end_to_end": card["end_to_end"]})
        SCORECARD_BASELINE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"baseline '{a.write_baseline}' recorded in {SCORECARD_BASELINE.relative_to(ROOT)}")
    if a.check:
        base = json.loads(SCORECARD_BASELINE.read_text(encoding="utf-8"))["entries"][-1]
        bad = regressions(card, base)
        for b in bad:
            print("REGRESSION:", b)
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
