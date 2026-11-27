"""Evaluate the coverage matrix readings (T5.1): rules only vs rules + model reading, and (optionally) record a baseline.

    python evals/runners/eval_coverage.py                       # dev, offline: model answers come from the recorded cassette
    python evals/runners/eval_coverage.py --split fresh         # the unseen cases (run once, after the prompt is frozen)
    python evals/runners/eval_coverage.py --record              # ask the model for anything not recorded yet (needs LLM_API_KEY)
    python evals/runners/eval_coverage.py --details             # list wrong rows
    python evals/runners/eval_coverage.py --write-baseline "label"
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

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from claimtower.core.config import load_settings  # noqa: E402
from claimtower.core.llm import Cassette, LLMClient  # noqa: E402
from claimtower.coverage.issue_matrix import IssueMatrixBuilder  # noqa: E402
from claimtower.coverage.llm_reading import make_builder  # noqa: E402
from evals.metrics.coverage import evaluate, load_cases  # noqa: E402

BASELINES = ROOT / "evals" / "baselines" / "coverage.json"
CASSETTE = ROOT / "data" / "cassettes" / "coverage_eval.json"
PACK = ROOT / "packs" / "property_water_damage"


def config_hash() -> str:
    h = hashlib.sha256()
    for p in (PACK / "coverage_issues.yaml", PACK / "prompts" / "coverage_reading.md"):
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def pct(x):
    return "  n/a" if x is None else f"{x:5.0%}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "fresh", "all"])
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--details", action="store_true")
    ap.add_argument("--write-baseline", metavar="LABEL")
    a = ap.parse_args()
    settings = load_settings(ROOT)
    llm = LLMClient.from_settings(settings)
    llm.mode, llm.cassette = ("auto" if a.record else "replay"), Cassette(CASSETTE)
    cases = [c for c in load_cases() if a.split in ("all", c.split)]
    modes = {"rules only": IssueMatrixBuilder(settings.pack.coverage, None, 1), "rules + model": make_builder(settings.pack, llm, 1)}
    results = {}
    print(f"{'mode':14s} {'split':6s} {'rows':>4s} {'accuracy':>8s} {'est. vs label':>13s} {'abstain ok':>10s} {'over-abst.':>10s} {'refused':>7s} {'lint':>5s} {'problems':>8s}")
    for name, b in modes.items():
        results[name] = evaluate(b, cases)
        for split, t in sorted(results[name].items()):
            s = t.summary()
            print(f"{name:14s} {split:6s} {s['rows']:4d} {pct(s['status_accuracy']):>8s} {s['established_against_the_label']:13d} {s['abstained_when_unclear']:10d} "
                  f"{s['over_abstained']:10d} {s['refused_readings']:7d} {s['lint_failures']:5d} {s['self_check_problems']:8d}")
            if a.details and name == "rules + model":
                for line in s["wrong"]:
                    print(f"      WRONG {line}")
    live = [c for c in llm.calls if not c.replayed]
    print(f"model calls: {len(llm.calls)} ({len(live)} live, {sum(c.input_tokens or 0 for c in live)} tokens in)")
    if a.record:
        llm.cassette.save()
    if a.write_baseline:
        data = json.loads(BASELINES.read_text(encoding="utf-8")) if BASELINES.exists() else {"entries": []}
        data["entries"].append({"label": a.write_baseline, "date": date.today().isoformat(), "config_sha": config_hash(), "model": llm.text_model,
                                "modes": {m: {k: t.summary() for k, t in sorted(ts.items())} for m, ts in results.items()}})
        BASELINES.parent.mkdir(parents=True, exist_ok=True)
        BASELINES.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"baseline '{a.write_baseline}' recorded in {BASELINES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
