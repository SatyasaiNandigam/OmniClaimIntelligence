"""Evaluate the settlement input extraction (T5.5): rules only vs rules + model reading of the amounts, and (optionally) record a baseline.

    python evals/runners/eval_settlement_inputs.py                       # dev, offline: model answers come from the recorded cassette
    python evals/runners/eval_settlement_inputs.py --split fresh         # the unseen cases (run once, after the prompt is frozen)
    python evals/runners/eval_settlement_inputs.py --record              # ask the model for anything not recorded yet (needs LLM_API_KEY)
    python evals/runners/eval_settlement_inputs.py --details             # list wrong rows
    python evals/runners/eval_settlement_inputs.py --write-baseline "label"
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
from claimtower.agents.settlement import SettlementAgent, make_settlement_agent  # noqa: E402
from claimtower.settlement.engine import SettlementCalculator  # noqa: E402
from claimtower.settlement.rules_loader import rules_by_version  # noqa: E402
from evals.metrics.settlement_inputs import evaluate, load_cases  # noqa: E402

BASELINES = ROOT / "evals" / "baselines" / "settlement_inputs.json"
CASSETTE = ROOT / "data" / "cassettes" / "settlement_inputs_eval.json"
PACK = ROOT / "packs" / "property_water_damage"


def config_hash() -> str:
    h = hashlib.sha256()
    for p in (PACK / "prompts" / "settlement_inputs.md",):
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
    calc = SettlementCalculator(rules_by_version(settings.pack.calculation))
    modes = {"rules only": SettlementAgent(calc, "sample-1.0"), "rules + model": make_settlement_agent(settings.pack, calc, "sample-1.0", None, None, llm)}
    results = {}
    print(f"{'mode':14s} {'split':6s} {'cases':>5s} {'assessed':>8s} {'non-cov.':>8s} {'WRONG no.':>9s} {'missing':>7s} {'model-only':>10s} {'refused':>7s}")
    for name, ag in modes.items():
        results[name] = evaluate(ag, cases)
        for split, t in sorted(results[name].items()):
            s = t.summary()
            print(f"{name:14s} {split:6s} {s['cases']:5d} {pct(s['assessed_correct']):>8s} {pct(s['non_covered_correct']):>8s} {s['wrong_numbers']:9d} "
                  f"{s['missing_when_labelled']:7d} {s['model_only_values']:10d} {s['refused_readings']:7d}")
            if a.details:
                for line in s["wrong"]:
                    print(f"      {name}: {line}")
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
