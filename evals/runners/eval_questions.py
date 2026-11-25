"""Evaluate the claim-specific questions the model adds to the register (T4.1) and (optionally) record a baseline.

    python evals/runners/eval_questions.py                       # dev cases, offline: model answers come from the recorded cassette
    python evals/runners/eval_questions.py --split fresh         # the unseen cases (run once, after the prompt is frozen)
    python evals/runners/eval_questions.py --record              # ask the model for anything not recorded yet (needs LLM_API_KEY)
    python evals/runners/eval_questions.py --details             # list every miss, spurious and extra question
    python evals/runners/eval_questions.py --write-baseline "label"

Model answers are cached in data/cassettes/question_eval.json, so a normal run costs nothing and is deterministic.
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
from claimtower.evidence.llm_questions import make_register  # noqa: E402
from evals.metrics.questions import evaluate, load_cases  # noqa: E402

BASELINES = ROOT / "evals" / "baselines" / "questions.json"
CASSETTE = ROOT / "data" / "cassettes" / "question_eval.json"
PACK = ROOT / "packs" / "property_water_damage"


def config_hash() -> str:
    h = hashlib.sha256()
    for p in (PACK / "question_templates.yaml", PACK / "prompts" / "question_proposal.md", PACK / "prompts" / "question_duplicate_check.md"):
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def pct(x):
    return "  n/a" if x is None else f"{x:5.0%}"


def main() -> int:
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
    results = evaluate(make_register(settings.pack, llm), cases)
    print(f"{'split':6s} {'cases':>5s} {'needed':>7s} {'hits':>5s} {'recall':>7s} {'none-needed':>12s} {'spurious':>9s} {'accepted/case':>14s}")
    for split, t in sorted(results.items()):
        s = t.summary()
        print(f"{split:6s} {s['cases']:5d} {s['needed_groups']:7d} {s['hits']:5d} {pct(s['recall']):>7s} {s['none_needed_cases']:12d} "
              f"{s['spurious_cases']:9d} {s['accepted_per_case']:14.2f}   refused: {s['refused_by_reason']}")
        if a.details:
            for key in ("misses", "spurious", "extras"):
                for line in s[key]:
                    print(f"      {key.upper():9s} {line}")
    live = [c for c in llm.calls if not c.replayed]
    print(f"model calls: {len(llm.calls)} ({len(live)} live, {sum(c.input_tokens or 0 for c in live)} tokens in)")
    if a.record:
        llm.cassette.save()
    if a.write_baseline:
        data = json.loads(BASELINES.read_text(encoding="utf-8")) if BASELINES.exists() else {"entries": []}
        data["entries"].append({"label": a.write_baseline, "date": date.today().isoformat(), "config_sha": config_hash(), "model": llm.text_model,
                                "splits": {k: t.summary() for k, t in sorted(results.items())}})
        BASELINES.parent.mkdir(parents=True, exist_ok=True)
        BASELINES.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"baseline '{a.write_baseline}' recorded in {BASELINES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
