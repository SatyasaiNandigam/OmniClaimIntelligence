"""Evaluate the sufficiency classifier (T4.4): rules only (no model) vs rules + model, and (optionally) record a baseline.

    python evals/runners/eval_sufficiency.py                        # dev, offline: model answers come from the recorded cassette
    python evals/runners/eval_sufficiency.py --split fresh          # the unseen cases (run once, after the prompt is frozen)
    python evals/runners/eval_sufficiency.py --record               # ask the model for anything not recorded yet (needs LLM_API_KEY)
    python evals/runners/eval_sufficiency.py --details              # list every wrong question and over-claim
    python evals/runners/eval_sufficiency.py --write-baseline "label"

Model answers are cached in data/cassettes/sufficiency_eval.json, so a normal run costs nothing and is deterministic.
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
from claimtower.evidence.llm_sufficiency import make_classifier  # noqa: E402
from claimtower.evidence.questions import QuestionRegister  # noqa: E402
from claimtower.evidence.sufficiency import SufficiencyClassifier  # noqa: E402
from evals.metrics.sufficiency import evaluate, load_cases  # noqa: E402

BASELINES = ROOT / "evals" / "baselines" / "sufficiency.json"
CASSETTE = ROOT / "data" / "cassettes" / "sufficiency_eval.json"
PACK = ROOT / "packs" / "property_water_damage"


def config_hash() -> str:
    h = hashlib.sha256()
    for p in (PACK / "question_templates.yaml", PACK / "prompts" / "sufficiency_check.md"):
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
    cases = [c for c in load_cases(QuestionRegister(settings.pack.question_templates)) if a.split in ("all", c.split)]
    cfg = settings.pack.question_templates.sufficiency
    modes_by_question = {q.id: q.criteria_mode for q in settings.pack.question_templates.questions}
    modes = {"rules only": SufficiencyClassifier(cfg), "rules + model": make_classifier(settings.pack, llm)}
    results = {}
    print(f"{'mode':14s} {'split':6s} {'scored':>6s} {'status':>7s} {'gap type':>9s} {'abstained':>9s} {'over-claims':>11s} {'refused':>7s}")
    for name, clf in modes.items():
        results[name] = evaluate(clf, cases, modes_by_question)
        for split, t in sorted(results[name].items()):
            s = t.summary()
            print(f"{name:14s} {split:6s} {s['questions_scored']:6d} {pct(s['status_accuracy']):>7s} {pct(s['gap_accuracy']):>9s} "
                  f"{s['abstained']:9d} {s['over_claims']:11d} {s['refused_judgements']:7d}")
            if a.details and name == "rules + model":
                for key in ("overclaims", "wrong"):
                    for line in s[key]:
                        print(f"      {key.upper():10s} {line}")
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
