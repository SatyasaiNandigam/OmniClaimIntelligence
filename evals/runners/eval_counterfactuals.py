"""Evaluate the counterfactual engine (T4.6): rule text only vs rules + model rewrite, and (optionally) record a baseline.

    python evals/runners/eval_counterfactuals.py                      # dev, offline: model answers come from the recorded cassette
    python evals/runners/eval_counterfactuals.py --split fresh        # the unseen cases (run once, after the prompt is frozen)
    python evals/runners/eval_counterfactuals.py --record             # ask the model for anything not recorded yet (needs LLM_API_KEY)
    python evals/runners/eval_counterfactuals.py --details            # list wrong cases
    python evals/runners/eval_counterfactuals.py --show S-01          # print the scenarios of one case
    python evals/runners/eval_counterfactuals.py --export             # write evals/review/counterfactual_review.csv for an adjuster to rate
    python evals/runners/eval_counterfactuals.py --write-baseline "label"

Model answers are cached in data/cassettes/counterfactual_eval.json, so a normal run costs nothing and is deterministic.
"""
import argparse
import csv
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
from claimtower.evidence.counterfactual import CounterfactualEngine  # noqa: E402
from claimtower.evidence.llm_counterfactual import make_engine  # noqa: E402
from claimtower.evidence.questions import QuestionRegister  # noqa: E402
from evals.metrics.counterfactuals import evaluate, load_cases  # noqa: E402

BASELINES = ROOT / "evals" / "baselines" / "counterfactuals.json"
CASSETTE = ROOT / "data" / "cassettes" / "counterfactual_eval.json"
REVIEW = ROOT / "evals" / "review" / "counterfactual_review.csv"
PACK = ROOT / "packs" / "property_water_damage"


def config_hash() -> str:
    h = hashlib.sha256()
    for p in (PACK / "question_templates.yaml", PACK / "prompts" / "counterfactual_rewrite.md"):
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
    ap.add_argument("--show", metavar="CASE")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--write-baseline", metavar="LABEL")
    a = ap.parse_args()
    settings = load_settings(ROOT)
    llm = LLMClient.from_settings(settings)
    llm.mode, llm.cassette = ("auto" if a.record else "replay"), Cassette(CASSETTE)
    cases = [c for c in load_cases(QuestionRegister(settings.pack.question_templates)) if a.split in ("all", c.split)]
    rules = CounterfactualEngine(settings.pack.question_templates)
    hybrid = make_engine(settings.pack, llm)
    if a.show or a.export:
        rows = []
        for c in cases:
            if a.show and c.id != a.show:
                continue
            r = hybrid.generate(c.questions, c.sufficiency, c.issues, c.clauses, c.requests, c.facts, c.links)
            for s in r.scenarios:
                src = r.source[(s.question_id, s.scenario_id)]
                if a.show:
                    print(f"\n[{s.scenario_id}] ({src}) {s.assumed_finding}; re-assesses {s.affected_issue_ids}\n   {s.implication_text}\n   next: {s.next_step}")
                rows.append([c.id, c.split, s.question_id, s.scenario_id, src, s.assumed_finding, ", ".join(s.affected_issue_ids), s.implication_text, s.next_step, "", "", ""])
            if a.show:
                print("refused:", [(x.scenario_id, x.reason) for x in r.refused], "warnings:", r.warnings)
        if a.export:
            REVIEW.parent.mkdir(parents=True, exist_ok=True)
            with REVIEW.open("w", encoding="utf-8-sig", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["case", "split", "question", "scenario", "text from", "assumed finding", "issues re-assessed", "implication", "next step",
                            "accurate (yes/no)", "never predicts a decision (yes/no)", "comment"])
                w.writerows(rows)
            print(f"wrote {len(rows)} scenarios to {REVIEW.relative_to(ROOT)}")
        return 0
    results = {}
    print(f"{'mode':14s} {'split':6s} {'complete':>8s} {'issues':>7s} {'surplus':>7s} {'lint fail':>9s} {'problems':>8s} {'model text':>10s}  refused")
    for name, eng in {"rules only": rules, "rules + model": hybrid}.items():
        results[name] = evaluate(eng, cases)
        for split, t in sorted(results[name].items()):
            s = t.summary()
            print(f"{name:14s} {split:6s} {pct(s['complete_a_b_c']):>8s} {pct(s['affected_issues_exact']):>7s} {s['scenarios_where_none_expected']:7d} "
                  f"{s['lint_failures']:9d} {s['self_check_problems']:8d} {pct(s['model_text_share']):>10s}  {s['refused_rewrites']}")
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
