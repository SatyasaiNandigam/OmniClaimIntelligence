"""Evaluate the minimum-evidence request generator (T4.5): rule-written items only vs rules + model wording, and (optionally) record a baseline.

    python evals/runners/eval_requests.py                        # dev, offline: model answers come from the recorded cassette
    python evals/runners/eval_requests.py --split fresh          # the unseen cases (run once, after the prompt is frozen)
    python evals/runners/eval_requests.py --record               # ask the model for anything not recorded yet (needs LLM_API_KEY)
    python evals/runners/eval_requests.py --details              # list wrong addressees, misses and low-rated requests
    python evals/runners/eval_requests.py --show S-01            # print the generated requests of one case
    python evals/runners/eval_requests.py --write-baseline "label"

The "sufficient / minimal" columns are rated by a MODEL (a self-assessment, not a domain reviewer). Model answers are cached in
data/cassettes/request_eval.json, so a normal run costs nothing and is deterministic.
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
from claimtower.evidence.llm_requests import make_generator  # noqa: E402
from claimtower.evidence.min_evidence import MinimumEvidenceGenerator  # noqa: E402
from claimtower.evidence.questions import QuestionRegister  # noqa: E402
from evals.metrics.requests import RequestRater, evaluate, load_cases  # noqa: E402

BASELINES = ROOT / "evals" / "baselines" / "requests.json"
CASSETTE = ROOT / "data" / "cassettes" / "request_eval.json"
PACK = ROOT / "packs" / "property_water_damage"


def config_hash() -> str:
    h = hashlib.sha256()
    for p in (PACK / "question_templates.yaml", PACK / "prompts" / "min_evidence_request.md", PACK / "prompts" / "min_evidence_review.md"):
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def pct(x):
    return "  n/a" if x is None else f"{x:5.0%}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")                    # model wording can contain non-breaking hyphens and typographic quotes
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "fresh", "fresh2", "all"])
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--details", action="store_true")
    ap.add_argument("--show", metavar="CASE")
    ap.add_argument("--write-baseline", metavar="LABEL")
    a = ap.parse_args()
    settings = load_settings(ROOT)
    llm = LLMClient.from_settings(settings)
    llm.mode, llm.cassette = ("auto" if a.record else "replay"), Cassette(CASSETTE)
    cases = [c for c in load_cases(QuestionRegister(settings.pack.question_templates)) if a.split in ("all", c.split)]
    rules = MinimumEvidenceGenerator(settings.pack.question_templates)
    hybrid = make_generator(settings.pack, llm)
    if a.show:
        for c in cases:
            if c.id == a.show:
                r = hybrid.generate(c.questions, c.sufficiency, c.facts, c.links, c.contradictions, c.received)
                for d in r.details:
                    print(f"\n-> {d.role} ({'internal' if d.internal else 'external'}), resolves {d.question_ids}")
                    for it in d.items:
                        print(f"   [{it.source}/{it.asks_for}] {it.text}")
                print("refused:", [(x.role, x.text[:60], x.reason) for x in r.refused], "warnings:", r.warnings, "review:", r.review_questions)
        return 0
    rater = RequestRater(llm, settings.pack)
    results = {}
    print(f"{'mode':14s} {'split':6s} {'addr P':>6s} {'addr R':>6s} {'q recall':>8s} {'extra':>5s} {'problems':>8s} {'dup':>3s} {'leak':>4s} {'rule items':>10s} {'suff':>5s} {'minimal':>7s} {'both':>5s}")
    for name, gen in {"rules only": rules, "rules + model": hybrid}.items():
        results[name] = evaluate(gen, cases, rater)
        for split, t in sorted(results[name].items()):
            s = t.summary()
            print(f"{name:14s} {split:6s} {pct(s['addressee_precision']):>6s} {pct(s['addressee_recall']):>6s} {pct(s['question_recall']):>8s} {s['extra_question_pairs']:5d} "
                  f"{s['self_check_problems']:8d} {s['duplicates_of_received']:3d} {s['internal_needs_sent_externally']:4d} {pct(s['rule_written_item_share']):>10s} "
                  f"{pct(s['rated_sufficient']):>5s} {pct(s['rated_minimal']):>7s} {pct(s['rated_both']):>5s}")
            if a.details and name == "rules + model":
                for key in ("unexpected_requests", "missed", "low_rated"):
                    for line in s[key]:
                        print(f"      {key.upper():19s} {line}")
    live = [c for c in llm.calls if not c.replayed]
    print(f"model calls: {len(llm.calls)} ({len(live)} live, {sum(c.input_tokens or 0 for c in live)} tokens in)")
    if a.record:
        llm.cassette.save()
    if a.write_baseline:
        data = json.loads(BASELINES.read_text(encoding="utf-8")) if BASELINES.exists() else {"entries": []}
        data["entries"].append({"label": a.write_baseline, "date": date.today().isoformat(), "config_sha": config_hash(), "model": llm.text_model,
                                "rating": "model-rated (self-assessment), not a domain reviewer",
                                "modes": {m: {k: t.summary() for k, t in sorted(ts.items())} for m, ts in results.items()}})
        BASELINES.parent.mkdir(parents=True, exist_ok=True)
        BASELINES.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"baseline '{a.write_baseline}' recorded in {BASELINES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
