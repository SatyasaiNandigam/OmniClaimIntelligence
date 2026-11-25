"""Evaluate the fact <-> question linker (T4.2): tags only vs tags + model, and (optionally) record a baseline.

    python evals/runners/eval_links.py                          # dev (scenario ground truth), offline: model answers come from the cassette
    python evals/runners/eval_links.py --split fresh            # the unseen cases (run once, after the prompt is frozen)
    python evals/runners/eval_links.py --tags none              # remove the template tags: can the model find the links itself?
    python evals/runners/eval_links.py --record                 # ask the model for anything not recorded yet (needs LLM_API_KEY)
    python evals/runners/eval_links.py --details                # list wrong directions, critical errors, misses and extras
    python evals/runners/eval_links.py --write-baseline "label"

Model answers are cached in data/cassettes/link_eval.json, so a normal run costs nothing and is deterministic.
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
from claimtower.evidence.linker import FactQuestionLinker  # noqa: E402
from claimtower.evidence.llm_linking import make_linker  # noqa: E402
from claimtower.evidence.questions import QuestionRegister  # noqa: E402
from evals.metrics.links import evaluate, load_items  # noqa: E402

BASELINES = ROOT / "evals" / "baselines" / "links.json"
CASSETTE = ROOT / "data" / "cassettes" / "link_eval.json"
PACK = ROOT / "packs" / "property_water_damage"


def config_hash() -> str:
    h = hashlib.sha256()
    for p in (PACK / "question_templates.yaml", PACK / "prompts" / "fact_linking.md"):
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def pct(x):
    return "  n/a" if x is None else f"{x:5.0%}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "fresh", "all"])
    ap.add_argument("--tags", default="given", choices=["given", "none"])
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--details", action="store_true")
    ap.add_argument("--write-baseline", metavar="LABEL")
    a = ap.parse_args()
    settings = load_settings(ROOT)
    llm = LLMClient.from_settings(settings)
    llm.mode, llm.cassette = ("auto" if a.record else "replay"), Cassette(CASSETTE)
    items = [i for i in load_items(QuestionRegister(settings.pack.question_templates)) if a.split in ("all", i.split)]
    modes = {"tags only": FactQuestionLinker(settings.pack.question_templates.fact_links), "tags + model": make_linker(settings.pack, llm)}
    results = {}
    print(f"tags: {a.tags}")
    print(f"{'mode':13s} {'split':6s} {'labelled':>8s} {'predicted':>9s} {'precision':>9s} {'recall':>7s} {'direction':>9s} {'critical':>8s} {'refused':>7s} {'dropped':>7s}")
    for name, linker in modes.items():
        results[name] = evaluate(linker, items, a.tags)
        for split, t in sorted(results[name].items()):
            s = t.summary()
            print(f"{name:13s} {split:6s} {s['labelled_links']:8d} {s['predicted_links']:9d} {pct(s['link_precision']):>9s} {pct(s['link_recall']):>7s} "
                  f"{pct(s['direction_accuracy']):>9s} {s['critical_errors']:8d} {s['refused_judgements']:7d} {s['dropped_links']:7d}")
            if a.details and name == "tags + model":
                for key in ("critical", "wrong_direction", "missed", "extra"):
                    for line in s[key]:
                        print(f"      {key.upper():16s} {line}")
    live = [c for c in llm.calls if not c.replayed]
    print(f"model calls: {len(llm.calls)} ({len(live)} live, {sum(c.input_tokens or 0 for c in live)} tokens in)")
    if a.record:
        llm.cassette.save()
    if a.write_baseline:
        data = json.loads(BASELINES.read_text(encoding="utf-8")) if BASELINES.exists() else {"entries": []}
        data["entries"].append({"label": a.write_baseline, "date": date.today().isoformat(), "config_sha": config_hash(), "model": llm.text_model,
                                "tags": a.tags, "modes": {m: {k: t.summary() for k, t in sorted(ts.items())} for m, ts in results.items()}})
        BASELINES.parent.mkdir(parents=True, exist_ok=True)
        BASELINES.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"baseline '{a.write_baseline}' recorded in {BASELINES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
