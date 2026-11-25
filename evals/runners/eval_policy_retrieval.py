"""Evaluate clause retrieval (T3.4): keyword only vs hybrid (keyword + embeddings), and (optionally) record a baseline.

    python evals/runners/eval_policy_retrieval.py                      # offline: embeddings come from the recorded cassette
    python evals/runners/eval_policy_retrieval.py --record             # embed any new text with the model in configs/app.yaml (needs LLM_API_KEY)
    python evals/runners/eval_policy_retrieval.py --details            # list every miss, modifier failure and false answer
    python evals/runners/eval_policy_retrieval.py --write-baseline "label"

Embeddings are cached in data/cassettes/policy_retrieval.json, so a normal run costs nothing and is deterministic.
"""
import argparse
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
from claimtower.policy.decompose import PolicyDecomposer  # noqa: E402
from claimtower.policy.relations import RelationExtractor  # noqa: E402
from claimtower.policy.retrieval import ClauseRetriever  # noqa: E402
from evals.metrics.policy_retrieval import build_world, evaluate  # noqa: E402
from evals.runners.eval_policy_clauses import retrieval_hash  # noqa: E402

BASELINES = ROOT / "evals" / "baselines" / "policy_retrieval.json"
CASSETTE = ROOT / "data" / "cassettes" / "policy_retrieval.json"


def embedder(settings, record: bool) -> LLMClient:
    llm = LLMClient.from_settings(settings)
    llm.mode, llm.cassette = ("auto" if record else "replay"), Cassette(CASSETTE)
    return llm


def run(record: bool = False):
    settings = load_settings(ROOT)
    ps = settings.pack.policy_structure
    world = build_world(PolicyDecomposer(ps), RelationExtractor(ps))
    llm = embedder(settings, record)
    return {"keyword": evaluate(ClauseRetriever(ps.retrieval), world),
            "hybrid": evaluate(ClauseRetriever(ps.retrieval, embedder=llm), world)}, llm


def pct(x):
    return "  n/a" if x is None else f"{x:5.1%}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--details", action="store_true")
    ap.add_argument("--write-baseline", metavar="LABEL")
    a = ap.parse_args()
    results, llm = run(a.record)
    print(f"{'mode':8s} {'split':6s} {'answerable':>10s} {'hit@1':>7s} {'hit@3':>7s} {'recall@5':>9s} {'MRR':>6s} {'modifiers':>10s} {'no-answer':>10s} {'degraded':>9s}")
    for mode, tallies in results.items():
        for split in sorted(tallies):
            s = tallies[split].summary()
            print(f"{mode:8s} {split:6s} {s['answerable']:10d} {pct(s['hit@1']):>7s} {pct(s['hit@3']):>7s} {pct(s['recall@5']):>9s} "
                  f"{pct(s['mrr']):>6s} {pct(s['modifier_completeness']):>10s} {pct(s['no_answer_rate']):>10s} {s['degraded_queries']:9d}")
            if a.details:
                for key in ("misses", "modifier_failures", "false_answers"):
                    for line in s[key]:
                        print(f"      {key.upper():18s} {line}")
    live = [c for c in llm.calls if c.kind == "embed" and not c.replayed]
    print(f"embedding calls: {len(llm.calls)} ({len(live)} live, {sum(c.input_tokens for c in live)} tokens)")
    if a.write_baseline:
        data = json.loads(BASELINES.read_text(encoding="utf-8")) if BASELINES.exists() else {"entries": []}
        data["entries"].append({"label": a.write_baseline, "date": date.today().isoformat(), "signals_sha": retrieval_hash(),
                                "embedding_model": llm.embedding_model, "embedding_dimensions": llm.embedding_dimensions,
                                "modes": {m: {s: t.summary() for s, t in sorted(ts.items())} for m, ts in results.items()}})
        BASELINES.parent.mkdir(parents=True, exist_ok=True)
        BASELINES.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"baseline '{a.write_baseline}' recorded in {BASELINES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
