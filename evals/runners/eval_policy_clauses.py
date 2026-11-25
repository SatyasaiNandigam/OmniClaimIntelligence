"""Evaluate policy decomposition (T3.1) and (optionally) record a baseline.

    python evals/runners/eval_policy_clauses.py                    # print the report
    python evals/runners/eval_policy_clauses.py --details          # also list wrong types, abstentions, misses, spurious clauses
    python evals/runners/eval_policy_clauses.py --write-baseline "label"
    python evals/runners/eval_policy_clauses.py --llm               # rules + the optional model typing (answers replayed from the cassette)
    python evals/runners/eval_policy_clauses.py --llm --record      # ... asking the model for anything not yet recorded (needs LLM_API_KEY)

Rules only by default: no model is called. Baselines go to evals/baselines/policy_clauses.json (rules) or policy_clauses_llm.json (--llm),
with a hash of policy_structure.yaml. Model answers are cached in data/cassettes/policy_typing.json.
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
from claimtower.policy.decompose import PolicyDecomposer  # noqa: E402
from claimtower.policy.llm_typing import LLMClauseTyper  # noqa: E402
from evals.metrics.policy_clauses import all_cases, evaluate  # noqa: E402

BASELINES = ROOT / "evals" / "baselines" / "policy_clauses.json"
LLM_BASELINES = ROOT / "evals" / "baselines" / "policy_clauses_llm.json"
CASSETTE = ROOT / "data" / "cassettes" / "policy_typing.json"


def signals_hash() -> str:
    """Hash of what DECOMPOSITION uses: noise, section / clause patterns, endorsement titles, type rules. The `relations`, `retrieval` and `llm_typing`
    sections are deliberately left out (they do not change a clause), so editing them does not invalidate this baseline."""
    ps = load_settings(ROOT).pack.policy_structure
    return hashlib.sha256(json.dumps(ps.model_dump(mode="json", exclude={"relations", "retrieval", "llm_typing"}), sort_keys=True).encode()).hexdigest()[:12]


def full_hash() -> str:
    """Hash of everything that decides clauses, relations and effective sets: all of policy_structure.yaml except the retrieval settings."""
    ps = load_settings(ROOT).pack.policy_structure
    return hashlib.sha256(json.dumps(ps.model_dump(mode="json", exclude={"retrieval", "llm_typing"}), sort_keys=True).encode()).hexdigest()[:12]


def retrieval_hash() -> str:
    """Hash of policy_structure.yaml without the optional model typing: used by the retrieval baseline (retrieval runs on rule-typed clauses)."""
    ps = load_settings(ROOT).pack.policy_structure
    return hashlib.sha256(json.dumps(ps.model_dump(mode="json", exclude={"llm_typing"}), sort_keys=True).encode()).hexdigest()[:12]


def run():
    return evaluate(PolicyDecomposer(load_settings(ROOT).pack.policy_structure))


def run_llm(record: bool = False, effort: str | None = None, splits: list[str] | None = None):
    """Rules + the model tie-breaker. Replay by default: a clause whose answer was never recorded is declined (and counted), not guessed."""
    settings = load_settings(ROOT)
    llm = LLMClient.from_settings(settings)
    llm.mode, llm.cassette = ("auto" if record else "replay"), Cassette(CASSETTE)
    cfg = settings.pack.policy_structure.llm_typing
    typer = LLMClauseTyper(llm, settings.pack, cfg.model_copy(update={"reasoning_effort": effort}) if effort else cfg)
    cases = [c for c in all_cases() if not splits or c.split in splits]
    return evaluate(PolicyDecomposer(settings.pack.policy_structure, typer=typer), cases), typer, llm


def pct(x):
    return "  n/a" if x is None else f"{x:5.1%}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", action="store_true")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--splits", nargs="+", help="only these splits (the unseen split should stay untouched while experimenting)")
    ap.add_argument("--effort", choices=["minimal", "low", "medium", "high"], help="override llm_typing.reasoning_effort for this run")
    ap.add_argument("--write-baseline", metavar="LABEL")
    a = ap.parse_args()
    typer = llm = None
    if a.llm:
        reports, typer, llm = run_llm(a.record, a.effort, a.splits)
    else:
        reports = run()
    print(f"{'split':8s} {'labelled':>8s} {'found':>6s} {'prec':>6s} {'recall':>7s} {'F1':>6s} {'type acc':>9s} {'abstain':>8s} "
          f"{'wrong type':>10s} {'cite problems':>13s}")
    for name in sorted(reports):
        r = reports[name]
        print(f"{name:8s} {r.labelled:8d} {r.detected:6d} {pct(r.precision):>6s} {pct(r.recall):>7s} {pct(r.f1):>6s} "
              f"{pct(r.type_accuracy):>9s} {len(r.abstained):8d} {len(r.wrong_types):10d} {len(r.citation_problems):13d}")
        if a.details:
            for key, items in (("WRONG TYPE", r.wrong_types), ("ABSTAINED", r.abstained), ("MISSED", r.missed), ("SPURIOUS", r.spurious),
                               ("CITATION", r.citation_problems), ("WARNING", r.warnings)):
                for line in items:
                    print(f"    {key:10s} {line}")
    if typer is not None:
        live = [c for c in llm.calls if not c.replayed]
        print(f"model typing: {typer.asked} clause(s) offered, {len(live)} live call(s), {sum(c.input_tokens + c.output_tokens for c in live)} tokens")
    if a.write_baseline:
        target = LLM_BASELINES if a.llm else BASELINES
        data = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {"entries": []}
        data["entries"].append({"label": a.write_baseline, "date": date.today().isoformat(), "signals_sha": signals_hash(),
                                "splits": {k: v.summary() for k, v in sorted(reports.items())}})
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"baseline '{a.write_baseline}' recorded in {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
