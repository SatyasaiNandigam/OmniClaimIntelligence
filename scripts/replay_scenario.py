"""Replay one end-to-end scenario case and print what happened (T8.2). Offline: the model's answers come from the recorded cassette.

    python scripts/replay_scenario.py --list
    python scripts/replay_scenario.py S-01                       # one case: the checks, the metrics against the labelled scenario, the run log
    python scripts/replay_scenario.py S-01-with-model            # the same claim with the model's recorded answers
    python scripts/replay_scenario.py S-01-with-model --record   # (re)record the model's answers (needs LLM_API_KEY in .env; sends only synthetic data)
    python scripts/replay_scenario.py S-02-approve --pack        # also print the decision pack built from the audit trail
"""
import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from evals.e2e import CASES, MODEL_CASES, Context, has_cassette  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--pack", action="store_true")
    a = ap.parse_args()
    if a.list or not a.case:
        for n in CASES:
            print(f"{n:28s}" + ("  (model answers replayed from data/cassettes/e2e_%s.json%s)" % (n, "" if has_cassette(n) else ": NOT RECORDED YET") if n in MODEL_CASES else ""))
        return 0
    if a.case not in CASES:
        print(f"unknown case '{a.case}'; use --list")
        return 2
    if a.case in MODEL_CASES and not a.record and not has_cassette(a.case):
        print("no recorded model answers for this case yet: run it once with --record")
        return 2
    ctx = Context.create(Path(tempfile.mkdtemp(prefix="claimtower-replay-")), record=a.record)
    r = CASES[a.case](ctx)
    print(f"case {r.name} ({r.mode}): {'PASS' if r.passed else 'FAIL'}\n")
    for k, v in r.checks.items():
        print(f"  [{'x' if v else ' '}] {k}")
    if r.metrics:
        print("\nagainst the labelled scenario:")
        for k, (n, d) in r.metrics.items():
            print(f"  {k:30s} {n}/{d}" + (f"  ({n / d:.0%})" if d else ""))
    print("\nrun log:")
    for e in r.runner.events_for(r.claim_id):
        print(f"  {e.seq:3d} {e.kind:22s} {e.node:11s} {e.message[:110]}")
    if a.pack and r.trail is not None:
        from claimtower.audit.explain_pack import build_pack
        print("\n" + build_pack(r.trail, r.claim_id))
    return 0 if r.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
