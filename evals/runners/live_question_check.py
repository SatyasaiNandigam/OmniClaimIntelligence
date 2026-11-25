"""Live look at the claim-specific questions the model proposes (T4.1). Needs LLM_API_KEY in .env; only synthetic data is sent.

    python evals/runners/live_question_check.py                # the five scenario fixtures S-01..S-05
    python evals/runners/live_question_check.py --cases dev    # the labelled cases (see data/synthetic/question_eval/cases.yaml)

Mode is "auto": a request already in data/cassettes/question_proposal.json is replayed (free, offline); a new one is sent and recorded.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from claimtower.core.config import load_settings  # noqa: E402
from claimtower.core.llm import Cassette, LLMClient  # noqa: E402
from claimtower.core.schemas import ClaimHeader, Fact, NotInForce, PolicySnapshot  # noqa: E402
from claimtower.evidence.llm_questions import make_register  # noqa: E402

CASSETTE = ROOT / "data" / "cassettes" / "question_proposal.json"


def fixture(scenario: str):
    d = ROOT / "data" / "fixtures" / scenario
    header = ClaimHeader(**json.loads((d / "intake.json").read_text(encoding="utf-8"))["claim_header"])
    pol = json.loads((d / "policy.json").read_text(encoding="utf-8"))
    snap = PolicySnapshot(**pol["policy_snapshot"]) if pol.get("policy_snapshot") else NotInForce(
        policy_id="?", loss_date=header.loss_date or "2026-01-01", reason="not in force")
    facts = [Fact(**f) for f in json.loads((d / "evidence.json").read_text(encoding="utf-8"))["facts"]]
    return header, snap, facts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    settings = load_settings()
    llm = LLMClient.from_settings(settings)
    llm.mode, llm.cassette = "auto", Cassette(CASSETTE)
    register = make_register(settings.pack, llm)
    for s in ["S-01", "S-02", "S-03", "S-04", "S-05"]:
        if args.only and s != args.only:
            continue
        header, snap, facts = fixture(s)
        r = register.build(header, snap, facts)
        print(f"\n== {s}: {len(r.questions)} question(s), {sum(v == 'model' for v in r.origin.values())} from the model ==")
        for qid, p in r.proposed.items():
            print(f"  + {qid} [{p.category}, {p.confidence}] {p.text}\n      clauses {list(p.clause_ids)} facts {list(p.fact_ids)} quote {p.evidence_quote!r}")
            print(f"      criteria {list(p.criteria)}")
        for x in r.rejected:
            print(f"  - REFUSED: {x.text!r}: {x.reason}")
        for w in r.warnings:
            print(f"  ! {w}")
    llm.cassette.save()


if __name__ == "__main__":
    main()
