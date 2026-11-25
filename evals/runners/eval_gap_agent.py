"""Incremental reassessment with the real pipeline (T4.7): late evidence arrives on S-01 and only the questions it touches are re-run.

    python evals/runners/eval_gap_agent.py                  # offline: model answers come from the recorded cassette
    python evals/runners/eval_gap_agent.py --record         # ask the model for anything not recorded yet (needs LLM_API_KEY)
    python evals/runners/eval_gap_agent.py --write-baseline "label"

For each late-evidence scenario: the Gap agent assesses S-01 (full), the new document is added, the agent runs again (incremental) and a fresh agent
assesses the new state (full) for comparison. Reported: the questions re-run against the ones the document should touch (labelled), whether every
other question kept an identical result, and the model calls the incremental run needed against the full one. Statuses before and after are shown
(no label: the model may reach a different status on a different day); the structural guarantees are the result.
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

from claimtower.agents.gap import make_gap_agent  # noqa: E402
from claimtower.core.config import load_settings  # noqa: E402
from claimtower.core.llm import Cassette, LLMClient  # noqa: E402
from claimtower.core.schemas import ClaimState, DocumentRecord, Fact  # noqa: E402
from claimtower.core.state import apply_patch  # noqa: E402

BASELINES = ROOT / "evals" / "baselines" / "gap_agent.json"
CASSETTE = ROOT / "data" / "cassettes" / "gap_agent.json"

# late evidence on S-01: (id, document type, facts [(text, tagged questions)], the questions the document must touch)
SCENARIOS = [
    ("L1-plumber-explains-the-corrosion", "plumber_report",
     [("Plumber supplementary report: the corrosion is surface rust only and the joint failed suddenly", ["Q-SUDDEN", "Q-START"]),
      ("Plumber supplementary report: the joint was inspected on 10 January 2026 and no leakage predates 14 March 2026", ["Q-START"])], {"Q-SUDDEN", "Q-START"}),
    ("L2-plumber-confirms-long-term-leak", "plumber_report",
     [("Plumber supplementary report: the corrosion had been present for many months and the joint was seeping long before 14 March 2026", ["Q-SUDDEN", "Q-START"])],
     {"Q-SUDDEN", "Q-START"}),
    ("L3-surveyor-reassesses-extent-and-cost", "surveyor_report",
     [("Surveyor supplementary report: the moisture in the bedroom wall is a result of the 14 March 2026 escape of water", ["Q-EXTENT"]),
      ("Surveyor supplementary report: the assessed cost of repair covering all affected areas is INR 175,000", ["Q-AMOUNT"])], {"Q-EXTENT", "Q-AMOUNT"}),
]


def add_evidence(state: ClaimState, scenario: tuple) -> ClaimState:
    sid, doc_type, facts, _ = scenario
    doc = DocumentRecord(document_id=f"DOC-{sid}", claim_id=state.claim_id, filename=f"{sid}.pdf", sha256="0" * 64, detected_type=doc_type)
    new = [Fact(fact_id=f"F-{sid}-{n}", claim_id=state.claim_id, proposition=t, source_document_id=doc.document_id, page=1, evidence_text=t,
                evidence_type="neutral", related_question_ids=tags, extraction_confidence=0.9) for n, (t, tags) in enumerate(facts)]
    return state.model_copy(update={"documents": [*state.documents, doc], "facts": [*state.facts, *new]})


def step(agent, state):
    return apply_patch(state, agent.run(state))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--write-baseline", metavar="LABEL")
    a = ap.parse_args()
    settings = load_settings(ROOT)
    llm = LLMClient.from_settings(settings)
    llm.mode, llm.cassette = ("auto" if a.record else "replay"), Cassette(CASSETTE)
    base = ClaimState(**json.loads((ROOT / "data/fixtures/S-01/claim_state.json").read_text(encoding="utf-8")))
    rows = []
    print(f"{'scenario':40s} {'re-run':>24s} {'expected':>24s} {'others identical':>16s} {'calls inc/full':>14s}")
    for sc in SCENARIOS:
        agent = make_gap_agent(settings.pack, llm)
        s1 = step(agent, base)
        n1 = len(llm.calls)
        s2_in = add_evidence(s1, sc)
        s2 = step(agent, s2_in)
        n2 = len(llm.calls)
        run = agent.last_run
        full_agent = make_gap_agent(settings.pack, llm)
        step(full_agent, s2_in)
        n3 = len(llm.calls)
        before, after = {q.question_id: q for q in s1.open_questions}, {q.question_id: q for q in s2.open_questions}
        others = [q for q in before if q not in run.affected]
        identical = all(before[q] == after[q] for q in others)
        contra_same = all(c in s2.contradictions for c in s1.contradictions if c.question_id not in run.affected)
        req_same = all(m in s2.missing_evidence for m in s1.missing_evidence if m.minimum_evidence_request.addressee_role in run.kept_requests)
        row = {"scenario": sc[0], "re_run": sorted(run.affected), "expected": sorted(sc[3]), "exact": run.affected == sc[3], "others_identical": identical and contra_same and req_same,
               "incremental_calls": n2 - n1, "full_calls": n3 - n2, "status_changes": run.diff.describe(), "problems": run.problems,
               "statuses_before": {k: v.status for k, v in before.items()}, "statuses_after": {k: v.status for k, v in after.items()}}
        rows.append(row)
        print(f"{sc[0]:40s} {','.join(row['re_run']):>24s} {','.join(row['expected']):>24s} {str(row['others_identical']):>16s} {row['incremental_calls']:>6d}/{row['full_calls']:<7d}")
        print(f"    {row['status_changes']}")
    print(f"model calls: {len(llm.calls)} ({len([c for c in llm.calls if not c.replayed])} live)")
    if a.record:
        llm.cassette.save()
    if a.write_baseline:
        data = json.loads(BASELINES.read_text(encoding="utf-8")) if BASELINES.exists() else {"entries": []}
        data["entries"].append({"label": a.write_baseline, "date": date.today().isoformat(), "model": llm.text_model, "scenarios": rows})
        BASELINES.parent.mkdir(parents=True, exist_ok=True)
        BASELINES.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"baseline '{a.write_baseline}' recorded in {BASELINES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
