"""Export the generated requests for an ADJUSTER to rate ("sufficient and minimal", decision 2026-11-25: the adjuster is the reviewer).

    python evals/runners/export_request_review.py                    # writes evals/review/request_review.csv from the recorded model answers (offline)
    python evals/runners/export_request_review.py --split fresh2     # one split only

One row per generated request. The adjuster fills the last three columns: sufficient (yes/no: if the addressee answered every item fully, would
every need be met?), minimal (yes/no: does every item serve a need, nothing redundant?), comment. Open the CSV in a spreadsheet; it is UTF-8 with a BOM.
Rating results are recorded by hand in docs/assumptions.md (Q-P11) and in evals/baselines/requests.json.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from claimtower.core.config import load_settings  # noqa: E402
from claimtower.core.llm import Cassette, LLMClient  # noqa: E402
from claimtower.evidence.llm_requests import make_generator  # noqa: E402
from claimtower.evidence.questions import QuestionRegister  # noqa: E402
from evals.metrics.requests import load_cases  # noqa: E402

OUT = ROOT / "evals" / "review" / "request_review.csv"
CASSETTE = ROOT / "data" / "cassettes" / "request_eval.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["dev", "fresh", "fresh2", "all"])
    a = ap.parse_args()
    settings = load_settings(ROOT)
    llm = LLMClient.from_settings(settings)
    llm.mode, llm.cassette = "replay", Cassette(CASSETTE)
    gen = make_generator(settings.pack, llm)
    rows = 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "split", "addressee", "delivery", "questions it resolves", "what is still needed", "request items", "sufficient (yes/no)", "minimal (yes/no)", "comment"])
        for c in load_cases(QuestionRegister(settings.pack.question_templates)):
            if a.split not in ("all", c.split):
                continue
            qmap = {q.question_id: q for q in c.questions}
            for d in gen.generate(c.questions, c.sufficiency, c.facts, c.links, c.contradictions, c.received).details:
                needs = "\n".join(("CONFLICT: " + n.explanation) if n.kind == "conflict" else f"{n.question_id}: {n.criterion}" for n in d.needs)
                items = "\n".join(f"{i + 1}. {it.text}" for i, it in enumerate(d.items))
                w.writerow([c.id, c.split, d.role, d.delivery, ", ".join(f"{q} ({qmap[q].text})" for q in d.question_ids), needs, items, "", "", ""])
                rows += 1
    print(f"wrote {rows} requests to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
