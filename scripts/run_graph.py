"""Run one synthetic claim through the whole graph and print what happened, node by node.

    python scripts/run_graph.py                                  # CLM-0001: real Evidence, Policy and Gap agents, the rest YET TO IMPLEMENT (mocks)
    python scripts/run_graph.py --claim CLM-0003                 # the lapsed policy: coverage and settlement are skipped
    python scripts/run_graph.py --llm                            # the Gap agent also uses the model (needs LLM_API_KEY in .env; sends only synthetic data)
    python scripts/run_graph.py --mock-only                      # every agent a mock: the flow on the labelled fixtures alone
    python scripts/run_graph.py --decide approve --role senior_adjuster
    python scripts/run_graph.py --decide reject --role supervisor --reason "not covered by the policy as reviewed"
    python scripts/run_graph.py --diagram                        # print the graph as a Mermaid diagram

Without --decide the run stops at the human gate (the claim stays paused: that is the normal state of a claim waiting for an adjuster).
Nothing is sent to any system: the commit step is a mock.
"""
import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claimtower.composition import SYNTHETIC, build_registry, ingest_claim_folder, policy_library_from_pdfs  # noqa: E402
from claimtower.core.config import load_settings  # noqa: E402
from claimtower.core.schemas import ClaimState  # noqa: E402
from claimtower.ingestion.store import DocumentStore  # noqa: E402
from claimtower.orchestration.graph import ClaimRunner  # noqa: E402
from claimtower.orchestration.supervisor import HumanDecision  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", default="CLM-0001", help="a synthetic claim: CLM-0001 .. CLM-0005")
    ap.add_argument("--mock-only", action="store_true", help="run every agent as a mock (fixtures only)")
    ap.add_argument("--llm", action="store_true", help="give the Gap agent the live model (default: rules only, fact questions abstain)")
    ap.add_argument("--decide", choices=["approve", "reject", "request_more_evidence"])
    ap.add_argument("--role", default="senior_adjuster")
    ap.add_argument("--reviewer", default="demo-reviewer")
    ap.add_argument("--reason", default="")
    ap.add_argument("--diagram", action="store_true")
    a = ap.parse_args()

    settings = load_settings(ROOT)
    llm = None
    if a.llm:
        from dotenv import load_dotenv

        from claimtower.core.llm import LLMClient
        load_dotenv(ROOT / ".env")
        llm = LLMClient.from_settings(settings)
    if a.mock_only:
        registry = build_registry(settings, force_mock=("gap",), llm=llm)
    else:
        store = DocumentStore(Path(tempfile.mkdtemp()) / "store")
        ingest_claim_folder(store, SYNTHETIC / "claims" / a.claim)
        registry = build_registry(settings, document_store=store, policy_library=policy_library_from_pdfs(settings), llm=llm)
    runner = ClaimRunner(registry)
    if a.diagram:
        print(runner.diagram())
        return 0

    print("AGENTS")
    for row in registry.describe():
        print(f"  {row['node']:11s} {row['status']:26s} {row['task']:6s} {row['will_do'][:70]}")
    print(f"\nRUNNING {a.claim}")
    status = runner.start(ClaimState(claim_id=a.claim))
    for e in runner.events_for(a.claim):
        tag = "mock" if e.mock else "real" if e.kind == "agent_finished" else "    "
        print(f"  {e.seq:2d} [{tag}] {e.kind:15s} {e.node:11s} {e.message[:150]}")
    print(f"\nPHASE {status.phase} (state v{status.state_version}, resolution_status {status.resolution_status})")
    if status.gate:
        g = status.gate
        print(f"  tier {g['tier']}, reasons {g['reason_codes']}, approver {g['required_approver_role']}")
        print(f"  open questions {g['open_questions']}")
        print(f"  mock agents behind this dossier: {g['mock_agents']}")
    if status.halted_reason:
        print(f"  HALTED at {status.paused_at}: {status.halted_reason}")
    if a.decide and status.phase == "awaiting_human_decision":
        status = runner.decide(a.claim, HumanDecision(decision=a.decide, reviewer_id=a.reviewer, reviewer_role=a.role, reason=a.reason))
        for e in runner.events_for(a.claim)[-3:]:
            print(f"  {e.seq:2d} {e.kind:22s} {e.message[:140]}")
        if status.gate and status.gate.get("error"):
            print(f"\nREFUSED: {status.gate['error']} (the gate is still open)")
        else:
            print(f"\nPHASE {status.phase}, resolution_status {status.resolution_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
