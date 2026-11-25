"""Build per-scenario fixtures from the ground truth (BUILD_PLAN section 11 'fixture rule').

    python scripts/build_fixtures.py          # (re)write data/fixtures/
    python scripts/build_fixtures.py --check  # fail if data/fixtures is out of date

For every scenario S-xx it writes data/fixtures/S-xx/:
    claim_state.json   the complete EXPECTED final ClaimState (valid against the contract)
    <agent>.json       one slice per agent holding exactly the fields that agent owns (core.state.OWNERSHIP);
                       applying the nine slices in pipeline order to an empty ClaimState reproduces claim_state.json
    get_snapshot_result.json   (only S-03) the NotInForce answer a PolicyService returns
Consumers develop against these instead of waiting for upstream agents. All content is SYNTHETIC.
"""
import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from claimtower.core.config import load_settings  # noqa: E402
from claimtower.core.schemas import (  # noqa: E402
    ClaimState, SettlementInputs, ClauseCitation, ClauseRelation, CoverageIssue, DocumentRecord, Endorsement, NotInForce,
    OpenQuestion, PolicyClause, PolicySnapshot, ProposedAction, QAReport, RoutingDecision, SettlementResult,
)
from claimtower.core.state import OWNERSHIP, StatePatch, apply_patch  # noqa: E402
from claimtower.settlement.engine import SettlementCalculator  # noqa: E402
from claimtower.settlement.rules_loader import rules_by_version  # noqa: E402
from evals.ground_truth import load_all_ground_truth, load_scenarios  # noqa: E402

OUT = ROOT / "data" / "fixtures"
PIPELINE = ["intake", "policy", "evidence", "gap", "coverage", "settlement", "action", "audit", "supervisor"]
STATUS = {"S-01": "awaiting_evidence", "S-02": "awaiting_human", "S-03": "awaiting_human", "S-04": "awaiting_human",
          "S-05": "awaiting_evidence"}
IDX = json.loads((ROOT / "data/synthetic/policies/clause_index.json").read_text(encoding="utf-8"))
CLAUSES = {c["clause_id"]: c for c in IDX["clauses"]}
DOC_TITLE = {d["doc_id"]: d["title"] for d in IDX["documents"]}
CALC = SettlementCalculator(rules_by_version(load_settings(ROOT).pack.calculation))
RULES_VERSION = "sample-1.0"


def citation(cid: str) -> ClauseCitation:
    c = CLAUSES[cid]
    modified_by = [r["source"] for r in IDX["relations"] if r["target"] == cid and r["relation"] == "modifies"]
    return ClauseCitation(clause_id=cid, document=c["doc_id"], page=c["page"], span=tuple(c["span"]), text=c["text"], modified_by=modified_by)


def snapshot(sc, gt) -> PolicySnapshot | NotInForce:
    pas, loss = sc.pas_expectation, sc.header().loss_date
    if gt.snapshot.status == "not_in_force":
        return NotInForce(policy_id=pas["policy"], loss_date=loss, reason=f"PAS status '{pas['status']}' since {pas['lapsed_on']}; loss on {loss}")
    start, end = pas["period"]
    eff = set(gt.snapshot.effective_clause_ids)
    clauses = [PolicyClause(clause_id=cid, clause_type=CLAUSES[cid]["clause_type"], heading=CLAUSES[cid]["heading"], text=CLAUSES[cid]["text"],
                            document=CLAUSES[cid]["doc_id"], page=CLAUSES[cid]["page"], span=tuple(CLAUSES[cid]["span"])) for cid in sorted(eff)]
    dates = {e["id"]: e["effective_from"] for e in pas["endorsements"] if isinstance(e, dict)}
    endorsements = []
    for doc in IDX["documents"]:
        ids = [c["clause_id"] for c in IDX["clauses"] if c["doc_id"] == doc["doc_id"] and c["clause_id"] in eff]
        if doc["kind"] == "endorsement" and ids:
            endorsements.append(Endorsement(endorsement_id=doc["doc_id"], title=doc["title"],
                                            effective_from=dates.get(doc["doc_id"], start), clause_ids=ids))
    return PolicySnapshot(
        policy_id=pas["policy"], version="2025-01", effective_from=start, effective_to=end, pas_status="in_force", clauses=clauses,
        endorsements=endorsements, precedence_rules=[gt.snapshot.precedence_note],
        relations=[ClauseRelation(source_clause_id=r["source"], target_clause_id=r["target"], relation=r["relation"])
                   for r in IDX["relations"] if r["source"] in eff and r["target"] in eff],
        source_system_response_ref=f"PAS-MOCK/{pas['policy']}/{loss}",
    )


def settlement(gt) -> SettlementResult | None:
    st = gt.settlement
    if st is None:
        return None
    inputs = SettlementInputs(**{k: Decimal(v) for k, v in st.inputs.items()}, unavailable_inputs=st.unavailable_inputs)
    return CALC.compute_settlement(inputs, RULES_VERSION)


def actions(claim_id: str, names: list[str]) -> list[ProposedAction]:
    out = []
    for n in names:
        kind, _, party = n.partition(":")
        out.append(ProposedAction(action_id=f"ACT-{claim_id}-{len(out) + 1}", action_type=kind, responsible_party=party or "adjuster",
                                  execution_mode="draft" if kind.startswith(("draft", "present")) else "controlled_api",
                                  idempotency_key=f"{claim_id}:{n}"))
    return out


def build_state(sc, gt) -> tuple[ClaimState, dict]:
    manifest = json.loads((ROOT / "data/synthetic/claims" / sc.claim_id / "manifest.json").read_text(encoding="utf-8"))
    file_sha = {m["document_id"]: m["sha256"] for m in manifest["documents"]}          # sha256 of the original PDF bytes
    docs = [DocumentRecord(document_id=d.document_id, claim_id=sc.claim_id, filename=d.filename, sha256=file_sha[d.document_id],
                           page_count=len(d.pages), detected_type=d.doc_type, type_confidence=0.95, uploaded_by=d.uploaded_by,
                           blob_path=f"data/synthetic/claims/{sc.claim_id}/{d.filename}") for d in sc.documents]
    snap = snapshot(sc, gt)
    fields = {
        "intake": {"claim_header": sc.header(), "intake_checklist": [], "duplicate_candidates": []},
        "policy": {"policy_snapshot": snap if isinstance(snap, PolicySnapshot) else None},
        "evidence": {"documents": docs, "facts": gt.facts, "image_findings": []},
        "gap": {"open_questions": [OpenQuestion(**q.model_dump(exclude={"expected_answer"})) for q in gt.questions],
                "contradictions": gt.contradictions, "missing_evidence": gt.missing_evidence, "counterfactuals": gt.counterfactuals},
        "coverage": {"coverage_issues": [CoverageIssue(issue_id=i.issue_id, issue=i.issue, status=i.status, fact_ids=i.fact_ids, note=i.note,
                                                         clause_citations=[citation(c) for c in i.clause_ids]) for i in gt.issues]},
        "settlement": {"settlement": settlement(gt)},
        "action": {"proposed_actions": actions(sc.claim_id, gt.expected_actions), "drafts": []},
        "audit": {"qa_report": QAReport(passed=True), "audit_ref": f"AUDIT-{sc.claim_id}"},
        "supervisor": {"routing": RoutingDecision(**gt.routing.model_dump()), "human_approval_required": True,
                       "resolution_status": STATUS[sc.id]},
    }
    state = ClaimState(claim_id=sc.claim_id, correlation_id=f"corr-{sc.claim_id}")
    for agent in PIPELINE:
        assert set(fields[agent]) == OWNERSHIP[agent], f"{agent} slice must hold exactly its owned fields"
        state = apply_patch(state, StatePatch(agent_id=agent, updates=fields[agent]))
    return state, {a: {k: _plain(v) for k, v in fields[a].items()} for a in PIPELINE}, snap


def _plain(v):
    if isinstance(v, list):
        return [_plain(x) for x in v]
    return v.model_dump(mode="json") if hasattr(v, "model_dump") else v


def render() -> dict[Path, str]:
    sc_file, gts, files = load_scenarios(), load_all_ground_truth(), {}
    for sc in sc_file.scenarios:
        state, slices, snap = build_state(sc, gts[sc.id])
        d = OUT / sc.id
        files[d / "claim_state.json"] = json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
        for agent, content in slices.items():
            files[d / f"{agent}.json"] = json.dumps(content, indent=2, ensure_ascii=False) + "\n"
        if isinstance(snap, NotInForce):
            files[d / "get_snapshot_result.json"] = json.dumps({"not_in_force": snap.model_dump(mode="json")}, indent=2) + "\n"
    return files


def main() -> int:
    files = render()
    if "--check" in sys.argv:
        stale = [str(p.relative_to(OUT)) for p, t in files.items() if not p.exists() or p.read_text(encoding="utf-8") != t]
        print("Fixtures out of date - run scripts/build_fixtures.py: " + ", ".join(stale) if stale else "Fixtures up to date.")
        return 1 if stale else 0
    for p, text in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {len(files)} fixture files to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
