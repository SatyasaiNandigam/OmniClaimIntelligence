"""End-to-end scenario cases (T8.1 / T8.2): whole claims through the real graph, offline and deterministic, with the model's answers replayed from recorded cassettes.

    ctx = Context(settings, library, workdir)
    result = CASES["S-01"](ctx)            # CaseResult: the final state, the runner, the checks (name -> passed) and, for a labelled scenario, the metrics against ground truth

Used three ways: `tests/e2e` asserts every check of every case (CI replays them), `evals/runners/run_eval.py` turns the same runs into the scorecard, and `scripts/replay_scenario.py`
prints one run. No case calls a live model: the two model cases read `data/cassettes/e2e_<case>.json` (record them once with `--record`, which needs LLM_API_KEY and sends only synthetic data).
"""
from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from claimtower.audit.trail import AuditTrail
from claimtower.composition import SYNTHETIC, build_commit_service, build_registry, ingest_claim_folder, policy_library_from_pdfs
from claimtower.core.config import Settings, load_settings
from claimtower.core.llm import Cassette, LLMClient
from claimtower.core.schemas import ClaimHeader, ClaimState
from claimtower.ingestion.store import DocumentStore
from claimtower.integrations import Adapters
from claimtower.integrations.dispatcher import Dispatcher
from claimtower.integrations.mock.claims_mock import MockClaimsAdapter
from claimtower.integrations.mock.pas_mock import MockPasAdapter
from claimtower.integrations.mock.recon_mock import MockReconAdapter
from claimtower.orchestration.graph import ClaimRunner
from claimtower.orchestration.overrides import OverrideStore

ROOT = Path(__file__).resolve().parents[1]
CASSETTES = ROOT / "data" / "cassettes"
CLAIM_OF = {"S-01": "CLM-0001", "S-02": "CLM-0002", "S-03": "CLM-0003", "S-04": "CLM-0004", "S-05": "CLM-0005"}


@dataclass
class Context:
    settings: Settings
    library: Any
    workdir: Path
    record: bool = False
    runs: int = 0                                                              # a fresh store directory per build, so a case can be run again in the same context

    @classmethod
    def create(cls, workdir: Path, record: bool = False) -> Context:
        s = load_settings(ROOT)
        return cls(s, policy_library_from_pdfs(s), workdir, record)


@dataclass
class CaseResult:
    name: str
    state: ClaimState
    runner: ClaimRunner
    claim_id: str
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, tuple[int, int]] = field(default_factory=dict)         # name -> (numerator, denominator)
    mode: str = "rules"                                                       # rules | rules + model (replayed)
    adapters: Adapters | None = None
    trail: AuditTrail | None = None

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


class Down:
    """A model that is unreachable."""

    text_model = vision_model = "down"
    calls: list = []
    mode = "replay"

    def derive(self, **kw: Any) -> Down:
        return self

    def __getattr__(self, name: str):
        def fail(*a: Any, **k: Any):
            raise ConnectionError("the model is not reachable")
        return fail


def build(ctx: Context, claim: str, *, folder: Path | None = None, without: tuple[str, ...] = (), llm: Any | None = None, tag: str = "") -> tuple[ClaimRunner, Adapters, AuditTrail, DocumentStore]:
    ctx.runs += 1
    store = DocumentStore(ctx.workdir / f"{claim}{tag}-{ctx.runs}" / "store")
    if folder is None:
        ingest_claim_folder(store, SYNTHETIC / "claims" / claim)
    else:
        for f in sorted(folder.glob("*.pdf")):
            store.ingest(claim, f.name, f.read_bytes(), uploaded_by="claimant")
    adapters = Adapters(MockPasAdapter(), MockClaimsAdapter(), MockReconAdapter())
    disp, ov, trail = Dispatcher(sleeper=lambda s: None), OverrideStore(), AuditTrail()
    lib = ctx.library.without(*without) if without else ctx.library
    reg = build_registry(ctx.settings, document_store=store, policy_library=lib, adapters=adapters, dispatcher=disp, overrides=ov, llm=llm)
    runner = ClaimRunner(reg, commit_service=build_commit_service(ctx.settings, adapters, disp), overrides=ov, settings=ctx.settings, trail=trail, llm=llm)
    return runner, adapters, trail, store


def decide(runner: ClaimRunner, claim: str, role: str, kind: str = "approve", reason: str = "", reviewer: str = "rev-1", **edits: str):
    return runner.decide(claim, {"decision": kind, "reviewer_id": reviewer, "reviewer_role": role, "reason": reason, "edits": edits})


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


# ------------------------------------------------------------------------------------------------------------- scoring against ground truth
def score_against_truth(state: ClaimState, sid: str, claim: str) -> dict[str, tuple[int, int]]:
    from evals.ground_truth import load_all_ground_truth

    gt = load_all_ground_truth()[sid]
    manifest = json.loads((SYNTHETIC / "claims" / claim / "manifest.json").read_text(encoding="utf-8"))
    want_type = {d["document_id"]: d["doc_type"] for d in manifest["documents"] if d.get("in_scenario", True)}
    m: dict[str, tuple[int, int]] = {}
    got_type = {d.document_id: d.detected_type for d in state.documents}
    m["document_type_accuracy"] = (sum(got_type.get(k) == v for k, v in want_type.items()), len(want_type))
    evidence = [_norm(f.evidence_text) for f in state.facts]
    m["fact_recall"] = (sum(any(_norm(g.evidence_text) in e or e in _norm(g.evidence_text) for e in evidence) for g in gt.facts), len(gt.facts))
    qs = {q.question_id: q.status for q in state.open_questions}
    m["question_status_agreement"] = (sum(qs.get(q.question_id) == q.status for q in gt.questions), len(gt.questions))
    m["contradiction_recall"] = (sum(any(c.question_id == g.question_id for c in state.contradictions) for g in gt.contradictions), len(gt.contradictions))
    roles = {r.minimum_evidence_request.addressee_role for r in state.missing_evidence}
    m["request_addressee_recall"] = (sum(g.minimum_evidence_request.addressee_role in roles for g in gt.missing_evidence), len(gt.missing_evidence))
    issues = {i.issue_id: i.status for i in state.coverage_issues}
    m["coverage_status_agreement"] = (sum(issues.get(i.issue_id) == i.status for i in gt.issues), len(gt.issues))
    abst = [i for i in gt.issues if i.status == "abstain"]
    if abst:
        m["abstention_appropriateness"] = (sum(issues.get(i.issue_id) == "abstain" for i in abst), len(abst))
    m["settlement_exact_match"] = (int(_settlement_ok(state, sid)), 1)
    r = state.routing
    m["routing_tier_agreement"] = (int(r is not None and r.tier == gt.routing.tier), 1)
    m["routing_codes_agreement"] = (int(r is not None and r.reason_codes == gt.routing.reason_codes), 1)
    have = {f"{a.action_type}:{a.responsible_party}" for a in state.proposed_actions} | {a.action_type for a in state.proposed_actions}
    m["action_agreement"] = (sum(g in have for g in gt.expected_actions), len(gt.expected_actions))
    m["citation_correctness"] = (int(state.qa_report is not None and state.qa_report.passed), 1)
    return m


# ------------------------------------------------------------------------------------------------------------- the cases
def gate_case(sid: str, *, llm_mode: bool = False) -> Callable[[Context], CaseResult]:
    claim = CLAIM_OF[sid]
    name = sid + ("-with-model" if llm_mode else "")

    def run(ctx: Context) -> CaseResult:
        llm = None
        if llm_mode:
            llm = LLMClient.from_settings(ctx.settings)
            llm.mode, llm.cassette = ("auto" if ctx.record else "replay"), Cassette(CASSETTES / f"e2e_{name}.json")
        runner, _, trail, _ = build(ctx, claim, without=("END-01",) if sid == "S-05" else (), llm=llm, tag="-m" if llm_mode else "")
        st = runner.start(ClaimState(claim_id=claim))
        if llm is not None and ctx.record:
            llm.cassette.save()
        state = runner.state(claim)
        res = CaseResult(name, state, runner, claim, mode="rules + model (replayed)" if llm_mode else "rules", trail=trail)
        res.checks = {"reaches the gate": st.phase == "awaiting_human_decision", "no agent failed": not runner.events.of_kind(claim, "agent_failed"),
                      "quality checks passed": state.qa_report is not None and state.qa_report.passed,
                      "rebuilt from the audit trail alone": trail.reconstruct(claim).model_dump(mode="json") == state.model_dump(mode="json") and trail.verify() == [],
                      "no payment before a decision": not runner.events.of_kind(claim, "commit")}
        if not llm_mode:                                                       # the model-free numbers must not move: these are the deterministic parts of the labelled claim
            res.checks["settlement matches the labelled calculation"] = _settlement_ok(state, sid)
            if sid != "S-03":
                res.checks["coverage rows cite the real snapshot"] = bool(state.coverage_issues) and all(i.clause_citations for i in state.coverage_issues)
        res.metrics = score_against_truth(state, sid, claim)
        return res
    return run


def _settlement_ok(state: ClaimState, sid: str) -> bool:
    """The calculation equals the labelled one (status, missing inputs, payable). A claim with nothing in force has no settlement, and the label has none either."""
    from evals.ground_truth import load_all_ground_truth

    want, s = load_all_ground_truth()[sid].settlement, state.settlement
    if want is None or s is None:
        return want is None and s is None
    payable = Decimal(want.payable) if want.payable else None
    return s.status == want.status and list(s.missing_inputs) == list(want.missing_inputs) and s.payable == payable


def late_evidence(ctx: Context) -> CaseResult:
    """S-01: a supplementary plumber report arrives after the first assessment: only what depends on it re-runs."""
    claim = "CLM-0001"
    runner, _, trail, store = build(ctx, claim, tag="-late")
    runner.start(ClaimState(claim_id=claim))
    first = runner.state(claim)
    decide(runner, claim, "supervisor", "request_more_evidence", "the plumber must explain the corrosion")
    store.ingest(claim, "plumber_supplementary.pdf", (SYNTHETIC / "claims" / claim / "plumber_report.pdf").read_bytes(), document_id="DOC-LATE-1", uploaded_by="plumber")
    st = runner.reassess(claim, from_node="evidence")
    state = runner.state(claim)
    old = {f.fact_id: f for f in first.facts}
    res = CaseResult("S-01-late-evidence", state, runner, claim, trail=trail)
    res.checks = {"back at the gate": st.phase == "awaiting_human_decision", "at least one node was skipped": len(st.skipped_nodes) >= 1,
                  "intake and policy were not re-run": not {"intake", "policy"} & set([e.node for e in runner.events.of_kind(claim, "agent_finished")][-6:]),
                  "unchanged documents keep their facts": all(state_f == old[state_f.fact_id] for state_f in state.facts if state_f.fact_id in old),
                  "the new document added facts": any(f.source_document_id == "DOC-LATE-1" for f in state.facts) and len(state.facts) > len(first.facts),
                  "the state version moved on": state.state_version > first.state_version,
                  "rebuilt from the audit trail alone": trail.reconstruct(claim).model_dump(mode="json") == state.model_dump(mode="json")}
    res.metrics = {"nodes_skipped_on_new_evidence": (len(st.skipped_nodes), 7)}
    return res


def approve_pays_only_for_a_supervisor(ctx: Context) -> CaseResult:
    claim = "CLM-0002"
    runner, ad, trail, _ = build(ctx, claim, tag="-pay")
    runner.start(ClaimState(claim_id=claim))
    decide(runner, claim, "senior_adjuster", "approve", reviewer="rev-2")
    none_yet = ad.recon.instruction_count == 0
    runner2, ad2, trail2, _ = build(ctx, claim, tag="-pay2")
    runner2.start(ClaimState(claim_id=claim))
    decide(runner2, claim, "supervisor", "approve", reviewer="rev-9")
    inst = next(iter(ad2.recon._instructions.values()), None)
    res = CaseResult("S-02-approve", runner2.state(claim), runner2, claim, adapters=ad2, trail=trail2)
    res.checks = {"a senior adjuster's approval pays nothing": none_yet, "a supervisor's approval instructs one payment": ad2.recon.instruction_count == 1,
                  "the payment is the engine's amount": inst is not None and inst.amount == Decimal(80000) and inst.approved_by == "rev-9",
                  "the claim is registered and approved in the claims system": ad2.claims.get_claim(claim).status == "approved",
                  "the commit is in the audit trail": any(r.data.get("kind") == "commit" for r in trail2.records(claim, "event"))}
    return res


def edit_then_approve(ctx: Context) -> CaseResult:
    claim = "CLM-0002"
    runner, ad, trail, _ = build(ctx, claim, tag="-edit")
    runner.start(ClaimState(claim_id=claim))
    decide(runner, claim, "senior_adjuster", "edit", "a second upgrade was confirmed", non_covered_items_total="28000")
    edited = runner.state(claim).settlement.payable
    decide(runner, claim, "supervisor", "approve", reviewer="rev-9")
    inst = next(iter(ad.recon._instructions.values()), None)
    res = CaseResult("S-02-edit-then-approve", runner.state(claim), runner, claim, adapters=ad, trail=trail)
    res.checks = {"the engine recalculated with the reviewer's input": edited == Decimal(60000), "the edit is recorded with who and why": any(
        e.data["data"].get("edits") and e.data["data"].get("reason") for e in trail.records(claim, "event") if e.data["kind"] == "human_decision"),
        "the edited amount is what is paid": inst is not None and inst.amount == Decimal(60000),
        "rebuilt from the audit trail alone": trail.reconstruct(claim).model_dump(mode="json") == runner.state(claim).model_dump(mode="json")}
    return res


def endorsement_arrives(ctx: Context) -> CaseResult:
    """S-05: the endorsement wording is missing at first (no number is produced); when it arrives the claim is re-assessed from the policy node."""
    claim = "CLM-0005"
    runner, _, trail, _ = build(ctx, claim, without=("END-01",), tag="-end")
    runner.start(ClaimState(claim_id=claim))
    before = runner.state(claim).settlement
    decide(runner, claim, "supervisor", "request_more_evidence", "wait for the endorsement")
    runner.registry["policy"].agent._snapshots._library = ctx.library                     # the wording is now in the library
    runner.reassess(claim, from_node="policy")
    after = runner.state(claim)
    res = CaseResult("S-05-endorsement-arrives", after, runner, claim, trail=trail)
    res.checks = {"without the wording no payable is produced": before.status == "cannot_calculate" and before.payable is None and before.missing_inputs == ["water_damage_sublimit"],
                  "with the wording the settlement is calculated": after.settlement.status == "calculated" and after.settlement.payable == Decimal(100000),
                  "the endorsement is in the snapshot": any(c.clause_id.startswith("END-01/") for c in after.policy_snapshot.clauses),
                  "the sublimit question is no longer missing": {q.question_id: q.status for q in after.open_questions}["Q-SUBLIMIT"] != "missing"}
    return res


def sample_claim(ctx: Context) -> CaseResult:
    """The claim in samples/claim_sample_1 (a claim the system has never seen), created with a header and its documents uploaded."""
    claim = "CLM-SAMPLE-1"
    folder = ROOT / "samples" / "claim_sample_1"
    runner, ad, trail, _ = build(ctx, claim, folder=folder, tag="-sample")
    header = ClaimHeader(claim_id=claim, policy_ref="POL-1013", loss_date="2026-05-22", loss_type="water_damage", loss_description="Dishwasher inlet hose burst")
    st = runner.start(ClaimState(claim_id=claim, claim_header=header))
    state = runner.state(claim)
    res = CaseResult("sample-claim-1", state, runner, claim, adapters=ad, trail=trail)
    res.checks = {"reaches the gate": st.phase == "awaiting_human_decision", "the policy was found and in force": state.policy_snapshot is not None,
                  "the settlement uses the surveyor's assessment, not the claimant's estimate": state.settlement.assessed_loss == Decimal(64500),
                  "quality checks passed": state.qa_report.passed, "the claim is registered in the claims system": ad.claims.claim_count == 1}
    return res


def model_unavailable(ctx: Context) -> CaseResult:
    claim = "CLM-0002"
    runner, _, trail, _ = build(ctx, claim, llm=Down(), tag="-down")
    st = runner.start(ClaimState(claim_id=claim))
    state = runner.state(claim)
    res = CaseResult("S-02-model-unavailable", state, runner, claim, trail=trail)
    res.checks = {"reaches the gate on rules alone": st.phase == "awaiting_human_decision" and not runner.events.of_kind(claim, "agent_failed"),
                  "the calculation is unchanged": state.settlement.payable == Decimal(80000), "nothing is concluded without a reading": {
                      i.issue_id: i.status for i in state.coverage_issues}["ISS-COVER-WATER"] != "established",
                  "a person decides (complex)": state.routing.tier == "complex"}
    return res


CASES: dict[str, Callable[[Context], CaseResult]] = {
    "S-01": gate_case("S-01"), "S-02": gate_case("S-02"), "S-03": gate_case("S-03"), "S-04": gate_case("S-04"), "S-05": gate_case("S-05"),
    "S-01-late-evidence": late_evidence, "S-02-approve": approve_pays_only_for_a_supervisor, "S-02-edit-then-approve": edit_then_approve,
    "S-05-endorsement-arrives": endorsement_arrives, "sample-claim-1": sample_claim, "S-02-model-unavailable": model_unavailable,
    "S-01-with-model": gate_case("S-01", llm_mode=True), "S-02-with-model": gate_case("S-02", llm_mode=True),
}
MODEL_CASES = ("S-01-with-model", "S-02-with-model")


def has_cassette(name: str) -> bool:
    return (CASSETTES / f"e2e_{name}.json").is_file()


def cleanup(workdir: Path) -> None:
    shutil.rmtree(workdir, ignore_errors=True)


__all__ = ["CASES", "CaseResult", "Context", "MODEL_CASES", "has_cassette", "score_against_truth"]
