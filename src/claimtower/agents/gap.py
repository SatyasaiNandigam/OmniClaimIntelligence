"""Gap agent (T4.7): evidence -> open questions, contradictions, minimum-evidence requests and counterfactual scenarios.

Owns (writes) exactly `open_questions`, `contradictions`, `missing_evidence`, `counterfactuals` (core.state.OWNERSHIP). It reads the claim header, the
policy snapshot, the facts and documents of the evidence agent and, for the scenarios, the coverage issues. It is a thin wrapper: WHAT each output is
comes from the capabilities of T4.1-T4.6 (register, linker, detector, classifier, request generator, counterfactual engine).

Incremental. A claim is assessed in full once (cold start). After that, when evidence arrives, only what it touches is re-run:
  1. the NEW or CHANGED facts are linked to the questions (tags + model); nothing else is sent to the model for linking;
  2. `affected_questions` names the questions those facts (or a changed policy) touch;
  3. contradictions, sufficiency and scenarios are recomputed for those questions only; requests only for the addressees whose needs changed;
  4. every other question, contradiction, request and scenario is carried over EXACTLY as it was (not re-judged: a model could answer differently).
The link directions and the per-criterion assessments are not part of ClaimState, so the agent keeps them in `GapMemory` (serialisable: save it with the
checkpoint with `memory.model_dump_json()`, restore it with `GapMemory.model_validate_json`). Without memory (a restart, another claim) the run is a full one:
slower, never wrong. The patch rationale names what was affected, what was kept and which statuses changed.

Not in force / no policy: `policy_snapshot` None is read as "the policy system answered: nothing in force" when the claim has a policy number and a loss
date (the Policy agent's convention), and as "no answer yet" otherwise.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from claimtower.core.schemas import ClaimHeader, ClaimState, NotInForce, OpenQuestion, PolicySnapshot
from claimtower.core.state import OWNERSHIP, StatePatch
from claimtower.evidence.contradiction import ContradictionDetector
from claimtower.evidence.counterfactual import CounterfactualEngine
from claimtower.evidence.incremental import GapDiff, affected_questions, diff_gap, new_or_changed_facts, snapshot_changed
from claimtower.evidence.linker import FactQuestionLinker, Link
from claimtower.evidence.min_evidence import GAP_ORDER, MinimumEvidenceGenerator
from claimtower.evidence.questions import QuestionRegister
from claimtower.evidence.sufficiency import QuestionAssessment, SufficiencyClassifier, SufficiencyResult

MODEL_QUESTION_PREFIX = "Q-M-"                       # ids of the claim-specific questions a model proposed (evidence.llm_questions.question_id)


class GapMemory(BaseModel):
    """What the agent must remember between runs (not in ClaimState): the state it last saw with its own outputs applied, the link directions,
    the per-question assessments and the fingerprints of the needs behind each request. Serialisable."""

    state: ClaimState
    links: list[Link] = []
    assessments: list[QuestionAssessment] = []
    request_fingerprints: dict[str, str] = {}


@dataclass
class GapRun:
    """What the last run did (for in-process callers, the run log and tests)."""

    cold: bool
    affected: set[str]
    kept: list[str]
    diff: GapDiff
    regenerated_requests: list[str] = field(default_factory=list)
    kept_requests: list[str] = field(default_factory=list)
    recomputed_scenarios: list[str] = field(default_factory=list)
    review_questions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    retained_contradictions: list[str] = field(default_factory=list)       # earlier conflicts kept: no checked evidence resolved them
    retracted_contradictions: list[str] = field(default_factory=list)      # earlier conflicts a new fact resolved, with the explanation
    settled_on_evidence: list[str] = field(default_factory=list)           # questions the facts together answered although a listed example was not met (with the facts)


class GapAgent:
    agent_id: ClassVar[str] = "gap"
    owns: ClassVar[frozenset[str]] = OWNERSHIP["gap"]

    def __init__(self, *, register: QuestionRegister, register_rules: QuestionRegister, linker: FactQuestionLinker, detector: ContradictionDetector,
                 classifier: SufficiencyClassifier, generator: MinimumEvidenceGenerator, engine: CounterfactualEngine,
                 modes: dict[str, str] | None = None, policy_questions: Sequence[str] = ()) -> None:
        self.register, self.register_rules = register, register_rules
        self.linker, self.detector, self.classifier, self.generator, self.engine = linker, detector, classifier, generator, engine
        self.modes = modes or {}
        self.policy_questions = set(policy_questions)       # questions whose criteria are decided from the policy (they re-run when it changes)
        self._memories: dict[str, GapMemory] = {}            # one memory per claim: an API serves many claims with one agent
        self._last_claim: str | None = None
        self.last_run: GapRun | None = None
        self._memory_dir: Path | None = None

    @property
    def memory(self) -> GapMemory | None:
        """The memory of the claim assessed last (the one a caller that handles a single claim means)."""
        return self._memories.get(self._last_claim) if self._last_claim else None

    @memory.setter
    def memory(self, value: GapMemory | None) -> None:
        if value is None:
            self._memories.pop(self._last_claim or "", None)
        else:
            self._memories[value.state.claim_id] = value
            self._last_claim = value.state.claim_id

    # ------------------------------------------------------------------------------------------------ keeping the memory on disk
    def enable_persistence(self, directory: Path | str) -> None:
        """Keep each claim's memory in `<directory>/<claim id>.json` (written at the end of every run), so that after a restart with the same folder a re-assessment is incremental again."""
        self._memory_dir = Path(directory)
        self._memory_dir.mkdir(parents=True, exist_ok=True)

    def _save_memory(self, mem: GapMemory) -> None:
        if self._memory_dir is None:
            return
        path = self._memory_dir / f"{mem.state.claim_id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(mem.model_dump_json(), encoding="utf-8", newline=chr(10))
        tmp.replace(path)                                                          # never half-written

    def _load_memory(self, state: ClaimState) -> GapMemory | None:
        """The memory saved for this claim, if it still describes the claim: its outputs must be exactly the ones the claim holds now. If the server stopped between this agent's run and the
        save of the claim, the file is AHEAD of the claim and is not used: the run is a full one (slower, never wrong)."""
        if self._memory_dir is None:
            return None
        path = self._memory_dir / f"{state.claim_id}.json"
        try:
            mem = GapMemory.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        s = mem.state
        same = (s.open_questions == state.open_questions and s.contradictions == state.contradictions and s.missing_evidence == state.missing_evidence
                and s.counterfactuals == state.counterfactuals)
        return mem if same else None

    def memory_of(self, claim_id: str) -> GapMemory | None:
        return self._memories.get(claim_id)

    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def _snapshot_view(state: ClaimState) -> PolicySnapshot | NotInForce | None:
        if state.policy_snapshot is not None:
            return state.policy_snapshot
        h = state.claim_header
        if h is not None and h.policy_ref and h.loss_date:
            return NotInForce(policy_id=h.policy_ref, loss_date=h.loss_date, reason="no policy snapshot is in force on the loss date")
        return None

    def run(self, state: ClaimState) -> StatePatch:
        mem = self._memories.get(state.claim_id) or self._load_memory(state)
        if mem is not None and state.claim_id not in self._memories:
            self._memories[state.claim_id] = mem                                                      # restored from the store
        cold = mem is None
        view = self._snapshot_view(state)
        usable = [f for f in state.facts if f.verification_status != "rejected"]
        usable_ids = {f.fact_id for f in usable}
        docs = {d.document_id: d.detected_type for d in state.documents}
        new_facts = list(state.facts) if cold else new_or_changed_facts(mem.state, state)           # type: ignore[union-attr]
        header = state.claim_header or ClaimHeader(claim_id=state.claim_id)

        # 1. the questions: templates (cheap, rules) every run; the model's claim-specific ones only when there is something new to read
        reg = self.register if (cold or new_facts) else self.register_rules
        built = reg.build(header, view, usable)
        prev_by_id = {q.question_id: q for q in state.open_questions}
        questions = list(built.questions)
        have = {q.question_id for q in questions}
        questions += [q for q in state.open_questions if q.question_id.startswith(MODEL_QUESTION_PREFIX) and q.question_id not in have]    # earlier model questions stay
        order = {q.question_id: i for i, q in enumerate(questions)}
        warnings = list(built.warnings)

        # 2. link only the new or changed facts; the others keep their earlier links
        gone = {f.fact_id for f in new_facts}
        kept_links = [] if cold else [x for x in mem.links if x.fact_id not in gone and x.fact_id in usable_ids and x.question_id in order]   # type: ignore[union-attr]
        linked = self.linker.link([f for f in new_facts if f.fact_id in usable_ids], questions)
        warnings += linked.warnings
        links = kept_links + linked.links

        # 3. which questions does the new evidence (or a changed policy) touch?
        probe = state.model_copy(update={"open_questions": questions})
        if cold:
            affected = set(order)
        else:
            changed_policy = snapshot_changed(mem.state, state)                                      # type: ignore[union-attr]
            affected = affected_questions(mem.state, new_facts, links=linked.links, state=probe,     # type: ignore[union-attr]
                                          policy_questions=self.policy_questions if changed_policy else ())
            have_assessment = {a.question_id for a in mem.assessments}                               # type: ignore[union-attr]
            affected |= {q for q in order if q not in prev_by_id or q not in have_assessment}        # a question never assessed before is new work
        assessed = {} if cold else {a.question_id: a for a in mem.assessments if a.question_id in order}   # type: ignore[union-attr]

        # 4. recompute the affected questions only
        det = self.detector.detect(usable, questions, links, only=affected, previous=() if cold else state.contradictions, changed_fact_ids=gone)
        warnings += det.warnings
        retracted = [f"{r.contradiction_id} ({r.explanation})" for r in det.retracted]
        contradictions = [c for c in state.contradictions if c.question_id not in affected and c.question_id in order] + det.contradictions
        suff = self.classifier.classify([q for q in questions if q.question_id in affected], usable, links, contradictions, view, docs, self.modes)
        warnings += suff.warnings
        assessed.update({a.question_id: a for a in suff.assessments})
        sufficiency_all = SufficiencyResult([assessed[q.question_id] for q in questions if q.question_id in assessed])

        out_questions: list[OpenQuestion] = []
        for q in questions:
            if q.question_id in affected:
                ids = list(dict.fromkeys([*q.linked_fact_ids, *[x.fact_id for x in links if x.question_id == q.question_id]]))
                out_questions.append(q.model_copy(update={"status": assessed[q.question_id].status, "linked_fact_ids": ids}))
            else:
                out_questions.append(prev_by_id.get(q.question_id, q))

        # 5. requests: rewrite only those addressees whose needs changed
        plan = self.generator.plan(sufficiency_all, contradictions)
        fingerprints = {r: self.generator.fingerprint(n) for r, n in plan.items()}
        prev_requests = {m.minimum_evidence_request.addressee_role: m for m in state.missing_evidence}
        changed = {r for r in plan if cold or fingerprints[r] != mem.request_fingerprints.get(r) or r not in prev_requests}   # type: ignore[union-attr]
        gen = self.generator.generate(out_questions, sufficiency_all, usable, links, contradictions, set(docs.values()), docs, only_roles=changed)
        warnings += gen.warnings
        kept_requests = [prev_requests[r] for r in plan if r not in changed]
        requests = sorted([*kept_requests, *gen.requests], key=lambda m: (GAP_ORDER.index(m.gap_type), order.get(m.question_id, 0)))

        # 6. scenarios: the framed questions that are affected (all of them when the coverage issues changed)
        framed = {f.question for f in self.engine.cfg.framings}
        coverage_moved = cold or mem.state.coverage_issues != state.coverage_issues                  # type: ignore[union-attr]
        recompute = framed if coverage_moved else affected & framed
        clauses = state.policy_snapshot.clauses if state.policy_snapshot else []
        cf = self.engine.generate(out_questions, sufficiency_all, state.coverage_issues, clauses, requests, usable, links, only_questions=recompute)
        warnings += cf.warnings
        scenarios = sorted([s for s in state.counterfactuals if s.question_id not in recompute and s.question_id in framed] + cf.scenarios,
                           key=lambda s: (order.get(s.question_id, 0), s.scenario_id))

        updates: dict[str, Any] = {"open_questions": out_questions, "contradictions": contradictions, "missing_evidence": requests, "counterfactuals": scenarios}
        new_state = state.model_copy(update=updates)
        diff = diff_gap(state, new_state)
        self.memory = GapMemory(state=new_state, links=links, assessments=list(assessed.values()), request_fingerprints=fingerprints)
        self._save_memory(self.memory)
        self.last_run = GapRun(cold, affected, sorted(set(order) - affected, key=order.__getitem__), diff, sorted(changed), sorted(set(plan) - changed),
                               sorted(recompute), gen.review_questions, warnings, gen.problems + cf.problems, det.retained, retracted,
                               [f"{a.question_id} (on {', '.join(a.settled_fact_ids)})" for a in assessed.values() if a.settled_on_evidence])
        return StatePatch(agent_id=self.agent_id, updates=updates, rationale=self._rationale(self.last_run, bool(new_facts)))

    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def _rationale(r: GapRun, had_new_facts: bool) -> str:
        head = "FULL assessment (nothing remembered from an earlier run)" if r.cold else (
            f"INCREMENTAL: re-assessed {sorted(r.affected)}" if r.affected else "NOTHING NEW: no question is touched, every output is carried over")
        parts = [head]
        if not r.cold and r.kept:
            parts.append(f"kept unchanged: {', '.join(r.kept)}")
        parts.append(f"status changes: {r.diff.describe()}" if not r.diff.empty else "no status change")
        if r.regenerated_requests or r.kept_requests:
            parts.append(f"requests rewritten for: {', '.join(r.regenerated_requests) or 'none'}; kept for: {', '.join(r.kept_requests) or 'none'}")
        if r.retained_contradictions:
            parts.append(f"earlier contradictions kept (nothing resolved them): {', '.join(r.retained_contradictions)}")
        if r.retracted_contradictions:
            parts.append(f"earlier contradictions resolved by new evidence: {'; '.join(r.retracted_contradictions)}")
        if r.settled_on_evidence:
            parts.append(f"settled on the evidence as a whole (a listed example was not met word for word): {', '.join(r.settled_on_evidence)}")
        if r.review_questions:
            parts.append(f"REVIEW (classifier abstained, a person decides): {', '.join(r.review_questions)}")
        if r.warnings:
            parts.append("warnings: " + " | ".join(r.warnings))
        if r.problems:
            parts.append("SELF-CHECK PROBLEMS: " + " | ".join(r.problems))
        return "; ".join(parts)


def make_gap_agent(pack: Any, llm: Any | None = None) -> GapAgent:
    """The Gap agent for a pack, with the model-using parts switched on when `llm` is given and the pack enables them (each is a rule-only fallback otherwise)."""
    from claimtower.evidence.llm_contradiction import make_detector
    from claimtower.evidence.llm_counterfactual import make_engine
    from claimtower.evidence.llm_linking import make_linker
    from claimtower.evidence.llm_questions import make_register
    from claimtower.evidence.llm_requests import make_generator
    from claimtower.evidence.llm_sufficiency import make_classifier

    t = pack.question_templates
    policy_q = [cr.question for cr in (t.sufficiency.criteria_rules if t.sufficiency else [])]
    return GapAgent(register=make_register(pack, llm), register_rules=QuestionRegister(t), linker=make_linker(pack, llm), detector=make_detector(pack, llm),
                    classifier=make_classifier(pack, llm), generator=make_generator(pack, llm), engine=make_engine(pack, llm),
                    modes={q.id: q.criteria_mode for q in t.questions}, policy_questions=policy_q)


__all__ = ["GapAgent", "GapMemory", "GapRun", "make_gap_agent"]
