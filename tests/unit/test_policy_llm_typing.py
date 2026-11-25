"""4a - optional model typing of policy clauses the rules could not type: checked, never trusted, always visible."""
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from claimtower.agents.policy import PolicyAgent
from claimtower.core.config import LLMTypingConfig, load_settings
from claimtower.core.errors import ConfigError
from claimtower.core.schemas import ClaimHeader, ClaimState, PolicyVersionResponse
from claimtower.policy.decompose import PolicyDecomposer, Unclassified, verify_clauses
from claimtower.policy.llm_typing import ALLOWED, LLMClauseTyper, TypingAnswer
from claimtower.policy.relations import RelationExtractor
from claimtower.policy.snapshot import PolicyLibrary, SnapshotBuilder
from claimtower.policy.store import PolicyStore

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()
PACK = SETTINGS.pack
STRUCTURE = PACK.policy_structure
CFG = STRUCTURE.llm_typing


@dataclass
class P:
    number: int
    text: str


class FakeLLM:
    """Answers `structured(TypingAnswer, prompt)` from a table keyed by a phrase of the clause text; records what it was asked."""

    def __init__(self, answers, fail=None):
        self.answers, self.fail, self.prompts = answers, fail, []

    def structured(self, model_cls, prompt, **kw):
        self.prompts.append(prompt.text)
        if self.fail:
            raise self.fail
        for key, ans in self.answers.items():
            if key in prompt.text:
                return TypingAnswer(**ans)
        return TypingAnswer(clause_type="unknown", confidence=0.3)


def typer(answers, fail=None):
    return LLMClauseTyper(FakeLLM(answers, fail), PACK)


TEXT = """Mini Terms
SECTION 1 TERMS EXPLAINED
1.1 Tank
Tank means a container that stores water for the home.
SECTION 4 PAYMENT AMOUNTS
4.1 Yearly ceiling
Our total payment in any policy year will never go beyond the Sum Insured in the Schedule.
4.2 Rounding
Payments are rounded to the nearest rupee.
SECTION 6 ADMINISTRATION
6.1 Language
This policy is written in English.
"""
ANSWERS = {
    "Our total payment": dict(clause_type="limit", evidence_snippet="never go beyond the Sum Insured", confidence=0.95),
    "Payments are rounded": dict(clause_type="limit", evidence_snippet="Payments are rounded to the nearest rupee.", confidence=0.9),
    "written in English": dict(clause_type="unknown", confidence=0.4),
}


def decompose(text=TEXT, t=None):
    return PolicyDecomposer(STRUCTURE, typer=t).decompose("DOC-1", [P(1, text)])


# ------------------------------------------------------------------ the checks on a model answer
def unclassified(text="We will pay at most INR 1,000.", heading="2.1 Cap"):
    return Unclassified("D", 1, "2.1", heading, "SECTION", text, "no rule", "D/2.1", (0, len(text)))


def verdict(**kw):
    base = dict(clause_type="sublimit", evidence_snippet="We will pay at most INR 1,000.", confidence=0.9)
    return typer({}).check(TypingAnswer(**{**base, **kw}), unclassified())


def test_a_valid_answer_is_accepted_and_its_confidence_is_capped():
    v = verdict()
    assert (v.clause_type, v.confidence) == ("sublimit", CFG.confidence_cap) and v.confidence < 0.9


@pytest.mark.parametrize("override, reason", [
    (dict(clause_type="unknown"), "answered unknown"),
    (dict(clause_type="governing_law"), "not an allowed"),
    (dict(clause_type="other"), "not an allowed"),
    (dict(evidence_snippet="We will pay at most INR 9,999."), "not verbatim"),
    (dict(evidence_snippet="pay"), "not verbatim"),
    (dict(evidence_snippet=""), "not verbatim"),
    (dict(confidence=0.69), "below"),
])
def test_an_answer_that_fails_any_check_leaves_the_clause_unclassified_with_a_reason(override, reason):
    v = verdict(**override)
    assert v.clause_type is None and reason in v.reason


def test_the_evidence_may_come_from_the_heading_and_whitespace_is_ignored():
    v = typer({}).check(TypingAnswer(clause_type="deductible", evidence_snippet="Your   share", confidence=0.9),
                        unclassified("Borne by you as shown in the Schedule.", "4.2 Your share"))
    assert v.clause_type == "deductible"


def test_the_model_can_only_name_the_seven_clause_types():
    assert set(ALLOWED) == {"coverage", "exclusion", "definition", "condition", "limit", "sublimit", "deductible"}


@pytest.mark.parametrize("clause_type", ["limit", "sublimit", "deductible"])
def test_a_payout_type_must_state_an_amount_or_point_to_one(clause_type):
    text = "Payments are rounded to the nearest rupee."
    v = typer({}).check(TypingAnswer(clause_type=clause_type, evidence_snippet=text, confidence=0.95), unclassified(text))
    assert v.clause_type is None and "states no amount" in v.reason
    for ok in ("The cap is INR 5,000.", "Up to the Sum Insured.", "As shown in the Schedule.", "Rs. 500 applies.", "The excess applies."):
        v = typer({}).check(TypingAnswer(clause_type=clause_type, evidence_snippet=ok, confidence=0.95), unclassified(ok))
        assert v.clause_type == clause_type, ok


def test_other_types_do_not_need_an_amount():
    t = "Keep the plumbing in good order."
    assert typer({}).check(TypingAnswer(clause_type="condition", evidence_snippet=t, confidence=0.9), unclassified(t)).clause_type == "condition"


# ------------------------------------------------------------------ inside the decomposer
def test_without_a_typer_nothing_changes():
    plain = decompose()
    assert [c.clause_id for c in plain.clauses] == ["DOC-1/1.1"] and len(plain.unclassified) == 3
    assert all(u.clause_id.startswith("DOC-1/") and u.span[1] > u.span[0] for u in plain.unclassified)


def test_only_clauses_the_rules_could_not_type_are_sent_to_the_model():
    t = typer(ANSWERS)
    decompose(t=t)
    assert t.asked == 3 and len(t._llm.prompts) == 3
    assert not any("Tank means" in p for p in t._llm.prompts)                      # the rule-typed definition never leaves the building


def test_a_checked_answer_types_the_clause_marks_it_llm_and_keeps_order_ids_and_spans():
    t = typer(ANSWERS)
    out = decompose(t=t)
    assert [c.clause_id for c in out.clauses] == ["DOC-1/1.1", "DOC-1/4.1"]        # order follows the document
    c = out.clauses[1]
    assert (c.clause_type, c.heading) == ("limit", "4.1 Yearly ceiling")
    assert (out.info[c.clause_id].type_source, out.info[c.clause_id].confidence) == ("llm", CFG.confidence_cap)
    assert out.info["DOC-1/1.1"].type_source == "text" and out.info["DOC-1/1.1"].confidence is None
    assert verify_clauses(out.clauses, [P(1, TEXT)]) == []                         # the span still points at the verbatim text
    assert c.clause_id == next(u.clause_id for u in decompose().unclassified if u.number == "4.1")     # the id never depends on the type


def test_a_declined_clause_stays_unclassified_and_says_why():
    out = decompose(t=typer(ANSWERS))
    assert [u.number for u in out.unclassified] == ["4.2", "6.1"]
    assert "states no amount" in out.unclassified[0].reason and "answered unknown" in out.unclassified[1].reason
    assert out.unclassified[0].reason.startswith("no type rule matched")


def test_a_model_failure_never_breaks_the_decomposition():
    for err in (RuntimeError("boom"), ConfigError("TBD model"), ValueError("bad json")):
        out = decompose(t=typer(ANSWERS, fail=err))
        assert [c.clause_id for c in out.clauses] == ["DOC-1/1.1"] and len(out.unclassified) == 3
        assert all("unavailable" in u.reason for u in out.unclassified)


def test_the_model_never_overrules_a_rule():
    llm = FakeLLM({"Tank means": dict(clause_type="exclusion", evidence_snippet="Tank means", confidence=0.99)})
    out = decompose(t=LLMClauseTyper(llm, PACK))
    assert out.clauses[0].clause_type == "definition" and out.info["DOC-1/1.1"].type_source == "text"


def test_the_prompt_carries_the_document_section_heading_and_text_and_a_version():
    t = typer(ANSWERS)
    decompose(t=t)
    p = next(x for x in t._llm.prompts if "Our total payment" in x)
    assert "Mini Terms" in p and "SECTION 4" not in p and "PAYMENT AMOUNTS" in p and "4.1 Yearly ceiling" in p


# ------------------------------------------------------------------ relational store and library
def test_store_keeps_the_model_typed_confidence_and_the_unclassified_ids(tmp_path):
    out = decompose(t=typer(ANSWERS))
    store = PolicyStore(tmp_path / "p.db")
    store.save(out, 1)
    info = store.info("DOC-1/4.1")
    assert (info.type_source, info.confidence) == ("llm", CFG.confidence_cap)
    assert store.info("DOC-1/1.1").confidence is None
    un = store.unclassified("DOC-1")
    assert [(u.clause_id, u.number) for u in un] == [("DOC-1/4.2", "4.2"), ("DOC-1/6.1", "6.1")] and all(u.span[1] > u.span[0] for u in un)
    lib = PolicyLibrary.from_store(store)
    assert lib.get("DOC-1").info["DOC-1/4.1"].type_source == "llm" and len(lib.get("DOC-1").unclassified) == 2


# ------------------------------------------------------------------ the snapshot says what a person must review
class Pas:
    def get_policy_version(self, policy_no, as_of):
        return PolicyVersionResponse(policy_no=policy_no, as_of=as_of, status="in_force", version="v1", effective_from=date(2026, 1, 1),
                                     effective_to=date(2026, 12, 31), base_wording_id="DOC-1", mock=True, source_system="PAS-MOCK",
                                     response_ref="PAS-MOCK/x", endorsements=[])


def builder(out):
    return SnapshotBuilder(Pas(), PolicyLibrary([out]), RelationExtractor(STRUCTURE), clock=lambda: datetime(2026, 7, 1, tzinfo=timezone.utc))


def test_the_snapshot_names_model_typed_clauses_and_clauses_that_are_missing_from_it():
    result = builder(decompose(t=typer(ANSWERS))).build("POL-X", date(2026, 3, 1))
    rules = " | ".join(result.result.precedence_rules)
    assert result.model_typed == ["DOC-1/4.1"] and [u.clause_id for u in result.unclassified] == ["DOC-1/4.2", "DOC-1/6.1"]
    assert f"REVIEW: the type of these clauses was decided by a language model, not by rules: DOC-1/4.1 ({CFG.confidence_cap:.2f})." in rules
    assert "NOT IN THIS SNAPSHOT: no type could be decided for DOC-1/4.2 '4.2 Rounding'" in rules and "DOC-1/6.1 '6.1 Language'" in rules
    assert {c.clause_id for c in result.result.clauses} == {"DOC-1/1.1", "DOC-1/4.1"}


def test_without_the_model_the_snapshot_still_says_which_clauses_are_missing():
    result = builder(decompose()).build("POL-X", date(2026, 3, 1))
    assert result.model_typed == [] and len(result.unclassified) == 3
    assert not any(r.startswith("REVIEW") for r in result.result.precedence_rules)
    assert any(r.startswith("NOT IN THIS SNAPSHOT") and "DOC-1/4.1 '4.1 Yearly ceiling'" in r for r in result.result.precedence_rules)


def test_a_clean_policy_gets_no_review_notes():
    clean = decompose("Mini Terms\nSECTION 1 TERMS EXPLAINED\n1.1 Tank\nTank means a container that stores water.\n")
    notes = builder(clean).build("POL-X", date(2026, 3, 1)).result.precedence_rules
    assert not any(r.startswith(("REVIEW", "NOT IN THIS SNAPSHOT")) for r in notes)


def test_the_policy_agent_reports_both_in_its_rationale():
    out = decompose(t=typer(ANSWERS))
    agent = PolicyAgent(builder(out))
    state = ClaimState(claim_id="C1", claim_header=ClaimHeader(claim_id="C1", policy_ref="POL-X", loss_date=date(2026, 3, 1)))
    patch = agent.run(state)
    assert "1 clause type(s) decided by a model: needs review" in patch.rationale and "2 clause(s) NOT in the snapshot" in patch.rationale
    assert set(patch.updates) == {"policy_snapshot"}


# ------------------------------------------------------------------ config
@pytest.mark.parametrize("patch, match", [
    ({"min_confidence": 0.9, "confidence_cap": 0.8}, "cannot be above"),
    ({"amount_cues": []}, "at least one amount cue"),
    ({"amount_cues": ["(unclosed"]}, "bad pattern"),
    ({"types_needing_amount": ["fine_print"]}, "types_needing_amount"),
    ({"reasoning_effort": "extreme"}, "reasoning_effort"),
    ({"max_clause_chars": 10}, "max_clause_chars"),
])
def test_invalid_llm_typing_config_is_rejected(patch, match):
    with pytest.raises(ValidationError, match=match):
        LLMTypingConfig.model_validate({**CFG.model_dump(), **patch})


def test_the_shipped_config_asks_for_more_reasoning_and_guards_payout_types():
    assert CFG.reasoning_effort == "low" and set(CFG.types_needing_amount) == {"limit", "sublimit", "deductible"}
    assert 0 < CFG.min_confidence <= CFG.confidence_cap < 1


def test_the_typer_uses_the_configured_reasoning_effort_without_changing_the_callers_client():
    class Client(FakeLLM):
        reasoning_effort = "minimal"

        def derive(self, **kw):
            c = Client(self.answers)
            c.reasoning_effort = kw["reasoning_effort"]
            return c

    base = Client({})
    t = LLMClauseTyper(base, PACK)
    assert t._llm.reasoning_effort == "low" and base.reasoning_effort == "minimal"


# ------------------------------------------------------------------ the recorded live run, replayed offline
@pytest.fixture(scope="module")
def replayed():
    from evals.runners.eval_policy_clauses import run_llm

    return run_llm(record=False)


def test_the_recorded_model_answers_replay_offline_and_never_go_live(replayed):
    reports, typer_, llm = replayed
    assert typer_.asked == 56 and all(c.replayed for c in llm.calls) and llm.mode == "replay"
    assert all(r.citation_problems == [] and r.recall == 1.0 for r in reports.values())


def test_the_model_never_made_a_wrong_type_on_the_clauses_it_was_actually_given_except_the_one_recorded(replayed):
    reports = replayed[0]
    assert reports["corpus"].f1 == 1.0 and reports["corpus"].type_accuracy == 1.0              # nothing to type: the model is never asked
    for split in ("dev_llm", "dev_llm2", "fresh"):
        assert reports[split].wrong_types == [] and reports[split].abstained == [], split      # tuned on these: a floor, not evidence
    fresh = reports["fresh_llm"]
    assert fresh.abstained == [] and len(fresh.wrong_types) == 1 and "6.3 Joint holders" in fresh.wrong_types[0]   # the honest unseen result


def test_the_llm_baseline_is_recorded_and_not_regressed(replayed):
    from evals.runners.eval_policy_clauses import LLM_BASELINES, signals_hash

    latest = json.loads(LLM_BASELINES.read_text(encoding="utf-8"))["entries"][-1]
    assert latest["signals_sha"] == signals_hash()
    for name, r in replayed[0].items():
        old = latest["splits"][name]
        assert r.type_accuracy >= old["type_accuracy"] - 1e-3 and len(r.wrong_types) <= len(old["wrong_types"]), name


def test_rules_only_results_are_unchanged_by_the_typing_feature():
    from evals.metrics.policy_clauses import evaluate

    reports = evaluate(PolicyDecomposer(STRUCTURE))
    assert reports["corpus"].type_accuracy == 1.0 and all(r.wrong_types == [] for r in reports.values())
    assert sum(len(r.abstained) for r in reports.values()) == 2 + 17 + 9 + 9                       # RIVERSIDE, ELMWOOD+WILLOW, WALNUT, BEECH


def test_the_policy_package_still_depends_only_on_core_and_itself():
    import re

    for f in (ROOT / "src/claimtower/policy").glob("*.py"):
        for imp in re.findall(r"^(?:from|import) (claimtower[\w.]*)", f.read_text(encoding="utf-8"), re.MULTILINE):
            assert imp.startswith(("claimtower.core", "claimtower.policy")), f"{f.name} imports {imp}"


# ------------------------------------------------------------------ switched ON (decision 2026-11-25)
def pack_with(**typing):
    return PACK.model_copy(update={"policy_structure": STRUCTURE.model_copy(update={"llm_typing": CFG.model_copy(update=typing)})})


def test_the_shipped_pack_has_model_typing_switched_on():
    assert CFG.enabled is True and LLMTypingConfig.model_fields["enabled"].default is False        # on by this pack's choice, off by default


def test_make_decomposer_uses_the_model_only_when_enabled_and_a_client_is_given():
    from claimtower.policy.llm_typing import make_decomposer

    assert make_decomposer(PACK, FakeLLM({})).typer is not None
    assert make_decomposer(PACK, None).typer is None                                  # no client: rules only
    assert make_decomposer(pack_with(enabled=False), FakeLLM({})).typer is None       # switched off in the pack


def test_when_the_model_is_unreachable_ingestion_still_works_and_says_the_clauses_are_missing():
    out = decompose(t=typer(ANSWERS, fail=RuntimeError("network down")))
    assert [c.clause_id for c in out.clauses] == ["DOC-1/1.1"] and len(out.unclassified) == 3
    assert any("model typing was unavailable for 3 clause(s)" in w and "NOT in a snapshot" in w for w in out.warnings)
    assert not any("unavailable" in w for w in decompose(t=typer(ANSWERS)).warnings)           # a clause the model merely declined is not an outage


def test_the_ingestor_stores_what_the_model_typed_and_the_snapshot_flags_it(tmp_path):
    from claimtower.policy.ingest import PolicyIngestor
    from claimtower.policy.llm_typing import make_decomposer

    ingestor = PolicyIngestor(PolicyStore(tmp_path / "p.db"), make_decomposer(PACK, FakeLLM(ANSWERS)))
    out = ingestor.ingest("DOC-1", [P(1, TEXT)])
    assert out.info["DOC-1/4.1"].type_source == "llm"
    assert ingestor.store.info("DOC-1/4.1").confidence == CFG.confidence_cap
    snap = builder_from(ingestor.library()).build("POL-X", date(2026, 3, 1))
    assert snap.model_typed == ["DOC-1/4.1"] and any(r.startswith("REVIEW") and "DOC-1/4.1" in r for r in snap.result.precedence_rules)
    assert [u.clause_id for u in snap.unclassified] == ["DOC-1/4.2", "DOC-1/6.1"]


def builder_from(library):
    return SnapshotBuilder(Pas(), library, RelationExtractor(STRUCTURE), clock=lambda: datetime(2026, 7, 1, tzinfo=timezone.utc))


def test_the_ingestor_refuses_a_document_whose_clauses_do_not_match_its_pages(tmp_path):
    from claimtower.policy.ingest import PolicyIngestError, PolicyIngestor

    class Liar(PolicyDecomposer):
        def decompose(self, doc_id, pages):
            out = super().decompose(doc_id, pages)
            out.clauses[0] = out.clauses[0].model_copy(update={"text": "something the page never said"})
            return out

    store = PolicyStore(tmp_path / "p.db")
    with pytest.raises(PolicyIngestError, match="DOC-1/1.1"):
        PolicyIngestor(store, Liar(STRUCTURE)).ingest("DOC-1", [P(1, TEXT)])
    assert store.documents() == []                                                     # nothing was stored


def test_end_to_end_on_a_real_unfamiliar_policy_with_the_recorded_model_answers(tmp_path):
    """ELMWOOD-1 uses wording no rule knows. Ingested with the model ON (answers replayed from the cassette), its limit, sublimit,
    deductible, exclusions and conditions reach the snapshot - flagged for review - and the two clauses that are none of the types do not."""
    from claimtower.core.llm import Cassette, LLMClient
    from claimtower.ingestion.loaders import load
    from claimtower.policy.ingest import PolicyIngestor
    from claimtower.policy.llm_typing import make_decomposer
    from evals.metrics.policy_retrieval import FakePas
    from evals.runners.eval_policy_clauses import CASSETTE

    llm = LLMClient.from_settings(SETTINGS)
    llm.mode, llm.cassette = "replay", Cassette(CASSETTE)
    ingestor = PolicyIngestor(PolicyStore(tmp_path / "p.db"), make_decomposer(PACK, llm))
    pages = load((ROOT / "data/synthetic/policy_eval/ELMWOOD-1.txt").read_bytes(), "ELMWOOD-1.txt").pages
    ingestor.ingest("ELMWOOD-1", pages)
    pas = FakePas({"POL-E1": {"base": "ELMWOOD-1", "endorsements": []}})
    out = SnapshotBuilder(pas, ingestor.library(), RelationExtractor(STRUCTURE),
                          clock=lambda: datetime(2026, 7, 1, tzinfo=timezone.utc)).build("POL-E1", date(2026, 3, 1))
    snap = out.result
    by_type = {}
    for c in snap.clauses:
        by_type.setdefault(c.clause_type, []).append(c.heading)
    assert {t: len(v) for t, v in by_type.items()} == {"definition": 2, "coverage": 2, "exclusion": 2, "limit": 1, "deductible": 1,
                                                        "sublimit": 1, "condition": 2}
    assert len(out.model_typed) == 9 and all(llm_call.replayed for llm_call in llm.calls)
    rules = " | ".join(snap.precedence_rules)
    assert "REVIEW:" in rules and "NOT IN THIS SNAPSHOT" in rules and "6.1 Governing law" in rules and "6.2 Dishonest claims" in rules
