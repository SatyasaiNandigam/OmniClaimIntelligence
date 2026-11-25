"""T3.2 relations, precedence and the effective clause set: evidence-based links, stated (never assumed) precedence, dated applicability."""
import copy
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from claimtower.core.config import PolicyStructure, RelationConfig, load_settings
from claimtower.policy.decompose import PolicyDecomposer
from claimtower.policy.effective import EffectiveSetBuilder, EndorsementListing
from claimtower.policy.relations import RelationExtractor, stem
from claimtower.policy.store import PolicyStore
from evals.metrics.policy_effective import run_cases, run_scenarios
from evals.metrics.policy_relations import evaluate

ROOT = Path(__file__).resolve().parents[2]
STRUCTURE = load_settings().pack.policy_structure
DECOMPOSER = PolicyDecomposer(STRUCTURE)
EXTRACTOR = RelationExtractor(STRUCTURE)
ENGINE = EffectiveSetBuilder(EXTRACTOR)


@dataclass
class P:
    number: int
    text: str


def docs(**texts):
    return [DECOMPOSER.decompose(name.replace("_", "-"), [P(1, text)]) for name, text in texts.items()]


WORDING = """Mini Wording
SECTION 2 - WHAT IS COVERED
2.1 Escape of water
We cover damage caused by escape of water from fixed pipes.
SECTION 5 - CONDITIONS
5.4 Endorsements
Where an endorsement conflicts with this wording or with an earlier endorsement, the later endorsement applies from its effective date and not before.
"""
WORDING_NO_RULE = WORDING.split("SECTION 5")[0]
E1 = "Water Sublimit Endorsement\n1. Water sublimit\nThe most we will pay for escape of water is INR 100,000.\n"
E2 = "Revised Water Sublimit Endorsement\n1. Revised sublimit\nThe sublimit in Endorsement E-1 is increased to INR 200,000.\n"
E2_PREVAILS = E2 + "This endorsement prevails over Endorsement E-1.\n"
LST = lambda i, d, **k: EndorsementListing(i, date.fromisoformat(d), **k)  # noqa: E731


def graph(wording=WORDING, e2=E2):
    return EXTRACTOR.build("W-1", docs(W_1=wording, E_1=E1, E_2=e2))


# ------------------------------------------------------------------ the labelled policies
def test_relations_on_the_corpus_and_unseen_policies_meet_the_recorded_baseline():
    from evals.runners.eval_policy_clauses import full_hash
    from evals.runners.eval_policy_relations import BASELINES

    reports = evaluate(DECOMPOSER, EXTRACTOR)
    latest = json.loads(BASELINES.read_text(encoding="utf-8"))["entries"][-1]
    assert latest["signals_sha"] == full_hash(), "policy_structure.yaml changed: re-run eval_policy_relations.py --write-baseline"
    for name, r in reports.items():
        old = latest["splits"][name]
        assert r.recall >= old["recall"] - 1e-3 and r.precision >= old["precision"] - 1e-3, name       # baselines store 4 decimals
        assert len(r.unlabelled) <= len(old["unlabelled"]), f"{name}: new unlabelled relations - audit them or fix the rule"
    assert reports["corpus"].recall >= 0.9 and reports["corpus"].lenient_precision == 1.0
    assert reports["fresh"].recall >= 0.8 and reports["fresh"].precision == 1.0


def test_every_relation_type_is_found_on_the_corpus():
    r = evaluate(DECOMPOSER, EXTRACTOR)["corpus"]
    for rel in ("modifies", "excepts", "defines", "conditions"):
        assert r.by_type[rel]["matched"] >= 2, rel


def test_effective_sets_match_the_scenario_ground_truth_and_every_dated_case():
    bad = [(r.case_id, r.failures) for r in [*run_scenarios(DECOMPOSER, EXTRACTOR), *run_cases(DECOMPOSER, EXTRACTOR)] if not r.passed]
    assert bad == []
    assert len(run_scenarios(DECOMPOSER, EXTRACTOR)) == 4 and len(run_cases(DECOMPOSER, EXTRACTOR)) >= 13


def test_recorded_effective_baseline_is_not_regressed():
    from evals.runners.eval_policy_clauses import full_hash
    from evals.runners.eval_policy_effective import BASELINES

    entry = json.loads(BASELINES.read_text(encoding="utf-8"))["entries"][-1]
    assert entry["signals_sha"] == full_hash(), "policy_structure.yaml changed: re-run eval_policy_effective.py --write-baseline"
    latest = entry["results"]
    assert latest["scenarios"]["failed"] == [] and latest["cases"]["failed"] == []


# ------------------------------------------------------------------ how a target is found
def test_stemming_matches_the_word_forms_policies_use():
    assert stem("escape") == stem("escapes") == stem("escaping") and stem("pipe") == stem("pipes")


def test_an_endorsement_naming_another_endorsement_modifies_that_one_not_the_wording():
    g = graph()
    pairs = {(r.source, r.target, r.relation, r.basis) for r in g.relations}
    assert ("E-2/1", "E-1/1", "modifies", "reference") in pairs
    assert ("E-1/1", "W-1/2.1", "modifies", "unique") in pairs
    assert not any(r.source == "E-2/1" and r.target == "W-1/2.1" for r in g.relations)


def test_an_endorsement_cannot_modify_another_one_by_similar_words_alone():
    e2_no_ref = "Another Sublimit Endorsement\n1. Another sublimit\nThe most we will pay for escape of water is INR 150,000.\n"
    g = EXTRACTOR.build("W-1", docs(W_1=WORDING, E_1=E1, E_2=e2_no_ref))
    assert not any(r.target.startswith("E-1/") for r in g.relations)          # it acts on the wording, not on its look-alike


def test_the_precedence_clause_conditions_exactly_the_endorsement_that_conflicts_with_another():
    g = graph()
    assert [(r.source, r.target, r.relation) for r in g.relations if r.relation == "conditions"] == [("W-1/5.4", "E-2/1", "conditions")]


def test_an_exception_after_an_exclusion_carves_out_that_exclusion():
    w = ("Wording\nPART B - COVER FOR WATER\nB1 Water damage\nWe cover damage from water escaping from a pipe.\nPART C - EXCLUSIONS\n"
         "C1 Gradual escape\nWe do not cover damage from water that escapes gradually.\nC2 Resulting damage\n"
         "However, we will cover resulting damage unless the escape was caused by neglect.\n")
    g = EXTRACTOR.build("W", docs(W=w))
    rels = {(r.source, r.target, r.relation, r.basis) for r in g.relations}
    assert ("W/C2", "W/C1", "excepts", "exception") in rels and ("W/C1", "W/B1", "excepts", "unique") in rels


def test_a_definition_governs_the_clauses_that_use_its_term_and_only_those():
    w = ("Wording\nSECTION 1 - DEFINITIONS\n1.1 Sudden\nSudden means happening without warning.\nSECTION 2 - WHAT IS COVERED\n"
         "2.1 Fire\nWe cover fire damage.\n2.2 Burst pipes\nWe cover a sudden burst of a pipe.\n")
    g = EXTRACTOR.build("W", docs(W=w))
    assert [(r.source, r.target, r.relation, r.basis) for r in g.relations] == [("W/1.1", "W/2.2", "defines", "term")]


def test_a_tie_between_targets_gives_no_relation_and_is_listed_as_ambiguous():
    w = ("Wording\nSECTION 2 - WHAT IS COVERED\n2.1 Pipe cover\nWe cover bursts of a pipe.\n2.2 Pipe repair\nWe cover repair of a pipe.\n"
         "SECTION 3 - WHAT IS NOT COVERED\n3.1 Old pipes\nWe do not cover an old pipe.\n")
    g = EXTRACTOR.build("W", docs(W=w))
    assert g.relations == []
    assert [(a.source, a.relation, a.candidates) for a in g.ambiguities] == [("W/3.1", "excepts", ("W/2.1", "W/2.2"))]


def test_no_shared_words_gives_no_relation_not_a_guess():
    w = ("Wording\nSECTION 2 - WHAT IS COVERED\n2.1 Fire\nWe cover fire.\n2.2 Storm\nWe cover storm.\n"
         "SECTION 3 - WHAT IS NOT COVERED\n3.1 Wear\nWe do not cover rust.\n")
    g = EXTRACTOR.build("W", docs(W=w))
    assert g.relations == [] and "no shared words" in g.ambiguities[0].reason


# ------------------------------------------------------------------ precedence is read, never assumed
def test_the_precedence_rule_is_read_from_the_wording_or_reported_as_unstated():
    g = graph()
    assert g.precedence.kind == "later_effective_date_prevails" and g.precedence.clause_id == "W-1/5.4"
    assert graph(WORDING_NO_RULE).precedence.kind == "unstated" and graph(WORDING_NO_RULE).precedence.clause_id is None
    other = WORDING.replace("the later endorsement applies from its effective date and not before.", "the endorsement is given priority.")
    assert graph(other).precedence.kind == "stated_other"                      # a rule exists but we do not understand it: abstain later


def test_an_endorsement_that_says_it_prevails_is_recorded_and_stated_dates_are_read():
    g = graph(e2=E2_PREVAILS + "With effect from 1 February 2026 it applies.\n")
    assert [(p.document, p.over) for p in g.prevails_over] == [("E-2", "E-1")]
    assert g.stated_dates == {"E-2": date(2026, 2, 1)}
    assert graph().prevails_over == [] and graph().stated_dates == {}


# ------------------------------------------------------------------ the effective set
def clause_ids(eff):
    return set(eff.clause_ids)


def test_the_wording_rule_applies_the_later_endorsement_only_from_its_date():
    g, ls = graph(), [LST("E-1", "2026-01-01"), LST("E-2", "2026-03-01")]
    before, on, after = (ENGINE.build(g, "W-1", ls, date.fromisoformat(d)) for d in ("2026-02-28", "2026-03-01", "2026-06-01"))
    assert "E-1/1" in clause_ids(before) and "E-2/1" not in clause_ids(before)
    assert "E-2/1" in clause_ids(on) and "E-1/1" not in clause_ids(on) and clause_ids(on) == clause_ids(after)
    assert on.modified_by["W-1/2.1"] == ["E-2/1"] and before.modified_by["W-1/2.1"] == ["E-1/1"]
    assert [(x.document, x.reason) for x in before.excluded] == [("E-2", "not_yet_effective")]
    assert [(x.document, x.reason) for x in on.excluded] == [("E-1", "superseded")]
    assert on.abstain_clause_ids == [] and on.conflicts == []


def test_without_a_stated_rule_the_engine_keeps_both_and_abstains_instead_of_picking_a_winner():
    g, ls = graph(WORDING_NO_RULE), [LST("E-1", "2026-01-01"), LST("E-2", "2026-03-01")]
    eff = ENGINE.build(g, "W-1", ls, date(2026, 6, 1))
    assert {"E-1/1", "E-2/1"} <= clause_ids(eff)                                 # neither is silently dropped
    assert [c.kind for c in eff.conflicts] == ["precedence_unstated"] and eff.abstain_clause_ids == ["W-1/2.1"]
    assert "No precedence rule is stated" in eff.precedence_rules[0]


def test_an_unclear_rule_also_makes_the_engine_abstain():
    other = WORDING.replace("the later endorsement applies from its effective date and not before.", "the endorsement is given priority.")
    eff = ENGINE.build(graph(other), "W-1", [LST("E-1", "2026-01-01"), LST("E-2", "2026-03-01")], date(2026, 6, 1))
    assert [c.kind for c in eff.conflicts] == ["precedence_unstated"] and eff.abstain_clause_ids == ["W-1/2.1"]


def test_an_endorsement_that_states_its_own_priority_resolves_the_conflict_even_without_a_wording_rule():
    g = graph(WORDING_NO_RULE, E2_PREVAILS)
    eff = ENGINE.build(g, "W-1", [LST("E-1", "2026-01-01"), LST("E-2", "2026-03-01")], date(2026, 6, 1))
    assert "E-1/1" not in clause_ids(eff) and "E-2/1" in clause_ids(eff) and eff.conflicts == [] and eff.abstain_clause_ids == []
    assert any("states that it prevails over E-1" in s for s in eff.precedence_rules)


def test_two_endorsements_starting_the_same_day_with_no_stated_priority_are_a_conflict():
    eff = ENGINE.build(graph(), "W-1", [LST("E-1", "2026-03-01"), LST("E-2", "2026-03-01")], date(2026, 6, 1))
    assert [c.kind for c in eff.conflicts] == ["same_effective_date"] and eff.abstain_clause_ids == ["W-1/2.1"]


def test_a_later_endorsement_whose_target_is_not_in_force_is_flagged():
    eff = ENGINE.build(graph(), "W-1", [LST("E-2", "2026-03-01")], date(2026, 6, 1))
    assert [c.kind for c in eff.conflicts] == ["modified_clause_not_in_force"]


def test_effective_window_edges_are_inclusive():
    g = graph()
    ls = [LST("E-1", "2026-01-01", effective_to=date(2026, 1, 31))]
    assert "E-1/1" in clause_ids(ENGINE.build(g, "W-1", ls, date(2026, 1, 31)))
    gone = ENGINE.build(g, "W-1", ls, date(2026, 2, 1))
    assert "E-1/1" not in clause_ids(gone) and gone.excluded[0].reason == "expired"
    assert "E-1/1" not in clause_ids(ENGINE.build(g, "W-1", ls, date(2025, 12, 31)))


def test_a_missing_endorsement_is_reported_and_only_abstains_when_its_title_says_what_it_touches():
    g = graph(WORDING_NO_RULE)
    titled = ENGINE.build(g, "W-1", [LST("E-9", "2026-01-01", title="Water Sublimit Endorsement")], date(2026, 6, 1))
    assert titled.missing_documents == ["E-9"] and titled.abstain_clause_ids == ["W-1/2.1"]
    untitled = ENGINE.build(g, "W-1", [LST("E-9", "2026-01-01")], date(2026, 6, 1))
    assert untitled.missing_documents == ["E-9"] and untitled.abstain_clause_ids == []
    assert clause_ids(titled) == clause_ids(untitled) == {c.clause_id for c in g.of_document("W-1")}     # it never changes what is in force
    not_yet = ENGINE.build(g, "W-1", [LST("E-9", "2027-01-01", title="Water Sublimit Endorsement")], date(2026, 6, 1))
    assert not_yet.missing_documents == [] and not_yet.abstain_clause_ids == []                     # not in force: not relevant to this loss


def test_a_stated_date_that_disagrees_with_the_policy_system_is_reported_and_the_system_wins():
    g = graph(e2=E2 + "With effect from 1 April 2026 it applies.\n")
    eff = ENGINE.build(g, "W-1", [LST("E-1", "2026-01-01"), LST("E-2", "2026-03-01")], date(2026, 3, 15))
    assert "E-2/1" in clause_ids(eff)                                           # the policy system's date decides what is in force
    assert [c.kind for c in eff.conflicts] == ["date_mismatch"]
    assert eff.abstain_clause_ids == ["W-1/2.1"]                                # E-2 changes E-1's sublimit, which changes the water cover


def test_the_engine_is_deterministic_and_ignores_listing_order():
    g = graph()
    a = ENGINE.build(g, "W-1", [LST("E-1", "2026-01-01"), LST("E-2", "2026-03-01")], date(2026, 6, 1))
    b = ENGINE.build(g, "W-1", [LST("E-2", "2026-03-01"), LST("E-1", "2026-01-01")], date(2026, 6, 1))
    assert a == b


# ------------------------------------------------------------------ config and storage
@pytest.mark.parametrize("mutate, match", [
    (lambda d: d["relations"]["explicit_reference_patterns"].append(r"^(?P<x>\d+)$"), "id"),
    (lambda d: d["relations"]["effective_date_cues"].append(r"from (?P<id>\d+)"), "date"),
    (lambda d: d["relations"]["exception_cues"].append("(unclosed"), "bad pattern"),
    (lambda d: d["relations"].update(rules=[]), "must not be empty"),
    (lambda d: d["relations"]["rules"][0].update(relation="defines"), "exception"),
    (lambda d: d["relations"]["rules"][2].update(relation="modifies"), "term"),
    (lambda d: d["relations"]["rules"][0].update(relation="supersedes"), "relation"),
    (lambda d: d["relations"].update(min_score=0), "min_score"),
])
def test_invalid_relation_config_is_rejected(mutate, match):
    d = yaml.safe_load((ROOT / "packs/property_water_damage/policy_structure.yaml").read_text(encoding="utf-8"))
    mutate(d)
    with pytest.raises(ValidationError, match=match):
        PolicyStructure.model_validate(d)


def test_relation_config_words_survive_yaml_booleans():
    """Regression: a bare `on` / `no` in a YAML list is a boolean. Every stopword must have loaded as a string."""
    cfg = load_settings().pack.policy_structure.relations
    assert isinstance(cfg, RelationConfig) and "on" in cfg.stopwords and all(isinstance(w, str) for w in cfg.stopwords)


def test_graph_round_trips_through_the_relational_store(tmp_path):
    g = graph()
    store = PolicyStore(tmp_path / "p.db")
    store.save_graph(g)
    assert store.relations("W-1") == g.relations
    assert [r.source for r in store.relations("W-1", "E-1/1")] == ["E-1/1", "E-2/1"]
    assert store.precedence("W-1") == g.precedence and store.precedence("nope") is None
    assert [(c.source_clause_id, c.target_clause_id, c.relation) for c in store.relation_contracts("W-1")][:1] == [("E-1/1", "W-1/2.1", "modifies")]
    store.save_graph(graph(WORDING_NO_RULE))                                       # re-saving replaces, it does not append
    assert store.precedence("W-1").kind == "unstated" and not any(r.relation == "conditions" for r in store.relations("W-1"))


def test_relation_config_is_a_deep_copy_safe_contract():
    cfg = load_settings().pack.policy_structure.relations
    assert copy.deepcopy(cfg) == cfg
