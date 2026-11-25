"""T3.1 policy decomposition: typed clauses with verbatim spans, honest abstention, stable ids, relational storage."""
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from claimtower.core.config import PolicyStructure, load_settings
from claimtower.ingestion.loaders import load
from claimtower.policy.decompose import PolicyDecomposer, verify_clauses
from claimtower.policy.store import PolicyStore
from evals.metrics.policy_clauses import all_cases, corpus_cases, evaluate

ROOT = Path(__file__).resolve().parents[2]
STRUCTURE = load_settings().pack.policy_structure
DECOMPOSER = PolicyDecomposer(STRUCTURE)


@dataclass
class P:
    number: int
    text: str


def decompose(text, doc_id="DOC-1", *more_pages):
    pages = [P(1, text), *[P(i, t) for i, t in enumerate(more_pages, start=2)]]
    return DECOMPOSER.decompose(doc_id, pages), pages


# ------------------------------------------------------------------ the labelled documents
def test_every_labelled_corpus_clause_is_found_typed_and_cited():
    r = evaluate(DECOMPOSER)["corpus"]
    assert (r.labelled, r.detected) == (31, 31) and r.f1 == 1.0
    assert r.wrong_types == [] and r.abstained == [] and r.spurious == [] and r.citation_problems == []
    assert r.type_accuracy == 1.0


def test_unseen_documents_are_never_mistyped_and_every_clause_is_still_found():
    r = evaluate(DECOMPOSER)["fresh"]
    assert r.recall == 1.0 and r.precision == 1.0 and r.spurious == [] and r.citation_problems == []
    assert r.wrong_types == []                              # an unknown wording is abstained on, not guessed
    assert len(r.abstained) <= 2                            # the recorded baseline; fewer is better


def test_recorded_baseline_is_not_regressed():
    from evals.runners.eval_policy_clauses import BASELINES, signals_hash

    latest = json.loads(BASELINES.read_text(encoding="utf-8"))["entries"][-1]
    assert latest["signals_sha"] == signals_hash(), "policy_structure.yaml changed: re-run eval_policy_clauses.py --write-baseline"
    for name, r in evaluate(DECOMPOSER).items():
        old = latest["splits"][name]
        assert r.f1 >= old["f1"] - 1e-9 and len(r.wrong_types) <= len(old["wrong_types"]), name
        assert len(r.abstained) <= len(old["abstained_type"]), name


def test_decomposition_is_deterministic_and_ids_follow_the_documents_own_numbering():
    case = next(c for c in corpus_cases() if c.doc_id == "BASE-HOME-1")
    pages = load(case.data, case.filename).pages
    a, b = DECOMPOSER.decompose("BASE-HOME-1", pages), DECOMPOSER.decompose("BASE-HOME-1", pages)
    assert a == b
    ids = [c.clause_id for c in a.clauses]
    assert ids[:2] == ["BASE-HOME-1/1.1", "BASE-HOME-1/1.2"] and "BASE-HOME-1/2.2" in ids and len(set(ids)) == len(ids)
    water = next(c for c in a.clauses if c.clause_id == "BASE-HOME-1/2.2")
    assert (water.clause_type, water.page, water.document) == ("coverage", 1, "BASE-HOME-1")
    assert a.info[water.clause_id].type_source == "section" and a.info[water.clause_id].section == "WHAT IS COVERED"


def test_endorsements_get_their_type_from_the_document_title_and_are_marked_as_endorsements():
    by = {c.doc_id: DECOMPOSER.decompose(c.doc_id, load(c.data, c.filename).pages) for c in corpus_cases()}
    assert by["END-04"].kind == "endorsement" and by["BASE-HOME-1"].kind == "policy_wording"
    assert by["END-04"].clauses[0].clause_type == "condition" and by["END-04"].info["END-04/1"].type_source == "title"
    assert by["END-05"].clauses[0].clause_type == "definition"
    assert by["END-02"].title == "Revised Water Damage Sublimit Endorsement"


# ------------------------------------------------------------------ citations are checked against the original pages
def test_clause_text_is_exactly_the_normalised_page_text_at_its_span():
    for case in all_cases():
        pages = load(case.data, case.filename).pages
        out = DECOMPOSER.decompose(case.doc_id, pages)
        assert verify_clauses(out.clauses, pages) == [], case.doc_id
        for c in out.clauses:
            assert " ".join(pages[c.page - 1].text.split())[c.span[0]:c.span[1]] == c.text


def test_verify_clauses_catches_a_changed_span_text_or_page():
    out, pages = decompose("Policy\nSECTION 1 - DEFINITIONS\n1.1 Home\nHome means the house.\n")
    assert verify_clauses(out.clauses, pages) == []
    c = out.clauses[0]
    assert verify_clauses([c.model_copy(update={"text": "Home means the flat."})], pages)
    assert verify_clauses([c.model_copy(update={"span": (c.span[0] + 1, c.span[1] + 1)})], pages)
    assert verify_clauses([c.model_copy(update={"page": 9})], pages)
    assert verify_clauses([c.model_copy(update={"span": None})], pages)


# ------------------------------------------------------------------ behaviour on awkward input
def test_a_clause_whose_type_cannot_be_decided_is_returned_unclassified_not_guessed():
    out, _ = decompose("Policy\n1.1 Mystery\nThe moon is made of cheese.\n")
    assert out.clauses == [] and len(out.unclassified) == 1
    u = out.unclassified[0]
    assert (u.number, u.page, u.text) == ("1.1", 1, "The moon is made of cheese.") and "no type rule" in u.reason


def test_the_heading_decides_before_the_section_and_the_section_before_the_text():
    text = ("Policy wording\nSECTION 2 - WHAT IS COVERED\n2.1 Standard deductible\nYou pay the deductible shown in the Schedule.\n"
            "2.2 Flood\nWe do not cover flood.\n")
    out, _ = decompose(text)
    kinds = {c.heading: (c.clause_type, out.info[c.clause_id].type_source) for c in out.clauses}
    assert kinds["2.1 Standard deductible"] == ("deductible", "heading")
    assert kinds["2.2 Flood"] == ("coverage", "section")           # the section beats the wording of the text


def test_wrapped_lines_are_joined_and_sub_items_stay_inside_their_clause():
    out, _ = decompose("Policy\nPART 3 WHAT WE DO NOT COVER\n3.2 Gradual damage\nWe do not cover damage that\nhappens gradually:\n"
                       "(a) rust;\n(b) a slow drip.\n3.3 Neglect\nWe do not cover neglect.\n")
    assert [c.text for c in out.clauses] == ["We do not cover damage that happens gradually: (a) rust; (b) a slow drip.",
                                             "We do not cover neglect."]


def test_text_that_continues_on_the_next_page_is_flagged_not_silently_dropped_or_merged():
    out, _ = decompose("Policy\nSECTION 1 - DEFINITIONS\n1.1 Home\nHome means the house and", "DOC-1", "its garden.\n1.2 Pipe\nPipe means a fixed pipe.\n")
    assert out.clauses[0].text == "Home means the house and"
    assert any("continues the previous page" in w and "its garden" in w for w in out.warnings)


def test_banners_footers_and_preamble_are_not_clauses():
    out, _ = decompose("SYNTHETIC DOCUMENT - for software testing only.\nWater Endorsement\nEndorsement E-1  |  Version 1\n"
                       "This endorsement forms part of the policy.\n1. Sublimit\nThe most we pay is INR 1.\nE-1  |  SYNTHETIC\nPage 1 of 1\n")
    assert out.title == "Water Endorsement" and out.kind == "endorsement"
    assert [(c.clause_type, c.text) for c in out.clauses] == [("sublimit", "The most we pay is INR 1.")]


def test_repeated_clause_numbers_still_get_distinct_ids():
    out, _ = decompose("Policy\nSECTION 1 - DEFINITIONS\n1.1 Home\nHome means the house.\n", "DOC-1", "SECTION 1 - DEFINITIONS\n1.1 Pipe\nPipe means a pipe.\n")
    assert [c.clause_id for c in out.clauses] == ["DOC-1/1.1", "DOC-1/1.1@p2"]


def test_a_heading_without_text_is_reported():
    out, _ = decompose("Policy\n1.1 Empty\n1.2 Home\nHome means the house.\n")
    assert [c.heading for c in out.clauses] == ["1.2 Home"] and any("no text" in w for w in out.warnings)


def test_body_lines_that_look_like_sentences_are_not_taken_for_headings():
    out, _ = decompose("Policy\nSECTION 4 - LIMITS\n4.1 Building Limit\nThe limit is 5.5 million.\n4.5 million is the cap for this section.\n")
    assert [c.text for c in out.clauses] == ["The limit is 5.5 million. 4.5 million is the cap for this section."]


# ------------------------------------------------------------------ config validation
@pytest.mark.parametrize("mutate, match", [
    (lambda d: d["clause_patterns"].append("(unclosed"), "bad pattern"),
    (lambda d: d["clause_patterns"].append(r"^(?P<number>\d+)$"), "number.*title"),
    (lambda d: d["section_patterns"].append(r"^PART \d+$"), "title"),
    (lambda d: d["clause_patterns"].clear(), "at least one clause pattern"),
    (lambda d: d["types"][0].update(heading_keywords=["[bad"]), "bad pattern"),
    (lambda d: d["types"].append(copy.deepcopy(d["types"][0])), "more than one rule"),
    (lambda d: d["types"][0].update(type="fine_print"), "type"),
])
def test_invalid_policy_structure_is_rejected(mutate, match):
    d = yaml.safe_load((ROOT / "packs/property_water_damage/policy_structure.yaml").read_text(encoding="utf-8"))
    mutate(d)
    with pytest.raises(ValidationError, match=match):
        PolicyStructure.model_validate(d)


# ------------------------------------------------------------------ relational store
def test_store_round_trip_replace_and_filters(tmp_path):
    case = next(c for c in corpus_cases() if c.doc_id == "BASE-HOME-1")
    pages = load(case.data, case.filename).pages
    out = DECOMPOSER.decompose("BASE-HOME-1", pages)
    store = PolicyStore(tmp_path / "policy.db")
    store.save(out, len(pages))
    assert store.clauses("BASE-HOME-1") == out.clauses                       # exact contract objects come back
    assert [c.clause_id for c in store.clauses(clause_type="exclusion")] == ["BASE-HOME-1/3.1", "BASE-HOME-1/3.2"]
    assert store.clause("BASE-HOME-1/2.2").text.startswith("We cover loss") and store.clause("nope") is None
    assert store.info("BASE-HOME-1/2.2").type_source == "section"
    assert store.documents() == [{"doc_id": "BASE-HOME-1", "title": "Home Protect - Policy Wording", "kind": "policy_wording", "page_count": 3}]
    store.save(DECOMPOSER.decompose("BASE-HOME-1", pages[:1]), 1)                 # re-ingest: nothing stale is left behind
    assert {c.page for c in store.clauses("BASE-HOME-1")} == {1}
    store.close()
    assert PolicyStore(tmp_path / "policy.db").clause("BASE-HOME-1/2.2") is not None       # persisted on disk


def test_store_keeps_the_unclassified_for_a_human():
    out, _ = decompose("Policy\n1.1 Mystery\nThe moon is made of cheese.\n")
    store = PolicyStore()
    store.save(out, 1)
    assert store.clauses() == [] and [u.heading for u in store.unclassified("DOC-1")] == ["1.1 Mystery"]


def test_policy_package_depends_only_on_core_and_itself():
    import re

    for f in (ROOT / "src/claimtower/policy").glob("*.py"):
        for imp in re.findall(r"^(?:from|import) (claimtower[\w.]*)", f.read_text(encoding="utf-8"), re.MULTILINE):
            assert imp.startswith(("claimtower.core", "claimtower.policy")), f"{f.name} imports {imp}"
