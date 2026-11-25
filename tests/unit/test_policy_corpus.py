"""T1.1 - the synthetic policy corpus is internally consistent and matches the scenario clause catalogue."""
import json
import re
from pathlib import Path

import pymupdf
import pytest
import yaml

from claimtower.core.schemas.policy import ClauseRelation, PolicyClause
from evals.ground_truth import load_scenarios

DIR = Path(__file__).resolve().parents[2] / "data/synthetic/policies"
IDX = json.loads((DIR / "clause_index.json").read_text(encoding="utf-8"))
CLAUSES = {c["clause_id"]: c for c in IDX["clauses"]}
DOCS = {d["doc_id"]: d for d in IDX["documents"]}
SOURCE = yaml.safe_load((DIR / "corpus.yaml").read_text(encoding="utf-8"))


def norm(s):
    return " ".join(s.split())


def test_corpus_size_matches_plan():
    kinds = [d["kind"] for d in IDX["documents"]]
    assert 3 <= kinds.count("base_wording") <= 4
    assert 4 <= kinds.count("endorsement") <= 6


def test_clause_ids_unique_stable_format_and_not_rendered():
    assert len(CLAUSES) == len(IDX["clauses"])
    for cid in CLAUSES:
        assert re.fullmatch(r"CL-[A-Z0-9]+(-[A-Z0-9]+)*", cid), cid
    for d in DOCS.values():
        text = " ".join(p.get_text() for p in pymupdf.open(DIR / d["pdf"]))
        assert not re.search(r"\bCL-[A-Z0-9-]+", text), f"clause id leaked into {d['pdf']}"


@pytest.mark.parametrize("doc_id", sorted(DOCS))
def test_pdf_is_page_numbered_synthetic_and_clauses_sit_at_recorded_spans(doc_id):
    d = DOCS[doc_id]
    pdf = pymupdf.open(DIR / d["pdf"])
    assert pdf.page_count == d["pages"]
    pages = [norm(p.get_text()) for p in pdf]
    for n, text in enumerate(pages, start=1):
        assert "SYNTHETIC" in text and f"Page {n} of {pdf.page_count}" in text
    for c in (c for c in IDX["clauses"] if c["doc_id"] == doc_id):
        a, b = c["span"]
        assert pages[c["page"] - 1][a:b] == norm(c["text"]), c["clause_id"]
    assert (DIR / d["docx"]).is_file()


def test_clauses_and_relations_fit_the_policy_contract():
    for c in IDX["clauses"]:
        PolicyClause(clause_id=c["clause_id"], clause_type=c["clause_type"], heading=c["heading"], text=c["text"],
                     document=c["doc_id"], page=c["page"], span=tuple(c["span"]))
    for r in IDX["relations"]:
        ClauseRelation(source_clause_id=r["source"], target_clause_id=r["target"], relation=r["relation"])
        assert r["source"] in CLAUSES and r["target"] in CLAUSES


def test_scenario_catalogue_matches_corpus():
    cat = load_scenarios().clause_catalogue
    for c in cat:
        real = CLAUSES[c.clause_id]
        assert real["clause_type"] == c.clause_type, c.clause_id
        expected_doc = "BASE-HOME-1" if c.document == "base_wording" else c.document
        assert real["doc_id"] == expected_doc, c.clause_id


def test_corpus_yaml_and_index_agree():
    src_ids = [b["clause"] for d in SOURCE["documents"] for pg in d["pages"] for b in pg if "clause" in b]
    assert sorted(src_ids) == sorted(CLAUSES)


def test_scenario_endorsement_facts_match_wording():
    """The sublimit amounts and END-02 effective date used by S-04 ground truth must be what the wording says."""
    assert "INR 100,000" in CLAUSES["CL-SUB-WATER-E1"]["text"]
    e2 = CLAUSES["CL-SUB-WATER-E2"]["text"]
    assert "INR 200,000" in e2 and "1 February 2026" in e2
    assert DOCS["END-02"]["effective_from"] == "2026-02-01"
    assert "later endorsement applies from its effective date" in CLAUSES["CL-COND-PRECEDENCE"]["text"]
