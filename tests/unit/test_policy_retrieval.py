"""T3.4 clause retrieval: hybrid keyword + embedding search over the clauses in force, with the modifying endorsements returned alongside."""
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from claimtower.core.config import RetrievalConfig, load_settings
from claimtower.core.errors import ConfigError
from claimtower.core.llm import CassetteMiss, LLMError
from claimtower.core.schemas import EndorsementRef, PolicyVersionResponse
from claimtower.policy.decompose import PolicyDecomposer
from claimtower.policy.relations import RelationExtractor
from claimtower.policy.retrieval import ClauseRetriever
from claimtower.policy.snapshot import PolicyLibrary, SnapshotBuilder
from evals.metrics.policy_retrieval import build_world, evaluate
from evals.runners.eval_policy_retrieval import embedder

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()
STRUCTURE = SETTINGS.pack.policy_structure
CFG = STRUCTURE.retrieval
DECOMPOSER, EXTRACTOR = PolicyDecomposer(STRUCTURE), RelationExtractor(STRUCTURE)
WORLD = build_world(DECOMPOSER, EXTRACTOR)


# ------------------------------------------------------------------ the labelled queries
@pytest.fixture(scope="module")
def results():
    llm = embedder(SETTINGS, record=False)                          # replay: no network, no cost
    return {"keyword": evaluate(ClauseRetriever(CFG), WORLD), "hybrid": evaluate(ClauseRetriever(CFG, llm), WORLD)}, llm


def test_the_offline_hybrid_run_uses_only_recorded_embeddings(results):
    _, llm = results
    assert llm.mode == "replay" and all(c.replayed for c in llm.calls)
    assert all(t.degraded == 0 for t in results[0]["hybrid"].values())


def test_every_labelled_modifier_comes_back_with_its_clause_in_both_modes(results):
    for mode in ("keyword", "hybrid"):
        for split, t in results[0][mode].items():
            assert t.mod_failures == [] and t.mod_checked >= 2, (mode, split)


def test_unrelated_questions_return_nothing_in_both_modes(results):
    for mode in ("keyword", "hybrid"):
        for split, t in results[0][mode].items():
            assert t.false_answers == [] and t.negatives >= 1, (mode, split)


def test_hybrid_is_at_least_as_good_as_keyword_and_better_on_paraphrases(results):
    kw, hy = results[0]["keyword"], results[0]["hybrid"]
    for split in ("dev", "fresh"):
        assert hy[split].recall_sum >= kw[split].recall_sum and hy[split].rr_sum >= kw[split].rr_sum - 1e-9, split
    assert hy["dev"].recall_sum > kw["dev"].recall_sum                 # 'excess' ~ 'deductible' needs the embeddings


def test_the_recorded_retrieval_baseline_is_not_regressed(results):
    from evals.runners.eval_policy_clauses import retrieval_hash
    from evals.runners.eval_policy_retrieval import BASELINES

    entry = json.loads(BASELINES.read_text(encoding="utf-8"))["entries"][-1]
    assert entry["signals_sha"] == retrieval_hash(), "policy_structure.yaml changed: re-run eval_policy_retrieval.py --write-baseline"
    for mode in ("keyword", "hybrid"):
        for split, t in results[0][mode].items():
            old, now = entry["modes"][mode][split], t.summary()
            for key in ("hit@3", "recall@5", "mrr", "modifier_completeness", "no_answer_rate"):
                assert now[key] >= old[key] - 1e-3, f"{mode}/{split}/{key}: {now[key]} < {old[key]}"


def test_clauses_missing_from_the_snapshot_are_reported_not_blamed_on_retrieval(results):
    """R-31 / R-34 ask about clauses the decomposer left unclassified (T3.1): they cannot be retrieved and are counted separately."""
    assert results[0]["hybrid"]["fresh"].unreachable == 2


# ------------------------------------------------------------------ behaviour on small, readable snapshots
@dataclass
class P:
    number: int
    text: str


WORDING = """Mini Wording
SECTION 1 - DEFINITIONS
1.1 Sudden and accidental
Sudden and accidental means happening abruptly and not through gradual deterioration.
SECTION 2 - WHAT IS COVERED
2.1 Escape of water
We cover damage caused by escape of water from fixed pipes.
2.2 Fire
We cover damage caused by fire or lightning.
SECTION 3 - WHAT IS NOT COVERED
3.1 Gradual leakage
We do not cover damage caused by gradual leakage or seepage of pipes.
SECTION 4 - LIMITS AND DEDUCTIBLE
4.1 Standard deductible
You pay the standard deductible shown in the Schedule on each claim.
SECTION 5 - CONDITIONS
5.4 Endorsements
Where an endorsement conflicts with this wording or with an earlier endorsement, the later endorsement applies from its effective date and not before.
"""
E1 = "Water Sublimit Endorsement\n1. Water sublimit\nThe most we will pay for escape of water is INR 100,000.\n"
E2 = "Revised Water Sublimit Endorsement\n1. Revised sublimit\nThe sublimit in Endorsement E-1 is increased to INR 200,000.\n"


class Pas:
    def get_policy_version(self, policy_no, as_of):
        return PolicyVersionResponse(policy_no=policy_no, as_of=as_of, status="in_force", version="v1", effective_from=date(2026, 1, 1),
                                     effective_to=date(2026, 12, 31), base_wording_id="W-1", mock=True, source_system="PAS-MOCK",
                                     response_ref="PAS-MOCK/x", endorsements=[EndorsementRef(endorsement_id="E-1", effective_from=date(2026, 1, 1)),
                                                                            EndorsementRef(endorsement_id="E-2", effective_from=date(2026, 3, 1))])


def snapshot(day):
    lib = PolicyLibrary(DECOMPOSER.decompose(k.replace("_", "-"), [P(1, t)]) for k, t in (("W_1", WORDING), ("E_1", E1), ("E_2", E2)))
    return SnapshotBuilder(Pas(), lib, EXTRACTOR, clock=lambda: datetime(2026, 7, 1, tzinfo=timezone.utc)).get_snapshot("POL-X", day)


SNAP_BEFORE, SNAP_AFTER = snapshot(date(2026, 2, 1)), snapshot(date(2026, 4, 1))
KW = ClauseRetriever(CFG)


def ids(found):
    return [c.clause_id for c in found.hits]


def test_a_keyword_query_ranks_the_clause_that_uses_the_words_first():
    assert ids(KW.search("is gradual leakage excluded", SNAP_BEFORE))[0] == "W-1/3.1"
    assert ids(KW.search("what is the standard deductible", SNAP_BEFORE))[0] == "W-1/4.1"
    assert ids(KW.search("sudden and accidental", SNAP_BEFORE))[0] == "W-1/1.1"


def test_word_forms_match_and_stopwords_alone_match_nothing():
    assert ids(KW.search("pipes leaking", SNAP_BEFORE)) and "W-1/3.1" in ids(KW.search("seeping pipe", SNAP_BEFORE))
    assert KW.search("what is the of and", SNAP_BEFORE).hits == []
    assert KW.search("   ", SNAP_BEFORE).hits == [] and KW.search("anything", snapshot_without_clauses()).hits == []


def snapshot_without_clauses():
    return SNAP_BEFORE.model_copy(update={"clauses": []})


def test_a_question_nothing_answers_returns_nothing_not_the_closest_clause():
    found = KW.search("who won the football match", SNAP_BEFORE)
    assert found.hits == [] and found.modifiers == [] and KW.retrieve_clauses("who won the football match", SNAP_BEFORE) == []


def test_the_modifying_endorsement_is_returned_with_the_base_clause():
    found = KW.search("is escape of water covered", SNAP_BEFORE)
    cover = next(c for c in found.hits if c.clause_id == "W-1/2.1")
    assert cover.modified_by == ["E-1/1"]
    assert [c.clause_id for c in found.citations if c.clause_id == "E-1/1"]          # the endorsement text itself is in the result


def test_a_modifier_that_is_not_a_hit_follows_the_hits_with_its_text():
    found = KW.search("fire or lightning", SNAP_BEFORE)
    assert ids(found) == ["W-1/2.2"] and found.modifiers == []                     # fire has no modifier
    found = KW.search("fixed pipes", SNAP_BEFORE, k=1)
    assert ids(found) == ["W-1/2.1"] and [c.clause_id for c in found.modifiers] == ["E-1/1"]
    assert found.modifiers[0].text.startswith("The most we will pay") and found.citations == [*found.hits, *found.modifiers]


def test_after_a_later_endorsement_replaces_the_earlier_one_the_cover_points_to_the_later_one():
    before, after = KW.search("escape of water", SNAP_BEFORE), KW.search("escape of water", SNAP_AFTER)
    assert next(c for c in before.hits if c.clause_id == "W-1/2.1").modified_by == ["E-1/1"]
    assert next(c for c in after.hits if c.clause_id == "W-1/2.1").modified_by == ["E-2/1"]
    assert "E-1/1" not in [c.clause_id for c in after.citations]                    # a replaced clause is never returned
    assert "E-2/1" not in [c.clause_id for c in before.citations]                   # nor one that is not yet effective


def test_only_clauses_in_the_snapshot_can_be_returned():
    inside = {c.clause_id for c in SNAP_AFTER.clauses}
    for q in ("sublimit", "escape of water", "endorsement", "deductible", "gradual", "increase"):
        assert {c.clause_id for c in KW.search(q, SNAP_AFTER).citations} <= inside


def test_citations_carry_the_clause_document_page_span_and_verbatim_text():
    cite = KW.search("standard deductible", SNAP_BEFORE).hits[0]
    clause = next(c for c in SNAP_BEFORE.clauses if c.clause_id == cite.clause_id)
    assert (cite.document, cite.page, cite.span, cite.text) == (clause.document, clause.page, clause.span, clause.text)


def test_k_limits_the_ranked_hits():
    assert len(KW.search("damage caused", SNAP_BEFORE, k=2).hits) == 2 and len(KW.search("damage caused", SNAP_BEFORE).hits) <= CFG.k


def test_search_is_deterministic():
    assert KW.search("escape of water", SNAP_AFTER) == KW.search("escape of water", SNAP_AFTER)


# ------------------------------------------------------------------ embeddings and fusion
class FakeEmbedder:
    """Maps a text to a vector by keyword, so tests control exactly what 'means the same'."""

    def __init__(self, table, fail=None):
        self.table, self.fail, self.calls = table, fail, []

    def embed(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise self.fail
        return [next((v for k, v in self.table.items() if k in t.lower()), [0.0, 0.0, 1.0]) for t in texts]


def test_embeddings_find_the_clause_that_means_the_same_without_sharing_words():
    emb = FakeEmbedder({"slow drip": [1.0, 0.0, 0.0], "gradual leakage": [1.0, 0.0, 0.0], "fire": [0.0, 1.0, 0.0]})
    r = ClauseRetriever(CFG, emb)
    assert KW.search("slow drip", SNAP_BEFORE).hits == []                          # keyword alone cannot
    found = r.search("slow drip", SNAP_BEFORE)
    assert found.mode == "hybrid" and ids(found)[0] == "W-1/3.1"


def test_a_weak_embedding_match_is_ignored():
    emb = FakeEmbedder({"banana": [0.29, 0.957, 0.0], "gradual leakage": [1.0, 0.0, 0.0]})    # cosine 0.29 < 0.30 with the leakage clause
    assert ClauseRetriever(CFG, emb).search("banana", SNAP_BEFORE).hits == []


def test_a_clause_both_rankings_agree_on_beats_one_only_one_ranking_likes():
    emb = FakeEmbedder({"gradual leakage": [1.0, 0.0, 0.0], "seepage": [1.0, 0.0, 0.0]})
    found = ClauseRetriever(CFG, emb).search("pipes seepage", SNAP_BEFORE)
    assert ids(found)[0] == "W-1/3.1"


def test_an_unavailable_embedding_service_degrades_to_keyword_and_says_so():
    for err in (LLMError("down"), CassetteMiss("not recorded"), ConfigError("no model")):
        found = ClauseRetriever(CFG, FakeEmbedder({}, fail=err)).search("gradual leakage", SNAP_BEFORE)
        assert found.mode == "keyword (embeddings unavailable)" and found.warnings and ids(found)[0] == "W-1/3.1"


def test_texts_are_embedded_once_per_retriever():
    emb = FakeEmbedder({})
    r = ClauseRetriever(CFG, emb)
    r.search("gradual leakage", SNAP_BEFORE)
    r.search("gradual leakage", SNAP_BEFORE)
    r.search("deductible", SNAP_BEFORE)
    flat = [t for call in emb.calls for t in call]
    assert len(flat) == len(set(flat))                                              # no text was sent twice


def test_modifiers_come_from_the_snapshot_relations_not_from_similarity():
    emb = FakeEmbedder({"escape of water": [1.0, 0.0, 0.0]})
    found = ClauseRetriever(CFG, emb).search("escape of water", SNAP_BEFORE)
    assert next(c for c in found.hits if c.clause_id == "W-1/2.1").modified_by == ["E-1/1"]


# ------------------------------------------------------------------ config
@pytest.mark.parametrize("patch, match", [
    ({"keyword_weight": 0, "dense_weight": 0}, "both be 0"), ({"k": 0}, "k"), ({"bm25_b": 2}, "bm25_b"),
    ({"min_dense_similarity": 1.5}, "min_dense_similarity"), ({"stopwords": [True]}, "stopwords"),
])
def test_invalid_retrieval_config_is_rejected(patch, match):
    with pytest.raises(ValidationError, match=match):
        RetrievalConfig.model_validate({**CFG.model_dump(), **patch})


def test_the_shipped_config_names_an_embedding_model_and_a_tuned_but_sample_threshold():
    assert SETTINGS.app.llm.embedding_model and SETTINGS.app.llm.embedding_dimensions
    assert 0 < CFG.min_dense_similarity < 1 and "on" in CFG.stopwords or "or" in CFG.stopwords


def test_the_policy_package_still_depends_only_on_core_and_itself():
    import re

    for f in (ROOT / "src/claimtower/policy").glob("*.py"):
        for imp in re.findall(r"^(?:from|import) (claimtower[\w.]*)", f.read_text(encoding="utf-8"), re.MULTILINE):
            assert imp.startswith(("claimtower.core", "claimtower.policy")), f"{f.name} imports {imp}"
