"""T3.3 snapshot as of the date of loss: PAS status first, then the clauses in force; explicit NotInForce; nothing guessed."""
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from claimtower.core.config import load_settings
from claimtower.core.schemas import EndorsementRef, PolicyVersionResponse
from claimtower.core.schemas.policy import NotInForce, PolicySnapshot
from claimtower.integrations.mock.data import load_pas_db
from claimtower.policy.decompose import PolicyDecomposer
from claimtower.policy.relations import RelationExtractor
from claimtower.policy.snapshot import PolicyDocumentMissing, PolicyLibrary, SnapshotBuilder
from claimtower.policy.store import PolicyStore
from evals.metrics.policy_snapshot import MockPas, corpus_library, run_cases, run_scenarios

ROOT = Path(__file__).resolve().parents[2]
STRUCTURE = load_settings().pack.policy_structure
DECOMPOSER, EXTRACTOR = PolicyDecomposer(STRUCTURE), RelationExtractor(STRUCTURE)
NOW = datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)


@dataclass
class P:
    number: int
    text: str


WORDING = """Mini Wording
SECTION 2 - WHAT IS COVERED
2.1 Escape of water
We cover damage caused by escape of water from fixed pipes.
SECTION 5 - CONDITIONS
5.4 Endorsements
Where an endorsement conflicts with this wording or with an earlier endorsement, the later endorsement applies from its effective date and not before.
"""
E1 = "Water Sublimit Endorsement\n1. Water sublimit\nThe most we will pay for escape of water is INR 100,000.\n"
E2 = "Revised Water Sublimit Endorsement\n1. Revised sublimit\nThe sublimit in Endorsement E-1 is increased to INR 200,000.\n"


def library(**texts):
    return PolicyLibrary(DECOMPOSER.decompose(k.replace("_", "-"), [P(1, t)]) for k, t in texts.items())


LIB = library(W_1=WORDING, E_1=E1, E_2=E2)


class FakePas:
    def __init__(self, **resp):
        self.resp, self.calls = resp, []

    def get_policy_version(self, policy_no, as_of):
        self.calls.append((policy_no, as_of))
        base = dict(policy_no=policy_no, as_of=as_of, mock=True, source_system="PAS-MOCK", response_ref=f"PAS-MOCK/{policy_no}/{as_of}")
        return PolicyVersionResponse(**{**base, **self.resp})


def covered(**over):
    d = dict(status="in_force", version="v1", effective_from=date(2026, 1, 1), effective_to=date(2026, 12, 31), base_wording_id="W-1",
             endorsements=[EndorsementRef(endorsement_id="E-1", effective_from=date(2026, 1, 1)),
                           EndorsementRef(endorsement_id="E-2", effective_from=date(2026, 3, 1))])
    return {**d, **over}


def builder(pas, lib=LIB):
    return SnapshotBuilder(pas, lib, EXTRACTOR, clock=lambda: NOW)


# ------------------------------------------------------------------ the labelled outcomes
def test_scenarios_and_every_labelled_case_return_the_expected_snapshot_or_not_in_force():
    bad = [(r.case_id, r.failures) for r in [*run_scenarios(DECOMPOSER, EXTRACTOR), *run_cases(DECOMPOSER, EXTRACTOR)] if not r.passed]
    assert bad == []
    assert len(run_scenarios(DECOMPOSER, EXTRACTOR)) == 5 and len(run_cases(DECOMPOSER, EXTRACTOR)) >= 19


def test_recorded_snapshot_baseline_is_not_regressed():
    from evals.runners.eval_policy_clauses import full_hash
    from evals.runners.eval_policy_snapshot import BASELINES

    entry = json.loads(BASELINES.read_text(encoding="utf-8"))["entries"][-1]
    assert entry["signals_sha"] == full_hash(), "policy_structure.yaml changed: re-run eval_policy_snapshot.py --write-baseline"
    assert entry["results"]["scenarios"]["failed"] == [] and entry["results"]["cases"]["failed"] == []


def test_every_labelled_pas_lookup_agrees_with_the_snapshot_outcome():
    """The PAS mock's own labelled cases (T1.4) drive the snapshot: covered statuses give a snapshot, all others NotInForce."""
    library, _ = corpus_library(DECOMPOSER)
    b = SnapshotBuilder(MockPas(), library, EXTRACTOR, clock=lambda: NOW)
    for case in load_pas_db().cases:
        got = b.get_snapshot(case.policy_no, case.as_of)
        if case.expected_status in ("in_force", "reinstated"):
            assert isinstance(got, PolicySnapshot) and got.version == case.expected_version, case.why
        else:
            assert isinstance(got, NotInForce), case.why


# ------------------------------------------------------------------ status decides first
@pytest.mark.parametrize("status, since", [("lapsed", date(2026, 2, 10)), ("cancelled", date(2026, 2, 1))])
def test_a_policy_that_is_not_covered_gets_not_in_force_with_the_pas_reason(status, since):
    pas = FakePas(**covered(status=status, status_effective_from=since))
    got = builder(pas).get_snapshot("POL-X", date(2026, 3, 1))
    assert got == NotInForce(policy_id="POL-X", loss_date=date(2026, 3, 1), reason=f"PAS status '{status}' since {since}; loss on 2026-03-01")


def test_an_unknown_or_uncovered_date_is_not_in_force_not_an_error():
    got = builder(FakePas(status="not_found")).get_snapshot("POL-X", date(2026, 3, 1))
    assert isinstance(got, NotInForce) and "not_found" in got.reason and "exact number" in got.reason


def test_a_version_window_that_does_not_cover_the_loss_date_is_not_in_force():
    pas = FakePas(**covered(effective_to=date(2026, 2, 28)))
    got = builder(pas).get_snapshot("POL-X", date(2026, 3, 1))
    assert isinstance(got, NotInForce) and "no version applies" in got.reason


def test_reinstated_is_covered_and_carried_on_the_snapshot():
    got = builder(FakePas(**covered(status="reinstated"))).get_snapshot("POL-X", date(2026, 3, 15))
    assert isinstance(got, PolicySnapshot) and got.pas_status == "reinstated"


def test_a_covered_status_without_a_version_is_a_data_error_not_a_guess():
    with pytest.raises(PolicyDocumentMissing, match="no policy version"):
        builder(FakePas(status="in_force")).get_snapshot("POL-X", date(2026, 3, 1))


def test_a_missing_base_wording_raises_instead_of_building_a_partial_policy():
    with pytest.raises(PolicyDocumentMissing, match="W-1"):
        builder(FakePas(**covered()), library(E_1=E1)).get_snapshot("POL-X", date(2026, 3, 1))


# ------------------------------------------------------------------ what the snapshot contains
def test_the_snapshot_carries_the_pas_reference_timestamp_mock_label_and_version():
    s = builder(FakePas(**covered())).get_snapshot("POL-X", date(2026, 3, 1))
    assert (s.policy_id, s.version, s.source_system_response_ref, s.retrieved_at, s.mock) == (
        "POL-X", "v1", "PAS-MOCK/POL-X/2026-03-01", NOW, True)
    assert (s.effective_from, s.effective_to) == (date(2026, 1, 1), date(2026, 12, 31))


def test_only_clauses_in_force_on_the_loss_date_are_in_the_snapshot():
    pas = FakePas(**covered())
    before = builder(pas).get_snapshot("POL-X", date(2026, 2, 1))
    after = builder(pas).get_snapshot("POL-X", date(2026, 3, 1))
    assert "E-1/1" in {c.clause_id for c in before.clauses} and "E-2/1" not in {c.clause_id for c in before.clauses}
    assert "E-2/1" in {c.clause_id for c in after.clauses} and "E-1/1" not in {c.clause_id for c in after.clauses}
    assert [e.endorsement_id for e in before.endorsements] == ["E-1"] and [e.endorsement_id for e in after.endorsements] == ["E-2"]
    assert any("replaces E-1" in r for r in after.precedence_rules) and any("not effective until 2026-03-01" in r for r in before.precedence_rules)


def test_relations_in_the_snapshot_join_only_clauses_that_are_in_the_snapshot():
    pas = FakePas(**covered())
    for day in (date(2026, 2, 1), date(2026, 3, 1)):
        s = builder(pas).get_snapshot("POL-X", day)
        ids = {c.clause_id for c in s.clauses}
        assert s.relations and all(r.source_clause_id in ids and r.target_clause_id in ids for r in s.relations)


def test_a_replacing_endorsement_is_linked_to_the_clause_the_replaced_one_modified():
    """Regression found by retrieval (T3.4): once E-2 replaces E-1, the cover must still point at the endorsement that governs it."""
    s = builder(FakePas(**covered())).get_snapshot("POL-X", date(2026, 3, 1))
    assert {(r.source_clause_id, r.target_clause_id, r.relation) for r in s.relations} >= {("E-2/1", "W-1/2.1", "modifies")}
    before = builder(FakePas(**covered())).get_snapshot("POL-X", date(2026, 2, 1))
    assert {(r.source_clause_id, r.target_clause_id) for r in before.relations} >= {("E-1/1", "W-1/2.1")}
    assert not any(r.source_clause_id == "E-2/1" for r in before.relations)


def test_endorsement_entries_list_exactly_their_clauses_in_force():
    s = builder(FakePas(**covered())).get_snapshot("POL-X", date(2026, 3, 1))
    (e,) = s.endorsements
    assert (e.endorsement_id, e.title, e.effective_from, e.clause_ids) == ("E-2", "Revised Water Sublimit Endorsement", date(2026, 3, 1), ["E-2/1"])


def test_a_listed_endorsement_without_wording_is_reported_and_never_invented():
    lib = library(W_1=WORDING)
    out = builder(FakePas(**covered())).build("POL-X", date(2026, 3, 1), {"E-1": "Water Sublimit Endorsement"})
    assert out.in_force and out.effective.missing_documents == []                       # the library above has both
    out = SnapshotBuilder(FakePas(**covered()), lib, EXTRACTOR, clock=lambda: NOW).build("POL-X", date(2026, 3, 1), {"E-1": "Water Sublimit Endorsement"})
    assert out.effective.missing_documents == ["E-1", "E-2"] and out.result.endorsements == []
    assert out.effective.abstain_clause_ids == ["W-1/2.1"]
    assert any("Endorsement E-1 is listed in the schedule but its wording is not available" in r for r in out.result.precedence_rules)
    assert any(r.startswith("Clauses to abstain on") for r in out.result.precedence_rules)


def test_unstated_precedence_is_written_into_the_snapshot_not_resolved_silently():
    lib = library(W_1=WORDING.split("SECTION 5")[0], E_1=E1, E_2=E2)
    s = builder(FakePas(**covered()), lib).get_snapshot("POL-X", date(2026, 6, 1))
    assert {"E-1/1", "E-2/1"} <= {c.clause_id for c in s.clauses}                 # neither dropped
    assert any(r.startswith("UNRESOLVED (precedence_unstated)") for r in s.precedence_rules)
    assert any(r.startswith("Clauses to abstain on") and r.endswith("W-1/2.1.") for r in s.precedence_rules)


# ------------------------------------------------------------------ determinism, contract and persistence
def test_the_snapshot_is_deterministic_and_a_valid_contract_after_a_json_round_trip():
    a = builder(FakePas(**covered())).get_snapshot("POL-X", date(2026, 3, 1))
    b = builder(FakePas(**covered())).get_snapshot("POL-X", date(2026, 3, 1))
    assert a == b and PolicySnapshot.model_validate_json(a.model_dump_json()) == a


def test_the_pas_is_asked_once_with_the_exact_policy_number_and_loss_date():
    pas = FakePas(**covered())
    builder(pas).get_snapshot("POL-X", date(2026, 3, 1))
    assert pas.calls == [("POL-X", date(2026, 3, 1))]


def test_a_library_rebuilt_from_the_relational_store_gives_the_same_snapshot(tmp_path):
    store = PolicyStore(tmp_path / "p.db")
    for doc in ("W-1", "E-1", "E-2"):
        d = LIB.get(doc)
        store.save(d, 1)
    rebuilt = PolicyLibrary.from_store(store)
    assert rebuilt.ids() == LIB.ids()
    a = builder(FakePas(**covered())).get_snapshot("POL-X", date(2026, 3, 1))
    b = builder(FakePas(**covered()), rebuilt).get_snapshot("POL-X", date(2026, 3, 1))
    assert a == b


def test_the_policy_package_still_depends_only_on_core_and_itself():
    import re

    for f in (ROOT / "src/claimtower/policy").glob("*.py"):
        for imp in re.findall(r"^(?:from|import) (claimtower[\w.]*)", f.read_text(encoding="utf-8"), re.MULTILINE):
            assert imp.startswith(("claimtower.core", "claimtower.policy")), f"{f.name} imports {imp}"
