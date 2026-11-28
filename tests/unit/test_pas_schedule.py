"""A policy schedule read into the mock PAS: the model proposes, the document confirms, a person registers. Nothing the model says is kept unless the document says it too."""
import json
from datetime import date

import pymupdf
import pytest
from fastapi.testclient import TestClient

from claimtower.api.app import create_app
from claimtower.composition import SYNTHETIC, build_registry
from claimtower.core.config import load_settings
from claimtower.ingestion.store import DocumentStore
from claimtower.integrations.mock.data import PAS_FILE
from claimtower.integrations.mock.pas_mock import MockPasAdapter, PasConflict, PasRegisterError
from claimtower.orchestration.graph import ClaimRunner
from claimtower.integrations.mock.schedule import ScheduleAmount, ScheduleAnswer, ScheduleEndorsement, ScheduleReader, date_in_text, parse_date
from claimtower.ui.client import ApiClient, ApiError
from claimtower.ui.views.policies import policy_from_form

SETTINGS = load_settings()
SAMPLE = (SYNTHETIC / "claims" / "CLM-0005" / "policy_schedule.pdf").read_bytes()
LINES = ["POLICY SCHEDULE", "Policyholder: Mrs A Rao", "Policy number: HSP-9000", "Period of insurance: 1 March 2026 to 28 February 2027", "Policy wording: HS-W-2026",
         "Endorsements attached: E-01 Escape of water; E-02 Higher deductible effective 1 September 2026", "Building limit: INR 4,500,000", "Standard deductible: INR 10,000"]


def pdf(lines):
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 60), "\n".join(lines), fontsize=10)
    return doc.tobytes()


class FakeLLM:
    def __init__(self, answer=None, fail=None):
        self.answer, self.fail = answer, fail

    def structured(self, cls, prompt, **kw):
        if self.fail:
            raise self.fail
        return self.answer


GOOD = ScheduleAnswer(policy_number="HSP-9000", holder_name="Mrs A Rao", period_text="1 March 2026 to 28 February 2027", period_start="2026-03-01", period_end="2027-02-28",
                      base_wording_id="HS-W-2026", endorsements=[ScheduleEndorsement(id="E-01"), ScheduleEndorsement(id="E-02", effective_from="2026-09-01")],
                      amounts=[ScheduleAmount(name="building_limit", value="4500000"), ScheduleAmount(name="standard_deductible", value="10000")], confidence=0.9)


# ---------------------------------------------------------------------------------------------------------------- dates
def test_dates_are_read_in_the_usual_written_forms_and_must_appear_in_the_text():
    assert parse_date("1 March 2026") == date(2026, 3, 1) and parse_date("March 1, 2026") == date(2026, 3, 1) and parse_date("2026-03-01") == date(2026, 3, 1)
    assert parse_date("31 Dec 2026") == date(2026, 12, 31) and parse_date("soon") is None and parse_date("31 February 2026") is None
    assert date_in_text(date(2026, 3, 1), "01 March 2026 to 28 February 2027 (both days inclusive)")          # a leading zero is the same day
    assert date_in_text(date(2027, 2, 28), "to 28 February 2027") and date_in_text(date(2027, 2, 28), "28/02/2027") and not date_in_text(date(2027, 2, 27), "to 28 February 2027")


# ---------------------------------------------------------------------------------------------------------------- the reader
def test_a_grounded_answer_becomes_a_pas_record_with_the_status_assumed():
    p = ScheduleReader(FakeLLM(GOOD), SETTINGS.pack).read(pdf(LINES), "s.pdf")
    assert not p.problems and p.source == "model" and p.policy is not None
    v = p.policy.versions[0]
    assert (p.policy.policy_no, p.policy.holder_name, v.base_wording_id) == ("HSP-9000", "Mrs A Rao", "HS-W-2026")
    assert (v.effective_from, v.effective_to) == (date(2026, 3, 1), date(2027, 2, 28)) and v.schedule == {"building_limit": "4500000", "standard_deductible": "10000"}
    assert [(e.endorsement_id, e.effective_from) for e in v.endorsements] == [("E-01", date(2026, 3, 1)), ("E-02", date(2026, 9, 1))]
    assert p.policy.status_history[0].status == "in_force" and "assumed" in p.policy.note


def test_values_the_document_does_not_contain_are_left_out_not_kept():
    bad = GOOD.model_copy(update={"endorsements": [ScheduleEndorsement(id="E-01"), ScheduleEndorsement(id="E-99")],
                                  "amounts": [ScheduleAmount(name="building_limit", value="9999999"), ScheduleAmount(name="standard_deductible", value="10000")],
                                  "base_wording_id": "MADE-UP-1", "holder_name": "Somebody Else"})
    p = ScheduleReader(FakeLLM(bad), SETTINGS.pack).read(pdf(LINES), "s.pdf")
    v = p.policy.versions[0]
    assert [e.endorsement_id for e in v.endorsements] == ["E-01"] and v.schedule == {"standard_deductible": "10000"} and v.base_wording_id == "" and p.policy.holder_name == "(not stated)"
    text = " ".join(p.warnings)
    assert "E-99" in text and "9999999" in text and "does not name the base wording" in text and "policyholder is not stated" in text
    assert any(c.field == "endorsement" and c.value == "E-99" and not c.found for c in p.checks)


@pytest.mark.parametrize("change,why", [({"policy_number": "NOPE-1"}, "policy number"), ({"period_text": "1 March 2026 to 1 March 2027"}, "period"), ({"period_end": "2027-02-27"}, "period"),
                                        ({"confidence": 0.2}, "reliable"), ({"period_start": "2027-03-01", "period_end": "2027-02-28"}, "period")])
def test_no_record_is_proposed_when_the_policy_number_or_period_cannot_be_confirmed(change, why):
    p = ScheduleReader(FakeLLM(GOOD.model_copy(update=change)), SETTINGS.pack).read(pdf(LINES), "s.pdf")
    assert p.policy is None and why in " ".join(p.problems)


def test_a_model_that_fails_or_a_scan_gives_a_reason_not_a_crash():
    assert "could not be reached" in ScheduleReader(FakeLLM(fail=RuntimeError("down")), SETTINGS.pack).read(pdf(LINES)).problems[0]
    empty = pymupdf.open()
    empty.new_page()
    assert "no text layer" in ScheduleReader(None).read(empty.tobytes()).problems[0]
    with pytest.raises(ValueError, match="not a readable PDF"):
        ScheduleReader(None).read(b"nope")


def test_without_a_model_the_usual_layout_is_read_by_fixed_patterns():
    p = ScheduleReader(None).read(SAMPLE, "policy_schedule.pdf")
    v = p.policy.versions[0]
    assert p.source == "rules" and p.policy.policy_no == "POL-1005" and v.schedule == {"building_limit": "2000000", "standard_deductible": "10000"}
    assert [e.endorsement_id for e in v.endorsements] == ["END-01"] and (v.effective_from, v.effective_to) == (date(2026, 1, 1), date(2026, 12, 31))
    assert "base wording" in " ".join(p.warnings)                               # this schedule does not name it: the person must enter it


# ---------------------------------------------------------------------------------------------------------------- registering
def adapter(tmp_path):
    return MockPasAdapter(PAS_FILE, tmp_path / "registered.json")


def record(wording="HS-W-2026", number="HSP-9000"):
    p = ScheduleReader(FakeLLM(GOOD), SETTINGS.pack).read(pdf(LINES), "s.pdf").policy
    return p.model_copy(update={"policy_no": number}) if number != "HSP-9000" else p.model_copy(update={"versions": [p.versions[0].model_copy(update={"base_wording_id": wording})]})


def test_a_registered_policy_is_used_at_once_survives_a_restart_and_needs_replace_to_change(tmp_path):
    a = adapter(tmp_path)
    assert a.get_policy_version("HSP-9000", date(2026, 6, 1)).status == "not_found"
    assert a.register_policy(record()) == "added"
    r = a.get_policy_version("HSP-9000", date(2026, 6, 1))
    assert r.status == "in_force" and r.base_wording_id == "HS-W-2026" and [e.endorsement_id for e in r.endorsements] == ["E-01", "E-02"] and a.get_schedule("HSP-9000", date(2026, 6, 1))
    assert adapter(tmp_path).get_policy_version("HSP-9000", date(2026, 6, 1)).status == "in_force"           # a new adapter reads the file again
    with pytest.raises(PasConflict):
        a.register_policy(record())
    assert a.register_policy(record(), replace=True) == "replaced"
    with pytest.raises(PasConflict):
        a.register_policy(record(number="POL-1001"))                                                       # a synthetic policy is not overwritten silently
    with pytest.raises(PasRegisterError, match="base wording"):
        a.register_policy(record(wording=" "), replace=True)
    sources = {p["policy_no"]: p["source"] for p in a.policies()}
    assert sources["HSP-9000"] == "registered" and sources["POL-1001"] == "synthetic"
    assert json.loads((tmp_path / "registered.json").read_text(encoding="utf-8"))["policies"][0]["policy_no"] == "HSP-9000"


def test_the_form_corrections_are_applied_to_the_proposal():
    base = record().model_dump(mode="json")
    out = policy_from_form(base, " Mr B Rao ", "2026-04-01", "2027-03-31", " HS-W-2026 ", [{"id": "E-01", "effective_from": ""}, {"id": "", "effective_from": ""}, {"id": "E-03", "effective_from": "2026-10-01"}],
                           [{"name": "building_limit", "value": "5,000,000"}, {"name": "", "value": "1"}, {"name": "standard_deductible", "value": ""}])
    v = out["versions"][0]
    assert out["holder_name"] == "Mr B Rao" and (v["effective_from"], v["effective_to"]) == ("2026-04-01", "2027-03-31") and v["base_wording_id"] == "HS-W-2026"
    assert [(e["endorsement_id"], e["effective_from"]) for e in v["endorsements"]] == [("E-01", "2026-04-01"), ("E-03", "2026-10-01")] and v["schedule"] == {"building_limit": "5000000"}
    assert out["status_history"][0]["effective_from"] == "2026-04-01" and base["versions"][0]["effective_from"] == "2026-03-01"          # the proposal itself is not changed


# ---------------------------------------------------------------------------------------------------------------- the API
@pytest.fixture
def api(tmp_path) -> ApiClient:
    store = DocumentStore(tmp_path / "store")
    runner = ClaimRunner(build_registry(SETTINGS, document_store=store), settings=SETTINGS)
    app = create_app(runner, store, synthetic_claims=SYNTHETIC / "claims", pas=adapter(tmp_path), schedule_reader=ScheduleReader(FakeLLM(GOOD), SETTINGS.pack))
    return ApiClient(http=TestClient(app))


def test_the_api_reads_a_schedule_stores_nothing_then_registers_what_a_person_confirmed(api):
    r = api.read_schedule("schedule.pdf", pdf(LINES))
    assert r["policy"]["policy_no"] == "HSP-9000" and r["existing"] == "none" and r["read_by"] == "model" and not r["problems"] and any(c["found"] for c in r["checks"])
    assert "HSP-9000" not in {p["policy_no"] for p in api.pas_policies()}                                  # reading stores nothing
    done = api.register_pas_policy(r["policy"])
    assert done["outcome"] == "added" and "HSP-9000" in {p["policy_no"] for p in api.pas_policies()}
    assert api.read_schedule("again.pdf", pdf(LINES))["existing"] == "registered"
    with pytest.raises(ApiError) as e:
        api.register_pas_policy(r["policy"])
    assert e.value.status == 409
    assert api.register_pas_policy(r["policy"], replace=True)["outcome"] == "replaced"


def test_the_api_refuses_what_cannot_be_registered_or_read(api):
    r = api.read_schedule("schedule.pdf", pdf(LINES))
    r["policy"]["versions"][0]["base_wording_id"] = ""
    with pytest.raises(ApiError) as e:
        api.register_pas_policy(r["policy"])
    assert e.value.status == 422 and "base wording" in str(e.value.detail)
    with pytest.raises(ApiError) as e:
        api.read_schedule("junk.pdf", b"nope")
    assert e.value.status == 422
    with pytest.raises(ApiError) as e:
        api.read_schedule("empty.pdf", b"")
    assert e.value.status == 422


def test_without_a_mock_pas_the_routes_say_so(tmp_path):
    store = DocumentStore(tmp_path / "store")
    c = TestClient(create_app(ClaimRunner(build_registry(SETTINGS, document_store=store), settings=SETTINGS), store, synthetic_claims=SYNTHETIC / "claims"))
    assert c.get("/pas/policies").status_code == 404 and "mock PAS" in c.get("/pas/policies").json()["detail"]


def test_a_zero_padded_period_and_an_amount_given_in_words_are_read_without_false_alarms():
    lines = ["Policy number: HSP-9001", "Period of insurance: 01 March 2026 to 28 February 2027 (both days inclusive)", "Policy wording: HS-W-2026", "Building limit: INR 3,200,000",
             "Escape of water cover: as per wording", "Deductible: as per wording clause 5.1"]
    ans = GOOD.model_copy(update={"policy_number": "HSP-9001", "period_text": "01 March 2026 to 28 February 2027 (both days inclusive)", "endorsements": [],
                                  "amounts": [ScheduleAmount(name="building_limit", value="3200000"), ScheduleAmount(name="escape_of_water_cover", value=""), ScheduleAmount(name="deductible", value="")]})
    p = ScheduleReader(FakeLLM(ans), SETTINGS.pack).read(pdf(lines), "s.pdf")
    assert not p.problems and p.policy.versions[0].schedule == {"building_limit": "3200000"}
    assert not any("escape_of_water" in w or "'deductible" in w for w in p.warnings) and any("no standard deductible" in w for w in p.warnings)


def test_an_item_the_schedule_gives_in_words_is_kept_as_those_words_when_the_document_says_them():
    from claimtower.integrations.mock.schedule import is_sum

    assert is_sum("INR 3,200,000") and is_sum("10,000") and is_sum("4500000") and not is_sum("as per wording clause 5.1") and not is_sum("included")
    lines = ["Policy number: HSP-9002", "Period of insurance: 1 March 2026 to 28 February 2027", "Policy wording: HS-W-2026", "Building limit: INR 3,200,000",
             "Standard deductible: as per wording clause 5.1", "Escape of water cover: included"]
    ans = GOOD.model_copy(update={"policy_number": "HSP-9002", "endorsements": [], "amounts": [
        ScheduleAmount(name="building_limit", value="3200000"), ScheduleAmount(name="standard_deductible", value="as per wording clause 5.1"),
        ScheduleAmount(name="escape_of_water_cover", value="included"), ScheduleAmount(name="premium", value="as per quotation 77")]})
    p = ScheduleReader(FakeLLM(ans), SETTINGS.pack).read(pdf(lines), "s.pdf")
    assert p.policy.versions[0].schedule == {"building_limit": "3200000", "standard_deductible": "as per wording clause 5.1", "escape_of_water_cover": "included"}
    assert any("premium" in w and "not in the document" in w for w in p.warnings) and any("in words" in w and "read it from the wording" in w for w in p.warnings)
