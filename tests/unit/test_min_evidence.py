"""T4.5 - minimum-evidence requests: grouped per addressee, every item serves a need, nothing already on file is asked for again."""
import json
from pathlib import Path

import pytest

from claimtower.core.config import QuestionTemplates, RequestConfig, load_settings
from claimtower.core.schemas import ClaimHeader, Contradiction, Fact, PolicySnapshot
from claimtower.evidence.linker import Link
from claimtower.evidence.llm_requests import DraftAnswer, DraftModel, LLMRequestWriter, make_generator
from claimtower.evidence.min_evidence import DraftItem, GeneratedRequest, MinimumEvidenceGenerator, Need, RequestItem, WriterOutcome
from claimtower.evidence.questions import QuestionRegister
from claimtower.evidence.sufficiency import CriterionAssessment, QuestionAssessment, SufficiencyResult

ROOT = Path(__file__).resolve().parents[2]
PACK = load_settings().pack
TEMPLATES = PACK.question_templates
CRIT = {q.id: q.establishing_criteria for q in TEMPLATES.questions}


def snapshot(s="S-01"):
    return PolicySnapshot(**json.loads((ROOT / f"data/fixtures/{s}/policy.json").read_text(encoding="utf-8"))["policy_snapshot"])


QUESTIONS = QuestionRegister(TEMPLATES).build(ClaimHeader(claim_id="C"), snapshot()).questions


def assess(qid, status, gap, verdicts, contradiction_ids=(), abstained=False):
    crit = tuple(CriterionAssessment(i, CRIT[qid][i], v, "model", "because") for i, v in enumerate(verdicts))
    return QuestionAssessment(qid, status, gap, "R", crit, tuple(contradiction_ids), (), abstained, "x")


def result(*assessments):
    return SufficiencyResult(assessments=list(assessments))


def fact(fid, text):
    return Fact(fact_id=fid, claim_id="C", proposition=text, source_document_id="D1", page=1, evidence_text=text, evidence_type="neutral", extraction_confidence=0.9)


CONFLICT = Contradiction(contradiction_id="C-1", question_id="Q-SUDDEN", fact_ids_a=["F1"], fact_ids_b=["F2"], kind="cause", severity="high",
                         explanation="the claimant reports a sudden burst, the plumber reports long-term corrosion")
S01 = result(
    assess("Q-INFORCE", "established", None, ["met"]),
    assess("Q-SUDDEN", "requires_validation", "contradictory", ["met", "met"], ["C-1"]),
    assess("Q-START", "missing", "missing", ["unmet", "unmet"]),
    assess("Q-EXTENT", "partially_established", "insufficient", ["met", "unmet"]),
    assess("Q-AMOUNT", "unverified", "insufficient", ["unmet", "partly"]),
    assess("Q-SUBLIMIT", "established", None, ["met"]))
FACTS = [fact("F1", "Claimant reports the kitchen supply pipe burst suddenly"), fact("F2", "Significant corrosion was present at the joint of the kitchen supply pipe")]
LINKS = [Link("F1", "Q-SUDDEN", "supporting", "x", "template"), Link("F2", "Q-SUDDEN", "contradicting", "x", "template"), Link("F2", "Q-START", "neutral", "x", "template")]


class FakeWriter:
    def __init__(self, drafts=None, unavailable=False):
        self.drafts, self.unavailable, self.contexts = drafts or {}, unavailable, []

    def write(self, ctx):
        self.contexts.append(ctx)
        return WriterOutcome(list(self.drafts.get(ctx.role, [])), unavailable=self.unavailable, detail="down" if self.unavailable else "")


def generate(sufficiency=S01, writer=None, received=("fnol", "plumber_report", "surveyor_report", "estimate"), **kw):
    return MinimumEvidenceGenerator(TEMPLATES, writer).generate(QUESTIONS, sufficiency, FACTS, LINKS, [CONFLICT], received, **kw)


# ------------------------------------------------------------------------------------------------ who is asked, grouped
def test_unresolved_needs_are_grouped_into_one_request_per_addressee():
    r = generate()
    assert [d.role for d in r.details] == ["plumber", "claimant", "surveyor"] and r.problems == []
    plumber, claimant, surveyor = r.requests
    assert (plumber.question_id, plumber.gap_type, plumber.resolves_question_ids) == ("Q-SUDDEN", "contradictory", ["Q-SUDDEN", "Q-START"])
    assert (claimant.question_id, claimant.gap_type, claimant.resolves_question_ids) == ("Q-START", "missing", ["Q-START"])
    assert (surveyor.question_id, surveyor.gap_type, surveyor.resolves_question_ids) == ("Q-EXTENT", "insufficient", ["Q-EXTENT", "Q-AMOUNT"])
    assert plumber.minimum_evidence_request.must_answer_sub_questions == ["Q-SUDDEN", "Q-START"]


def test_the_conflict_goes_to_the_role_that_can_settle_it_and_maps_to_that_roles_criteria():
    plumber = generate().requests[0].minimum_evidence_request
    assert plumber.addressee_role == "plumber"
    assert plumber.maps_to_criteria == [*CRIT["Q-SUDDEN"], CRIT["Q-START"][0]]
    assert any("sudden burst" in i and "long-term corrosion" in i for i in plumber.items)          # the rule item quotes the conflict


def test_only_criteria_that_are_not_met_become_items():
    r = generate()
    surveyor = r.for_role("surveyor")
    assert {n.criterion for n in surveyor.needs} == {CRIT["Q-EXTENT"][1], CRIT["Q-AMOUNT"][0], CRIT["Q-AMOUNT"][1]}      # Q-EXTENT's met criterion is not asked again


def test_a_missing_endorsement_is_an_internal_request_never_sent_to_the_claimant():
    s5 = result(assess("Q-SUBLIMIT", "missing", "missing", ["unmet"]))
    r = generate(s5)
    (req,) = r.details
    assert (req.role, req.internal, req.label) == ("policy_administration", True, "the policy administration team")
    assert req.question_ids == ["Q-SUBLIMIT"] and r.requests[0].minimum_evidence_request.maps_to_criteria == CRIT["Q-SUBLIMIT"]
    assert "claimant" not in {d.role for d in r.details}


def test_alternative_criteria_need_no_more_evidence_once_one_is_met():
    one_met = result(assess("Q-START", "partially_established", "insufficient", ["met", "unmet"]))
    none_met = result(assess("Q-START", "missing", "missing", ["unmet", "unmet"]))
    assert generate(one_met).requests == []
    assert {d.role for d in generate(none_met).details} == {"plumber", "claimant"}


def test_established_and_abstained_questions_make_no_request_and_abstained_ones_go_to_review():
    r = generate(result(assess("Q-SUDDEN", "established", None, ["met", "met"]), assess("Q-START", "unverified", None, ["unclear", "met"], abstained=True)))
    assert r.requests == [] and r.review_questions == ["Q-START"]


def test_the_output_is_the_frozen_contract_and_maps_only_to_pack_criteria():
    r = generate()
    valid = {c for cs in CRIT.values() for c in cs}
    for m in r.requests:
        assert m.minimum_evidence_request.items and set(m.minimum_evidence_request.maps_to_criteria) <= valid


# ------------------------------------------------------------------------------------------------ the writer is checked
def draft(text, ids=(0,), asks="information", doc=""):
    return DraftItem(tuple(ids), text, asks, doc)


def plumber_needs():
    return generate().for_role("plumber").needs


def test_a_good_model_item_is_used_and_marked_as_the_models():
    w = FakeWriter({"plumber": [draft("Whether the corrosion at the kitchen supply pipe joint predates 14 March 2026 and how you can tell", (0, 1))]})
    r = generate(writer=w)
    items = r.for_role("plumber").items
    assert items[0].source == "model" and items[0].need_ids == (0, 1) and r.problems == [] and r.refused == []


@pytest.mark.parametrize("d,why", [
    (draft("Hmm"), "too short"), (draft("x" * 300), "longer"), (draft("Send the report", (9,)), "does not serve"), (draft("Send the report", ()), "does not serve"),
    (draft("Please confirm whether the claim will be approved"), "pre-judges"), (draft("Tell us if we should deny this claim"), "pre-judges"),
    (draft("Send something please", asks="demand"), "allowed kind"),
    (draft("A survey report on the building", asks="new_document", doc="surveyor_report"), "can supply"),
    (draft("A new plumber report on the failed joint", asks="new_document", doc="plumber_report"), "already on file"),
])
def test_a_bad_model_item_is_refused_with_the_reason_and_its_needs_still_get_an_item(d, why):
    r = generate(writer=FakeWriter({"plumber": [d]}))
    assert why in r.refused[0].reason and r.problems == []
    assert all(i.source == "rule" for i in r.for_role("plumber").items)


def test_a_document_type_not_yet_on_file_may_be_requested():
    r = generate(writer=FakeWriter({"plumber": [draft("Close-up photographs of the failed kitchen supply pipe joint", (0, 1), "new_document", "photo")]}),
                 received=("fnol",))
    assert r.refused == [] and r.for_role("plumber").items[0].document_type == "photo"


def test_a_repeat_item_and_items_over_the_limit_are_refused():
    twin = [draft("Whether the corrosion at the joint predates the loss date", (0,)), draft("Whether the corrosion at the joint predates the loss", (1,))]
    assert "repeats" in generate(writer=FakeWriter({"plumber": twin})).refused[0].reason
    topics = ["the age of the supply pipe", "the make of the failed coupling", "any photographs taken before the repair", "the pressure of the mains supply",
              "the name of the engineer who attended", "the time the water was turned off"]
    many = [draft(f"Please state {t}", (0,)) for t in topics]
    r = generate(writer=FakeWriter({"plumber": many}))
    assert sum(i.source == "model" for i in r.for_role("plumber").items) == 4 and any("over the limit" in x.reason for x in r.refused)


def test_a_need_the_model_leaves_out_gets_a_rule_written_item_so_nothing_is_dropped():
    r = generate(writer=FakeWriter({"plumber": [draft("Whether the corrosion at the joint predates the loss", (0,))]}))
    items = r.for_role("plumber").items
    assert [i.source for i in items] == ["model", "rule"] and items[1].need_ids == (1,)
    assert {n for i in items for n in i.need_ids} == {0, 1} and r.problems == []


def test_model_outage_still_gives_a_complete_request_and_a_warning():
    r = generate(writer=FakeWriter(unavailable=True))
    assert len(r.requests) == 3 and r.problems == [] and any("unavailable" in w for w in r.warnings)
    assert all(i.source == "rule" for d in r.details for i in d.items)


def test_no_writer_or_a_switched_off_config_is_the_rules_alone():
    off = TEMPLATES.model_copy(update={"requests": TEMPLATES.requests.model_copy(update={"enabled": False})})
    w = FakeWriter({"plumber": [draft("Whether the corrosion at the joint predates the loss", (0,))]})
    MinimumEvidenceGenerator(off, w).generate(QUESTIONS, S01, FACTS, LINKS, [CONFLICT], ())
    assert w.contexts == [] and make_generator(PACK, None).writer is None and make_generator(PACK, FakeLLM()).writer is not None


# ------------------------------------------------------------------------------------------------ self-check and prompt
def test_self_check_finds_an_item_without_a_need_and_a_need_without_an_item():
    gen = generate().for_role("surveyor")
    broken = GeneratedRequest(gen.role, gen.label, gen.internal, gen.delivery, gen.needs, [RequestItem("Do something about it", (), (), "information", "", "rule")], gen.question_ids)
    problems = MinimumEvidenceGenerator(TEMPLATES).self_check(broken, {q.question_id: q for q in QUESTIONS})
    assert any("serves no need" in p for p in problems) and any("not covered" in p for p in problems)


class FakeLLM:
    def __init__(self, items=None, fail=None):
        self.items, self.fail, self.prompts = items or [], fail, []

    def structured(self, model_cls, prompt, **kw):
        self.prompts.append(prompt.text)
        if self.fail:
            raise self.fail
        return DraftAnswer(items=self.items)


def test_the_prompt_carries_the_needs_the_known_facts_the_documents_on_file_and_the_role():
    llm = FakeLLM()
    MinimumEvidenceGenerator(TEMPLATES, LLMRequestWriter(llm, PACK)).generate(QUESTIONS, S01, FACTS, LINKS, [CONFLICT], ("fnol", "plumber_report"),
                                                                              {"D1": "plumber_report"})
    plumber = next(p for p in llm.prompts if "the plumber." in p)
    for needle in ["CONFLICT", "sudden burst", "kitchen supply pipe", "(plumber_report)", "fnol, plumber_report", "Q-START", "photo", "At most 4 items"]:
        assert needle in plumber, needle
    internal = MinimumEvidenceGenerator(TEMPLATES, LLMRequestWriter(llm, PACK)).generate(
        QUESTIONS, result(assess("Q-SUBLIMIT", "missing", "missing", ["unmet"])), FACTS, LINKS, [], ())
    assert internal.problems == [] and "internal request" in llm.prompts[-1]


def test_a_real_llm_writer_answer_goes_through_the_same_checks():
    llm = FakeLLM([DraftModel(need_ids=[0], text="Will the claim be approved?", asks_for="information")])
    r = MinimumEvidenceGenerator(TEMPLATES, LLMRequestWriter(llm, PACK)).generate(QUESTIONS, result(assess("Q-START", "missing", "missing", ["unmet", "unmet"])),
                                                                                  FACTS, LINKS, [], ())
    assert r.refused and r.problems == []
    r2 = MinimumEvidenceGenerator(TEMPLATES, LLMRequestWriter(FakeLLM(fail=RuntimeError("down")), PACK)).generate(
        QUESTIONS, result(assess("Q-START", "missing", "missing", ["unmet", "unmet"])), FACTS, LINKS, [], ())
    assert r2.warnings and len(r2.requests) == 2


# ------------------------------------------------------------------------------------------------ config
def test_request_config_is_validated():
    base = TEMPLATES.requests.model_dump()
    with pytest.raises(ValueError, match="default_role"):
        RequestConfig(**{**base, "default_role": "nobody"})
    with pytest.raises(ValueError, match="unknown role"):
        RequestConfig(**{**base, "conflict_roles": {"Q-SUDDEN": "nobody"}})
    with pytest.raises(ValueError, match="pattern"):
        RequestConfig(**{**base, "forbidden_patterns": ["("]})
    bad = {**base, "criteria_roles": [{"question": "Q-SUDDEN", "criterion": "not a criterion", "role": "plumber"}]}
    with pytest.raises(ValueError, match="no criterion"):
        QuestionTemplates(questions=TEMPLATES.questions, requests=RequestConfig(**bad))
    assert TEMPLATES.requests.enabled and TEMPLATES.requests.roles["policy_administration"].internal
    _ = Need


# ------------------------------------------------------------------------------------------------ repair round
class RepairWriter:
    """Answers the first call with a refused (too long) item and the second, which carries the refusal feedback, with a good one."""

    def __init__(self):
        self.contexts = []

    def write(self, ctx):
        self.contexts.append(ctx)
        if not ctx.feedback:
            return WriterOutcome([draft("x" * 300, (0, 1))])
        return WriterOutcome([draft("Whether the corrosion at the kitchen supply pipe joint predates 14 March 2026", tuple(ctx.uncovered))])


def test_when_items_are_refused_and_needs_uncovered_the_writer_is_asked_again_with_the_reasons():
    w = RepairWriter()
    r = generate(writer=w)
    plumber_calls = [c for c in w.contexts if c.role == "plumber"]
    assert len(plumber_calls) == 2 and plumber_calls[0].feedback == () and "longer than" in plumber_calls[1].feedback[0] and list(plumber_calls[1].uncovered) == [0, 1]
    items = r.for_role("plumber").items
    assert [i.source for i in items] == ["model"] and items[0].need_ids == (0, 1) and r.problems == []


def test_no_repair_round_when_nothing_was_refused_or_when_it_is_switched_off():
    ok = FakeWriter({"plumber": [draft("Whether the corrosion at the joint predates the loss", (0, 1))]})
    generate(writer=ok)
    assert len([c for c in ok.contexts if c.role == "plumber"]) == 1
    off = TEMPLATES.model_copy(update={"requests": TEMPLATES.requests.model_copy(update={"repair_rounds": 0})})
    w = RepairWriter()
    MinimumEvidenceGenerator(off, w).generate(QUESTIONS, S01, FACTS, LINKS, [CONFLICT], ())
    assert len([c for c in w.contexts if c.role == "plumber"]) == 1


def test_the_repair_prompt_names_the_refusals_and_the_uncovered_needs():
    class Llm:
        def __init__(self):
            self.prompts = []

        def structured(self, model_cls, prompt, **kw):
            self.prompts.append(prompt.text)
            return DraftAnswer(items=[DraftModel(need_ids=[0, 1], text="y" * 300)] if len(self.prompts) % 2 else [])

    llm = Llm()
    MinimumEvidenceGenerator(TEMPLATES, LLMRequestWriter(llm, PACK)).generate(QUESTIONS, result(assess("Q-START", "missing", "missing", ["unmet", "unmet"])),
                                                                              FACTS, LINKS, [], ())
    repair = next(p for p in llm.prompts if "previous attempt" in p)
    assert "longer than 250 characters" in repair and "ids: 0" in repair


# ------------------------------------------------------------------------------------------------ delivery (decision 2026-11-25)
def test_a_third_party_is_never_contacted_directly_only_the_claimant_is():
    r = generate()
    assert {d.role: d.delivery for d in r.details} == {"plumber": "via_claimant_or_adjuster", "claimant": "direct", "surveyor": "via_claimant_or_adjuster"}
    internal = generate(result(assess("Q-SUBLIMIT", "missing", "missing", ["unmet"])))
    assert [(d.role, d.delivery) for d in internal.details] == [("policy_administration", "internal")]
    roles = TEMPLATES.requests.roles
    assert {k for k, v in roles.items() if v.delivery == "direct"} == {"claimant"}
    assert all(v.delivery == "via_claimant_or_adjuster" for k, v in roles.items() if k in ("plumber", "surveyor", "contractor"))


def test_an_internal_role_must_be_delivered_internally_and_only_it():
    from claimtower.core.config import RequestRole
    with pytest.raises(ValueError, match="internal"):
        RequestRole(label="x", internal=True, delivery="direct")
    with pytest.raises(ValueError, match="internal"):
        RequestRole(label="x", internal=False, delivery="internal")
