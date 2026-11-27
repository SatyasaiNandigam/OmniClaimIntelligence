"""A wording whose clauses are not numbered the way the patterns expect is cut at the headings the model marks; the marks are checked and kept."""
import pymupdf
import pytest

from claimtower.core.config import load_settings
from claimtower.policy.decompose import PolicyDecomposer
from claimtower.policy.llm_identity import IdentityAnswer
from claimtower.policy.llm_segments import ClauseMark, LLMSegmenter, Marks, SegmentAnswer
from claimtower.policy.manager import IdentityReader, PolicyDocumentError, PolicyDocumentManager

PACK = load_settings().pack
LINES = ["Maple Mutual HOME SHIELD PLUS", "Document ref HSP/2026/B (edition 4)", "Policy number MM-55-0098213", "WHAT IS COVERED", "Escape of water",
         "We cover damage caused by water escaping from a fixed pipe.", "Wear and tear", "We do not cover damage caused by wear and tear."]


def pdf(lines):
    d = pymupdf.open()
    d.new_page().insert_text((50, 60), "\n".join(lines), fontsize=10)
    return d.tobytes()


class FakeLLM:
    def __init__(self, seg=None, fail_seg=False):
        self.seg, self.fail_seg, self.seg_calls = seg, fail_seg, 0

    def structured(self, cls, prompt, **kw):
        if cls is IdentityAnswer:
            return IdentityAnswer(document_id="HSP/2026/B", evidence_quote="Document ref HSP/2026/B (edition 4)", confidence=0.95, kind="policy_wording")
        self.seg_calls += 1
        if self.fail_seg:
            raise ConnectionError("down")
        return self.seg


SEG = SegmentAnswer(clauses=[ClauseMark(heading="Escape of water"), ClauseMark(heading="Wear and tear"), ClauseMark(heading="invented heading")], sections=["WHAT IS COVERED"])


def mgr(tmp_path, llm):
    from claimtower.policy.llm_typing import make_decomposer
    return PolicyDocumentManager(make_decomposer(PACK, llm), IdentityReader.load(PACK, llm), None, tmp_path / "up")


def test_only_headings_that_are_lines_of_the_page_are_kept():
    m = Marks()
    LLMSegmenter.check(SEG, [x for x in LINES], m)
    assert set(m.clauses) == {"Escape of water", "Wear and tear"} and m.sections == {"WHAT IS COVERED"}


def test_an_unnumbered_wording_is_cut_at_the_marked_headings_and_a_reload_does_not_ask_again(tmp_path):
    llm = FakeLLM(SEG)
    m = mgr(tmp_path, llm)
    res = m.ingest(pdf(LINES), "x.pdf")
    assert res.entry.clauses + res.entry.unclassified == 2 and llm.seg_calls == 1
    assert any("marked by the model" in w for w in res.entry.warnings)
    again = FakeLLM(fail_seg=True)
    m2 = mgr(tmp_path, again)
    assert again.seg_calls == 0 and next(e for e in m2.entries() if e.doc_id == "HSP-2026-B").clauses + next(e for e in m2.entries()).unclassified == 2


def test_when_the_model_marks_nothing_the_error_says_so(tmp_path):
    m = mgr(tmp_path, FakeLLM(SegmentAnswer()))
    with pytest.raises(PolicyDocumentError, match="no clauses were found.*model said: the model found no clause headings"):
        m.ingest(pdf(LINES), "x.pdf")


def test_without_a_model_the_numbering_patterns_decide():
    d = PolicyDecomposer(PACK.policy_structure)
    class P:
        number, text = 1, "\n".join(LINES)
    assert not d.decompose("X", [P()], use_segmenter=True).clauses


def test_a_one_line_clause_is_kept_verbatim_and_a_bare_title_is_not():
    lines = ["Policy wording", "SECTION 3 EXCLUSIONS", "3.3 Flood, storm and sea water. We do not cover loss or damage caused by flood, storm, tidal or sea water.",
             "1.3 Escape of Water means the discharge of water from a fixed water pipe, tank, or domestic appliance."]
    marks = Marks({lines[2]: "3.3", lines[3]: "1.3", lines[1]: ""}, set())
    class P:
        number, text = 1, "\n".join(lines)
    from claimtower.policy.decompose import verify_clauses
    d = PolicyDecomposer(PACK.policy_structure).decompose("X", [P()], marks=marks)
    assert [c.clause_id for c in d.clauses + [u for u in []]] or d.unclassified
    texts = {c.text for c in d.clauses} | {u.text for u in d.unclassified}
    assert lines[2] in texts and lines[3] in texts and lines[1] not in texts
    assert verify_clauses(d.clauses, [P()]) == [] and any("only a title" in w for w in d.warnings)
    flood = next(c for c in d.clauses if c.text == lines[2])
    assert flood.clause_type == "exclusion" and flood.heading == "3.3 Flood, storm and sea water"


def test_the_kind_is_what_the_model_read_not_a_title_pattern(tmp_path):
    class Endorse(FakeLLM):
        def structured(self, cls, prompt, **kw):
            if cls is IdentityAnswer:
                return IdentityAnswer(document_id="E-01", evidence_quote="Endorsement E-01", confidence=0.95, kind="endorsement")
            return super().structured(cls, prompt, **kw)

    lines = ["Endorsement E-01", "Escape of water sublimit", "We will pay no more than INR 80,000 in total for escape of water in any policy year."]
    seg = SegmentAnswer(clauses=[ClauseMark(heading="Escape of water sublimit")])
    m = mgr(tmp_path, Endorse(seg))
    assert m.ingest(pdf(lines), "x.pdf").entry.kind == "endorsement"
    assert mgr(tmp_path, FakeLLM(fail_seg=True)).library.get("E-01").kind == "endorsement"          # kept across a reload


def test_a_wrapped_continuation_line_is_never_a_clause_start():
    lines = ["Clause 5.2 amended: the maximum We will pay for claims under Clause 2.1", "(Escape of Water) in any one period is INR 300,000.", "The sublimit forms part of the Sum Insured."]
    m = Marks()
    LLMSegmenter.check(SegmentAnswer(clauses=[ClauseMark(heading=x) for x in lines]), lines, m)
    assert set(m.clauses) == {lines[0], lines[2]}


def test_a_payout_clause_without_an_amount_is_not_typed_when_the_model_cut_the_document():
    lines = ["Escape of Water Aggregate Sublimit Endorsement", "Policy number HSP-2026-004417", "Effective date 01 January 2026", "5.2 Water sublimit. The most We will pay is INR 300,000 in aggregate."]
    marks = Marks({lines[0]: "", lines[3]: ""}, set())
    class P:
        number, text = 1, "\n".join(lines)
    d = PolicyDecomposer(PACK.policy_structure).decompose("X", [P()], marks=marks)
    assert [(c.heading, c.clause_type) for c in d.clauses] == [("5.2 Water sublimit", "sublimit")]
    assert "states no amount" in next(u for u in d.unclassified if u.heading == lines[0]).reason


def test_an_amount_in_the_heading_line_counts_for_a_payout_clause():
    lines = ["5.1 Deductible. For each claim under Clause 2.1 You bear the first INR 10,000 of the loss, unless the Schedule or", "an Endorsement states otherwise."]
    class P:
        number, text = 1, "\n".join(lines)
    d = PolicyDecomposer(PACK.policy_structure).decompose("X", [P()], marks=Marks({lines[0]: "5.1"}, set()))
    assert [c.clause_type for c in d.clauses] == ["deductible"] and not d.unclassified
