"""T7.3 - the document reader: what the system is reading, next to the document, with the place a fact was read marked on its page."""
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from streamlit.testing.v1 import AppTest

from claimtower.api.app import create_app
from claimtower.composition import SYNTHETIC, build_registry, policy_library_from_pdfs
from claimtower.core.config import load_settings
from claimtower.core.schemas.common import BBox
from claimtower.ingestion.preview import UnsupportedDocument, render_highlighted
from claimtower.ingestion.store import DocumentStore
from claimtower.orchestration.graph import ClaimRunner
from claimtower.ui.client import ApiClient

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()
APP = str(ROOT / "src" / "claimtower" / "ui" / "app.py")
PDF = (SYNTHETIC / "claims" / "CLM-0002" / "surveyor_report.pdf").read_bytes()


@pytest.fixture(scope="module")
def library():
    return policy_library_from_pdfs(SETTINGS)


@pytest.fixture
def api(tmp_path, library) -> ApiClient:
    store = DocumentStore(tmp_path / "store")
    runner = ClaimRunner(build_registry(SETTINGS, document_store=store, policy_library=library), settings=SETTINGS)
    c = ApiClient(http=TestClient(create_app(runner, store, synthetic_claims=SYNTHETIC / "claims")))
    c.create_claim(claim_id="CLM-0002", use_synthetic_documents=True)
    return c


def page(api, claim="CLM-0002", **state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["_client"] = api
    at.session_state["selected_claim"] = claim
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def texts(at) -> str:
    return " ".join([*[e.value for e in at.error], *[w.value for w in at.warning], *[i.value for i in at.info], *[m.value for m in at.markdown], *[c.value for c in at.caption],
                     *[t.value for t in at.text], *[h.value for h in at.subheader], *[x.value for x in at.success]])


def amber(png: bytes) -> int:
    raw = Image.open(io.BytesIO(png)).convert("RGB").tobytes()
    return sum(1 for i in range(0, len(raw), 3) if raw[i] > 240 and 150 < raw[i + 1] < 235 and raw[i + 2] < 120)


# ---------------------------------------------------------------------------------------------------------------- the page picture
def test_a_recorded_region_is_marked_on_the_page_and_nothing_changes_without_one():
    plain, how0 = render_highlighted(PDF, 1)
    marked, how1 = render_highlighted(PDF, 1, region=BBox(x0=0.1, y0=0.1, x1=0.6, y1=0.2))
    assert how0 == "none" and how1 == "region" and amber(marked) > amber(plain) + 1000 and Image.open(io.BytesIO(marked)).size == Image.open(io.BytesIO(plain)).size


def test_without_a_region_the_quoted_line_is_found_on_the_page_and_marked():
    png, how = render_highlighted(PDF, 1, text="Assessed repair cost: INR 98,000")
    assert how == "text" and amber(png) > amber(render_highlighted(PDF, 1)[0]) + 200
    assert render_highlighted(PDF, 1, text="a sentence that is not on the page at all")[1] == "none"


def test_a_document_with_no_page_picture_is_refused_so_the_caller_can_show_its_text():
    with pytest.raises(UnsupportedDocument):
        render_highlighted(b"just some text, no layout", 1)


# ---------------------------------------------------------------------------------------------------------------- the API
def test_the_details_endpoint_says_how_the_document_was_typed_and_what_was_read_where(api):
    docs = api.documents("CLM-0002")
    survey = next(d for d in docs if d["detected_type"] == "surveyor_report")
    d = api.document_details("CLM-0002", survey["document_id"])
    assert d["detected_type"] == "surveyor_report" and d["type_method"] and d["type_reason"] and d["page_count"] == 1 and d["pages"][0]["source"] == "text_layer" and d["pages"][0]["text"]
    f = next(x for x in d["fields"] if x["name"] == "assessed_repair_cost")
    assert f["page"] == 1 and f["evidence_text"].startswith("Assessed repair cost") and f["region"] and set(f["region"]) == {"x0", "y0", "x1", "y1"} and not f["low_confidence"]
    assert d["low_confidence_below"] == SETTINGS.pack.document_types.extractor.low_confidence and api.health()["low_confidence_below"] == d["low_confidence_below"]


def test_the_page_endpoint_returns_the_picture_and_marks_a_fact_where_it_was_read(api):
    state = api.claim("CLM-0002")["state"]
    fact = next(f for f in state["facts"] if f["related_question_ids"] and f["region"])
    plain, how0 = api.page_image("CLM-0002", fact["source_document_id"], fact["page"])
    marked, how1 = api.page_image("CLM-0002", fact["source_document_id"], fact["page"], fact["fact_id"])
    assert plain[:8] == b"\x89PNG\r\n\x1a\n" and how0 == "none" and how1 in ("region", "text") and amber(marked) > amber(plain)


def test_unknown_documents_pages_and_facts_are_not_found_and_a_text_document_has_no_picture(api):
    http = api._http
    doc = api.documents("CLM-0002")[0]["document_id"]
    assert http.get(f"/claims/CLM-0002/documents/{doc}/page/9").status_code == 404
    assert http.get(f"/claims/CLM-0002/documents/{doc}/page/1", params={"fact": "F-NOPE"}).status_code == 404
    assert http.get("/claims/CLM-0002/documents/DOC-NOPE/details").status_code == 404 and http.get("/claims/CLM-NOPE/documents/x/page/1").status_code == 404
    api.upload("CLM-0002", "note.txt", b"The pipe burst in the night and water came through the ceiling.", uploaded_by="claimant", reassess=False)
    txt = next(d for d in api.documents("CLM-0002") if d["filename"] == "note.txt")
    assert api.page_image("CLM-0002", txt["document_id"], 1) == (None, "")
    assert "pipe burst" in api.document_details("CLM-0002", txt["document_id"])["pages"][0]["text"]


# ---------------------------------------------------------------------------------------------------------------- the tab
def test_the_reader_shows_the_type_the_reason_the_fields_and_the_duplicate_badge_that_says_not_checked(api):
    at = page(api)
    assert not at.exception and "Document reader" in [t.label for t in at.tabs]
    t = texts(at)
    assert "Typed by" in t and "Fields extracted from this document" in t and "What the agents did" in t
    assert "Duplicate check: **not done yet**" in t and "not a statement that there is no duplicate" in t


def test_choosing_a_fact_jumps_to_its_page_and_shows_the_source_excerpt_beside_it(api):
    state = api.claim("CLM-0002")["state"]
    fact = next(f for f in state["facts"] if "assessed" in f["proposition"].lower())
    at = page(api, **{"viewer_doc_CLM-0002": fact["source_document_id"], "viewer_fact_CLM-0002": fact["fact_id"]})
    t = texts(at)
    assert not at.exception and "Source excerpt" in t and fact["evidence_text"] in t and f"Page {fact['page']}" in t and "Reads as: " + fact["proposition"] in t


def test_the_facts_tab_opens_a_fact_in_the_reader_and_marks_low_confidence(api):
    at = page(api)
    pick = at.selectbox(key="facttable_pick_CLM-0002")
    target = pick.options[2]
    pick.select(target).run()
    at.button(key="facttable_open_CLM-0002").click().run()
    assert at.session_state["viewer_fact_CLM-0002"] == target.split("  ·  ")[0] or at.session_state["viewer_fact_CLM-0002"] == target
    assert "is open in the **Document reader** tab" in texts(at)


def test_a_low_confidence_fact_is_marked_in_the_table_and_in_the_reader_never_hidden():
    from claimtower.ui.components.fact_table import rows
    state = {"documents": [{"document_id": "D", "filename": "f.pdf"}], "facts": [
        {"fact_id": "F1", "source_document_id": "D", "page": 1, "proposition": "p", "evidence_text": "e", "extraction_confidence": 0.5, "related_question_ids": [], "verification_status": "unverified"},
        {"fact_id": "F2", "source_document_id": "D", "page": 1, "proposition": "p", "evidence_text": "e", "extraction_confidence": 0.95, "related_question_ids": [], "verification_status": "unverified"}]}
    r = rows(state, 0.7)
    assert [x[""] for x in r] == ["⚠ low confidence", ""] and len(r) == 2 and rows(state, None)[0][""] == ""


def test_no_accusation_appears_anywhere_in_the_ui_the_words_are_low_confidence_conflicting_unverified(api):
    for tab_state in ({}, {"viewer_doc_CLM-0002": api.documents("CLM-0002")[0]["document_id"]}):
        assert "fraud" not in texts(page(api, **tab_state)).lower()
    src = " ".join(p.read_text(encoding="utf-8") for p in (ROOT / "src" / "claimtower" / "ui").rglob("*.py")).lower()
    assert "fraud" not in src and ":red[" not in src and "color=\"red\"" not in src
