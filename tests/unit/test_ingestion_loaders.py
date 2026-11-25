"""T2.1 loaders: every format becomes addressable pages; failures are explicit; the scanned page is flagged, not guessed."""
import io
import json
from email.message import EmailMessage
from pathlib import Path

import pymupdf
import pytest
from docx import Document
from docx.enum.text import WD_BREAK
from PIL import Image

from claimtower.core.llm import Cassette, LLMClient, LLMResponse
from claimtower.ingestion.loaders import (
    CorruptDocument, ProtectedDocument, UnsupportedDocument, claimed_media_type, detect_kind, html_to_text, load, normalise,
    render_page_image,
)
from claimtower.ingestion.ocr import VisionLLMOcr
from evals.ground_truth import load_scenarios

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data/synthetic"
SC = load_scenarios()
CORPUS = json.loads((SYN / "policies/clause_index.json").read_text(encoding="utf-8"))
DOCS = [(s, d) for s in SC.scenarios for d in s.documents]


def pdf_bytes(*page_texts, image=False):
    pdf = pymupdf.open()
    for t in page_texts:
        page = pdf.new_page()
        if t:
            page.insert_text((50, 80), t, fontsize=11)
        if image:
            buf = io.BytesIO()
            Image.new("RGB", (40, 40), "grey").save(buf, "PNG")
            page.insert_image(pymupdf.Rect(300, 300, 340, 340), stream=buf.getvalue())
    return pdf.tobytes()


def png_bytes(size=(30, 20)):
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, "PNG")
    return buf.getvalue()


# ------------------------------------------------------------------ detection from content
def test_format_is_detected_from_the_bytes_not_the_extension():
    assert detect_kind(pdf_bytes("hi"), "notes.txt")[0] == "pdf"
    assert detect_kind(png_bytes(), "scan.pdf")[0] == "image"
    assert detect_kind((SYN / "policies/END-01.docx").read_bytes(), "x.bin")[0] == "docx"
    assert detect_kind(b"From: a@b.example\nTo: c@d.example\nSubject: hi\n\nbody", "mail")[0] == "email"
    assert detect_kind(b"just some notes\nsecond line", "notes.txt")[0] == "text"
    assert claimed_media_type("scan.PDF") == "application/pdf" and claimed_media_type("noext") is None


# ------------------------------------------------------------------ PDFs from the T1.3 corpus
@pytest.mark.parametrize("sc, doc", DOCS, ids=lambda x: getattr(x, "document_id", getattr(x, "id", "")))
def test_scenario_pdfs_give_text_per_page_with_every_source_line(sc, doc):
    loaded = load((SYN / "claims" / sc.claim_id / doc.filename).read_bytes(), doc.filename)
    assert (loaded.kind, loaded.page_count) == ("pdf", len(doc.pages))
    for src, page in zip(doc.pages, loaded.pages):
        assert page.source == "text_layer" and not page.needs_ocr and (page.width, page.height) == (595.0, 842.0)
        for line in src.strip().split("\n"):
            assert normalise(line) in normalise(page.text)


@pytest.mark.parametrize("ext", ["pdf", "docx"])
def test_policy_documents_split_into_the_same_pages_with_clauses_on_the_right_page(ext):
    for d in CORPUS["documents"]:
        loaded = load((SYN / "policies" / f"{d['doc_id']}.{ext}").read_bytes(), f"{d['doc_id']}.{ext}")
        assert loaded.page_count == d["pages"], (d["doc_id"], ext)
        for c in (c for c in CORPUS["clauses"] if c["doc_id"] == d["doc_id"]):
            assert normalise(c["text"]) in normalise(loaded.pages[c["page"] - 1].text), (c["clause_id"], ext)


def test_scanned_copy_is_flagged_for_ocr_not_guessed():
    loaded = load((SYN / "claims/CLM-0002/surveyor_report_scanned.pdf").read_bytes(), "scan.pdf")
    p = loaded.pages[0]
    assert (loaded.page_count, p.text, p.source, p.needs_ocr) == (1, "", "none", True)


def test_pdf_page_rules():
    loaded = load(pdf_bytes("A normal page with plenty of readable text on it", "", "tiny"), "x.pdf")
    normal, blank, tiny = loaded.pages
    assert (normal.source, normal.needs_ocr) == ("text_layer", False)
    assert (blank.source, blank.needs_ocr, blank.text) == ("none", False, "")          # blank, nothing to OCR
    assert (tiny.source, tiny.needs_ocr) == ("text_layer", False)                      # short text, no image: keep it
    scanned = load(pdf_bytes("tiny", image=True), "x.pdf").pages[0]
    assert (scanned.source, scanned.needs_ocr, scanned.text) == ("none", True, "")     # short text + image = no usable text layer


# ------------------------------------------------------------------ images
def test_photo_is_one_page_without_text_and_marked_for_the_classifier():
    loaded = load((SYN / "claims/CLM-0001/photos/pipe_joint_closeup.jpg").read_bytes(), "p.jpg")
    assert (loaded.kind, loaded.media_type, loaded.page_count) == ("image", "image/jpeg", 1)
    p = loaded.pages[0]
    assert (p.width, p.height, p.text, p.needs_ocr) == (960.0, 640.0, "", True)


def test_multi_frame_tiff_has_one_page_per_frame():
    frames = [Image.new("RGB", (20 + i, 20), c) for i, c in enumerate(["red", "green", "blue"])]
    buf = io.BytesIO()
    frames[0].save(buf, "TIFF", save_all=True, append_images=frames[1:])
    loaded = load(buf.getvalue(), "multi.tif")
    assert loaded.page_count == 3 and [p.width for p in loaded.pages] == [20.0, 21.0, 22.0]
    assert Image.open(io.BytesIO(render_page_image(buf.getvalue(), 2)[0])).size == (21, 20)


# ------------------------------------------------------------------ DOCX
def docx_bytes(build):
    d = Document()
    build(d)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_docx_pages_follow_explicit_page_breaks_and_include_tables():
    def build(d):
        d.add_paragraph("first page line")
        t = d.add_table(rows=2, cols=2)
        for r, row in enumerate((("Item", "Amount"), ("Repairs", "INR 5,000"))):
            for c, v in enumerate(row):
                t.cell(r, c).text = v
        d.add_page_break()
        d.add_paragraph("second page line")
        d.add_page_break()                                   # trailing break must not create an empty third page

    loaded = load(docx_bytes(build), "x.docx")
    assert loaded.page_count == 2 and all(not p.needs_ocr and p.source == "text_layer" for p in loaded.pages)
    assert "first page line" in loaded.pages[0].text and "Repairs\tINR 5,000" in loaded.pages[0].text
    assert loaded.pages[1].text == "second page line"


def test_docx_page_break_before_and_merged_cells():
    def build(d):
        d.add_paragraph("one")
        p = d.add_paragraph("two")
        p.paragraph_format.page_break_before = True
        t = d.add_table(rows=1, cols=2)
        t.cell(0, 0).merge(t.cell(0, 1)).text = "merged"

    loaded = load(docx_bytes(build), "x.docx")
    assert loaded.page_count == 2 and loaded.pages[0].text == "one"
    assert loaded.pages[1].text.count("merged") == 1


def test_run_level_page_break_inside_a_paragraph_is_honoured():
    def build(d):
        p = d.add_paragraph("before")
        p.add_run().add_break(WD_BREAK.PAGE)
        d.add_paragraph("after")

    assert load(docx_bytes(build), "x.docx").page_count == 2


# ------------------------------------------------------------------ e-mail and text
def test_plain_email_page_has_headers_and_body():
    m = EmailMessage()
    m["From"], m["To"], m["Subject"], m["Date"] = "a@x.example", "b@y.example", "Water damage", "Tue, 10 Mar 2026 10:00:00 +0000"
    m.set_content("The pipe burst.\nPlease advise.")
    loaded = load(bytes(m), "claim.eml")
    assert (loaded.kind, loaded.page_count, loaded.attachments) == ("email", 1, [])
    text = loaded.pages[0].text
    assert "From: a@x.example" in text and "Subject: Water damage" in text and "The pipe burst.\nPlease advise." in text
    assert loaded.metadata["subject"] == "Water damage"


def test_html_only_email_is_converted_to_text():
    m = EmailMessage()
    m["From"], m["To"], m["Subject"] = "a@x.example", "b@y.example", "Hi"
    html = "<html><style>p{color:red}</style><body><p>Line one</p><p>Line <b>two</b></p><script>x()</script></body></html>"
    m.set_content(html, subtype="html")
    text = load(bytes(m), "m.eml").pages[0].text
    assert "Line one\nLine two" in text and "color" not in text and "x()" not in text


def test_email_attachments_are_extracted_byte_for_byte():
    pdf = pdf_bytes("attached report")
    m = EmailMessage()
    m["From"], m["To"], m["Subject"] = "a@x.example", "b@y.example", "Report attached"
    m.set_content("See attached.")
    m.add_attachment(pdf, maintype="application", subtype="pdf", filename="report.pdf")
    m.add_attachment(png_bytes(), maintype="image", subtype="png", filename="photo.png")
    loaded = load(bytes(m), "m.eml")
    assert [(a.filename, a.media_type) for a in loaded.attachments] == [("report.pdf", "application/pdf"), ("photo.png", "image/png")]
    assert loaded.attachments[0].data == pdf and "See attached." in loaded.pages[0].text


def test_email_with_unknown_charset_still_loads():
    raw = (b"From: a@x.example\nTo: b@y.example\nSubject: x\nMIME-Version: 1.0\nContent-Type: text/plain; charset=made-up-9\n"
           b"Content-Transfer-Encoding: 8bit\n\nplain body caf\xe9\n")
    assert "plain body" in load(raw, "m.eml").pages[0].text


def test_plain_text_pages_split_on_form_feed():
    loaded = load(b"page one\n\x0cpage two\n", "n.txt")
    assert [p.text for p in loaded.pages] == ["page one", "page two"]


def test_legacy_single_byte_encoded_text_is_text_not_binary():
    loaded = load("Claimant café owner - née Smith".encode("cp1252"), "note.txt")
    assert loaded.kind == "text" and "café" in loaded.pages[0].text


def test_normalise_and_html_to_text():
    assert normalise("a  b\n c\t d") == "a b c d"
    assert html_to_text("<div>a</div><div>b&amp;c</div>") == "a\nb&c"


# ------------------------------------------------------------------ failures are explicit and typed
def test_failures_raise_typed_errors():
    with pytest.raises(CorruptDocument, match="empty"):
        load(b"", "x.pdf")
    with pytest.raises(CorruptDocument):
        load(b"%PDF-1.4\nthis is not really a pdf", "x.pdf")
    with pytest.raises(CorruptDocument):
        load(png_bytes()[:40], "x.png")
    with pytest.raises(CorruptDocument):
        load(b"PK\x03\x04 junk junk junk", "x.docx")
    with pytest.raises(UnsupportedDocument, match="binary"):
        load(bytes(range(256)) * 4, "x.dat")
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.txt", "x")
    with pytest.raises(UnsupportedDocument, match="zip"):
        load(buf.getvalue(), "x.zip")
    with pytest.raises(CorruptDocument):
        buf2 = io.BytesIO()
        with zipfile.ZipFile(buf2, "w") as z:
            z.writestr("word/document.xml", "<not-word/>")
        load(buf2.getvalue(), "x.docx")


def test_password_protected_pdf_is_reported_not_opened():
    pdf = pymupdf.open()
    pdf.new_page()
    data = pdf.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    with pytest.raises(ProtectedDocument):
        load(data, "locked.pdf")


# ------------------------------------------------------------------ rendering for OCR
def test_render_page_image_for_pdf_and_errors():
    data = (SYN / "claims/CLM-0002/surveyor_report_scanned.pdf").read_bytes()
    png, mime = render_page_image(data, 1)
    width, height = Image.open(io.BytesIO(png)).size
    assert mime == "image/png" and width == 1240 and abs(height - 1754) <= 1                # A4 at 150 dpi
    with pytest.raises(ValueError):
        render_page_image(data, 2)
    with pytest.raises(UnsupportedDocument):
        render_page_image((SYN / "policies/END-01.docx").read_bytes(), 1)


# ------------------------------------------------------------------ vision-LLM OCR engine
class FakeProvider:
    name = "fake"

    def __init__(self, text):
        self.text, self.requests = text, []

    def complete(self, req):
        self.requests.append(req)
        return LLMResponse(text=self.text, model="vision-test", input_tokens=5, output_tokens=5)


def test_vision_llm_ocr_sends_the_page_image_and_strips_code_fences():
    provider = FakeProvider("```text\nSite visit: 5 May 2026\n```")
    ocr = VisionLLMOcr(LLMClient(provider, text_model="t", vision_model="v"))
    png, mime = render_page_image((SYN / "claims/CLM-0002/surveyor_report_scanned.pdf").read_bytes(), 1)
    assert ocr.recognise(png, mime) == "Site visit: 5 May 2026"
    req = provider.requests[0]
    assert req.model == "v" and len(req.images) == 1 and req.images[0].data == png
    assert "Transcribe ALL visible text" in req.messages[-1]["content"]


def test_vision_ocr_is_deterministic_with_a_cassette(tmp_path):
    png, mime = render_page_image((SYN / "claims/CLM-0002/surveyor_report_scanned.pdf").read_bytes(), 1)
    path = tmp_path / "ocr.json"
    recorder_llm = LLMClient(FakeProvider("recorded text"), text_model="t", vision_model="v", mode="record", cassette=Cassette(path))
    recorder = VisionLLMOcr(recorder_llm)
    assert recorder.recognise(png, mime) == "recorded text"
    offline = FakeProvider("MUST NOT BE CALLED")
    replayer = VisionLLMOcr(LLMClient(offline, text_model="t", vision_model="v", mode="replay", cassette=Cassette(path)))
    assert replayer.recognise(png, mime) == "recorded text" and offline.requests == []
