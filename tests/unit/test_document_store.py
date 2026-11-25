"""T2.1 document store: every file gets a DocumentRecord; originals are byte-identical on retrieval and tamper-evident."""
import hashlib
import io
import json
import os
import stat
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from claimtower.core.schemas import DocumentRecord
from claimtower.ingestion.loaders import normalise
from claimtower.ingestion.store import DocumentStore, IntegrityError, StoreError
from evals.ground_truth import load_scenarios

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data/synthetic"
SC = load_scenarios()
T0 = datetime(2026, 3, 16, 9, 30, tzinfo=timezone.utc)
CORPUS_FILES = sorted([*SYN.glob("claims/*/*.pdf"), *SYN.glob("claims/*/photos/*.jpg"), *SYN.glob("photos/quality_set/*.jpg"),
                       *SYN.glob("policies/*.pdf"), *SYN.glob("policies/*.docx"), *SYN.glob("templates/*.pdf")])


@pytest.fixture
def store(tmp_path):
    return DocumentStore(tmp_path / "store", clock=lambda: T0)


def pdf(text="hello world this is a page of text"):
    d = pymupdf.open()
    d.new_page().insert_text((50, 80), text)
    return d.tobytes()


def png():
    buf = io.BytesIO()
    Image.new("RGB", (30, 20), "white").save(buf, "PNG")
    return buf.getvalue()


# ------------------------------------------------------------------ T2.1 done-when: every file -> record; original byte-identical
def test_corpus_is_large_enough_to_mean_something():
    assert len(CORPUS_FILES) >= 40


@pytest.mark.parametrize("path", CORPUS_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_corpus_file_yields_a_valid_record_and_comes_back_byte_identical(store, path):
    data = path.read_bytes()
    rec = store.ingest("CLM-T", path.name, data, uploaded_by="tester", source_channel="portal")
    assert isinstance(rec, DocumentRecord)
    assert rec.sha256 == hashlib.sha256(data).hexdigest() and rec.page_count >= 1
    assert (rec.uploaded_by, rec.source_channel, rec.uploaded_at) == ("tester", "portal", T0)
    assert store.original(rec.document_id) == data                      # byte-identical on retrieval
    pages = store.pages(rec.document_id)
    assert len(pages) == rec.page_count
    for p in pages:                                                     # text per page OR an OCR flag, never silently nothing
        assert p.text.strip() or p.needs_ocr or p.source == "none"
    assert store.stored(rec.document_id).load_error is None


def test_scenario_documents_match_the_fixtures_for_hash_and_page_count(store):
    for sc in SC.scenarios:
        fixture = json.loads((ROOT / f"data/fixtures/{sc.id}/evidence.json").read_text(encoding="utf-8"))["documents"]
        for d, fx in zip(sc.documents, fixture):
            rec = store.ingest(sc.claim_id, d.filename, (SYN / "claims" / sc.claim_id / d.filename).read_bytes(), document_id=d.document_id)
            assert (rec.document_id, rec.sha256, rec.page_count) == (fx["document_id"], fx["sha256"], fx["page_count"])
            for src, page in zip(d.pages, store.pages(rec.document_id)):
                assert all(normalise(line) in normalise(page.text) for line in src.strip().split("\n"))
    assert len(store.records("CLM-0001")) == 4 and store.records("CLM-9999") == []


def test_record_fields_for_a_text_pdf_and_the_scanned_copy(store):
    text = store.ingest("CLM-T", "fnol.pdf", (SYN / "claims/CLM-0001/fnol.pdf").read_bytes())
    scan = store.ingest("CLM-T", "scan.pdf", (SYN / "claims/CLM-0002/surveyor_report_scanned.pdf").read_bytes())
    assert (text.needs_ocr, text.detected_type, text.blob_path.startswith("blobs/")) == (False, "unknown", True)
    assert scan.needs_ocr is True                                      # flagged, not guessed
    assert store.pages(scan.document_id)[0].source == "none"


# ------------------------------------------------------------------ immutability and tamper evidence
def test_blob_is_content_addressed_and_read_only(store):
    rec = store.ingest("CLM-T", "a.pdf", pdf())
    path = store.root / rec.blob_path
    assert path.name == rec.sha256 and path.parent.name == rec.sha256[:2]
    assert not os.access(path, os.W_OK) or not (path.stat().st_mode & stat.S_IWRITE)


def test_tampering_with_a_stored_original_is_detected_on_read_and_by_the_sweep(store):
    rec = store.ingest("CLM-T", "a.pdf", pdf())
    assert store.verify_all() == []
    path = store.root / rec.blob_path
    os.chmod(path, 0o666)
    path.write_bytes(b"tampered")
    with pytest.raises(IntegrityError, match="no longer match"):
        store.original(rec.document_id)
    assert store.verify_all() == [rec.document_id]


def test_a_missing_blob_is_reported_by_the_sweep(store):
    rec = store.ingest("CLM-T", "a.pdf", pdf())
    path = store.root / rec.blob_path
    os.chmod(path, 0o666)
    path.unlink()
    assert store.verify_all() == [rec.document_id]


def test_reingesting_over_a_corrupt_blob_refuses_to_overwrite_evidence(store):
    data = pdf()
    rec = store.ingest("CLM-T", "a.pdf", data)
    path = store.root / rec.blob_path
    os.chmod(path, 0o666)
    path.write_bytes(b"bit rot")
    with pytest.raises(IntegrityError, match="refusing to overwrite"):
        store.ingest("CLM-T", "again.pdf", data)


def test_identical_bytes_share_one_blob_but_each_upload_is_its_own_document(store):
    data = pdf()
    a = store.ingest("CLM-T", "a.pdf", data, uploaded_by="claimant", source_channel="portal")
    b = store.ingest("CLM-T", "copy.pdf", data, uploaded_by="adjuster", source_channel="email")
    assert a.document_id != b.document_id and a.blob_path == b.blob_path
    assert len([p for p in (store.root / "blobs").rglob("*") if p.is_file()]) == 1                             # one blob file
    assert store.stored(b.document_id).same_content_as == [a.document_id] and store.stored(a.document_id).same_content_as == []
    assert {r.document_id for r in store.find_by_sha256(a.sha256)} == {a.document_id, b.document_id}


# ------------------------------------------------------------------ metadata, ids, persistence
def test_metadata_survives_a_restart(tmp_path):
    s1 = DocumentStore(tmp_path / "s", clock=lambda: T0)
    data = pdf()
    rec = s1.ingest("CLM-T", "r.pdf", data, uploaded_by="plumber", source_channel="email", document_id="DOC-X-1")
    s2 = DocumentStore(tmp_path / "s")
    assert s2.record("DOC-X-1") == rec and s2.original("DOC-X-1") == data
    assert s2.stored("DOC-X-1").uploaded_at == T0 and s2.pages("DOC-X-1")[0].text.startswith("hello")
    assert s2.find_by_sha256(rec.sha256)[0].document_id == "DOC-X-1"


def test_document_ids_are_validated_and_unique(store):
    store.ingest("CLM-T", "a.pdf", pdf(), document_id="DOC-1")
    with pytest.raises(StoreError, match="already exists"):
        store.ingest("CLM-T", "b.pdf", pdf("other text on this page"), document_id="DOC-1")
    for bad in ("../evil", "a/b", "", "x" * 200, ".hidden"):
        with pytest.raises(StoreError, match="invalid"):
            store.ingest("CLM-T", "c.pdf", pdf(), document_id=bad)
    with pytest.raises(StoreError, match="invalid claim id"):
        store.ingest("../CLM", "c.pdf", pdf())
    with pytest.raises(StoreError, match="unknown document"):
        store.record("DOC-NOPE")
    assert not (store.root.parent / "evil.json").exists()


def test_filename_is_reduced_to_a_base_name_and_extension_mismatch_is_flagged(store):
    rec = store.ingest("CLM-T", "C:\\Users\\x\\..\\scan.pdf", png())
    s = store.stored(rec.document_id)
    assert s.filename == "scan.pdf" and s.media_type == "image/png" and s.claimed_media_type == "application/pdf" and s.media_type_mismatch
    assert store.original(rec.document_id) == png()
    assert store.stored(store.ingest("CLM-T", "ok.pdf", pdf()).document_id).media_type_mismatch is False


# ------------------------------------------------------------------ nothing is ever lost: bad files are stored and flagged
@pytest.mark.parametrize("name, data, error", [
    ("empty.pdf", b"", "CorruptDocument"),
    ("broken.pdf", b"%PDF-1.4 not really", "CorruptDocument"),
    ("cut.png", png()[:30], "CorruptDocument"),
    ("blob.bin", bytes(range(256)) * 8, "UnsupportedDocument"),
])
def test_unreadable_files_are_kept_and_flagged_for_a_human(store, name, data, error):
    rec = store.ingest("CLM-T", name, data)
    s = store.stored(rec.document_id)
    assert s.load_error.startswith(error) and rec.page_count == 1 and rec.needs_ocr is False
    assert store.original(rec.document_id) == data                       # the evidence is still there, byte for byte
    assert s.pages[0].text == "" and s.pages[0].source == "none"


def test_password_protected_pdf_is_kept_and_flagged(store):
    d = pymupdf.open()
    d.new_page()
    data = d.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    rec = store.ingest("CLM-T", "locked.pdf", data)
    assert store.stored(rec.document_id).load_error.startswith("ProtectedDocument") and store.original(rec.document_id) == data


# ------------------------------------------------------------------ e-mail attachments become child documents
def test_email_attachments_are_ingested_as_children_with_their_own_hashes(store):
    report = (SYN / "claims/CLM-0001/plumber_report.pdf").read_bytes()
    m = EmailMessage()
    m["From"], m["To"], m["Subject"] = "plumber@x.example", "claims@y.example", "Report"
    m.set_content("Report attached.")
    m.add_attachment(report, maintype="application", subtype="pdf", filename="plumber_report.pdf")
    m.add_attachment(png(), maintype="image", subtype="png", filename="photo.png")
    rec = store.ingest("CLM-T", "mail.eml", bytes(m), uploaded_by="plumber", source_channel="email", document_id="DOC-MAIL")
    kids = store.children("DOC-MAIL")
    assert [k.sha256 for k in kids] == [hashlib.sha256(report).hexdigest(), hashlib.sha256(png()).hexdigest()]
    for k in kids:
        s = store.stored(k.document_id)
        assert s.parent_document_id == "DOC-MAIL" and (s.uploaded_by, s.source_channel, s.uploaded_at) == ("plumber", "email", T0)
    assert store.original(kids[0].document_id) == report
    assert store.stored("DOC-MAIL").email_headers["subject"] == "Report" and "Report attached." in store.pages("DOC-MAIL")[0].text
    assert rec.page_count == 1 and len(store.records("CLM-T")) == 3


def test_email_nesting_is_bounded(store):
    inner = b"From: a@x.example\nTo: b@y.example\nSubject: leaf\n\nbody"
    data = inner
    for _ in range(6):
        m = EmailMessage()
        m["From"], m["To"], m["Subject"] = "a@x.example", "b@y.example", "wrapper"
        m.set_content("see attachment")
        m.add_attachment(data, maintype="message", subtype="rfc822", filename="inner.eml")
        data = bytes(m)
    store.ingest("CLM-T", "outer.eml", data)
    assert len(store.records("CLM-T")) <= 5                              # MAX_EMAIL_DEPTH keeps a hostile chain from running away


# ------------------------------------------------------------------ OCR path
class FixedOcr:
    name = "fixed-test-engine"

    def __init__(self, text="Site visit: 5 May 2026"):
        self.text, self.calls = text, 0

    def recognise(self, image, mime):
        self.calls += 1
        assert mime == "image/png" and Image.open(io.BytesIO(image)).width == 1240
        return self.text


def test_ocr_fills_in_text_without_touching_the_original(tmp_path):
    scan = (SYN / "claims/CLM-0002/surveyor_report_scanned.pdf").read_bytes()
    st = DocumentStore(tmp_path / "s", clock=lambda: T0)
    rec = st.ingest("CLM-0002", "scan.pdf", scan, document_id="DOC-SCAN")
    assert rec.needs_ocr is True
    engine = FixedOcr()
    done = st.run_ocr("DOC-SCAN", engine)
    page = st.pages("DOC-SCAN")[0]
    assert done.needs_ocr is False and (page.text, page.source, page.needs_ocr) == ("Site visit: 5 May 2026", "ocr", False)
    assert st.stored("DOC-SCAN").ocr_engine == "fixed-test-engine" and engine.calls == 1
    assert st.original("DOC-SCAN") == scan and done.sha256 == rec.sha256                   # the evidence itself is unchanged
    assert DocumentStore(tmp_path / "s").pages("DOC-SCAN")[0].source == "ocr"              # and the result persists


def test_ingest_can_run_ocr_in_one_step_and_text_pages_are_not_sent_to_the_engine(store):
    engine = FixedOcr("scanned words")
    store.ingest("CLM-T", "text.pdf", (SYN / "claims/CLM-0001/fnol.pdf").read_bytes(), ocr=engine)
    assert engine.calls == 0
    rec = store.ingest("CLM-T", "scan.pdf", (SYN / "claims/CLM-0002/surveyor_report_scanned.pdf").read_bytes(), ocr=engine)
    assert engine.calls == 1 and rec.needs_ocr is False


def test_a_failing_ocr_page_stays_flagged_and_does_not_lose_the_others(store):
    frames = [Image.new("RGB", (20, 20), c) for c in ("red", "green")]
    buf = io.BytesIO()
    frames[0].save(buf, "TIFF", save_all=True, append_images=frames[1:])
    rec = store.ingest("CLM-T", "two.tif", buf.getvalue(), document_id="DOC-TIF")
    assert rec.page_count == 2

    class Flaky:
        name = "flaky"

        def __init__(self):
            self.n = 0

        def recognise(self, image, mime):
            self.n += 1
            if self.n == 1:
                raise TimeoutError("model unavailable")
            return "page two text"

    done = store.run_ocr("DOC-TIF", Flaky())
    p1, p2 = store.pages("DOC-TIF")
    assert (p1.needs_ocr, p1.text) == (True, "") and (p2.needs_ocr, p2.text, p2.source) == (False, "page two text", "ocr")
    assert done.needs_ocr is True and "page 1: TimeoutError" in store.stored("DOC-TIF").ocr_errors[0]
    assert store.run_ocr("DOC-TIF", FixedOcrAny()).needs_ocr is False                 # a retry can finish the job


class FixedOcrAny:
    name = "any"

    def recognise(self, image, mime):
        return "now readable"
