"""T1.3 - the rendered claim corpus: provenance survives rendering, output is reproducible, layouts vary, photos are labelled truthfully."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pymupdf
import pytest
from PIL import Image, ImageFilter, ImageStat

from evals.ground_truth import load_all_ground_truth, load_scenarios

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data/synthetic"
SC = load_scenarios()
GT = load_all_ground_truth()


def norm(s):
    return " ".join(s.split())


def manifest(claim_id):
    return json.loads((SYN / "claims" / claim_id / "manifest.json").read_text(encoding="utf-8"))


def pages(path):
    return [norm(p.get_text()) for p in pymupdf.open(path)]


def sharpness(img):
    return ImageStat.Stat(img.convert("L").filter(ImageFilter.FIND_EDGES)).var[0]


DOCS = [(s, d) for s in SC.scenarios for d in s.documents]


# ------------------------------------------------------------------ provenance survives rendering
@pytest.mark.parametrize("sc, doc", DOCS, ids=lambda x: getattr(x, "document_id", getattr(x, "id", "")))
def test_every_source_line_is_extractable_and_page_is_marked_synthetic(sc, doc):
    text = pages(SYN / "claims" / sc.claim_id / doc.filename)
    assert len(text) == len(doc.pages)
    for src, got in zip(doc.pages, text):
        for line in src.strip().split("\n"):
            assert norm(line) in got, f"{doc.document_id}: line lost in rendering: {line!r}"
        assert "SYNTHETIC DOCUMENT" in got and f"Page 1 of {len(doc.pages)}" in got


def test_ground_truth_evidence_text_is_found_in_the_rendered_pdfs():
    checked = 0
    for sid, gt in GT.items():
        sc = next(s for s in SC.scenarios if s.id == sid)
        by_id = {d.document_id: d for d in sc.documents}
        for f in gt.facts:
            text = pages(SYN / "claims" / sc.claim_id / by_id[f.source_document_id].filename)[f.page - 1]
            assert norm(f.evidence_text) in text, f.fact_id
            checked += 1
    assert checked >= 30


# ------------------------------------------------------------------ reproducibility and manifests
def test_generation_is_reproducible_byte_for_byte():
    r = subprocess.run([sys.executable, "scripts/generate_synthetic_docs.py", "--check"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("sc", SC.scenarios, ids=lambda s: s.id)
def test_manifest_hashes_match_the_files(sc):
    base, m = SYN / "claims" / sc.claim_id, manifest(sc.claim_id)
    assert m["synthetic"] is True and "CC0" in m["license"]
    assert [d["document_id"] for d in m["documents"]] == [d.document_id for d in sc.documents]
    for d in m["documents"]:
        data = (base / d["filename"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == d["sha256"]
        assert hashlib.sha256("\n".join(pages(base / d["filename"])).encode()).hexdigest() == d["text_sha256"]
    for ph in m["photos"]:
        assert hashlib.sha256((base / ph["file"]).read_bytes()).hexdigest() == ph["sha256"]


def test_fixtures_carry_the_real_file_hash_and_path():
    for sid in GT:
        ev = json.loads((ROOT / f"data/fixtures/{sid}/evidence.json").read_text(encoding="utf-8"))
        for d in ev["documents"]:
            assert hashlib.sha256((ROOT / d["blob_path"]).read_bytes()).hexdigest() == d["sha256"], d["document_id"]


# ------------------------------------------------------------------ layout variance (tables, stamps, fonts, scanned page)
def test_documents_use_different_layouts_fonts_and_stamps():
    fonts, stamps = set(), set()
    for sc, doc in DOCS:
        pdf = pymupdf.open(SYN / "claims" / sc.claim_id / doc.filename)
        fonts |= {f[3] for f in pdf[0].get_fonts()}
        for word in ("RECEIVED", "INSPECTED", "CONFIDENTIAL", "ON FILE"):
            if word in pdf[0].get_text():
                stamps.add(word)
    assert {"Helvetica", "Times", "Courier"} <= {f.split("+")[-1].split("-")[0] for f in fonts}      # sans, serif and monospace all appear
    assert stamps == {"RECEIVED", "INSPECTED", "CONFIDENTIAL", "ON FILE"}


def test_estimate_is_a_ruled_table_with_right_aligned_amounts():
    pdf = pymupdf.open(SYN / "claims/CLM-0001/estimate.pdf")
    assert len(pdf[0].get_drawings()) >= 6                        # header band, row boxes, rules
    words = {w[4]: w[0] for w in pdf[0].get_text("words")}
    assert words["110,000"] > 400 and words["Kitchen"] < 100      # label left, amount far right


def test_scanned_rotated_variant_is_image_only_and_needs_ocr():
    m = manifest("CLM-0002")
    v = m["variants"][0]
    assert (v["kind"], v["needs_ocr"], v["text_extractable"], v["rotation_deg"]) == ("scanned_rotated", True, False, 1.8)
    pdf = pymupdf.open(SYN / "claims/CLM-0002" / v["filename"])
    assert pdf[0].get_text().strip() == "" and len(pdf[0].get_images()) == 1
    src = next(d for d in SC.scenarios[1].documents if d.document_id == v["variant_of"])
    assert v["expected_text"] == [norm(p) for p in src.pages]     # what OCR should recover
    assert v["in_scenario"] is False                              # ground truth is unchanged by the extra file


# ------------------------------------------------------------------ photos
def test_quality_set_roles_are_measurable_not_just_asserted():
    d = SYN / "photos/quality_set"
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    by = {p["role"]: p for p in m["photos"] if p["file"] != "good_closeup.jpg"}
    assert set(by) == {"good", "blurry", "low_res", "glare", "dark", "unrelated"} and "CC0" in m["license"]
    img = {r: Image.open(d / p["file"]).convert("RGB") for r, p in by.items()}
    assert sharpness(img["good"]) > 2 * sharpness(img["blurry"])
    assert img["low_res"].width < 300 and img["good"].width >= 900
    assert ImageStat.Stat(img["dark"].convert("L")).mean[0] < 0.35 * ImageStat.Stat(img["good"].convert("L")).mean[0]
    def bright(im):                                                                # pixels that are (nearly) pure white
        return sum(im.convert("L").histogram()[246:])

    assert bright(img["glare"]) > 5 * max(1, bright(img["good"]))
    r, g, b = ImageStat.Stat(img["unrelated"]).mean
    assert b > r and g > r - 5                                    # sky/grass colours, unlike the brown stain scenes
    assert {p["expected_quality_verdict"] for p in m["photos"]} == {"usable", "poor", "unrelated"}
    for p in m["photos"]:
        assert hashlib.sha256((d / p["file"]).read_bytes()).hexdigest() == p["sha256"]


def test_claim_photos_are_labelled_good_and_marked_synthetic():
    for cid in ("CLM-0001", "CLM-0002", "CLM-0004", "CLM-0005"):
        m = manifest(cid)
        assert m["photos"], cid
        for p in m["photos"]:
            im = Image.open(SYN / "claims" / cid / p["file"])
            assert im.size == (960, 640) and p["role"] == "good" and p["in_scenario"] is False
    assert manifest("CLM-0003")["photos"] == []                   # S-03 has only the FNOL


def _generator():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen_docs", ROOT / "scripts/generate_synthetic_docs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FIXED_ID = b"/ID[<" + b"0" * 32 + b"><" + b"0" * 32 + b">]"


@pytest.mark.parametrize("id_body", [
    rb"<AABB><CCDD>", rb"(abc)(def)", rb"<3A11C3BB>(%\332,\005HZ\332\)*m\005}SK1o)", rb"(a\)b)(c)", rb"(a(b)c)<BB>", rb"<AA>(x\\)",
    rb"( \(\) )<AA>", rb" <AA>  <BB> ", rb"(\()(\))",
])
def test_pdf_id_pinning_handles_every_string_form_a_pdf_may_use(id_body):
    """Regression: a regex that mishandled the escaped parenthesis made generated PDFs intermittently non-reproducible."""
    gen = _generator()
    head, tail = b"%PDF-1.7\ntrailer\n<</Size 5/Root 1 0 R", b">>\nstartxref\n629\n%%EOF\n"
    pinned = gen.pin_pdf_id(head + b"/ID[" + id_body + b"]" + tail)
    assert pinned == head + FIXED_ID + tail                         # only the ID changed; everything around it is byte-identical
    assert gen.pin_pdf_id(head + b"/ID [<11><22>]" + tail) == head + FIXED_ID + tail


def test_pdf_id_pinning_refuses_ambiguous_or_malformed_trailers():
    gen = _generator()
    for bad in (b"%PDF no id here", b"/ID[<AA><BB>]/ID[<CC><DD>]", b"/ID[<AA>]", b"/ID[(unterminated"):
        with pytest.raises((SystemExit, ValueError)):
            gen.pin_pdf_id(bad)


def test_template_samples_exist_for_invoice_and_email():
    m = json.loads((SYN / "templates/manifest.json").read_text(encoding="utf-8"))
    by = {d["doc_type"]: d for d in m["documents"]}
    inv = pages(SYN / "templates" / by["invoice"]["filename"])[0]
    assert "Total: INR 76,700" in inv and "Labour: INR 40,000" in inv
    mail = pages(SYN / "templates" / by["email"]["filename"])[0]
    assert "Subject:" in mail and "kitchen" in mail


# ------------------------------------------------------------------ the check is strict only where the committed bytes come from (Windows)
def _generator():
    sys.path.insert(0, str(ROOT / "scripts"))
    import generate_synthetic_docs as g
    return g


def test_image_derived_files_are_the_platform_dependent_ones():
    g = _generator()
    dependent = ("photos/a.jpg", "claims/CLM-0002/surveyor_report_scanned.pdf", "claims/CLM-0001/manifest.json",
                 "photos/quality_set/manifest.json")
    independent = ("claims/CLM-0001/fnol.pdf", "policies/BASE-HOME-1.pdf", "templates/invoice_sample.pdf")
    assert all(g.platform_dependent(Path(n)) for n in dependent)
    assert not any(g.platform_dependent(Path(n)) for n in independent)


def test_off_windows_a_differing_jpeg_does_not_fail_the_check_but_a_changed_text_document_still_does(tmp_path):
    g = _generator()
    (tmp_path / "photo.jpg").write_bytes(b"as encoded on windows")
    (tmp_path / "manifest.json").write_text('{"sha": "windows"}')
    (tmp_path / "fnol.pdf").write_bytes(b"same text pdf")
    fresh = {tmp_path / "photo.jpg": b"as encoded on linux", tmp_path / "manifest.json": b'{"sha": "linux"}',
             tmp_path / "fnol.pdf": b"same text pdf"}
    stale, skipped = g.stale_files(fresh, strict=False)
    assert stale == [] and {p.name for p in skipped} == {"photo.jpg", "manifest.json"}
    stale, _ = g.stale_files(fresh, strict=True)                          # on Windows (or SYNTHETIC_STRICT=1) every byte is compared
    assert {p.name for p in stale} == {"photo.jpg", "manifest.json"}
    changed = {**fresh, tmp_path / "fnol.pdf": b"someone edited the scenario text"}
    assert [p.name for p in g.stale_files(changed, strict=False)[0]] == ["fnol.pdf"]      # a stale text document is still caught everywhere
    missing = {**fresh, tmp_path / "new.pdf": b"never generated"}
    assert [p.name for p in g.stale_files(missing, strict=False)[0]] == ["new.pdf"]
