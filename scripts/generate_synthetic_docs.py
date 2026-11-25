"""Render the synthetic claim corpus from data/scenarios/scenarios.yaml (T1.3).

    python scripts/generate_synthetic_docs.py           # (re)write data/synthetic/claims/, photos/ and templates/
    python scripts/generate_synthetic_docs.py --check   # fail if anything on disk differs from a fresh build

Outputs (ALL SYNTHETIC, deterministic - same inputs give byte-identical files):
  data/synthetic/claims/CLM-xxxx/<filename>.pdf        every document of the scenario, rendered with a per-type layout template
  data/synthetic/claims/CLM-xxxx/photos/*.jpg          generated illustrations (good / close-up) for some claims
  data/synthetic/claims/CLM-xxxx/manifest.json         sha256 of every file, extracted-text hash, photo labels
  data/synthetic/claims/CLM-0002/surveyor_report_scanned.pdf   image-only, rotated, noisy copy of a document (needs OCR)
  data/synthetic/photos/quality_set/*.jpg + manifest   labelled good / blurry / low-res / glare / dark / unrelated photos (T2.5)
  data/synthetic/templates/*.pdf                       invoice and e-mail templates (no scenario uses them yet)

The text on every page is the scenario page text, line for line: only decoration (letterheads, stamps, boxes, footers) is added, so
ground-truth `evidence_text` stays extractable. Photos are drawn by this script (no third-party imagery): CC0-1.0.
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import re
import sys
from datetime import date
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evals.ground_truth import ScenarioDocument, load_scenarios  # noqa: E402

OUT = ROOT / "data" / "synthetic"
W, H, M = 595, 842, 56
FIXED_DATE = "D:20260101000000Z"
LICENSE = "CC0-1.0 - drawn by scripts/generate_synthetic_docs.py, no third-party imagery"
WATERMARK = "SYNTHETIC ILLUSTRATION - CC0"
BANNER = "SYNTHETIC DOCUMENT - for software testing only. Not a real document. All amounts are SAMPLE."
RED = (0.7, 0.05, 0.05)
_ROW = re.compile(r"^(?P<label>.+?:)\s+(?P<value>INR [\d,]+)$")

# ---------------------------------------------------------------------------------------------------- style per document type
STYLES = {
    "fnol":            dict(template="form",   accent=(0.16, 0.32, 0.55), body="helv", bold="hebo", stamp="RECEIVED"),
    "policy_schedule": dict(template="form",   accent=(0.12, 0.45, 0.30), body="cour", bold="cobo", stamp="ON FILE"),
    "plumber_report":  dict(template="letter", accent=(0.05, 0.40, 0.45), body="tiro", bold="tibo", stamp="INSPECTED",
                            company="SYNTHETIC Plumbing Co. (fictional)", rules=False),
    "surveyor_report": dict(template="letter", accent=(0.45, 0.30, 0.10), body="helv", bold="hebo", stamp="CONFIDENTIAL",
                            company="SYNTHETIC Loss Surveyors Ltd (fictional)", rules=True),
    "estimate":        dict(template="table",  accent=(0.10, 0.20, 0.45), body="helv", bold="hebo", stamp="RECEIVED",
                            head=("Description", "Amount")),
    "invoice":         dict(template="table",  accent=(0.45, 0.10, 0.10), body="cour", bold="cobo", stamp="PAID",
                            head=("Item", "Amount")),
    "email":           dict(template="email",  accent=(0.30, 0.30, 0.30), body="helv", bold="hebo", stamp=None),
}


DEFAULT_STYLE = dict(template="letter", accent=(0.25, 0.25, 0.25), body="helv", bold="hebo", stamp=None,
                     company="SYNTHETIC Sender (fictional)", rules=False)


def _skip_ws(b: bytes, i: int) -> int:
    while i < len(b) and b[i] in b" \t\r\n\x0c\x00":
        i += 1
    return i


def _skip_pdf_string(b: bytes, i: int) -> int:
    """Index just after the PDF string starting at b[i]: hex <...> or literal (...) (backslash escapes the next byte; parentheses nest)."""
    if b[i:i + 1] == b"<":
        return b.index(b">", i) + 1
    if b[i:i + 1] != b"(":
        raise ValueError(f"not a PDF string at offset {i}")
    depth, j = 0, i
    while j < len(b):
        c = b[j:j + 1]
        if c == b"\\":
            j += 2
            continue
        depth += 1 if c == b"(" else -1 if c == b")" else 0
        j += 1
        if depth == 0:
            return j
    raise ValueError("unterminated PDF string")


def pin_pdf_id(data: bytes) -> bytes:
    """MuPDF writes a random trailer /ID (hex or literal strings, in any mix); replace it with a fixed one so output is reproducible.
    A small scanner follows the PDF string rules instead of a regex: escapes and nested parentheses are handled by construction.
    The ID sits in the trailer/xref dictionary after every object offset, so a different length shifts nothing."""
    start = data.find(b"/ID")
    if start < 0 or data.find(b"/ID", start + 1) >= 0:
        raise SystemExit("expected exactly one /ID in the PDF trailer")
    i = _skip_ws(data, start + 3)
    if data[i:i + 1] != b"[":
        raise SystemExit("malformed /ID in the PDF trailer")
    i = _skip_ws(data, _skip_pdf_string(data, _skip_ws(data, i + 1)))
    i = _skip_ws(data, _skip_pdf_string(data, i))
    if data[i:i + 1] != b"]":
        raise SystemExit("malformed /ID in the PDF trailer")
    return data[:start] + b"/ID[<" + b"0" * 32 + b"><" + b"0" * 32 + b">]" + data[i + 1:]


def finish_pdf(pdf: pymupdf.Document, title: str) -> bytes:
    pdf.set_metadata({"title": f"{title} (SYNTHETIC)", "subject": "SYNTHETIC test document", "author": "OmniClaim synthetic data",
                      "creator": "generate_synthetic_docs.py", "producer": "PyMuPDF", "creationDate": FIXED_DATE, "modDate": FIXED_DATE})
    return pin_pdf_id(pdf.tobytes(garbage=3, deflate=True))


def norm(s: str) -> str:
    return " ".join(s.split())


def text_len(text: str, font: str, size: float) -> float:
    return pymupdf.get_text_length(text, fontname=font, fontsize=size)


def wrap(text: str, font: str, size: float, width: float) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and text_len(trial, font, size) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return [*lines, cur] if cur else lines


def stamp(page: pymupdf.Page, text: str, at: tuple[float, float], angle: float = 12) -> None:
    """A rotated, red 'rubber stamp' box. Its text is extractable but never overlaps body text."""
    x, y = at
    mat = pymupdf.Matrix(angle)
    w = text_len(text, "hebo", 12) + 16
    page.draw_rect(pymupdf.Rect(x, y - 16, x + w, y + 6), color=RED, width=1.6, morph=(pymupdf.Point(x, y), mat))
    page.insert_text(pymupdf.Point(x + 8, y), text, fontname="hebo", fontsize=12, color=RED, morph=(pymupdf.Point(x, y), mat))


def chrome(page: pymupdf.Page, ctx: dict, i: int, n: int) -> None:
    page.insert_text((M, 28), BANNER, fontname="helv", fontsize=7, color=RED)
    page.insert_text((M, H - 34), f"{ctx['claim_id']} | {ctx['document_id']} | SYNTHETIC", fontname="helv", fontsize=8)
    page.insert_text((W - M - 62, H - 34), f"Page {i} of {n}", fontname="helv", fontsize=8)


def stamp_text(style: dict, ctx: dict) -> str | None:
    if not style["stamp"]:
        return None
    d = date.fromisoformat(ctx["received_on"])
    return f"{style['stamp']} {d.day:02d} {d.strftime('%b').upper()} {d.year}" if style["stamp"] in ("RECEIVED", "PAID") else style["stamp"]


def guard(y: float, doc: str) -> None:
    if y > H - 80:
        raise SystemExit(f"{doc}: content overflows the page - shorten the page text or split it over pages")


# ---------------------------------------------------------------------------------------------------- templates
def tpl_form(page, lines, style, ctx):
    """Boxed form: coloured title band, one bordered row per source line."""
    page.draw_rect(pymupdf.Rect(M, 56, W - M, 94), color=None, fill=style["accent"])
    page.insert_text((M + 12, 80), lines[0], fontname="hebo", fontsize=13, color=(1, 1, 1))
    y, width = 112.0, W - 2 * M - 20
    for n, line in enumerate(lines[1:]):
        rows = wrap(line, style["body"], 10, width)
        h = len(rows) * 14 + 10
        page.draw_rect(pymupdf.Rect(M, y, W - M, y + h), color=(0.6, 0.6, 0.6), fill=(0.96, 0.96, 0.98) if n % 2 else (1, 1, 1), width=0.6)
        for k, r in enumerate(rows):
            page.insert_text((M + 10, y + 16 + k * 14), r, fontname=style["body"], fontsize=10)
        y += h + 3
        guard(y, ctx["document_id"])
    return y


def tpl_letter(page, lines, style, ctx):
    """Letterhead + paragraphs; serif or sans, optional rules between paragraphs."""
    page.insert_text((W - M - text_len(style["company"], style["bold"], 10), 56), style["company"], fontname=style["bold"], fontsize=10,
                     color=style["accent"])
    page.insert_text((W - M - 120, 70), f"Ref: {ctx['claim_id']}/{ctx['doc_type'][:3].upper()}", fontname=style["body"], fontsize=8)
    page.draw_line((M, 82), (W - M, 82), color=style["accent"], width=1.4)
    page.insert_text((M, 108), lines[0], fontname=style["bold"], fontsize=14)
    y = 134.0
    for line in lines[1:]:
        for r in wrap(line, style["body"], 10.5, W - 2 * M - 14):
            page.insert_text((M + 14, y), r, fontname=style["body"], fontsize=10.5)
            y += 15
        if style["rules"]:
            page.draw_line((M + 14, y - 6), (W - M, y - 6), color=(0.75, 0.75, 0.75), width=0.5)
        y += 7
        guard(y, ctx["document_id"])
    if style["rules"]:
        page.draw_line((M - 8, 82), (M - 8, y), color=style["accent"], width=2.5)     # margin bar, clear of the text
    page.insert_text((M + 14, y + 36), "Signed: ____________________", fontname=style["body"], fontsize=10)
    return y


def tpl_table(page, lines, style, ctx):
    """Grid table: 'label: INR amount' lines become rows (label left, amount right-aligned); other lines become notes below."""
    page.insert_text((M, 70), lines[0], fontname="hebo", fontsize=14, color=style["accent"])
    page.draw_line((M, 80), (W - M, 80), color=style["accent"], width=1.6)
    y = 100.0
    page.draw_rect(pymupdf.Rect(M, y, W - M, y + 20), color=style["accent"], fill=style["accent"])
    page.insert_text((M + 8, y + 14), style["head"][0], fontname="hebo", fontsize=9.5, color=(1, 1, 1))
    page.insert_text((W - M - 8 - text_len(style["head"][1], "hebo", 9.5), y + 14), style["head"][1], fontname="hebo", fontsize=9.5, color=(1, 1, 1))
    y += 20
    notes = []
    for line in lines[1:]:
        m = _ROW.match(line)
        if not m:
            notes.append(line)
            continue
        total = m["label"].lower().startswith("total")
        font = style["bold"] if total else style["body"]
        page.draw_rect(pymupdf.Rect(M, y, W - M, y + 22), color=(0.35, 0.35, 0.35), width=1.4 if total else 0.5, fill=(0.93, 0.95, 1) if total else None)
        page.insert_text((M + 8, y + 15), m["label"], fontname=font, fontsize=10)
        value = m["value"]
        page.insert_text((W - M - 8 - text_len(value, font, 10), y + 15), " " + value, fontname=font, fontsize=10)   # leading space keeps the extracted line intact
        y += 22
        guard(y, ctx["document_id"])
    y += 16
    for note in notes:
        for r in wrap(note, style["body"], 10, W - 2 * M):
            page.insert_text((M, y), r, fontname=style["body"], fontsize=10)
            y += 14
    return y


def tpl_email(page, lines, style, ctx):
    page.draw_rect(pymupdf.Rect(M, 56, W - M, 130), color=(0.7, 0.7, 0.7), fill=(0.97, 0.97, 0.97), width=0.6)
    for k, (label, val) in enumerate([("From", "claimant@example.invalid"), ("To", "claims@insurer.example.invalid"),
                                      ("Date", ctx["received_on"]), ("Subject", lines[0])]):
        page.insert_text((M + 10, 74 + k * 15), f"{label}: {val}", fontname=style["body"], fontsize=9.5)
    y = 160.0
    for line in lines[1:]:
        for r in wrap(line, style["body"], 10.5, W - 2 * M):
            page.insert_text((M, y), r, fontname=style["body"], fontsize=10.5)
            y += 15
        y += 8
        guard(y, ctx["document_id"])
    return y


TEMPLATES = {"form": tpl_form, "letter": tpl_letter, "table": tpl_table, "email": tpl_email}


def render_document(doc: ScenarioDocument | dict, claim_id: str) -> bytes:
    d = doc.model_dump() if hasattr(doc, "model_dump") else doc
    style = STYLES.get(d["doc_type"], DEFAULT_STYLE)
    pdf = pymupdf.open()
    n = len(d["pages"])
    for i, text in enumerate(d["pages"], start=1):
        page = pdf.new_page(width=W, height=H)
        ctx = {"claim_id": claim_id, "document_id": d["document_id"], "doc_type": d["doc_type"], "received_on": d["received_on"]}
        chrome(page, ctx, i, n)
        y = TEMPLATES[style["template"]](page, text.rstrip("\n").split("\n"), style, ctx)
        st = stamp_text(style, ctx)
        if st:
            stamp(page, st, (W - 200, min(y + 70, H - 120)))
    return finish_pdf(pdf, d["filename"])


# ---------------------------------------------------------------------------------------------------- scanned / rotated variant
def render_scanned_variant(pdf_bytes: bytes, angle: float = 1.8, seed: int = 7) -> bytes:
    """Image-only PDF of the first page: rasterised, slightly rotated, noisy, soft - like a phone scan. Needs OCR."""
    src = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pix = src[0].get_pixmap(dpi=110)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=235)
    rng = random.Random(seed)
    noise = Image.effect_noise(img.size, 18)
    img = Image.blend(img, noise, 0.10).filter(ImageFilter.GaussianBlur(0.7))
    img = img.point(lambda v: min(255, int(v * 0.92 + rng.randint(0, 3))))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=72)
    out = pymupdf.open()
    page = out.new_page(width=W, height=H)
    page.insert_image(pymupdf.Rect(0, 0, W, H), stream=buf.getvalue())
    return finish_pdf(out, "scanned copy")


# ---------------------------------------------------------------------------------------------------- photos (generated illustrations)
def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                       # very old Pillow
        return ImageFont.load_default()


def _mark(img: Image.Image) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.rectangle((0, img.height - 20, img.width, img.height), fill=(0, 0, 0))
    d.text((6, img.height - 18), WATERMARK, fill=(255, 255, 255), font=_font(12))
    return img


def wall_scene(seed: int, size=(960, 640), floor: bool = False) -> Image.Image:
    """Painted wall (or floor boards) with a brown water stain, tide rings and dark mould-like specks."""
    rng = random.Random(seed)
    w, h = size
    base = (205, 200, 188) if not floor else (176, 140, 96)
    img = Image.new("RGB", size, base)
    img = Image.blend(img, Image.merge("RGB", [Image.effect_noise(size, 14)] * 3), 0.10)
    d = ImageDraw.Draw(img, "RGBA")
    if floor:
        for x in range(0, w, 80):
            d.line((x, 0, x, h), fill=(90, 62, 36, 120), width=2)
    else:
        d.rectangle((0, int(h * 0.88), w, h), fill=(235, 235, 232))
        d.line((0, int(h * 0.88), w, int(h * 0.88)), fill=(120, 120, 118), width=3)
    cx, cy = int(w * rng.uniform(0.35, 0.6)), int(h * rng.uniform(0.35, 0.55))
    for k in range(9, 0, -1):
        rx, ry = 40 + k * 34, 30 + k * 24
        jitter = [(cx + rng.randint(-14, 14), cy + rng.randint(-10, 10))]
        c = (150 - k * 6, 112 - k * 5, 62 - k * 2, 46)
        d.ellipse((jitter[0][0] - rx, jitter[0][1] - ry, jitter[0][0] + rx, jitter[0][1] + ry), fill=c, outline=(110, 80, 40, 120), width=2)
    for _ in range(55):
        x, y = cx + rng.randint(-190, 190), cy + rng.randint(-120, 120)
        r = rng.randint(2, 6)
        d.ellipse((x - r, y - r, x + r, y + r), fill=(40, 40, 36, 150))
    for _ in range(6):                                   # drip streaks
        x = cx + rng.randint(-120, 120)
        d.line((x, cy, x + rng.randint(-8, 8), min(h - 40, cy + rng.randint(150, 260))), fill=(120, 88, 48, 110), width=3)
    return _mark(img)


def pipe_closeup(seed: int, size=(960, 640)) -> Image.Image:
    """Close-up of a pipe joint with rust and green corrosion and a drip."""
    rng = random.Random(seed)
    w, h = size
    img = Image.new("RGB", size, (60, 64, 68))
    img = Image.blend(img, Image.merge("RGB", [Image.effect_noise(size, 20)] * 3), 0.12)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle((w // 2 - 70, 0, w // 2 + 70, h), fill=(150, 154, 158))
    d.rectangle((w // 2 - 100, h // 2 - 60, w // 2 + 100, h // 2 + 60), fill=(122, 126, 130), outline=(40, 40, 40), width=4)
    for _ in range(420):
        x, y = w // 2 + rng.randint(-110, 110), h // 2 + rng.randint(-70, 70)
        r = rng.randint(2, 9)
        colour = rng.choice([(176, 92, 40, 170), (96, 148, 110, 150), (110, 70, 40, 160)])
        d.ellipse((x - r, y - r, x + r, y + r), fill=colour)
    d.line((w // 2 + 20, h // 2 + 62, w // 2 + 24, h - 50), fill=(70, 110, 150, 180), width=5)
    d.ellipse((w // 2 + 14, h - 56, w // 2 + 34, h - 30), fill=(70, 110, 150, 200))
    return _mark(img)


def garden_scene(seed: int, size=(960, 640)) -> Image.Image:
    """Unrelated image (outdoor garden) - must be routed to a human, never read as damage."""
    rng = random.Random(seed)
    w, h = size
    img = Image.new("RGB", size, (140, 190, 235))
    d = ImageDraw.Draw(img, "RGBA")
    d.ellipse((w - 190, 40, w - 90, 140), fill=(255, 224, 120))
    d.polygon([(0, int(h * 0.62)), (w * 0.3, int(h * 0.45)), (w * 0.6, int(h * 0.6)), (w, int(h * 0.48)), (w, h), (0, h)], fill=(90, 150, 80))
    d.rectangle((0, int(h * 0.75), w, h), fill=(70, 130, 64))
    for tx in (int(w * 0.25), int(w * 0.7)):
        d.rectangle((tx - 14, int(h * 0.5), tx + 14, int(h * 0.78)), fill=(96, 70, 44))
        for _ in range(7):
            x, y = tx + rng.randint(-60, 60), int(h * 0.45) + rng.randint(-50, 40)
            d.ellipse((x - 45, y - 45, x + 45, y + 45), fill=(46, 120, 52, 220))
    return _mark(img)


def jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def quality_set() -> dict[str, tuple[Image.Image, dict]]:
    good = wall_scene(11)
    glare = good.copy()
    gd = ImageDraw.Draw(glare, "RGBA")
    for k in range(10, 0, -1):                 # stacked translucent discs: the centre clips to pure white (blown-out highlight)
        gd.ellipse((560 - k * 36, 120 - k * 26, 560 + k * 36, 120 + k * 26), fill=(255, 255, 255, 70))
    dark = Image.eval(good, lambda v: int(v * 0.18))
    low = _mark(good.resize((160, 107)).convert("RGB"))
    low = low.resize((160, 107))
    return {
        "good_wide.jpg": (good, dict(role="good", shows="water stain on a wall, wide view", expected_quality_verdict="usable")),
        "good_closeup.jpg": (pipe_closeup(12), dict(role="good", shows="corroded pipe joint, close-up", expected_quality_verdict="usable")),
        "blurry.jpg": (good.filter(ImageFilter.GaussianBlur(10)), dict(role="blurry", shows="water stain, out of focus", expected_quality_verdict="poor")),
        "low_res.jpg": (low, dict(role="low_res", shows="water stain, 160x107 pixels", expected_quality_verdict="poor")),
        "glare.jpg": (glare, dict(role="glare", shows="water stain partly washed out by glare", expected_quality_verdict="poor")),
        "dark.jpg": (dark, dict(role="dark", shows="water stain, severely under-exposed", expected_quality_verdict="poor")),
        "unrelated_garden.jpg": (garden_scene(13), dict(role="unrelated", shows="a garden - not the insured property damage",
                                                       expected_quality_verdict="unrelated")),
    }


CLAIM_PHOTOS = {   # claim_id -> [(file, builder, label)]
    "CLM-0001": [("kitchen_wall_stain.jpg", lambda: wall_scene(101), dict(role="good", shows="stain on kitchen wall", expected_quality_verdict="usable")),
                 ("pipe_joint_closeup.jpg", lambda: pipe_closeup(102), dict(role="good", shows="corroded pipe joint", expected_quality_verdict="usable"))],
    "CLM-0002": [("kitchen_floor_moisture.jpg", lambda: wall_scene(201, floor=True), dict(role="good", shows="moisture stain on kitchen floor",
                                                                                          expected_quality_verdict="usable"))],
    "CLM-0004": [("bathroom_wall_stain.jpg", lambda: wall_scene(401), dict(role="good", shows="stain on bathroom wall", expected_quality_verdict="usable"))],
    "CLM-0005": [("utility_room_stain.jpg", lambda: wall_scene(501), dict(role="good", shows="stain in utility room", expected_quality_verdict="usable"))],
}

SAMPLE_TEMPLATES = {   # no scenario uses these types yet; rendered so T2.x can develop against them
    "invoice_sample": dict(document_id="DOC-TPL-1", filename="invoice_sample.pdf", doc_type="invoice", received_on="2026-03-25", uploaded_by="contractor",
                           pages=["INVOICE - SYNTHETIC DOCUMENT\nInvoice number: INV-TPL-0001\nVendor: SYNTHETIC Repairs Pvt Ltd (fictional)\n"
                                  "Labour: INR 40,000\nMaterials: INR 25,000\nTax: INR 11,700\nTotal: INR 76,700\nPayment terms: 30 days\n"]),
    "email_sample": dict(document_id="DOC-TPL-2", filename="email_sample.pdf", doc_type="email", received_on="2026-03-26", uploaded_by="claimant",
                         pages=["Re: my water damage claim - SYNTHETIC DOCUMENT\nHello, I am writing about the damage to my kitchen.\n"
                                "The plumber told me the pipe may have been leaking for some time.\nPlease tell me what else you need.\nRegards, SYNTHETIC Claimant\n"]),
}


# ---------------------------------------------------------------------------------------------------- build
def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pdf_text(pdf_bytes: bytes) -> list[str]:
    d = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    return [norm(p.get_text()) for p in d]


def build_all() -> dict[Path, bytes]:
    files: dict[Path, bytes] = {}
    sc_file = load_scenarios()
    for sc in sc_file.scenarios:
        base = OUT / "claims" / sc.claim_id
        manifest = {"claim_id": sc.claim_id, "scenario_id": sc.id, "synthetic": True, "license": LICENSE, "documents": [], "variants": [], "photos": []}
        pdfs: dict[str, bytes] = {}
        for doc in sc.documents:
            data = render_document(doc, sc.claim_id)
            pdfs[doc.document_id] = data
            files[base / doc.filename] = data
            manifest["documents"].append({
                "document_id": doc.document_id, "filename": doc.filename, "doc_type": doc.doc_type, "page_count": len(doc.pages),
                "sha256": sha(data), "text_sha256": sha("\n".join(pdf_text(data)).encode()), "received_on": doc.received_on,
                "uploaded_by": doc.uploaded_by, "in_scenario": True})
        if sc.id == "S-02":                      # one scanned + rotated page (needs OCR)
            src = next(d for d in sc.documents if d.doc_type == "surveyor_report")
            data = render_scanned_variant(pdfs[src.document_id])
            files[base / "surveyor_report_scanned.pdf"] = data
            manifest["variants"].append({
                "filename": "surveyor_report_scanned.pdf", "variant_of": src.document_id, "kind": "scanned_rotated", "rotation_deg": 1.8,
                "needs_ocr": True, "text_extractable": False, "sha256": sha(data), "expected_text": [norm(p) for p in src.pages],
                "in_scenario": False})
        for fname, build, label in CLAIM_PHOTOS.get(sc.claim_id, []):
            data = jpeg(build())
            files[base / "photos" / fname] = data
            manifest["photos"].append({"file": f"photos/{fname}", "sha256": sha(data), "in_scenario": False, **label})
        files[base / "manifest.json"] = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()

    qset = {"synthetic": True, "license": LICENSE, "purpose": "image-quality evaluation set (T2.5)", "photos": []}
    for fname, (img, label) in quality_set().items():
        data = jpeg(img)
        files[OUT / "photos" / "quality_set" / fname] = data
        qset["photos"].append({"file": fname, "sha256": sha(data), "size": list(img.size), **label})
    files[OUT / "photos" / "quality_set" / "manifest.json"] = (json.dumps(qset, indent=2) + "\n").encode()

    tpl = {"synthetic": True, "documents": []}
    for name, doc in SAMPLE_TEMPLATES.items():
        data = render_document(doc, "CLM-TEMPLATE")
        files[OUT / "templates" / doc["filename"]] = data
        tpl["documents"].append({"document_id": doc["document_id"], "filename": doc["filename"], "doc_type": doc["doc_type"], "sha256": sha(data),
                                 "source_text": doc["pages"]})
    files[OUT / "templates" / "manifest.json"] = (json.dumps(tpl, indent=2) + "\n").encode()
    return files


def main() -> int:
    files = build_all()
    if "--check" in sys.argv:
        stale = sorted(str(p.relative_to(OUT)) for p, b in files.items() if not p.exists() or p.read_bytes() != b)
        print("Synthetic docs out of date - run scripts/generate_synthetic_docs.py: " + ", ".join(stale) if stale else "Synthetic docs up to date.")
        return 1 if stale else 0
    for p, b in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    print(f"Wrote {len(files)} files under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

