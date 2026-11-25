"""Render data/synthetic/policies/corpus.yaml -> PDFs, DOCX sources, clause_index.json and INDEX.md (T1.1).

    python scripts/generate_policy_corpus.py

Everything produced is SYNTHETIC. Pagination is explicit in corpus.yaml; the script fails loudly if a page overflows.
clause_index.json records, per clause, its page and character span inside the whitespace-normalised page text, which is
what a clause extractor sees after PDF text extraction (T3.1 uses it as labelled data).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf
import yaml
from docx import Document
from docx.shared import Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic" / "policies"
SRC = OUT / "corpus.yaml"

W, H, MARGIN = 595, 842, 56
BODY, HEAD, LEAD = 10, 11, 14
BANNER = "SYNTHETIC DOCUMENT - for software testing only. Not an insurer document. All amounts are SAMPLE."


def norm(s: str) -> str:
    return " ".join(s.split())


def wrap(text: str, font: str, size: float, width: float) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and pymupdf.get_text_length(trial, fontname=font, fontsize=size) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def render_pdf(doc: dict, path: Path) -> None:
    pdf = pymupdf.open()
    n_pages = len(doc["pages"])
    width = W - 2 * MARGIN
    for i, blocks in enumerate(doc["pages"], start=1):
        page = pdf.new_page(width=W, height=H)
        page.insert_text((MARGIN, 30), BANNER, fontname="helv", fontsize=7, color=(0.7, 0, 0))
        y = 70
        if i == 1:
            page.insert_text((MARGIN, y), doc["title"], fontname="hebo", fontsize=16)
            y += 20
            kind = "Endorsement" if doc["kind"] == "endorsement" else "Policy wording"
            page.insert_text((MARGIN, y), f"{kind} {doc['doc_id']}  |  Version {doc['version']}", fontname="helv", fontsize=9)
            y += 30
        for b in blocks:
            if "h" in b:
                y += 6
                page.insert_text((MARGIN, y), b["h"], fontname="hebo", fontsize=HEAD)
                y += LEAD + 2
            else:
                if "clause" in b:
                    page.insert_text((MARGIN, y), b["heading"], fontname="hebo", fontsize=BODY)
                    y += LEAD
                for line in wrap(b.get("text") or b["p"], "helv", BODY, width):
                    page.insert_text((MARGIN, y), line, fontname="helv", fontsize=BODY)
                    y += LEAD
                y += 8
            if y > H - 70:
                raise SystemExit(f"{doc['doc_id']} page {i} overflows; move content to another page in corpus.yaml")
        page.insert_text((MARGIN, H - 36), f"{doc['doc_id']}  |  SYNTHETIC", fontname="helv", fontsize=8)
        page.insert_text((W - MARGIN - 60, H - 36), f"Page {i} of {n_pages}", fontname="helv", fontsize=8)
    pdf.set_metadata({"title": f"{doc['title']} (SYNTHETIC)", "subject": "SYNTHETIC test document", "author": "OmniClaim synthetic data",
                      "creator": "generate_policy_corpus.py", "producer": "PyMuPDF"})
    pdf.save(path, garbage=3, deflate=True)
    pdf.close()


def render_docx(doc: dict, path: Path) -> None:
    d = Document()
    d.core_properties.title = f"{doc['title']} (SYNTHETIC)"
    p = d.add_paragraph()
    r = p.add_run(BANNER)
    r.font.size = Pt(8)
    d.add_heading(doc["title"], level=1)
    kind = "Endorsement" if doc["kind"] == "endorsement" else "Policy wording"
    d.add_paragraph(f"{kind} {doc['doc_id']} | Version {doc['version']}")
    for i, blocks in enumerate(doc["pages"]):
        if i:
            d.add_page_break()
        for b in blocks:
            if "h" in b:
                d.add_heading(b["h"], level=2)
            elif "clause" in b:
                d.add_paragraph().add_run(b["heading"]).bold = True
                d.add_paragraph(b["text"])
            else:
                d.add_paragraph(b["p"])
    d.save(path)


def build_index(corpus: dict) -> dict:
    docs_out, clauses = [], []
    for doc in corpus["documents"]:
        pdf = pymupdf.open(OUT / f"{doc['doc_id']}.pdf")
        page_text = [norm(pg.get_text()) for pg in pdf]
        pdf.close()
        for pno, blocks in enumerate(doc["pages"], start=1):
            for b in blocks:
                if "clause" not in b:
                    continue
                t = norm(b["text"])
                start = page_text[pno - 1].find(t)
                if start < 0:
                    raise SystemExit(f"{b['clause']}: text not found on {doc['doc_id']} p{pno} after PDF extraction")
                clauses.append({"clause_id": b["clause"], "doc_id": doc["doc_id"], "clause_type": b["type"],
                                "heading": b["heading"], "page": pno, "span": [start, start + len(t)], "text": b["text"]})
        docs_out.append({k: doc[k] for k in ("doc_id", "title", "kind", "version", "difficulty", "used_by", "intent")}
                        | {"pages": len(doc["pages"]), "pdf": f"{doc['doc_id']}.pdf", "docx": f"{doc['doc_id']}.docx"}
                        | ({"applies_to": doc["applies_to"], "effective_from": doc["effective_from"]}
                           if doc["kind"] == "endorsement" else {}))
    ids = [c["clause_id"] for c in clauses]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate clause ids: " + ", ".join(sorted({i for i in ids if ids.count(i) > 1})))
    known = set(ids)
    for r in corpus["relations"]:
        if r["source"] not in known or r["target"] not in known:
            raise SystemExit(f"relation references unknown clause: {r}")
    return {"synthetic": True, "span_basis": "character offsets in the whitespace-normalised PDF page text (pymupdf get_text)",
            "documents": docs_out, "clauses": clauses, "relations": corpus["relations"]}


def render_index_md(idx: dict) -> str:
    rel_by_src: dict[str, list[str]] = {}
    for r in idx["relations"]:
        rel_by_src.setdefault(r["source"], []).append(f"{r['relation']} {r['target']}")
    out = ["# Synthetic policy corpus - index", "",
           "> **ALL SYNTHETIC.** Generated by `scripts/generate_policy_corpus.py` from `corpus.yaml` - do not edit by hand.",
           "> Clause IDs are **stable** (used by scenarios, ground truth and evals). Amounts and day-counts are SAMPLE.", "",
           "## Documents", "",
           "| Doc | Kind | Version | Pages | Difficulty | Effective from | Used by | Intent |", "|---|---|---|---|---|---|---|---|"]
    for d in idx["documents"]:
        out.append(f"| [{d['doc_id']}]({d['pdf']}) ({d['title']}) | {d['kind']} | {d['version']} | {d['pages']} | "
                   f"{d['difficulty']} | "
                   f"{d.get('effective_from', '-')} | {', '.join(d['used_by']) or 'reserved'} | {d['intent']} |")
    out += ["", "Effective date `schedule` = taken from the policy Schedule / PAS per policy (T1.4), not from the endorsement text.", "",
            "## Clauses", "", "| Clause ID | Type | Doc | Page | Heading | Relations |", "|---|---|---|---|---|---|"]
    for c in idx["clauses"]:
        out.append(f"| `{c['clause_id']}` | {c['clause_type']} | {c['doc_id']} | {c['page']} | {c['heading']} | "
                   f"{'; '.join(rel_by_src.get(c['clause_id'], [])) or '-'} |")
    out += ["", "## Files", "", "- `*.pdf` - rendered documents (page-numbered, SYNTHETIC banner on every page)",
            "- `*.docx` - editable sources of the same content",
            "- `clause_index.json` - labelled clauses (page, span, text) and relations; ground truth for T3.1/T3.2",
            "- `corpus.yaml` - the single source; edit this, then regenerate", ""]
    return "\n".join(out)


def main() -> int:
    corpus = yaml.safe_load(SRC.read_text(encoding="utf-8"))
    for doc in corpus["documents"]:
        render_pdf(doc, OUT / f"{doc['doc_id']}.pdf")
        render_docx(doc, OUT / f"{doc['doc_id']}.docx")
    idx = build_index(corpus)
    (OUT / "clause_index.json").write_text(json.dumps(idx, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    (OUT / "INDEX.md").write_text(render_index_md(idx), encoding="utf-8", newline="\n")
    print(f"Wrote {len(corpus['documents'])} documents, {len(idx['clauses'])} clauses to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
