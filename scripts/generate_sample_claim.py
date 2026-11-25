"""Generate the sample claims under samples/ (SYNTHETIC documents the system has never seen, to try the API flow with your own uploads).

    python scripts/generate_sample_claim.py            # (re)write samples/claim_sample_1/
    python scripts/generate_sample_claim.py --check    # fail if the files differ from what this script produces, or are not new

Claim sample 1: a dishwasher inlet hose bursts in a utility room (policy POL-1013, loss 22 May 2026). A new claimant, policy, dates, amounts and wording:
nothing in data/synthetic, the scenarios or the tests contains it. It is a clean, consistent claim with one deliberate wrinkle: the claimant's own estimate
is 10% above the surveyor's assessed cost (a LOW-severity amount difference that is noted but does not block).
All amounts are SAMPLE. The documents are rendered by the same renderer as data/synthetic/claims (generate_synthetic_docs.py).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_synthetic_docs import pdf_text, render_document  # noqa: E402

OUT = ROOT / "samples" / "claim_sample_1"
CLAIM = {"claim_id": "CLM-SAMPLE-1", "policy_ref": "POL-1013", "loss_date": "2026-05-22", "loss_type": "water_damage",
         "loss_description": "Dishwasher inlet hose burst in the utility room"}
LABEL = "SAMPLE-1"

DOCS = [
    dict(document_id="DOC-SAMPLE1-1", filename="fnol.pdf", doc_type="fnol", received_on="2026-05-24", uploaded_by="claimant", pages=[
        "FIRST NOTICE OF LOSS - SYNTHETIC DOCUMENT\n"
        "Claimant: SYNTHETIC Claimant K\n"
        "Policy number: POL-1013\n"
        "Date of loss: 22 May 2026\n"
        "Location of loss: Utility room, 14 Rosewood Lane (fictional address)\n"
        "Description: On 22 May 2026 the inlet hose of the dishwasher in the utility room burst while the machine was running. Water spread across the "
        "utility room floor and into the hallway skirting before the isolation valve was closed.\n"
        "Estimated loss: INR 72,000\n"]),
    dict(document_id="DOC-SAMPLE1-2", filename="plumber_report.pdf", doc_type="plumber_report", received_on="2026-05-24", uploaded_by="plumber", pages=[
        "PLUMBER'S REPORT - SYNTHETIC DOCUMENT\n"
        "Inspection date: 23 May 2026\n"
        "Findings: The rubber inlet hose of the dishwasher split at its crimped end while under pressure. The tear was fresh and there was no sign of "
        "earlier leakage, staining or corrosion at the tap or on the hose. The hose was a recent fitting.\n"
        "Hose replaced and supply restored on 23 May 2026.\n"]),
    dict(document_id="DOC-SAMPLE1-3", filename="surveyor_report.pdf", doc_type="surveyor_report", received_on="2026-05-28", uploaded_by="surveyor", pages=[
        "SURVEYOR'S REPORT - SYNTHETIC DOCUMENT\n"
        "Site visit: 26 May 2026\n"
        "Cause of loss: Sudden failure of the dishwasher inlet hose during operation.\n"
        "Moisture readings were elevated in the utility room floor and the lower hallway wall only. The damage is consistent with the reported event "
        "and no other rooms were affected.\n"
        "Assessed repair cost: INR 64,500\n"]),
    dict(document_id="DOC-SAMPLE1-4", filename="estimate.pdf", doc_type="estimate", received_on="2026-05-27", uploaded_by="contractor", pages=[
        "REPAIR ESTIMATE - SYNTHETIC DOCUMENT\n"
        "Utility room flooring: INR 41,000\n"
        "Hallway skirting and plaster: INR 25,000\n"
        "Total: INR 66,000\n"
        "Scope: utility room and hallway\n"]),
    dict(document_id="DOC-SAMPLE1-5", filename="invoice.pdf", doc_type="invoice", received_on="2026-06-04", uploaded_by="contractor", pages=[
        "TAX INVOICE - SYNTHETIC DOCUMENT\n"
        "Invoice number: INV-5521\n"
        "Invoice date: 3 June 2026\n"
        "Vendor: Brightline Repairs (fictional)\n"
        "Payment terms: Net 15 days\n"
        "GSTIN: 00SYNTHETIC0000Z0\n"
        "Utility room flooring: INR 36,300\n"
        "Hallway skirting and plaster: INR 18,785\n"
        "GST (18%): INR 9,915\n"
        "Total: INR 65,000\n"]),
    dict(document_id="DOC-SAMPLE1-6", filename="claimant_email.pdf", doc_type="email", received_on="2026-06-05", uploaded_by="claimant", pages=[
        "Re: my dishwasher claim and the repair - SYNTHETIC DOCUMENT\n"
        "Hello, the plumber replaced the hose on 23 May and the contractor has now finished the floor and skirting repair.\n"
        "The damage was only in the utility room and the hallway. I have sent the contractor's invoice.\n"
        "Please tell me if you need anything else. Regards, SYNTHETIC Claimant K\n"]),
]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build() -> dict[Path, bytes]:
    files: dict[Path, bytes] = {}
    manifest = {"synthetic": True, "note": "SYNTHETIC. Never seen by the system: new claimant, policy, dates, amounts and wording. All amounts are SAMPLE.", **CLAIM, "documents": []}
    for d in DOCS:
        data = render_document(d, LABEL)
        files[OUT / d["filename"]] = data
        manifest["documents"].append({"filename": d["filename"], "expected_type": d["doc_type"], "uploaded_by": d["uploaded_by"], "pages": len(d["pages"]),
                                      "sha256": sha(data), "text_sha256": sha("\n".join(pdf_text(data)).encode())})
    files[OUT / "manifest.json"] = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return files


def known_hashes() -> tuple[set[str], set[str]]:
    """sha256 and text hashes of every document the repository already has (data/synthetic claims and policies)."""
    shas, texts = set(), set()
    for f in [*(ROOT / "data" / "synthetic").rglob("*.pdf")]:
        data = f.read_bytes()
        shas.add(sha(data))
        try:
            texts.add(sha("\n".join(pdf_text(data)).encode()))
        except Exception:
            pass
    return shas, texts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    files = build()
    seen_sha, seen_text = known_hashes()
    old = [d["filename"] for d, m in zip(DOCS, json.loads(files[OUT / "manifest.json"])["documents"]) if m["sha256"] in seen_sha or m["text_sha256"] in seen_text]
    if old:
        print("NOT NEW: already present in data/synthetic:", ", ".join(old))
        return 1
    if a.check:
        stale = [str(p.relative_to(ROOT)) for p, b in files.items() if not p.exists() or p.read_bytes() != b]
        print("samples are up to date and new." if not stale else "out of date: " + ", ".join(stale))
        return 1 if stale else 0
    OUT.mkdir(parents=True, exist_ok=True)
    for p, b in files.items():
        p.write_bytes(b)
    print(f"wrote {len(files)} files to {OUT.relative_to(ROOT)} (all new: none matches a document already in data/synthetic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
