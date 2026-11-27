"""samples/claim_sample_1: new SYNTHETIC documents for trying the API flow. They must stay new, stay in step with their generator, and work through the API."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from claimtower.api.app import create_app
from claimtower.composition import SYNTHETIC, build_registry, policy_library_from_pdfs
from claimtower.core.config import load_settings
from claimtower.ingestion.store import DocumentStore
from claimtower.orchestration.graph import ClaimRunner

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "samples" / "claim_sample_1"
SETTINGS = load_settings()
FILES = ["fnol.pdf", "plumber_report.pdf", "surveyor_report.pdf", "estimate.pdf", "invoice.pdf", "claimant_email.pdf"]
TYPES = {"fnol.pdf": "fnol", "plumber_report.pdf": "plumber_report", "surveyor_report.pdf": "surveyor_report", "estimate.pdf": "estimate",
         "invoice.pdf": "invoice", "claimant_email.pdf": "email"}


def test_the_sample_files_are_up_to_date_and_new():
    r = subprocess.run([sys.executable, "scripts/generate_sample_claim.py", "--check"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_manifest_lists_every_file_with_its_expected_type():
    m = json.loads((SAMPLE / "manifest.json").read_text(encoding="utf-8"))
    assert m["synthetic"] is True and m["policy_ref"] == "POL-1013" and {d["filename"]: d["expected_type"] for d in m["documents"]} == TYPES
    assert all((SAMPLE / f).is_file() for f in FILES) and (SAMPLE / "README.md").is_file()


@pytest.fixture(scope="module")
def library():
    return policy_library_from_pdfs(SETTINGS)


def test_the_documented_api_flow_works_with_the_sample_files(tmp_path, library):
    store = DocumentStore(tmp_path / "store")
    client = TestClient(create_app(ClaimRunner(build_registry(SETTINGS, document_store=store, policy_library=library)), store, synthetic_claims=SYNTHETIC / "claims"))
    r = client.post("/claims", json={"claim_id": "CLM-1001", "policy_ref": "POL-1013", "loss_date": "2026-05-22", "loss_type": "water_damage"})
    assert r.status_code == 201 and r.json()["state"]["policy_snapshot"]["pas_status"] == "in_force" and r.json()["state"]["facts"] == []
    for i, name in enumerate(FILES):
        up = client.post("/claims/CLM-1001/documents", params={} if i == len(FILES) - 1 else {"reassess": "false"},
                         files={"file": (name, (SAMPLE / name).read_bytes(), "application/pdf")}, data={"uploaded_by": "tester"})
        assert up.status_code == 201
    v = client.get("/claims/CLM-1001").json()
    s = v["state"]
    assert v["status"]["phase"] == "awaiting_human_decision" and {d["filename"]: d["detected_type"] for d in s["documents"]} == TYPES      # typed by the assessment
    assert len(s["facts"]) == 33 and all(f["evidence_text"] and f["page"] for f in s["facts"])
    q = {x["question_id"]: x["status"] for x in s["open_questions"]}
    assert q["Q-INFORCE"] == "established" and q["Q-SUBLIMIT"] == "established" and s["missing_evidence"] == []                           # clean: nothing is asked for
    assert [(c["kind"], c["severity"], c["question_id"]) for c in s["contradictions"]] == [("amount", "low", "Q-AMOUNT")]                  # the deliberate 10% difference
    assert client.post("/claims/CLM-1001/decision", json={"decision": "approve", "reviewer_id": "t1", "reviewer_role": "senior_adjuster"}).status_code == 200
