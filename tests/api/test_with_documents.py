"""POST /claims/with-documents: the claim and all its documents in one multipart request, assessed once; a bad file is skipped and reported; later evidence still goes to /documents."""

import pytest
from fastapi.testclient import TestClient

from claimtower.api.app import create_app
from claimtower.composition import SYNTHETIC, build_registry, policy_library_from_pdfs
from claimtower.core.config import load_settings
from claimtower.ingestion.store import DocumentStore
from claimtower.orchestration.graph import ClaimRunner

SETTINGS = load_settings()
FOLDER = SYNTHETIC / "claims" / "CLM-0001"
FILES = ["fnol.pdf", "plumber_report.pdf", "surveyor_report.pdf", "estimate.pdf"]


@pytest.fixture(scope="module")
def library():
    return policy_library_from_pdfs(SETTINGS)


@pytest.fixture
def client(tmp_path, library):
    store = DocumentStore(tmp_path / "store")
    runner = ClaimRunner(build_registry(SETTINGS, document_store=store, policy_library=library))
    return TestClient(create_app(runner, store, synthetic_claims=SYNTHETIC / "claims")), runner


def parts(names=FILES):
    return [("files", (n, (FOLDER / n).read_bytes(), "application/pdf")) for n in names]


def test_a_claim_is_created_from_its_documents_in_one_request_and_assessed_once(client):
    c, runner = client
    r = c.post("/claims/with-documents", files=parts(), data={"claim_id": "CLM-9001", "policy_ref": "POL-1001", "loss_date": "2026-03-14", "uploaded_by": "claimant"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["claim"]["claim_id"] == "CLM-9001" and body["skipped"] == []
    assert {d["filename"] for d in body["stored"]} == set(FILES) and all(d["uploaded_by"] == "claimant" for d in body["stored"])
    status = body["claim"]["status"]
    assert status["phase"] == "awaiting_human_decision" and status["nodes_run"].count("evidence") == 1          # one assessment, after every document was stored
    types = {d["detected_type"] for d in c.get("/claims/CLM-9001/documents").json()}
    assert {"fnol", "plumber_report", "surveyor_report", "estimate"} <= types                                   # the documents were read in that one run


def test_every_field_is_optional_the_header_is_read_from_the_claim_form(client):
    c, _ = client
    r = c.post("/claims/with-documents", files=parts(["fnol.pdf", "plumber_report.pdf"]))
    assert r.status_code == 201
    h = r.json()["claim"]["state"]["claim_header"]
    assert h["claim_id"].startswith("CLM-") and h["policy_ref"] and h["loss_date"]


def test_a_bad_file_is_skipped_and_listed_but_the_claim_is_still_created(client):
    c, _ = client
    files = parts(["fnol.pdf"]) + [("files", ("empty.pdf", b"", "application/pdf"))]
    r = c.post("/claims/with-documents", files=files, data={"claim_id": "CLM-9002"})
    assert r.status_code == 201
    body = r.json()
    assert [s["filename"] for s in body["skipped"]] == ["empty.pdf"] and "empty" in body["skipped"][0]["reason"] and len(body["stored"]) == 1


def test_when_no_file_can_be_stored_no_claim_is_created(client):
    c, runner = client
    r = c.post("/claims/with-documents", files=[("files", ("empty.pdf", b"", "application/pdf"))], data={"claim_id": "CLM-9003"})
    assert r.status_code == 422 and r.json()["detail"]["skipped"][0]["filename"] == "empty.pdf"
    assert not runner.knows("CLM-9003")


def test_an_existing_claim_id_is_refused_and_a_bad_claim_id_is_rejected(client):
    c, _ = client
    assert c.post("/claims/with-documents", files=parts(["fnol.pdf"]), data={"claim_id": "CLM-9004"}).status_code == 201
    assert c.post("/claims/with-documents", files=parts(["fnol.pdf"]), data={"claim_id": "CLM-9004"}).status_code == 409
    assert c.post("/claims/with-documents", files=parts(["fnol.pdf"]), data={"claim_id": "x"}).status_code == 422


def test_later_evidence_still_goes_to_the_documents_endpoint_and_re_assesses(client):
    c, _ = client
    assert c.post("/claims/with-documents", files=parts(["fnol.pdf"]), data={"claim_id": "CLM-9005"}).status_code == 201
    r = c.post("/claims/CLM-9005/documents", files={"file": ("invoice.pdf", (FOLDER / "plumber_report.pdf").read_bytes(), "application/pdf")})
    assert r.status_code == 201 and r.json()["reassessment"]["nodes_run"][:1] == ["intake"]


def test_the_files_field_is_described_in_a_way_swagger_ui_shows_a_file_picker_for(client):
    c, _ = client
    schema = c.get("/openapi.json").json()["components"]["schemas"]["Body_create_claim_with_documents_claims_with_documents_post"]["properties"]["files"]
    assert schema["type"] == "array" and schema["items"] == {"type": "string", "format": "binary"}


def test_a_restarted_server_does_not_store_the_same_files_again_for_the_same_claim_id(tmp_path, library):
    store = DocumentStore(tmp_path / "store")

    def server():
        runner = ClaimRunner(build_registry(SETTINGS, document_store=store, policy_library=library))
        return TestClient(create_app(runner, store, synthetic_claims=SYNTHETIC / "claims"))

    first = server().post("/claims/with-documents", files=parts(["fnol.pdf", "plumber_report.pdf"]), data={"claim_id": "CLM-9006"}).json()
    again = server().post("/claims/with-documents", files=parts(["fnol.pdf", "plumber_report.pdf"]), data={"claim_id": "CLM-9006"})      # the claim is forgotten on restart, its files are not
    assert again.status_code == 201
    assert {d["document_id"] for d in again.json()["stored"]} == {d["document_id"] for d in first["stored"]} and len(store.records("CLM-9006")) == 2
