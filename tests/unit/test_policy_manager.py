"""Policy documents added and reloaded while the system runs; the document id comes from INSIDE the document, never from the file name."""
import shutil
from decimal import Decimal

import pymupdf
import pytest
from fastapi.testclient import TestClient

from claimtower.api.app import create_app
from claimtower.composition import SYNTHETIC, build_policy_manager, build_registry, ingest_claim_folder
from claimtower.core.config import load_settings
from claimtower.core.schemas import ClaimState
from claimtower.ingestion.store import DocumentStore
from claimtower.orchestration.graph import ClaimRunner
from claimtower.policy.manager import PolicyConflict, PolicyDocumentError

SETTINGS = load_settings()
POLICIES = SYNTHETIC / "policies"


def pdf_bytes(name: str) -> bytes:
    return (POLICIES / f"{name}.pdf").read_bytes()


def make_pdf(lines: list[str]) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 60), "\n".join(lines), fontsize=10)
    return doc.tobytes()


@pytest.fixture
def folders(tmp_path):
    """A built-in folder holding every wording EXCEPT END-01, each under a meaningless file name, and an empty upload folder."""
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    for n, f in enumerate(sorted(POLICIES.glob("*.pdf"))):
        if f.stem != "END-01":
            shutil.copy(f, builtin / f"scan {n} (final).pdf")
    return builtin, tmp_path / "uploads"


def manager(folders):
    return build_policy_manager(SETTINGS, folders[1], folders[0])


# ---------------------------------------------------------------------------------------------------------------- identity is read from the document
def test_the_id_and_version_come_from_inside_the_document_not_from_the_file_name(folders):
    m = manager(folders)
    assert "END-01" not in {e.doc_id for e in m.entries()} and {"BASE-HOME-1", "BASE-HOME-2", "END-02"} <= {e.doc_id for e in m.entries()}
    assert all(e.id_from == "document" and e.filename.startswith("scan ") for e in m.entries())
    r = m.ingest(pdf_bytes("END-01"), "Water sublimit - FINAL v3 (copy).pdf", uploaded_by="ops-1")
    e = r.entry
    assert r.outcome == "added" and (e.doc_id, e.version, e.kind, e.source, e.id_from) == ("END-01", "2025-01", "endorsement", "uploaded", "document")
    assert e.filename == "Water sublimit - FINAL v3 (copy).pdf" and e.uploaded_by == "ops-1" and e.clauses >= 1 and (folders[1] / "END-01.pdf").is_file()


def test_a_document_with_no_id_inside_is_refused_unless_an_id_is_given(folders):
    m = manager(folders)
    body = make_pdf(["Some insurer - Water Wording", "1. Water damage", "We cover water escaping from a fixed pipe.", "2. Exclusions", "We do not cover gradual leakage."])
    with pytest.raises(PolicyDocumentError, match="no document id was found inside the document"):
        m.ingest(body, "BASE-HOME-9.pdf")                                              # the file name is not enough
    r = m.ingest(body, "whatever.pdf", document_id="WATER-9")
    assert r.entry.doc_id == "WATER-9" and r.entry.id_from == "given" and r.entry.clauses >= 1
    with pytest.raises(PolicyDocumentError, match="not a valid document id"):
        m.ingest(body, "x.pdf", document_id="bad id!")


def test_other_header_styles_are_read_too(folders):
    m = manager(folders)
    body = make_pdf(["Document reference: HOME-PLUS-7 | Version 2026-03", "1.1 Escape of water", "We cover water escaping from a fixed pipe or appliance."])
    e = m.ingest(body, "a.pdf").entry
    assert (e.doc_id, e.version) == ("HOME-PLUS-7", "2026-03")


@pytest.mark.parametrize("data,why", [(b"this is not a pdf", "not a readable PDF"), (make_pdf([]), "no text layer"),
                                      (make_pdf(["Endorsement END-77 | Version 1", "just a paragraph with no numbered clauses at all."]), "no clauses were found")])
def test_a_document_that_cannot_be_read_as_a_wording_is_refused_with_the_reason(folders, data, why):
    with pytest.raises(PolicyDocumentError, match=why):
        manager(folders).ingest(data, "x.pdf")


# ---------------------------------------------------------------------------------------------------------------- adding, conflicts, reload
def test_the_same_content_again_changes_nothing_and_different_content_under_an_id_is_a_conflict_unless_replaced(folders):
    m = manager(folders)
    first = m.ingest(pdf_bytes("END-01"), "a.pdf")
    again = m.ingest(pdf_bytes("END-01"), "b.pdf")
    assert first.outcome == "added" and again.outcome == "unchanged" and again.entry.filename == "a.pdf"
    changed = make_pdf(["Endorsement END-01 | Version 2026-09", "1. Water damage sublimit", "The most we will pay for any one claim under the escape of water cover is INR 150,000."])
    with pytest.raises(PolicyConflict, match="replace=true"):
        m.ingest(changed, "c.pdf")
    r = m.ingest(changed, "c.pdf", replace=True)
    assert r.outcome == "replaced" and r.entry.version == "2026-09" and m.library.get("END-01").clauses[0].text.endswith("INR 150,000.")


def test_an_uploaded_document_replacing_a_built_in_one_says_so(folders):
    m = manager(folders)
    r = m.ingest(make_pdf(["Policy wording BASE-HOME-1 | Version 2030-01", "1.1 Buildings", "Buildings means the dwelling."]), "new.pdf", replace=True)
    assert r.outcome == "replaced" and r.entry.replaces_built_in and {e.doc_id: e.source for e in m.entries()}["BASE-HOME-1"] == "uploaded"


def test_uploaded_documents_survive_a_restart_with_their_original_name_and_uploader(folders):
    manager(folders).ingest(pdf_bytes("END-01"), "Original name.pdf", uploaded_by="ops-1")
    again = manager(folders)
    e = next(x for x in again.entries() if x.doc_id == "END-01")
    assert (e.filename, e.uploaded_by, e.source) == ("Original name.pdf", "ops-1", "uploaded") and again.library.get("END-01") is not None


def test_a_reload_finds_a_file_dropped_into_the_folder_whatever_it_is_called_and_reports_the_change(folders):
    builtin, _ = folders
    m = manager(folders)
    assert m.reload().added == [] and m.reload().total == len(m.entries())
    shutil.copy(POLICIES / "END-01.pdf", builtin / "zzz random name.pdf")
    (builtin / "broken.pdf").write_bytes(b"not a pdf")
    r = m.reload()
    assert r.added == ["END-01"] and any("broken.pdf" in p for p in r.problems) and m.library.get("END-01") is not None
    (builtin / "zzz random name.pdf").unlink()
    assert m.reload().removed == ["END-01"] and m.library.get("END-01") is None


def test_a_builder_attached_to_the_manager_sees_the_new_library_at_once(folders):
    from claimtower.policy.relations import RelationExtractor
    from claimtower.policy.snapshot import SnapshotBuilder
    from claimtower.integrations.mock.pas_mock import MockPasAdapter
    from datetime import date

    m = manager(folders)
    b = SnapshotBuilder(MockPasAdapter(), m.library, RelationExtractor(SETTINGS.pack.policy_structure))
    m.attach(b)
    before = b.build("POL-1005", date(2026, 6, 10))
    m.ingest(pdf_bytes("END-01"), "x.pdf")
    after = b.build("POL-1005", date(2026, 6, 10))
    assert not any(c.clause_id.startswith("END-01/") for c in before.result.clauses) and any(c.clause_id.startswith("END-01/") for c in after.result.clauses)


# ---------------------------------------------------------------------------------------------------------------- the API, and the effect on a claim
@pytest.fixture
def client(tmp_path, folders):
    m = manager(folders)
    store = DocumentStore(tmp_path / "store")
    ingest_claim_folder(store, SYNTHETIC / "claims" / "CLM-0005")
    runner = ClaimRunner(build_registry(SETTINGS, document_store=store, policy_manager=m), settings=SETTINGS)
    return TestClient(create_app(runner, store, synthetic_claims=SYNTHETIC / "claims", policies=m)), m


def post(c, body, name="anything.pdf", **kw):
    return c.post("/policies/documents", files={"file": (name, body, "application/pdf")}, **kw)


def test_the_api_lists_adds_and_refuses_policy_documents_with_the_right_codes(client):
    c, _ = client
    listing = c.get("/policies/documents").json()
    assert {"BASE-HOME-1", "END-02"} <= {d["doc_id"] for d in listing} and all(d["id_from"] == "document" for d in listing)
    r = post(c, pdf_bytes("END-01"), "My endorsement.pdf", data={"uploaded_by": "ops-1"})
    assert r.status_code == 201 and r.json()["outcome"] == "added" and r.json()["document"]["doc_id"] == "END-01" and "re-assess" in r.json()["note"]
    assert post(c, pdf_bytes("END-01"), "other.pdf").status_code == 200 and post(c, pdf_bytes("END-01"), "other.pdf").json()["outcome"] == "unchanged"
    changed = make_pdf(["Endorsement END-01 | Version 2", "1. Water damage sublimit", "The most we will pay is INR 90,000 for any one claim."])
    assert post(c, changed).status_code == 409 and post(c, changed, params={"replace": "true"}).json()["outcome"] == "replaced"
    assert post(c, b"nope").status_code == 422 and "no document id" in post(c, make_pdf(["1. Water", "We cover water."])).json()["detail"]


def test_a_reload_through_the_api_and_an_unknown_node_in_reassess(client, folders):
    c, _ = client
    shutil.copy(POLICIES / "END-01.pdf", folders[0] / "dropped.pdf")
    assert c.post("/policies/reload").json()["added"] == ["END-01"]
    c.post("/claims", json={"claim_id": "CLM-0005"})
    assert c.post("/claims/CLM-0005/reassess", params={"from_node": "payment"}).status_code == 422 and c.post("/claims/CLM-NOPE/reassess").status_code == 404


def test_a_wording_added_later_changes_the_claim_when_it_is_re_assessed_from_the_policy_node(client):
    """S-05: the endorsement wording is missing at first (no payable); an upload with any file name supplies it and the claim is re-assessed."""
    c, _ = client
    v = c.post("/claims", json={"claim_id": "CLM-0005"}).json()
    s = v["state"]["settlement"]
    assert s["status"] == "cannot_calculate" and s["payable"] is None and s["missing_inputs"] == ["water_damage_sublimit"]
    assert post(c, pdf_bytes("END-01"), "From the broker (2).pdf").status_code == 201
    assert c.get("/claims/CLM-0005").json()["state"]["settlement"]["status"] == "cannot_calculate"            # an assessed claim is not changed behind its back
    after = c.post("/claims/CLM-0005/reassess").json()                                                        # default node: policy
    st = after["state"]["settlement"]
    assert st["status"] == "calculated" and Decimal(st["payable"]) == Decimal(100000)
    assert any(cl["clause_id"].startswith("END-01/") for cl in after["state"]["policy_snapshot"]["clauses"])


def test_without_a_manager_the_endpoints_say_so(tmp_path):
    store = DocumentStore(tmp_path / "s")
    runner = ClaimRunner(build_registry(SETTINGS, force_mock=("intake", "gap", "coverage", "settlement", "action", "audit", "routing")))
    c = TestClient(create_app(runner, store, synthetic_claims=SYNTHETIC / "claims"))
    assert c.get("/policies/documents").status_code == 404 and c.post("/policies/reload").status_code == 404 and ClaimState


# ---------------------------------------------------------------------------------------------------------------- many documents in one request
def batch(c, files, **kw):
    return c.post("/policies/documents/batch", files=[("files", (n, b, "application/pdf")) for n, b in files], **kw)


def test_a_batch_stores_every_good_file_and_reports_each_bad_one_on_its_own(client):
    c, m = client
    changed = make_pdf(["Endorsement END-01 | Version 2", "1. Water damage sublimit", "The most we will pay is INR 90,000 for any one claim."])
    r = batch(c, [("a.pdf", pdf_bytes("END-01")), ("junk.pdf", b"nope"), ("empty.pdf", b""), ("b.pdf", changed)], data={"uploaded_by": "ops-1"})
    assert r.status_code == 200
    body = r.json()
    got = {x["filename"]: x for x in body["results"]}
    assert [x["filename"] for x in body["results"]] == ["a.pdf", "junk.pdf", "empty.pdf", "b.pdf"]                   # the order sent
    assert got["a.pdf"]["outcome"] == "added" and got["a.pdf"]["document"]["doc_id"] == "END-01" and got["a.pdf"]["document"]["uploaded_by"] == "ops-1"
    assert got["junk.pdf"]["outcome"] == "failed" and got["junk.pdf"]["status"] == 422 and got["empty.pdf"]["error"] == "the file is empty"
    assert got["b.pdf"]["outcome"] == "failed" and got["b.pdf"]["status"] == 409 and "replace" in got["b.pdf"]["error"]    # same id, other content, in the same batch
    assert (body["added"], body["failed"]) == (1, 3) and "END-01" in {e.doc_id for e in m.entries()}


def test_a_second_batch_is_unchanged_and_replace_supersedes(client):
    c, _ = client
    assert batch(c, [("x.pdf", pdf_bytes("END-01"))]).json()["added"] == 1
    assert batch(c, [("y.pdf", pdf_bytes("END-01"))]).json()["unchanged"] == 1
    changed = make_pdf(["Endorsement END-01 | Version 2", "1. Water damage sublimit", "The most we will pay is INR 90,000 for any one claim."])
    r = batch(c, [("z.pdf", changed)], params={"replace": "true"}).json()
    assert r["replaced"] == 1 and r["results"][0]["document"]["version"] == "2"


def test_a_batch_reads_many_files_and_refuses_too_many(client):
    c, m = client
    names = ["END-01", "END-02", "BASE-HOME-1"]
    r = batch(c, [(f"{n}.pdf", pdf_bytes(n)) for n in names]).json()
    assert [x["document"]["doc_id"] for x in r["results"]] == names and r["failed"] == 0
    assert batch(c, [(f"f{i}.pdf", b"x") for i in range(31)]).status_code == 413


def test_the_client_sends_a_batch(client):
    from claimtower.ui.client import ApiClient

    c, _ = client
    api = ApiClient(http=c)
    res = api.add_policy_documents([("one.pdf", pdf_bytes("END-01")), ("bad.pdf", b"nope")], uploaded_by="ops-2")
    assert (res["added"], res["failed"]) == (1, 1)


def test_last_problem_is_kept_per_thread_so_parallel_reads_do_not_mix_their_reasons():
    import threading

    from claimtower.policy.problem import ThreadLocalProblem

    class Reader:
        last_problem = ThreadLocalProblem()

    r, seen, gate = Reader(), {}, threading.Barrier(2)
    assert r.last_problem is None

    def work(name: str) -> None:
        r.last_problem = f"reason of {name}"
        gate.wait()                                       # both threads have now set theirs
        seen[name] = r.last_problem

    threads = [threading.Thread(target=work, args=(n,)) for n in ("a", "b")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert seen == {"a": "reason of a", "b": "reason of b"} and r.last_problem is None          # this thread never set one
