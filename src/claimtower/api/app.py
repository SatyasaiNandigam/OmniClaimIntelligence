"""FastAPI service (T7.1): the only surface the UI and the integrations talk to.

    GET  /health                          the service, which agents are real and which are mocks
    GET  /agents                          the roster (implemented | YET TO IMPLEMENT (mock)) with the task that will implement each
    POST /claims                          create a claim and run it to the human gate
    POST /claims/with-documents           the same, with the claim's documents sent in the one multipart request (a bad file is skipped and reported)
    GET  /claims                          the claims, one line each
    GET  /claims/{id}                     status (phase, gate, mock nodes) and the full state at the current version
    GET  /claims/{id}/versions            every committed version and the node that produced it
    GET  /claims/{id}/diff?from=&to=      what changed between two versions
    GET  /claims/{id}/documents           the claim's documents          POST: upload one (re-assesses the claim from the intake node)
    GET  /claims/{id}/documents/{doc}/file   the original file
    GET  /claims/{id}/documents/{doc}/details   what the system read: type and why, pages, extracted fields with page, region and confidence
    GET  /claims/{id}/documents/{doc}/page/{n}   the page as a PNG; ?fact=<fact id> marks where that fact was read
    POST /claims/{id}/decision            the human gate: approve | reject | request_more_evidence by a named reviewer
    GET  /claims/{id}/audit               the event log (agents, mocks, routing, human decisions)
    GET  /policies/documents              the policy documents in the library (id and version read from inside each document)
    POST /policies/documents              add a policy wording or endorsement (any file name; the id is read from the document); ?replace=true to supersede
    POST /policies/reload                 read the policy folders again (what was added, removed, changed)
    POST /claims/{id}/reassess            run a claim again from a node (default policy: pick up a new wording)
    GET  /claims/{id}/trail               the persistent audit trail (hash-chained) and whether its chain is intact
    GET  /claims/{id}/decision-pack       the decision pack built from the trail alone: ?format=md (default) or pdf

Prototype limits, stated rather than hidden: reviewer identity and role are ASSERTED by the caller (no authentication yet); claims and events live in
memory; one claim is processed at a time; every response carries `X-Mock-Agents` naming the agents that are not implemented yet.
Errors: 404 unknown claim / document / version, 409 the claim is in the wrong state for the call, 413 file too large, 422 invalid or refused decision.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from claimtower.api.schemas import (
    PasPolicyView, PasRegisterResult, ScheduleCheckView, ScheduleReadResult, PolicyBatchItem, PolicyBatchResult, PolicyDocumentResult, PolicyDocumentView, PolicyReloadResult, AuditView, ClaimCreate, ClaimWithDocumentsResult, SkippedFile, ClaimSummary, ClaimView, DocumentDetails, DocumentUploadResult, ErrorDetail, EventView, FieldView, Health, PageView, TrailRecordView, TrailView,
)
from claimtower.core.schemas import ClaimHeader, ClaimState, DocumentRecord
from claimtower.core.state import OwnershipViolation
from claimtower.core.state_diff import StateDiff
from claimtower.ingestion.loaders import UnsupportedDocument
from claimtower.integrations.mock.data import PasPolicy
from claimtower.ingestion.preview import render_highlighted
from claimtower.ingestion.store import DocumentStore, StoreError
from claimtower.orchestration.graph import ClaimRunner, RunStatus, VersionInfo
from claimtower.orchestration.supervisor import HumanDecision

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_POLICY_BATCH = 30                                   # policy documents in one batch request
ERRORS = {404: {"model": ErrorDetail}, 409: {"model": ErrorDetail}, 422: {"model": ErrorDetail}}


def create_app(runner: ClaimRunner, store: DocumentStore, *, synthetic_claims: Path | None = None, max_upload_bytes: int = MAX_UPLOAD_BYTES, policies: Any | None = None,
               pas: Any | None = None, schedule_reader: Any | None = None) -> FastAPI:
    app = FastAPI(title="Insurance Claim Resolution Control Tower", version="0.1.0",
                  description="Prototype. SYNTHETIC data only. Agents marked `YET TO IMPLEMENT (mock)` return labelled fixtures. "
                              "A human decides every approval, denial and payment; the system only recommends.")
    app.state.runner, app.state.store = runner, store

    @app.middleware("http")
    async def flag_mocks(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Mock-Agents"] = ",".join(runner.registry.mocks)       # a visible MOCK flag on every response
        return response

    @app.exception_handler(OwnershipViolation)
    async def ownership(_: Request, e: OwnershipViolation):
        return JSONResponse(status_code=500, content={"detail": f"an agent wrote a field it does not own (a bug): {e}"})

    # ------------------------------------------------------------------ helpers
    def known(claim_id: str) -> None:
        if not runner.knows(claim_id):
            raise HTTPException(404, f"unknown claim {claim_id}")

    def view(claim_id: str) -> ClaimView:
        state = runner.state(claim_id)
        return ClaimView(claim_id=claim_id, state_version=state.state_version, status=runner.status(claim_id), state=state)

    def summary(claim_id: str) -> ClaimSummary:
        st, status = runner.state(claim_id), runner.status(claim_id)
        counts: dict[str, int] = {}
        for q in st.open_questions:
            counts[q.status] = counts.get(q.status, 0) + 1
        return ClaimSummary(claim_id=claim_id, phase=status.phase, state_version=st.state_version, resolution_status=st.resolution_status,
                            tier=st.routing.tier if st.routing else None, open_questions=counts, mock_agents=runner.registry.mocks)

    # ------------------------------------------------------------------ service
    @app.get("/health", response_model=Health, tags=["service"])
    def health() -> Health:
        return Health(mock_agents=runner.registry.mocks, implemented_agents=runner.registry.implemented, claims=len(runner.claims()),
                      low_confidence_below=runner.supervisor.low_confidence_below)

    @app.get("/agents", tags=["service"])
    def agents() -> list[dict[str, str]]:
        return runner.registry.describe()

    # ------------------------------------------------------------------ claims
    @app.post("/claims", response_model=ClaimView, status_code=201, tags=["claims"], responses={**ERRORS})
    def create_claim(body: ClaimCreate) -> ClaimView:
        claim_id = body.claim_id or f"CLM-{uuid.uuid4().hex[:8].upper()}"
        if runner.knows(claim_id):
            raise HTTPException(409, f"claim {claim_id} already exists")
        if body.use_synthetic_documents:
            folder = (synthetic_claims or Path()) / claim_id
            if synthetic_claims is None or not (folder / "manifest.json").is_file():
                raise HTTPException(404, f"there is no synthetic claim folder for {claim_id}")
            if not store.records(claim_id):                                              # idempotent: a restarted API keeps the stored documents
                from claimtower.composition import ingest_claim_folder
                ingest_claim_folder(store, folder)
        header = None
        if any((body.claimant_name, body.policy_ref, body.loss_date, body.loss_type, body.loss_description)):
            header = ClaimHeader(claim_id=claim_id, claimant_name=body.claimant_name, policy_ref=body.policy_ref, loss_date=body.loss_date,
                                 loss_type=body.loss_type, loss_description=body.loss_description, channel=body.channel)
        runner.start(ClaimState(claim_id=claim_id, claim_header=header))
        return view(claim_id)

    @app.post("/claims/with-documents", response_model=ClaimWithDocumentsResult, status_code=201, tags=["claims"], responses={**ERRORS, 413: {"model": ErrorDetail}})
    def create_claim_with_documents(
            files: Annotated[list[UploadFile], File(description="The claim's documents: claim form, reports, estimate, invoice, photos, schedule ...", json_schema_extra={"items": {"type": "string", "format": "binary"}})],
            claim_id: Annotated[str | None, Form(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,63}$")] = None, claimant_name: Annotated[str | None, Form()] = None,
            policy_ref: Annotated[str | None, Form()] = None, loss_date: Annotated[date | None, Form()] = None, loss_type: Annotated[str | None, Form()] = None,
            loss_description: Annotated[str | None, Form()] = None, channel: Annotated[str, Form()] = "other", uploaded_by: Annotated[str | None, Form()] = None,
    ) -> ClaimWithDocumentsResult:
        """Create a claim from its documents in ONE request and assess it once, to the human gate. Every field is optional: what is missing (name, policy, loss date, description) is read from the
        claim form by the intake agent, and what you supply is kept. A file that is empty, too large or unreadable is skipped and listed in `skipped`; the claim is created if at least one file was stored.
        Later evidence goes to `POST /claims/{id}/documents` (which re-assesses the claim)."""
        if channel not in ("portal", "call_centre", "email", "agent", "other"):
            raise HTTPException(422, "channel must be one of portal, call_centre, email, agent, other")
        claim_id = claim_id or f"CLM-{uuid.uuid4().hex[:8].upper()}"
        if runner.knows(claim_id):
            raise HTTPException(409, f"claim {claim_id} already exists")
        stored: list[DocumentRecord] = []
        skipped: list[SkippedFile] = []
        for f in files:
            name = f.filename or "upload"
            data = f.file.read(max_upload_bytes + 1)
            if len(data) > max_upload_bytes:
                skipped.append(SkippedFile(filename=name, reason=f"larger than {max_upload_bytes // (1024 * 1024)} MB"))
            elif not data:
                skipped.append(SkippedFile(filename=name, reason="the file is empty"))
            else:
                same = next((r for r in store.records(claim_id) if r.sha256 == hashlib.sha256(data).hexdigest()), None)
                if same is not None:                                                       # a restarted server forgets the claim, not its stored files: never store the same file twice
                    if all(r.document_id != same.document_id for r in stored):
                        stored.append(same)
                    continue
                try:
                    stored.append(store.ingest(claim_id, name, data, uploaded_by=uploaded_by))
                except (StoreError, UnsupportedDocument) as e:
                    skipped.append(SkippedFile(filename=name, reason=str(e)))
        if not stored:
            raise HTTPException(422, {"error": "no document could be stored, so no claim was created", "skipped": [s.model_dump() for s in skipped]})
        header = None
        if any((claimant_name, policy_ref, loss_date, loss_type, loss_description)):
            header = ClaimHeader(claim_id=claim_id, claimant_name=claimant_name, policy_ref=policy_ref, loss_date=loss_date, loss_type=loss_type,
                                 loss_description=loss_description, channel=channel)
        runner.start(ClaimState(claim_id=claim_id, claim_header=header))                 # one run, after every document is stored
        return ClaimWithDocumentsResult(claim=view(claim_id), stored=stored, skipped=skipped)

    @app.get("/claims", response_model=list[ClaimSummary], tags=["claims"])
    def list_claims() -> list[ClaimSummary]:
        return [summary(c) for c in runner.claims()]

    @app.get("/claims/{claim_id}", response_model=ClaimView, tags=["claims"], responses={**ERRORS})
    def get_claim(claim_id: str) -> ClaimView:
        known(claim_id)
        return view(claim_id)

    @app.get("/claims/{claim_id}/status", response_model=RunStatus, tags=["claims"], responses={**ERRORS})
    def get_status(claim_id: str) -> RunStatus:
        known(claim_id)
        return runner.status(claim_id)

    @app.get("/claims/{claim_id}/versions", response_model=list[VersionInfo], tags=["claims"], responses={**ERRORS})
    def get_versions(claim_id: str) -> list[VersionInfo]:
        known(claim_id)
        return runner.versions(claim_id)

    @app.get("/claims/{claim_id}/diff", response_model=StateDiff, tags=["claims"], responses={**ERRORS})
    def get_diff(claim_id: str, from_: Annotated[int, Query(alias="from", ge=0)], to: Annotated[int, Query(ge=0)]) -> StateDiff:
        known(claim_id)
        try:
            return runner.diff(claim_id, from_, to)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'\"")) from e

    # ------------------------------------------------------------------ documents
    @app.get("/claims/{claim_id}/documents", response_model=list[DocumentRecord], tags=["documents"], responses={**ERRORS})
    def list_documents(claim_id: str) -> list[DocumentRecord]:
        known(claim_id)
        return store.records(claim_id)

    @app.post("/claims/{claim_id}/documents", response_model=DocumentUploadResult, status_code=201, tags=["documents"], responses={**ERRORS, 413: {"model": ErrorDetail}})
    def upload_document(claim_id: str, file: Annotated[UploadFile, File()], uploaded_by: Annotated[str | None, Form()] = None,
                        source_channel: Annotated[str | None, Form()] = None, reassess: Annotated[bool, Query()] = True) -> DocumentUploadResult:
        """Store the document and, unless `reassess=false`, re-assess the claim from the intake node (the checklist and header are refreshed, the notice of loss is read if it is new; the policy node is skipped when nothing it reads changed): only the new document is processed, and a pending
        recommendation at the human gate is superseded (nobody had decided yet)."""
        known(claim_id)
        data = file.file.read(max_upload_bytes + 1)
        if len(data) > max_upload_bytes:
            raise HTTPException(413, f"the file is larger than {max_upload_bytes // (1024 * 1024)} MB")
        if not data:
            raise HTTPException(422, "the file is empty")
        try:
            record = store.ingest(claim_id, file.filename or "upload", data, uploaded_by=uploaded_by, source_channel=source_channel)
        except StoreError as e:
            raise HTTPException(409, str(e)) from e
        status = runner.reassess(claim_id, from_node="intake") if reassess else None
        return DocumentUploadResult(document=record, reassessment=status, note="re-assessed from the intake node" if reassess else "stored; not re-assessed")

    @app.get("/claims/{claim_id}/documents/{document_id}/file", tags=["documents"], responses={**ERRORS}, response_class=Response)
    def get_document_file(claim_id: str, document_id: str) -> Response:
        known(claim_id)
        record = next((d for d in store.records(claim_id) if d.document_id == document_id), None)
        if record is None:
            raise HTTPException(404, f"claim {claim_id} has no document {document_id}")
        stored = store.stored(document_id)
        return Response(content=store.original(document_id), media_type=getattr(stored, "media_type", None) or "application/octet-stream",
                        headers={"Content-Disposition": f'inline; filename="{record.filename}"'})

    def stored_of(claim_id: str, document_id: str):
        known(claim_id)
        if next((d for d in store.records(claim_id) if d.document_id == document_id), None) is None:
            raise HTTPException(404, f"claim {claim_id} has no document {document_id}")
        return store.stored(document_id)

    @app.get("/claims/{claim_id}/documents/{document_id}/details", response_model=DocumentDetails, tags=["documents"], responses={**ERRORS})
    def document_details(claim_id: str, document_id: str) -> DocumentDetails:
        d = stored_of(claim_id, document_id)
        ex = d.extraction
        fields = [FieldView(name=f.name, value=f.value, normalised=f.normalised, label=f.label, page=f.page, region=f.region.model_dump() if f.region else None, evidence_text=f.evidence_text,
                            confidence=f.confidence, low_confidence=f.low_confidence, method=f.method, flags=list(f.flags)) for f in (ex.fields if ex else [])]
        return DocumentDetails(document_id=d.document_id, filename=d.filename, detected_type=d.detected_type, type_confidence=d.type_confidence, type_method=d.type_method,
                               type_reason=d.type_reason, type_evidence_snippet=d.type_evidence_snippet, type_page=d.type_page, media_type=d.media_type, kind=d.kind,
                               page_count=d.page_count, uploaded_by=d.uploaded_by, source_channel=d.source_channel, uploaded_at=d.uploaded_at, same_content_as=list(d.same_content_as),
                               ocr_engine=d.ocr_engine, ocr_errors=list(d.ocr_errors), load_error=d.load_error,
                               pages=[PageView(number=p.number, source=p.source, needs_ocr=p.needs_ocr, text=p.text) for p in d.pages], fields=fields,
                               missing_required=list(ex.missing_required) if ex else [], warnings=list(ex.warnings) if ex else [], low_confidence_below=runner.supervisor.low_confidence_below)

    @app.get("/claims/{claim_id}/documents/{document_id}/page/{page}", tags=["documents"], responses={200: {"content": {"image/png": {}}}, **ERRORS}, response_class=Response)
    def document_page(claim_id: str, document_id: str, page: int, fact: Annotated[str | None, Query()] = None) -> Response:
        """The page as a picture. With `fact`, the place that fact was read from is marked (header `X-Highlight`: region | text | none)."""
        d = stored_of(claim_id, document_id)
        if not 1 <= page <= d.page_count:
            raise HTTPException(404, f"document {document_id} has pages 1 to {d.page_count}")
        region = text = None
        if fact is not None:
            f = next((x for x in runner.state(claim_id).facts if x.fact_id == fact and x.source_document_id == document_id), None)
            if f is None:
                raise HTTPException(404, f"document {document_id} has no fact {fact}")
            region, text = (f.region if f.page == page else None), (f.evidence_text if f.page == page else None)
        try:
            png, how = render_highlighted(store.original(document_id), page, region, text)
        except UnsupportedDocument as e:
            raise HTTPException(404, str(e)) from None
        return Response(png, media_type="image/png", headers={"X-Highlight": how})

    # ------------------------------------------------------------------ policy documents
    def manager() -> Any:
        if policies is None:
            raise HTTPException(404, "policy documents cannot be managed on this server")
        return policies

    def pview(e: Any) -> PolicyDocumentView:
        return PolicyDocumentView(**e.__dict__)

    @app.get("/policies/documents", response_model=list[PolicyDocumentView], tags=["policies"], responses={**ERRORS})
    def list_policy_documents() -> list[PolicyDocumentView]:
        return [pview(e) for e in manager().entries()]

    @app.post("/policies/documents", response_model=PolicyDocumentResult, status_code=201, tags=["policies"], responses={**ERRORS})
    async def add_policy_document(response: Response, file: Annotated[UploadFile, File()], uploaded_by: Annotated[str | None, Form()] = None,
                                  document_id: Annotated[str | None, Form()] = None, replace: Annotated[bool, Query()] = False) -> PolicyDocumentResult:
        """Add a policy wording or endorsement. The file may be called anything: the document id and version are read from INSIDE the document (a line such as
        'Policy wording BASE-HOME-1 | Version 2025-01'); `document_id` is only for a document that has none. The same content again is `unchanged`; different content under an
        existing id is 409 unless `replace=true`. Claims already assessed are not changed: re-assess one with POST /claims/{id}/reassess."""
        from claimtower.policy.manager import PolicyConflict, PolicyDocumentError

        data = await file.read(max_upload_bytes + 1)
        if len(data) > max_upload_bytes:
            raise HTTPException(413, f"the file is over {max_upload_bytes // (1024 * 1024)} MB")
        try:
            res = manager().ingest(data, file.filename or "upload.pdf", uploaded_by=uploaded_by, replace=replace, document_id=document_id)
        except PolicyConflict as e:
            raise HTTPException(409, str(e)) from None
        except PolicyDocumentError as e:
            raise HTTPException(422, str(e)) from None
        if res.outcome == "unchanged":
            response.status_code = 200
        e = res.entry
        note = {"added": "added to the library", "replaced": "replaced the earlier document with this id", "unchanged": "the same content is already in the library"}[res.outcome]
        if e.unclassified:
            note += f"; {e.unclassified} clause(s) could not be typed by the rules and are left for a person (they are not guessed)"
        return PolicyDocumentResult(outcome=res.outcome, document=pview(e), note=note + ". Claims already assessed are not changed: re-assess one from the policy node.")

    @app.post("/policies/documents/batch", response_model=PolicyBatchResult, tags=["policies"], responses={**ERRORS, 413: {"model": ErrorDetail}})
    async def add_policy_documents(files: Annotated[list[UploadFile], File(description="Policy wordings and endorsements, as many as needed", json_schema_extra={"items": {"type": "string", "format": "binary"}})],
                                   uploaded_by: Annotated[str | None, Form()] = None, replace: Annotated[bool, Query()] = False) -> PolicyBatchResult:
        """Add many policy wordings and endorsements in ONE request. Each file is read like `POST /policies/documents` (the id and version come from inside the document), the files are read in
        parallel, and every file gets its own result: one that is unreadable, or that clashes with a different document of the same id (unless `replace=true`), fails alone and the others are stored.
        Claims already assessed are not changed: re-assess one with POST /claims/{id}/reassess."""
        from claimtower.policy.manager import PolicyDocumentError

        m = manager()
        if len(files) > MAX_POLICY_BATCH:
            raise HTTPException(413, f"at most {MAX_POLICY_BATCH} documents in one request")
        items: list[tuple[str, bytes]] = []
        early: dict[int, PolicyBatchItem] = {}
        for n, f in enumerate(files):
            data = await f.read(max_upload_bytes + 1)
            name = f.filename or f"upload-{n + 1}.pdf"
            if len(data) > max_upload_bytes:
                early[n] = PolicyBatchItem(filename=name, outcome="failed", status=413, error=f"the file is over {max_upload_bytes // (1024 * 1024)} MB")
            elif not data:
                early[n] = PolicyBatchItem(filename=name, outcome="failed", status=422, error="the file is empty")
            else:
                items.append((name, data))
        try:
            done = iter(m.ingest_many(items, uploaded_by=uploaded_by, replace=replace))
        except PolicyDocumentError as e:
            raise HTTPException(422, str(e)) from None
        results: list[PolicyBatchItem] = []
        for n in range(len(files)):
            if n in early:
                results.append(early[n])
                continue
            r = next(done)
            results.append(PolicyBatchItem(filename=r.filename, outcome=r.outcome, document=pview(r.entry) if r.entry else None, status=r.status, error=r.error))
        count = {k: sum(1 for r in results if r.outcome == k) for k in ("added", "replaced", "unchanged", "failed")}
        return PolicyBatchResult(results=results, **count, note="Claims already assessed are not changed: re-assess one from the policy node.")

    # ------------------------------------------------------------------ the mock policy administration system (a real PAS is read-only for us: this is demo set-up)
    def mock_pas() -> Any:
        if pas is None or not hasattr(pas, "register_policy") or not pas.can_register():
            raise HTTPException(404, "policies cannot be registered on this server (it has no mock PAS to write to)")
        return pas

    @app.get("/pas/policies", response_model=list[PasPolicyView], tags=["mock PAS"], responses={**ERRORS})
    def list_pas_policies() -> list[PasPolicyView]:
        """The policies the mock PAS knows: the synthetic ones and the ones registered through this API."""
        return [PasPolicyView(**p) for p in mock_pas().policies()]

    @app.post("/pas/schedules/read", response_model=ScheduleReadResult, tags=["mock PAS"], responses={**ERRORS, 413: {"model": ErrorDetail}})
    async def read_policy_schedule(file: Annotated[UploadFile, File(description="A policy schedule (PDF)")]) -> ScheduleReadResult:
        """Read a policy schedule and PROPOSE the PAS record: policy number, period, endorsements, limits and deductible. Every value is checked against the text of the document and a value
        that is not in it is left out (see `checks`, `warnings`). NOTHING is stored: look at the proposal, correct it if needed, and send it to `POST /pas/policies`."""
        from claimtower.integrations.mock.schedule import ScheduleReader

        adapter = mock_pas()
        data = await file.read(max_upload_bytes + 1)
        if len(data) > max_upload_bytes:
            raise HTTPException(413, f"the file is over {max_upload_bytes // (1024 * 1024)} MB")
        if not data:
            raise HTTPException(422, "the file is empty")
        try:
            p = (schedule_reader or ScheduleReader(None)).read(data, file.filename or "schedule.pdf")
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        known = {r["policy_no"]: r["source"] for r in adapter.policies()}
        return ScheduleReadResult(filename=file.filename or "schedule.pdf", policy=p.policy, checks=[ScheduleCheckView(**c.__dict__) for c in p.checks], problems=p.problems,
                                  warnings=p.warnings, existing=known.get(p.policy.policy_no, "none") if p.policy else "none", read_by=p.source)  # type: ignore[arg-type]

    @app.post("/pas/policies", response_model=PasRegisterResult, status_code=201, tags=["mock PAS"], responses={**ERRORS})
    def register_pas_policy(policy: PasPolicy, replace: Annotated[bool, Query()] = False) -> PasRegisterResult:
        """Record a policy in the mock PAS (normally the proposal from `POST /pas/schedules/read`, after a person checked it). A policy number that already exists is 409 unless `replace=true`.
        Claims already assessed are not changed: re-assess one with POST /claims/{id}/reassess."""
        from claimtower.integrations.mock.pas_mock import PasConflict, PasRegisterError

        try:
            outcome = mock_pas().register_policy(policy, replace=replace)
        except PasConflict as e:
            raise HTTPException(409, str(e)) from None
        except PasRegisterError as e:
            raise HTTPException(422, str(e)) from None
        return PasRegisterResult(outcome=outcome, policy_no=policy.policy_no, note="Recorded in the mock PAS and in use at once. Claims already assessed are not changed: re-assess one from the policy node.")

    @app.post("/policies/reload", response_model=PolicyReloadResult, tags=["policies"], responses={**ERRORS})
    def reload_policy_documents() -> PolicyReloadResult:
        r = manager().reload()
        return PolicyReloadResult(total=r.total, added=r.added, removed=r.removed, changed=r.changed, problems=r.problems)

    @app.post("/claims/{claim_id}/reassess", response_model=ClaimView, tags=["claims"], responses={**ERRORS})
    def reassess_claim(claim_id: str, from_node: Annotated[str, Query()] = "policy") -> ClaimView:
        """Run the claim again from `from_node` (default: the policy node, to pick up a policy document added since). Nodes before it are not run; nodes after it whose inputs
        did not change are skipped. A recommendation waiting at the gate is superseded."""
        from claimtower.orchestration.registry import PIPELINE

        known(claim_id)
        if from_node not in PIPELINE:
            raise HTTPException(422, f"unknown node '{from_node}'; nodes are {', '.join(PIPELINE)}")
        runner.reassess(claim_id, from_node=from_node)
        return view(claim_id)

    # ------------------------------------------------------------------ the human gate
    @app.post("/claims/{claim_id}/decision", response_model=ClaimView, tags=["human gate"], responses={**ERRORS})
    def decide(claim_id: str, decision: HumanDecision) -> ClaimView:
        """approve | reject | request_more_evidence, by a named reviewer whose role reaches the claim's required approver role. The system only recommends:
        an approval pays and denies nothing. A refused decision (role too low, reason missing ...) leaves the gate open and returns 422 with the reason."""
        known(claim_id)
        try:
            status = runner.decide(claim_id, decision)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        if status.gate and status.gate.get("error"):
            raise HTTPException(422, {"error": status.gate["error"], "gate": "still open", "required_approver_role": status.gate.get("required_approver_role")})
        return view(claim_id)

    # ------------------------------------------------------------------ audit
    @app.get("/claims/{claim_id}/audit", response_model=AuditView, tags=["audit"], responses={**ERRORS})
    def audit(claim_id: str) -> AuditView:
        known(claim_id)
        events = [EventView(seq=e.seq, ts=e.ts, kind=e.kind, node=e.node, message=e.message, agent_id=e.agent_id, mock=e.mock, state_version=e.state_version,
                            data={k: v for k, v in e.data.items() if k != "key"}) for e in runner.events_for(claim_id)]
        return AuditView(claim_id=claim_id, events=events, agents=runner.registry.describe(),
                         human_decisions=[e for e in events if e.kind in ("human_decision", "human_decision_refused")])

    def trail_of(claim_id: str):
        known(claim_id)
        if runner.trail is None:
            raise HTTPException(404, "the audit trail is not enabled on this server")
        return runner.trail

    @app.get("/claims/{claim_id}/trail", response_model=TrailView, tags=["audit"], responses={**ERRORS})
    def trail(claim_id: str) -> TrailView:
        t = trail_of(claim_id)
        problems = t.verify()
        recs = [TrailRecordView(seq=r.seq, ts=r.ts, kind=r.kind, hash=r.hash, summary=(f"{r.data['kind']} ({r.data['node']}): {r.data['message']}"[:200] if r.kind == "event"
                                                                                  else f"patch by {r.data['agent_id']} -> v{r.data['version']}" if r.kind == "state_patch" else "claim created"))
                for r in t.records(claim_id)]
        return TrailView(claim_id=claim_id, records=recs, chain_ok=not problems, problems=problems[:10], head=t.head)

    @app.get("/claims/{claim_id}/decision-pack", tags=["audit"], responses={200: {"content": {"text/markdown": {}, "application/pdf": {}}}, **ERRORS})
    def decision_pack(claim_id: str, format: Annotated[str, Query(pattern="^(md|pdf)$")] = "md") -> Response:
        t = trail_of(claim_id)
        from claimtower.audit.explain_pack import build_pack, pack_to_pdf
        md = build_pack(t, claim_id)
        if format == "pdf":
            return Response(pack_to_pdf(md), media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="decision-pack-{claim_id}.pdf"'})
        return Response(md, media_type="text/markdown; charset=utf-8", headers={"Content-Disposition": f'inline; filename="decision-pack-{claim_id}.md"'})

    return app
