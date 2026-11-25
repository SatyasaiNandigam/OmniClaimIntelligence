"""FastAPI service (T7.1): the only surface the UI and the integrations talk to.

    GET  /health                          the service, which agents are real and which are mocks
    GET  /agents                          the roster (implemented | YET TO IMPLEMENT (mock)) with the task that will implement each
    POST /claims                          create a claim and run it to the human gate
    GET  /claims                          the claims, one line each
    GET  /claims/{id}                     status (phase, gate, mock nodes) and the full state at the current version
    GET  /claims/{id}/versions            every committed version and the node that produced it
    GET  /claims/{id}/diff?from=&to=      what changed between two versions
    GET  /claims/{id}/documents           the claim's documents          POST: upload one (re-assesses the claim from the evidence node)
    GET  /claims/{id}/documents/{doc}/file   the original file
    POST /claims/{id}/decision            the human gate: approve | reject | request_more_evidence by a named reviewer
    GET  /claims/{id}/audit               the event log (agents, mocks, routing, human decisions)

Prototype limits, stated rather than hidden: reviewer identity and role are ASSERTED by the caller (no authentication yet); claims and events live in
memory; one claim is processed at a time; every response carries `X-Mock-Agents` naming the agents that are not implemented yet.
Errors: 404 unknown claim / document / version, 409 the claim is in the wrong state for the call, 413 file too large, 422 invalid or refused decision.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from claimtower.api.schemas import (
    AuditView, ClaimCreate, ClaimSummary, ClaimView, DocumentUploadResult, ErrorDetail, EventView, Health,
)
from claimtower.core.schemas import ClaimHeader, ClaimState, DocumentRecord
from claimtower.core.state import OwnershipViolation
from claimtower.core.state_diff import StateDiff
from claimtower.ingestion.store import DocumentStore, StoreError
from claimtower.orchestration.graph import ClaimRunner, RunStatus, VersionInfo
from claimtower.orchestration.supervisor import HumanDecision

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ERRORS = {404: {"model": ErrorDetail}, 409: {"model": ErrorDetail}, 422: {"model": ErrorDetail}}


def create_app(runner: ClaimRunner, store: DocumentStore, *, synthetic_claims: Path | None = None, max_upload_bytes: int = MAX_UPLOAD_BYTES) -> FastAPI:
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
        return Health(mock_agents=runner.registry.mocks, implemented_agents=runner.registry.implemented, claims=len(runner.claims()))

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
        """Store the document and, unless `reassess=false`, re-assess the claim from the evidence node: only the new document is processed, and a pending
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
        status = runner.reassess(claim_id, from_node="evidence") if reassess else None
        return DocumentUploadResult(document=record, reassessment=status, note="re-assessed from the evidence node" if reassess else "stored; not re-assessed")

    @app.get("/claims/{claim_id}/documents/{document_id}/file", tags=["documents"], responses={**ERRORS}, response_class=Response)
    def get_document_file(claim_id: str, document_id: str) -> Response:
        known(claim_id)
        record = next((d for d in store.records(claim_id) if d.document_id == document_id), None)
        if record is None:
            raise HTTPException(404, f"claim {claim_id} has no document {document_id}")
        stored = store.stored(document_id)
        return Response(content=store.original(document_id), media_type=getattr(stored, "media_type", None) or "application/octet-stream",
                        headers={"Content-Disposition": f'inline; filename="{record.filename}"'})

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

    return app
