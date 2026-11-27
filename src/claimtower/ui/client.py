"""The UI's only way to the system: a thin client over the HTTP API (docs/api.md). It exposes exactly the endpoints of T7.1 and nothing else, so anything the
UI can do, the API can; and it is testable without a browser by handing it a FastAPI TestClient."""
from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_URL = os.environ.get("CLAIMTOWER_API_URL", "http://127.0.0.1:8000")


class ApiError(Exception):
    """The API refused or could not be reached. `status` is 0 when it could not be reached; `detail` is the API's explanation."""

    def __init__(self, status: int, detail: Any) -> None:
        self.status, self.detail = status, detail
        super().__init__(f"{status}: {detail}")

    @property
    def message(self) -> str:
        d = self.detail
        if isinstance(d, dict):
            return str(d.get("error") or d.get("detail") or d)
        if isinstance(d, list):                                   # FastAPI validation errors
            return "; ".join(f"{'.'.join(str(x) for x in e.get('loc', [])[1:])}: {e.get('msg')}" for e in d)
        return str(d)


class ApiClient:
    def __init__(self, base_url: str = DEFAULT_URL, http: httpx.Client | None = None, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = http or httpx.Client(base_url=self.base_url, timeout=timeout)

    def _call(self, method: str, path: str, **kw: Any) -> httpx.Response:
        try:
            r = self._http.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise ApiError(0, f"cannot reach the API at {self.base_url}: {e}") from e
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ApiError(r.status_code, detail)
        return r

    # ------------------------------------------------------------------ service
    def health(self) -> dict[str, Any]:
        return self._call("GET", "/health").json()

    def agents(self) -> list[dict[str, str]]:
        return self._call("GET", "/agents").json()

    # ------------------------------------------------------------------ claims
    def claims(self) -> list[dict[str, Any]]:
        return self._call("GET", "/claims").json()

    def claim(self, claim_id: str) -> dict[str, Any]:
        return self._call("GET", f"/claims/{claim_id}").json()

    def create_claim(self, **body: Any) -> dict[str, Any]:
        return self._call("POST", "/claims", json={k: v for k, v in body.items() if v not in (None, "")}).json()

    def versions(self, claim_id: str) -> list[dict[str, Any]]:
        return self._call("GET", f"/claims/{claim_id}/versions").json()

    def diff(self, claim_id: str, from_version: int, to_version: int) -> dict[str, Any]:
        return self._call("GET", f"/claims/{claim_id}/diff", params={"from": from_version, "to": to_version}).json()

    def audit(self, claim_id: str) -> dict[str, Any]:
        return self._call("GET", f"/claims/{claim_id}/audit").json()

    # ------------------------------------------------------------------ documents
    def documents(self, claim_id: str) -> list[dict[str, Any]]:
        return self._call("GET", f"/claims/{claim_id}/documents").json()

    def upload(self, claim_id: str, filename: str, data: bytes, *, uploaded_by: str | None = None, reassess: bool = True) -> dict[str, Any]:
        r = self._call("POST", f"/claims/{claim_id}/documents", params={"reassess": str(reassess).lower()}, files={"file": (filename, data)},
                       data={"uploaded_by": uploaded_by} if uploaded_by else None)
        return r.json()

    def document_file(self, claim_id: str, document_id: str) -> bytes:
        return self._call("GET", f"/claims/{claim_id}/documents/{document_id}/file").content

    def document_details(self, claim_id: str, document_id: str) -> dict[str, Any]:
        return self._call("GET", f"/claims/{claim_id}/documents/{document_id}/details").json()

    def page_image(self, claim_id: str, document_id: str, page: int, fact_id: str | None = None) -> tuple[bytes | None, str]:
        """(PNG bytes, how the fact was marked: region | text | none). (None, "") when the document has no page picture (text, e-mail, Word)."""
        try:
            r = self._call("GET", f"/claims/{claim_id}/documents/{document_id}/page/{page}", params={"fact": fact_id} if fact_id else None)
        except ApiError as e:
            if e.status == 404 and "page image" in e.message:
                return None, ""
            raise
        return r.content, r.headers.get("x-highlight", "none")

    # ------------------------------------------------------------------ the human gate
    def decide(self, claim_id: str, decision: str, reviewer_id: str, reviewer_role: str, reason: str = "", edits: dict[str, str] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"decision": decision, "reviewer_id": reviewer_id, "reviewer_role": reviewer_role, "reason": reason}
        if edits:
            body["edits"] = edits
        return self._call("POST", f"/claims/{claim_id}/decision", json=body).json()
