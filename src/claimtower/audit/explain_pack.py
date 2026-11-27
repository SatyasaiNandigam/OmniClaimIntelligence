"""The decision pack (T8.3): one document that says what was decided, by whom, on what evidence and wording, with which calculation, built from the AUDIT TRAIL ONLY.

    md = build_pack(trail, "CLM-0002")            # Markdown
    pdf = pack_to_pdf(md)                         # the same content as a PDF (bytes)

Nothing is read from the running system: the claim state is rebuilt from the trail (`AuditTrail.reconstruct`), the human decisions, denials and model calls come from its records, and the
pack ends with the result of verifying the trail's hash chain. If a decision cannot be seen in this document, it cannot be reconstructed from the log. The pack states plainly that the system
only recommends: the decision is the person's, named in section 1.
"""
from __future__ import annotations

import html
import re
from typing import Any

from claimtower.audit.trail import AuditTrail
from claimtower.core.schemas import ClaimState


def _cell(v: Any, n: int = 140) -> str:
    s = " ".join(str(v if v is not None else "").split()).replace("|", "/")
    return s if len(s) <= n else s[: n - 1] + "…"


def _table(head: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["_none_", ""]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return out + [""]


def build_pack(trail: AuditTrail, claim_id: str) -> str:
    state: ClaimState = trail.reconstruct(claim_id)
    recs = trail.records(claim_id)
    events = [r for r in recs if r.kind == "event"]
    patches = [r for r in recs if r.kind == "state_patch"]
    decisions = [r for r in events if r.data["kind"] == "human_decision"]
    problems = trail.verify()
    h = state.claim_header
    mock = sorted({r.data["agent_id"] for r in patches if r.data.get("mock")})
    L: list[str] = [f"# Decision pack: claim {claim_id}", ""]
    L.append("> Prototype, SYNTHETIC data. The system only recommends; a named person decides (section 1)." + (f" Mock agents took part: {', '.join(mock)}." if mock else ""))
    L += ["", f"Claimant: {h.claimant_name if h else '-'} · Policy: {h.policy_ref if h else '-'} · Loss date: {h.loss_date if h else '-'} · Status: **{state.resolution_status}** · State version: {state.state_version}", ""]

    L += ["## 1. Decision", ""]
    if decisions:
        for d in decisions:
            x = d.data["data"]
            L.append(f"- **{x['decision']}** by {x['reviewer_id']} ({x['reviewer_role']}) at trail record {d.seq}" + (f": {x['reason']}" if x.get("reason") else "")
                     + (f". Edits to settlement inputs: {x['edits']}" if x.get("edits") else ""))
    else:
        L.append("No human decision has been recorded yet.")
    r = state.routing
    L += ["", f"Recommendation for the decision-maker: tier **{r.tier}**, reasons {', '.join(r.reason_codes)}, approver {r.required_approver_role} (rules {r.rules_version})." if r else "No routing recommendation.", ""]

    L += ["## 2. Coverage issues (a status says how well an issue is resolved, never a verdict)", ""]
    L += _table(["Issue", "Status", "Clauses cited", "Note"], [[i.issue, i.status, ", ".join(c.clause_id for c in i.clause_citations), i.note] for i in state.coverage_issues])

    L += ["## 3. Settlement", ""]
    s = state.settlement
    if s is None:
        L += ["No settlement was calculated.", ""]
    else:
        L.append(f"Status **{s.status}**; assessed loss {s.assessed_loss}; payable {s.payable}; reserve forecast {s.reserve_forecast} (a forecast, never a payment); "
                 + (f"forecast range {s.forecast_low} to {s.forecast_high} (the repair cost is not validated); " if s.forecast_low is not None else "") + f"rules {s.rule_version}.")
        if s.missing_inputs:
            L.append(f"Missing inputs: {', '.join(s.missing_inputs)}.")
        L += ["", s.explanation or "", ""]
        L += _table(["Step", "Input", "Before", "After"], [[t.rule_id, t.input_ref, t.amount_before, t.amount_after] for t in s.steps])

    L += ["## 4. Evidence", ""]
    L += _table(["Document", "File", "Type", "SHA-256"], [[d.document_id, d.filename, d.detected_type, d.sha256[:16] + "…"] for d in state.documents])
    L += _table(["Fact", "Statement", "Source", "Verbatim evidence", "Confidence", "Status"],
                [[f.fact_id, f.proposition, f"{f.source_document_id} p{f.page}", f.evidence_text, f.extraction_confidence, f.verification_status] for f in state.facts])

    L += ["## 5. Policy wording used", ""]
    p = state.policy_snapshot
    if p is None:
        L += ["No policy was in force on the loss date.", ""]
    else:
        L.append(f"Policy {p.policy_id} version {p.version}, status {p.pas_status}, source {p.source_system_response_ref}; endorsements {', '.join(e.endorsement_id for e in p.endorsements) or 'none'}. "
                 + ("**Mock policy system.**" if p.mock else ""))
        cited = {c.clause_id: c for i in state.coverage_issues for c in i.clause_citations}
        L += [""] + _table(["Clause", "Document", "Text"], [[c.clause_id, c.document, c.text] for c in cited.values()])

    L += ["## 6. Open questions, contradictions and requests", ""]
    L += _table(["Question", "Status"], [[q.question_id, q.status] for q in state.open_questions])
    L += _table(["Contradiction", "Question", "Severity", "Explanation"], [[c.contradiction_id, c.question_id, c.severity, c.explanation] for c in state.contradictions])
    L += _table(["Asked of", "For question", "Items"], [[m.minimum_evidence_request.addressee_role, m.question_id, "; ".join(m.minimum_evidence_request.items)] for m in state.missing_evidence])

    L += ["## 7. Actions and drafts (nothing is sent without a person)", ""]
    L += _table(["Action", "Party", "Mode", "Status"], [[a.action_type, a.responsible_party, a.execution_mode, a.status] for a in state.proposed_actions])
    L += _table(["Draft", "To", "Status", "Subject"], [[d.draft_id, d.addressee_role, d.status, d.subject] for d in state.drafts])

    L += ["## 8. How the assessment was made", ""]
    L += _table(["Version", "Author", "Mock", "Model calls (model, prompt version)", "What it said"],
                [[x.data["version"], x.data["agent_id"], "mock" if x.data.get("mock") else "", "; ".join(f"{c['model']} {c['prompt'] or c['kind']} v{c['prompt_version'] or '-'}{' (replayed)' if c['replayed'] else ''}"
                                                                                            for c in x.data.get("model_calls", [])), x.data.get("rationale")] for x in patches])
    skipped = [e for e in events if e.data["kind"] == "node_skipped"]
    denied = [e for e in events if e.data["kind"] == "action_denied"]
    failed = [e for e in events if e.data["kind"] == "agent_failed"]
    L += _table(["Record", "Event", "Node", "Message"], [[e.seq, e.data["kind"], e.data["node"], e.data["message"]] for e in [*skipped, *failed, *denied]])

    L += ["## 9. Quality checks", ""]
    q = state.qa_report
    L.append("Not run." if q is None else ("Passed." if q.passed else "FAILED: " + "; ".join(f"{f.check}: {f.detail}" for f in q.failures)))
    L += ["", "## 10. Integrity of this record", ""]
    L.append(f"Trail records: {len(recs)} for this claim ({len(trail)} in the trail). Chain head {trail.head[:16]}… " + ("Chain verified: no record was edited, removed or reordered." if not problems
                                                                                                                       else "CHAIN BROKEN: " + "; ".join(problems[:3])))
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------------------- PDF
def _inline(s: str) -> str:
    return re.sub(r"_(.+?)_", r"<i>\1</i>", re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(s)))


def to_html(md: str) -> str:
    out: list[str] = ["<html><body style='font-family:sans-serif;font-size:9pt'>"]
    table: list[list[str]] = []

    def flush() -> None:
        if table:
            out.append("<table border='1' cellpadding='3' width='100%'>")
            for n, row in enumerate(table):
                tag = "th" if n == 0 else "td"
                out.append("<tr>" + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in row) + "</tr>")
            out.append("</table>")
            table.clear()

    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(set(c) <= set("-") for c in cells):
                table.append(cells)
            continue
        flush()
        if line.startswith("# "):
            out.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("> "):
            out.append(f"<p><i>{_inline(line[2:])}</i></p>")
        elif line.startswith("- "):
            out.append(f"<p>• {_inline(line[2:])}</p>")
        elif line.strip():
            out.append(f"<p>{_inline(line)}</p>")
    flush()
    out.append("</body></html>")
    return "\n".join(out)


def pack_to_pdf(md: str) -> bytes:
    import pymupdf

    story = pymupdf.Story(html=to_html(md))
    writer = pymupdf.DocumentWriter(buf := __import__("io").BytesIO())
    media = pymupdf.paper_rect("a4-l")
    where = media + (36, 36, -36, -36)
    more = 1
    while more:
        dev = writer.begin_page(media)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    return buf.getvalue()


__all__ = ["build_pack", "pack_to_pdf", "to_html"]
