"""Drafts (T6.4): the text of the messages the planner proposes. Templates from the pack; a model may reword, and the reworded text is checked.

    writer = DraftWriter(templates, forbidden_patterns, rewriter=None)
    draft = writer.write(planned, state)                     # -> Draft (status "draft": a person reviews it; nothing is sent)

A reworded draft is accepted only if ALL hold: every requested item is still in it (verbatim, ignoring case and spacing), no rupee amount appears that the template did not
have, no wording that states a coverage conclusion (the pack's forbidden patterns), and it is not much longer than the template. Otherwise the template text is used.
The status message never states an outcome or an amount.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from claimtower.actions.planner import ActionTemplates, PlannedAction
from claimtower.core.schemas import ClaimState, Draft

AMOUNT = re.compile(r"(?:INR|Rs\.?|₹)\s*[\d,]+")
MAX_GROWTH = 2.0


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def draft_id(key: str) -> str:
    return "DRF-" + re.sub(r"[^A-Za-z0-9]+", "-", key).strip("-")


class DraftWriter:
    def __init__(self, templates: ActionTemplates, forbidden_patterns: Sequence[str] = (), rewriter: Callable[[str, str, list[str]], str | None] | None = None) -> None:
        self.t = templates
        self._forbidden = [re.compile(p, re.IGNORECASE) for p in forbidden_patterns]
        self.rewriter = rewriter
        self.dropped: list[str] = []

    # ------------------------------------------------------------------ checks
    def lint(self, text: str) -> list[str]:
        return [f"states a conclusion: {m.group(0)!r}" for rx in self._forbidden for m in [rx.search(text)] if m]

    def acceptable(self, reworded: str, template_text: str, items: Sequence[str]) -> list[str]:
        problems = [f"item missing: {i[:40]}" for i in items if _norm(i) not in _norm(reworded)]
        problems += self.lint(reworded)
        extra = {_norm(a) for a in AMOUNT.findall(reworded)} - {_norm(a) for a in AMOUNT.findall(template_text)}
        problems += [f"new amount: {a}" for a in sorted(extra)]
        if len(reworded) > MAX_GROWTH * len(template_text) + 200:
            problems.append("much longer than the template")
        return problems

    # ------------------------------------------------------------------ writing
    def write(self, a: PlannedAction, state: ClaimState) -> Draft:
        h = state.claim_header
        loss = f" (loss on {h.loss_date:%d %B %Y})" if h and h.loss_date else ""
        tpl = self.t.drafts[a.draft_kind or "information_request"]
        fmt = {"claim_id": state.claim_id, "loss": loss}
        if a.draft_kind == "information_request":
            bullets = "\n".join(f"- {i}" for i in a.items)
            body = f"{tpl['intro'].format(**fmt)}\n{bullets}\n{tpl['outro']}"
        elif a.draft_kind == "status_message":
            roles = {m.minimum_evidence_request.addressee_role for m in state.missing_evidence}
            waiting = tpl["waiting_for_you"] if "claimant" in roles else tpl["waiting_for_others"] if roles else tpl["nothing_waiting"]
            body = tpl["body"].format(waiting=waiting, **fmt)
        else:
            body = tpl["body"].format(**fmt)
        body = self._reword(a, body)
        return Draft(draft_id=draft_id(a.key), kind=a.draft_kind or "other", addressee_role=a.party, subject=tpl["subject"].format(**fmt), body=body, status="draft")  # type: ignore[arg-type]

    def _reword(self, a: PlannedAction, template_text: str) -> str:
        if self.rewriter is None:
            return template_text
        reworded = self.rewriter(template_text, a.draft_kind or "", a.items)
        if not reworded:
            return template_text
        problems = self.acceptable(reworded, template_text, a.items)
        if problems:
            self.dropped.append(f"{a.key}: {'; '.join(problems)}")
            return template_text
        return reworded.strip()


__all__ = ["DraftWriter", "draft_id"]
