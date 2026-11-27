"""What every UI page shares: the API client for this session, how an API refusal is shown, and the words used for phases and question states."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from claimtower.ui.client import DEFAULT_URL, ApiClient, ApiError

STATUS_ICON = {"established": "✅", "partially_established": "🟡", "possible": "🟠", "unverified": "⚪", "missing": "🔴", "requires_validation": "🟣"}
PHASE_TEXT = {"awaiting_human_decision": "⏸ waiting for a human decision", "completed": "✔ decided", "halted": "⛔ halted (an agent failed)", "not_started": "not started"}
PHASE_SHORT = {"awaiting_human_decision": "⏸ awaiting decision", "completed": "✔ decided", "halted": "⛔ halted", "not_started": "not started"}


def client() -> ApiClient:
    injected = st.session_state.get("_client")                      # tests inject a client wired to an in-process API
    if injected is not None:
        return injected
    return ApiClient(st.session_state.get("api_url", DEFAULT_URL))


def show_error(e: ApiError) -> None:
    st.error(f"{'Cannot reach the API' if e.status == 0 else f'The API said no ({e.status})'}: {e.message}")


def df(rows: list[dict], columns: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns or [])
