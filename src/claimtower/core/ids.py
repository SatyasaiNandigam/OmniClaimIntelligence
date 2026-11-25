"""ID helpers."""
import uuid


def new_id(prefix: str) -> str:
    """Short, prefixed unique id, e.g. new_id("EVT") -> 'EVT-3f9a1c2b7d4e'."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"
