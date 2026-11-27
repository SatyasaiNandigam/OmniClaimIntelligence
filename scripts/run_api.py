"""Start the API (real Evidence, Policy and Gap agents; mocks for the rest; SYNTHETIC data).

    python scripts/run_api.py                       # http://127.0.0.1:8000  (interactive docs at /docs, the contract at /openapi.json)
    python scripts/run_api.py --port 8080 --llm     # the Evidence, Gap, Coverage and Settlement agents also use the model (needs LLM_API_KEY in .env)
    python scripts/run_api.py --store var/store     # keep uploaded documents and the audit trail between runs (claims and events are in memory either way)
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--store")
    a = ap.parse_args()
    if a.llm:
        os.environ["CLAIMTOWER_LLM"] = "1"
    if a.store:
        os.environ["CLAIMTOWER_STORE"] = str(Path(a.store).resolve())
    import uvicorn

    uvicorn.run("claimtower.api.main:build_default_app", factory=True, host=a.host, port=a.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
