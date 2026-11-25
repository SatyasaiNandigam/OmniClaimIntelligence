"""Start the Streamlit UI (it talks to the API only).

    python scripts/run_api.py          # terminal 1: the API
    python scripts/run_ui.py           # terminal 2: the UI at http://localhost:8501

    python scripts/run_ui.py --start-api         # start the API in the background too (stops with the UI)
    python scripts/run_ui.py --api http://127.0.0.1:8080 --port 8502
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--start-api", action="store_true", help="also start the API (rules only; use run_api.py --llm for the model)")
    ap.add_argument("--llm", action="store_true", help="with --start-api: the Gap agent also uses the model")
    a = ap.parse_args()
    env = {**os.environ, "CLAIMTOWER_API_URL": a.api, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    api_proc = None
    if a.start_api:
        port = a.api.rsplit(":", 1)[-1]
        cmd = [sys.executable, str(ROOT / "scripts" / "run_api.py"), "--port", port] + (["--llm"] if a.llm else [])
        api_proc = subprocess.Popen(cmd, env=env)
        time.sleep(4)                                              # the API loads the policy wordings before it answers
    try:
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "src" / "claimtower" / "ui" / "app.py"), "--server.port", str(a.port),
                                "--browser.gatherUsageStats", "false"], env=env)
    finally:
        if api_proc is not None:
            api_proc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
