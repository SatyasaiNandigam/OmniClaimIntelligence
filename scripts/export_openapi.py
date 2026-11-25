"""Write the API contract to docs/contracts/openapi.json (the file the UI and any integration code against).

    python scripts/export_openapi.py            # rewrite it
    python scripts/export_openapi.py --check    # fail if the committed file differs from the service (also run by tests/api/test_api.py)
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claimtower.api.app import create_app  # noqa: E402
from claimtower.composition import build_registry  # noqa: E402
from claimtower.core.config import load_settings  # noqa: E402
from claimtower.ingestion.store import DocumentStore  # noqa: E402
from claimtower.orchestration.graph import ClaimRunner  # noqa: E402

TARGET = ROOT / "docs" / "contracts" / "openapi.json"


def current() -> dict:
    store = DocumentStore(Path(tempfile.mkdtemp()) / "store")
    runner = ClaimRunner(build_registry(load_settings(ROOT), document_store=store))
    return create_app(runner, store).openapi()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    spec = current()
    if a.check:
        ok = TARGET.exists() and json.loads(TARGET.read_text(encoding="utf-8")) == spec
        print("OpenAPI contract is up to date." if ok else "OpenAPI contract is out of date: run python scripts/export_openapi.py")
        return 0 if ok else 1
    TARGET.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {TARGET.relative_to(ROOT)} ({len(spec['paths'])} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
