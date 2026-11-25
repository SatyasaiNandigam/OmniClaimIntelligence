"""Export the contract surface to docs/contracts/. `--check` fails if any file is out of date (CI drift check).

Exports:
  <Model>.schema.json   JSON Schema of every model in claimtower.core.schemas.EXPORTED, plus StatePatch (the agent protocol payload)
  ownership.json        which agent may write which ClaimState field (core.state.OWNERSHIP)
  protocols.json        method signatures of the agent protocol, the three adapter protocols and the section-11 interfaces
"""
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claimtower.agents.base import Agent  # noqa: E402
from claimtower.core import interfaces  # noqa: E402
from claimtower.core.schemas import EXPORTED, ClaimState  # noqa: E402
from claimtower.core.state import OWNERSHIP, StatePatch  # noqa: E402
from claimtower.integrations import base as adapters  # noqa: E402

OUT = ROOT / "docs" / "contracts"
PROTOCOLS = [Agent, adapters.PASClient, adapters.ClaimsClient, adapters.ReconClient, interfaces.FactExtractor,
             interfaces.ImageAnalyser, interfaces.PolicyService, interfaces.SettlementEngine, interfaces.Router, interfaces.AuditLog]


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def protocol_signatures() -> dict:
    out = {}
    for proto in PROTOCOLS:
        methods = {name: str(inspect.signature(fn)) for name, fn in inspect.getmembers(proto, inspect.isfunction) if not name.startswith("_")}
        attrs = {k: str(v) for k, v in getattr(proto, "__annotations__", {}).items()}
        out[proto.__name__] = {"module": proto.__module__, "attributes": attrs, "methods": methods}
    return out


def render() -> dict[Path, str]:
    files = {OUT / f"{m.__name__}.schema.json": _dump(m.model_json_schema()) for m in [*EXPORTED, StatePatch]}
    owned = sorted(f for fields in OWNERSHIP.values() for f in fields)
    files[OUT / "ownership.json"] = _dump({
        "agents": {a: sorted(f) for a, f in OWNERSHIP.items()},
        "system_fields": sorted(set(ClaimState.model_fields) - set(owned)),
    })
    files[OUT / "protocols.json"] = _dump(protocol_signatures())
    return files


def main() -> int:
    files = render()
    if "--check" in sys.argv:
        stale = [p.name for p, text in files.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
        extra = sorted(p.name for p in OUT.glob("*.schema.json") if p not in files)
        if stale or extra:
            print("Contract drift - run `python scripts/export_schemas.py`:", ", ".join(stale), "| orphaned:", ", ".join(extra) or "-")
            return 1
        print("Contracts up to date.")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for p, text in files.items():
        p.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {len(files)} contract files to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
