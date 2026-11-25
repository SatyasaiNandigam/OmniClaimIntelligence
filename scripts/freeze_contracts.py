"""Contract freeze (BUILD_PLAN: 'Schema changes after M0 need Tech-lead approval').

    python scripts/freeze_contracts.py --check
        Fails (exit 1) if any exported contract file differs from docs/contracts/CONTRACT_LOCK.json. Run by CI via pytest.
    python scripts/freeze_contracts.py --approve "reason" --by "name" [--bump major|minor|patch]
        Deliberately accepts the current contracts: re-exports them, rewrites the lock, bumps contract_version and appends an entry
        to docs/contracts/CHANGELOG.md. This is the 'tech-lead approval' step - the person running it owns the change.

Locked files: every docs/contracts/*.schema.json, ownership.json (who may write which field) and protocols.json (method signatures of the
agent, adapter and section-11 interface protocols).
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "docs" / "contracts"
LOCK = DIR / "CONTRACT_LOCK.json"
CHANGELOG = DIR / "CHANGELOG.md"


def contract_files() -> list[Path]:
    return sorted([*DIR.glob("*.schema.json"), DIR / "ownership.json", DIR / "protocols.json"], key=lambda p: p.name)


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()   # stable across Windows/Unix checkouts


def current() -> dict[str, str]:
    return {p.name: digest(p) for p in contract_files() if p.exists()}


def diff(locked: dict[str, str], now: dict[str, str]) -> dict[str, list[str]]:
    return {"changed": sorted(k for k in now if k in locked and now[k] != locked[k]),
            "added": sorted(set(now) - set(locked)), "removed": sorted(set(locked) - set(now))}


def check() -> int:
    if not LOCK.exists():
        print("No CONTRACT_LOCK.json - freeze first: --approve 'baseline' --by 'name'")
        return 1
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    d = diff(lock["files"], current())
    if any(d.values()):
        print(f"CONTRACTS CHANGED since freeze v{lock['contract_version']} ({lock['frozen_on']}):")
        for k, names in d.items():
            if names:
                print(f"  {k}: {', '.join(names)}")
        print("If intended, a tech lead approves it:  python scripts/freeze_contracts.py --approve \"<reason>\" --by \"<name>\"")
        return 1
    print(f"Contracts match freeze v{lock['contract_version']}.")
    return 0


def bump(version: str, how: str) -> str:
    major, minor, patch = (int(x) for x in version.split("."))
    return {"major": f"{major + 1}.0.0", "minor": f"{major}.{minor + 1}.0", "patch": f"{major}.{minor}.{patch + 1}"}[how]


def approve(reason: str, by: str, how: str) -> int:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "export_schemas.py")], check=True)
    now = current()
    old = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else None
    d = diff(old["files"], now) if old else {"changed": [], "added": sorted(now), "removed": []}
    if old and not any(d.values()):
        print("Nothing to approve: contracts already match the lock.")
        return 0
    version = bump(old["contract_version"], how) if old else "1.0.0"
    LOCK.write_text(json.dumps({"contract_version": version, "frozen_on": date.today().isoformat(), "approved_by": by, "files": now},
                               indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    lines = [f"## v{version} - {date.today().isoformat()} - approved by {by}", "", f"**Reason:** {reason}", ""]
    for k, names in d.items():
        if names:
            lines.append(f"- {k}: {', '.join(names)}")
    entry = "\n".join(lines) + "\n\n"
    header = "# Contract changelog\n\nEvery change to a frozen contract is recorded here by `scripts/freeze_contracts.py --approve`.\n\n"
    existing = CHANGELOG.read_text(encoding="utf-8") if CHANGELOG.exists() else header
    CHANGELOG.write_text(existing.replace(header, header + entry, 1) if existing.startswith(header) else header + entry + existing,
                         encoding="utf-8", newline="\n")
    print(f"Froze contracts at v{version} ({len(now)} files). See {CHANGELOG}.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--approve", metavar="REASON")
    ap.add_argument("--by", help="who approves (required with --approve)")
    ap.add_argument("--bump", choices=["major", "minor", "patch"], default="minor")
    a = ap.parse_args()
    if a.approve:
        if not a.by:
            ap.error("--approve requires --by")
        return approve(a.approve, a.by, a.bump)
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
