"""The scorecard (T8.1): the numbers in one place. Two parts.

  1. END TO END: every scenario case of `evals/e2e.py` run through the whole graph, offline, scored against the labelled ground truth (document type, facts, question statuses, contradictions,
     evidence requests, coverage statuses, abstention, settlement exact match, routing, actions, citations), once on rules alone and once with the model's recorded answers (the cases that have a cassette),
     plus the pass/fail of every behaviour check (governance, audit trail, idempotent payment, incremental re-assessment ...) and a per-scenario failure list.
  2. COMPONENTS: the latest committed baseline of every component eval (evals/baselines/*.json) with its headline numbers.

No target numbers are set here: "establish baselines first" (BUILD_PLAN section 13). A baseline is committed with `run_eval.py --write-baseline`, and `tests/unit/test_scorecard.py` fails if a metric drops below it.
"""
from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from evals.e2e import CASES, MODEL_CASES, CaseResult, Context, has_cassette

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "evals" / "baselines"
SCORECARD_BASELINE = BASELINES / "scorecard.json"


def run_cases(workdir: Path | None = None) -> dict[str, CaseResult]:
    ctx = Context.create(workdir or Path(tempfile.mkdtemp(prefix="claimtower-eval-")))
    return {name: fn(ctx) for name, fn in CASES.items() if name not in MODEL_CASES or has_cassette(name)}


GATE = ("S-01", "S-02", "S-03", "S-04", "S-05", "S-01-with-model", "S-02-with-model")


def aggregate(results: dict[str, CaseResult]) -> dict[str, dict[str, list[int]]]:
    """{mode: {metric: [numerator, denominator]}} over the labelled scenario runs. The model was recorded on S-01 and S-02 only, so the rules are also shown on exactly those two."""
    out: dict[str, dict[str, list[int]]] = {}

    def add(mode: str, r: CaseResult) -> None:
        m = out.setdefault(mode, {})
        for k, (n, d) in r.metrics.items():
            cur = m.setdefault(k, [0, 0])
            cur[0] += n
            cur[1] += d

    modeled = {n.removesuffix("-with-model") for n in results if n.endswith("-with-model")}
    for name in GATE:
        r = results.get(name)
        if r is None:
            continue
        add(r.mode, r)
        if name in modeled:
            add("rules on the same scenarios as the model runs", r)
    return out


def failures(results: dict[str, CaseResult]) -> list[str]:
    out: list[str] = []
    for r in results.values():
        out += [f"{r.name}: check failed: {k}" for k, v in r.checks.items() if not v]
        out += [f"{r.name} ({r.mode}): {k} {n}/{d}" for k, (n, d) in r.metrics.items() if d and n < d and k != "nodes_skipped_on_new_evidence"]
    return out


def headline(name: str, e: dict[str, Any]) -> list[tuple[str, str]]:
    """The headline number(s) of one component baseline entry; [] when its layout is not the expected one (the file is still listed)."""
    def pct(x: Any) -> str:
        return f"{x:.0%}" if isinstance(x, (int, float)) else str(x)
    try:
        if name == "classification":
            return [("accuracy, unseen set (fresh2)", pct(e["splits"]["fresh2"]["accuracy"])), ("accuracy, rendered corpus", pct(e["splits"]["corpus"]["accuracy"]))]
        if name == "extraction":
            return [("field score, unseen set (fresh2)", pct(e["splits"]["fresh2"]["field_score"]))]
        if name == "facts":
            return [("ground-truth facts recalled", pct(e["result"]["ground_truth_recall"])), ("citation verbatim rate", pct(e["result"]["citation"]["verbatim_rate"]))]
        if name == "policy_clauses":
            return [("clause F1, corpus", pct(e["splits"]["corpus"]["f1"]))]
        if name == "policy_relations":
            return [("relation recall, fresh", pct(e["splits"]["fresh"]["recall"])), ("relation precision, fresh", pct(e["splits"]["fresh"]["precision"]))]
        if name in ("policy_snapshot", "policy_effective"):
            return [("scenarios passed", f"{e['results']['scenarios']['passed']}/{e['results']['scenarios']['total']}"), ("cases passed", f"{e['results']['cases']['passed']}/{e['results']['cases']['total']}")]
        if name == "questions":
            return [("question recall, fresh", pct(e["splits"]["fresh"]["recall"]))]
        if name == "coverage":
            return [("status accuracy, fresh (rules + model)", pct(e["modes"]["rules + model"]["fresh"]["status_accuracy"])),
                    ("rows established against the label, fresh", str(e["modes"]["rules + model"]["fresh"]["established_against_the_label"]))]
        if name == "settlement_inputs":
            return [("assessed amount correct, fresh (rules + model)", pct(e["modes"]["rules + model"]["fresh"]["assessed_correct"])),
                    ("wrong numbers, fresh", str(e["modes"]["rules + model"]["fresh"]["wrong_numbers"]))]
    except (KeyError, TypeError):
        return []
    return []


def components() -> list[dict[str, Any]]:
    rows = []
    for f in sorted(BASELINES.glob("*.json")):
        if f.name == "scorecard.json":
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        e = d["entries"][-1] if isinstance(d, dict) and "entries" in d else d
        rows.append({"component": f.stem, "label": e.get("label", ""), "date": e.get("date", ""), "model": e.get("model", ""), "headline": headline(f.stem, e)})
    return rows


def build(results: dict[str, CaseResult]) -> dict[str, Any]:
    agg = aggregate(results)
    return {"date": date.today().isoformat(), "cases": {n: {"passed": r.passed, "checks": len(r.checks), "failed_checks": [k for k, v in r.checks.items() if not v]} for n, r in results.items()},
            "end_to_end": agg, "failures": failures(results), "components": components()}


def to_markdown(card: dict[str, Any], results: dict[str, CaseResult]) -> str:
    L = [f"# Scorecard ({card['date']})", "",
         "> Prototype on SYNTHETIC data. Baselines first: no target numbers are set. Every case runs offline; the model's answers are replayed from recorded cassettes.", ""]
    cases = card["cases"]
    L += ["## 1. Behaviour checks (end to end)", "", f"{sum(c['passed'] for c in cases.values())} of {len(cases)} cases pass.", "", "| Case | Mode | Checks | Result |", "|---|---|---|---|"]
    for n, c in cases.items():
        L.append(f"| {n} | {results[n].mode} | {c['checks']} | {'pass' if c['passed'] else 'FAIL: ' + ', '.join(c['failed_checks'])} |")
    L += ["", "## 2. Against the labelled scenarios", ""]
    modes = card["end_to_end"]
    names = sorted({k for m in modes.values() for k in m})
    if names:
        head = list(modes)
        modeled = {n.removesuffix("-with-model") for n in results if n.endswith("-with-model")}
        counts = {m: [n for n in GATE if n in results and ((results[n].mode == m) or (m.startswith("rules on the same") and n in modeled))] for m in head}
        L += ["| Metric | " + " | ".join(f"{m} ({len(counts[m])} scenarios)" for m in head) + " |", "|---|" + "---|" * len(head)]
        for k in names:
            L.append(f"| {k} | " + " | ".join((f"{modes[m][k][0]}/{modes[m][k][1]} ({modes[m][k][0] / modes[m][k][1]:.0%})" if k in modes[m] and modes[m][k][1] else "-") for m in head) + " |")
    L += ["", "Reading the table: rules alone leave facts unverified (no judgement without a model), so question, coverage and routing agreement are low by design; with the model they follow the evidence. The two model rows are only the scenarios that have a recorded run.", ""]
    L += ["## 3. Where the system and the labels differ (per scenario)", ""]
    L += [f"- {f}" for f in card["failures"]] or ["- none"]
    L += ["", "## 4. Component baselines (latest committed)", "", "| Component | Baseline | Date | Model | Headline |", "|---|---|---|---|---|"]
    for c in card["components"]:
        L.append(f"| {c['component']} | {c['label'][:60]} | {c['date']} | {c['model']} | " + "; ".join(f"{k}: {v}" for k, v in c["headline"]) + " |")
    skipped = results.get("S-01-late-evidence")
    if skipped is not None and skipped.metrics:
        n, d = skipped.metrics["nodes_skipped_on_new_evidence"]
        L += ["", f"Incremental efficiency: {n} of {d} nodes skipped when late evidence arrived (S-01).", ""]
    return "\n".join(L) + "\n"


def regressions(card: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Where the current scorecard is worse than the committed baseline."""
    out: list[str] = []
    for n, c in baseline["cases"].items():
        if c["passed"] and not card["cases"].get(n, {}).get("passed"):
            out.append(f"case {n} used to pass and does not")
    for mode, metrics in baseline["end_to_end"].items():
        for k, (bn, bd) in metrics.items():
            cur = card["end_to_end"].get(mode, {}).get(k)
            if cur is None or (bd and cur[1] and cur[0] / cur[1] < bn / bd - 1e-9):
                out.append(f"{mode}: {k} fell from {bn}/{bd} to {cur[0] if cur else '-'}/{cur[1] if cur else '-'}")
    return out


def by_mode_counts(results: dict[str, CaseResult]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for r in results.values():
        c[r.mode] += 1
    return dict(c)


__all__ = ["aggregate", "build", "components", "failures", "regressions", "run_cases", "to_markdown"]
