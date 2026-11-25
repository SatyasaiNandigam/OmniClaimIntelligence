"""T1.5 - the SAMPLE routing rules reproduce the expert tier + reason codes labelled in ground truth,
and every number in the sample rule files carries an inline '# SAMPLE' comment.

The evaluator below is a minimal TEST helper that mirrors the evaluation order documented in routing_rules.yaml.
The production evaluator is T5.6 (routing/tiering.py) and should replace it.
"""
import re
from pathlib import Path

import pytest

from claimtower.core.config import Condition, load_settings
from evals.ground_truth import load_all_ground_truth

ROOT = Path(__file__).resolve().parents[2]
RULES = load_settings().routing
GT = load_all_ground_truth()


def holds(c: Condition, feats: dict) -> bool:
    v = feats[c.feature]
    ok = True
    if c.is_ is not None:
        ok &= v == c.is_
    if c.in_ is not None:
        ok &= v in c.in_
    for op, test in (("gt", lambda a, b: a > b), ("gte", lambda a, b: a >= b), ("lt", lambda a, b: a < b), ("lte", lambda a, b: a <= b)):
        if getattr(c, op) is not None:
            ok &= test(v, getattr(c, op))
    return bool(ok)


def route(feats: dict) -> tuple[str, list[str], str]:
    stages = [("complex", RULES.mandatory_referral.any_of, "any"), ("complex", RULES.tiers.complex.any_of, "any"),
              ("simple", RULES.tiers.simple.all_of, "all"), ("medium", RULES.tiers.medium.any_of, "any")]
    for tier, conds, mode in stages:
        hits = [c for c in conds if holds(c, feats)]
        if (mode == "any" and hits) or (mode == "all" and len(hits) == len(conds)):
            return tier, [c.code for c in hits], RULES.approver_roles[tier]
    return "complex", ["DEFAULT_COMPLEX"], RULES.approver_roles["complex"]


@pytest.mark.parametrize("sid", sorted(GT))
def test_sample_rules_reproduce_ground_truth_routing(sid):
    gt = GT[sid]
    assert set(gt.routing_features) == set(RULES.features), "ground-truth features must cover every declared feature"
    tier, codes, role = route(gt.routing_features)
    assert (tier, codes, role) == (gt.routing.tier, gt.routing.reason_codes, gt.routing.required_approver_role), sid
    assert gt.routing.rules_version == RULES.rules_version


def test_exposure_alone_cannot_hide_a_high_contradiction():
    """Regression for the S-01 finding: a mid-band exposure must not downgrade a high-severity contradiction."""
    feats = dict(GT["S-01"].routing_features)
    assert route(feats)[0] == "complex"
    feats.update(contradiction_severity="none", evidence_completeness="complete", coverage_certainty="resolved")
    assert route(feats)[0] == "medium"      # only the exposure band remains


def test_mandatory_referral_beats_everything():
    feats = dict(GT["S-02"].routing_features, duplicate_status="multiple_linked")
    assert route(feats) == ("complex", ["DUPLICATE_MULTIPLE_LINKED"], "senior_adjuster")


@pytest.mark.parametrize("path", ["configs/routing_rules.yaml", "packs/property_water_damage/calculation/settlement_v1.yaml",
                                  "packs/property_water_damage/calculation/settlement_v2_concept.yaml",
                                  "packs/property_water_damage/document_types.yaml"])
def test_every_number_in_sample_rule_files_is_marked_sample(path):
    missing = []
    for n, raw in enumerate((ROOT / path).read_text(encoding="utf-8").splitlines(), start=1):
        code = raw.split("#", 1)[0]
        code = re.sub(r'"[^"]*"|\'[^\']*\'', "", code)           # ignore quoted strings such as "sample-2"
        if re.search(r"(?<![\w-])\d", code) and "# SAMPLE" not in raw:
            missing.append(f"{path}:{n}: {raw.strip()}")
    assert not missing, "numbers without '# SAMPLE':\n" + "\n".join(missing)
