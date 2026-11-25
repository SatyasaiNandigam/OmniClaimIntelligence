"""T5.4 property-based tests: invariants that must hold for ANY amounts, checked against an independent oracle formula."""
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from claimtower.core.schemas import SettlementInputs
from claimtower.settlement.engine import calculate
from claimtower.settlement.rules_loader import load_rules

D = Decimal
CALC = Path(__file__).resolve().parents[2] / "packs/property_water_damage/calculation"
cents = st.integers(min_value=0, max_value=10**10).map(lambda c: D(c).scaleb(-2))      # exact 2-decimal amounts up to 100,000,000.00
maybe = st.none() | cents
PROFILE = settings(max_examples=300, deadline=None)


def rules(file: str, places: int):
    raw = yaml.safe_load((CALC / file).read_text(encoding="utf-8"))
    raw["rounding"] = {"mode": "half_up", "places": places}
    return load_rules(raw)


V1 = {p: rules("settlement_v1.yaml", p) for p in (0, 2)}
V2 = {p: rules("settlement_v2_concept.yaml", p) for p in (0, 2)}


def rnd(x: Decimal, places: int) -> Decimal:
    return x.quantize(D(1).scaleb(-places), rounding=ROUND_HALF_UP)


def oracle_v1(a, n, d, s, lim, p, places):          # (loss - non-covered - deductible) -> sublimit -> limit -> - prior -> floor 0
    x = a - n - d
    for cap in (s, lim):
        x = x if cap is None else min(x, cap)
    return rnd(max(D(0), x - p), places)


def oracle_v2(a, n, d, s, lim, p, places):          # concept order: min(eligible loss, sublimit, limit) - deductible - prior -> floor 0
    x = a - n
    for cap in (s, lim):
        x = x if cap is None else min(x, cap)
    return rnd(max(D(0), x - d - p), places)


def inputs(a, n, d, s, lim, p) -> SettlementInputs:
    return SettlementInputs(assessed_covered_loss=a, non_covered_items_total=n, applicable_deductible=d, water_damage_sublimit=s,
                            remaining_policy_limit=lim, prior_payments_total=p)


def both():
    return ((V1, oracle_v1), (V2, oracle_v2))


@PROFILE
@given(a=cents, n=cents, d=cents, s=maybe, lim=cents, p=cents, places=st.sampled_from([0, 2]))
def test_engine_equals_independent_oracle_for_both_rule_orders(a, n, d, s, lim, p, places):
    for table, oracle in both():
        assert calculate(table[places], inputs(a, n, d, s, lim, p)).payable == oracle(a, n, d, s, lim, p, places)


@PROFILE
@given(a=cents, n=cents, d=cents, s=maybe, lim=cents, p=cents, places=st.sampled_from([0, 2]))
def test_payable_is_never_negative_and_never_exceeds_remaining_limit_sublimit_or_loss(a, n, d, s, lim, p, places):
    for table, _ in both():
        pay = calculate(table[places], inputs(a, n, d, s, lim, p)).payable
        assert pay >= 0
        assert pay <= rnd(lim, places)
        assert s is None or pay <= rnd(s, places)
        assert pay <= rnd(a, places)
        assert pay.as_tuple().exponent == -places


@PROFILE
@given(a=cents, n=cents, d=cents, s=maybe, lim=cents, p=cents, bump=cents)
def test_monotonicity(a, n, d, s, lim, p, bump):
    for table, _ in both():
        r = table[2]
        base = calculate(r, inputs(a, n, d, s, lim, p)).payable
        assert calculate(r, inputs(a, n, d + bump, s, lim, p)).payable <= base          # higher deductible never pays more
        assert calculate(r, inputs(a, n + bump, d, s, lim, p)).payable <= base          # more non-covered never pays more
        assert calculate(r, inputs(a, n, d, s, lim, p + bump)).payable <= base          # earlier payments never increase it
        assert calculate(r, inputs(a + bump, n, d, s, lim, p)).payable >= base          # a larger loss never pays less
        assert calculate(r, inputs(a, n, d, s, lim + bump, p)).payable >= base          # a larger limit never pays less


@PROFILE
@given(a=cents, n=cents, d=cents, s=maybe, lim=cents, p=cents)
def test_trace_is_a_consistent_chain_ending_at_the_payable(a, n, d, s, lim, p):
    for table, _ in both():
        r = calculate(table[0], inputs(a, n, d, s, lim, p))
        assert r.steps[0].amount_before == 0
        for prev, cur in zip(r.steps, r.steps[1:]):
            assert cur.amount_before == prev.amount_after
        assert r.steps[-1].rule_id == "rounding" and r.steps[-1].amount_after == r.payable
        assert [x.rule_id for x in r.steps[:-1]] == [st_.id for st_ in table[0].steps]


@PROFILE
@given(a=maybe, n=maybe, d=maybe, lim=maybe, p=maybe, s=maybe)
def test_any_missing_required_input_means_cannot_calculate_with_exactly_those_inputs(a, n, d, lim, p, s):
    r = calculate(V1[0], inputs(a, n, d, s, lim, p))
    required = {"assessed_covered_loss": a, "non_covered_items_total": n, "applicable_deductible": d,
                "remaining_policy_limit": lim, "prior_payments_total": p}
    missing = sorted(k for k, v in required.items() if v is None)
    if missing:
        assert r.status == "cannot_calculate" and r.missing_inputs == missing and r.payable is None
    else:
        assert r.status == "calculated" and r.payable is not None


@PROFILE
@given(a=cents, n=cents, d=cents, s=maybe, lim=cents, p=cents)
def test_unavailable_input_always_blocks_even_when_everything_else_is_known(a, n, d, s, lim, p):
    i = inputs(a, n, d, s, lim, p).model_copy(update={"unavailable_inputs": ["water_damage_sublimit"]})
    r = calculate(V1[0], i)
    assert r.status == "cannot_calculate" and r.missing_inputs == ["water_damage_sublimit"]
