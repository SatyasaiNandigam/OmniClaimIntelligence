"""T5.3 / T5.4 - deterministic settlement engine: worked examples, edge-case matrix, rule invariants, hygiene."""
import ast
import inspect
import json
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from claimtower.core.config import CalculationRules, load_settings
from claimtower.core.errors import ConfigError
from claimtower.core.schemas import SettlementInputs
from claimtower.settlement import money
from claimtower.settlement.engine import InvalidAmount, SettlementCalculator, calculate, calculate_values
from claimtower.settlement.rules_loader import load_rules, rules_by_version

ROOT = Path(__file__).resolve().parents[2]
PACK = load_settings().pack
RULES = rules_by_version(PACK.calculation)
V1, V2 = RULES["sample-1.0"], RULES["sample-concept-1.0"]
D = Decimal


def inp(**kw) -> SettlementInputs:
    base = dict(assessed_covered_loss=D(200000), non_covered_items_total=D(15000), applicable_deductible=D(10000),
                remaining_policy_limit=D(150000), prior_payments_total=D(0))
    return SettlementInputs(**{**base, **kw})


def payable(rules, **kw) -> Decimal:
    r = calculate(rules, inp(**kw))
    assert r.status == "calculated", r
    return r.payable


def raw_rules(**changes):
    d = yaml.safe_load((ROOT / "packs/property_water_damage/calculation/settlement_v1.yaml").read_text(encoding="utf-8"))
    d.update(changes)
    return d


# ------------------------------------------------------------------ worked examples (T5.3 done-when)
def test_guidelines_pdf_example_200k_capped_to_150k_with_full_trace():
    r = calculate(V1, inp())
    assert r.status == "calculated" and r.payable == D(150000) and r.rule_version == "sample-1.0" and r.currency == "INR"
    assert r.assessed_loss == D(200000)
    trace = [(s.rule_id, s.input_ref, s.amount_before, s.amount_after) for s in r.steps]
    assert trace == [
        ("start", "assessed_covered_loss", D(0), D(200000)),
        ("less_non_covered", "non_covered_items_total", D(200000), D(185000)),
        ("less_deductible", "applicable_deductible", D(185000), D(175000)),
        ("apply_sublimit", None, D(175000), D(175000)),                 # optional, absent -> did not apply
        ("cap_remaining_limit", "remaining_policy_limit", D(175000), D(150000)),
        ("less_prior_payments", "prior_payments_total", D(150000), D(150000)),
        ("floor_zero", None, D(150000), D(150000)),
        ("rounding", None, D(150000), D(150000)),
    ]


def test_concept_document_order_gives_a_different_answer_from_the_same_inputs():
    assert payable(V1) == D(150000) and payable(V2) == D(140000)       # min(185000, 150000) - 10000


def test_calculator_selects_rule_version_and_rejects_unknown():
    calc = SettlementCalculator(RULES)
    assert calc.versions == ["sample-1.0", "sample-concept-1.0"]
    assert calc.compute_settlement(inp(), "sample-1.0").payable == D(150000)
    assert calc.compute_settlement(inp(), "sample-concept-1.0").payable == D(140000)
    with pytest.raises(ConfigError, match="unknown settlement rules version 'nope'"):
        calc.compute_settlement(inp(), "nope")


def test_calculator_matches_the_frozen_engine_protocol():
    protocols = json.loads((ROOT / "docs/contracts/protocols.json").read_text(encoding="utf-8"))
    frozen = protocols["SettlementEngine"]["methods"]["compute_settlement"]
    sig = str(inspect.signature(SettlementCalculator.compute_settlement))
    assert sig.replace("'", "") == frozen.replace("'", "")


# ------------------------------------------------------------------ edge cases (T5.4)
def test_zero_amounts():
    assert payable(V1, assessed_covered_loss=D(0), non_covered_items_total=D(0)) == 0
    assert payable(V1, applicable_deductible=D(0), non_covered_items_total=D(0)) == D(150000)      # 200000 capped
    assert payable(V1, assessed_covered_loss=D(50000), non_covered_items_total=D(0), applicable_deductible=D(0)) == D(50000)


@pytest.mark.parametrize("name", ["assessed_covered_loss", "non_covered_items_total", "applicable_deductible",
                                  "remaining_policy_limit", "prior_payments_total", "water_damage_sublimit"])
def test_negative_amounts_are_rejected_not_clamped(name):
    with pytest.raises(InvalidAmount, match=f"{name}.*negative"):
        calculate(V1, inp(**{name: D(-1)}))


def test_floats_and_bad_values_are_rejected_at_the_boundary():
    for bad in (0.1, True, "abc", D("NaN"), D("Infinity"), None):
        with pytest.raises(InvalidAmount):
            money.to_money(bad, "x")
    with pytest.raises(InvalidAmount, match="float"):
        calculate_values(V1, {"assessed_covered_loss": 1.5})                        # type: ignore[dict-item]
    assert money.to_money("10.25") == D("10.25") and money.to_money(7) == D(7)


def test_deductible_larger_than_loss_pays_zero_not_negative():
    assert payable(V1, assessed_covered_loss=D(8000), non_covered_items_total=D(0), applicable_deductible=D(10000)) == 0


def test_limit_exhaustion():
    assert payable(V1, remaining_policy_limit=D(0)) == 0                              # limit fully used up
    assert payable(V1, remaining_policy_limit=D(1)) == 1
    assert payable(V1, remaining_policy_limit=D(10**9)) == D(175000)                  # limit not binding


def test_sublimit_binds_only_when_lower():
    assert payable(V1, water_damage_sublimit=D(100000)) == D(100000)
    assert payable(V1, water_damage_sublimit=D(400000)) == D(150000)                  # remaining limit binds instead
    r = calculate(V1, inp(water_damage_sublimit=D(100000)))
    step = next(s for s in r.steps if s.rule_id == "apply_sublimit")
    assert (step.input_ref, step.amount_before, step.amount_after) == ("water_damage_sublimit", D(175000), D(100000))


def test_prior_payments_reduce_the_payable_and_cannot_make_it_negative():
    assert payable(V1, prior_payments_total=D(40000)) == D(110000)                    # 150000 - 40000
    assert payable(V1, prior_payments_total=D(150000)) == 0                           # limit already paid out
    assert payable(V1, prior_payments_total=D(500000)) == 0


def test_deductible_is_applied_once_and_repeat_calls_give_the_same_answer():
    i = inp(remaining_policy_limit=D(10**9))
    first, second = calculate(V1, i), calculate(V1, i)
    assert first == second and first.payable == D(175000)                              # 200000 - 15000 - 10000, once
    assert i == inp(remaining_policy_limit=D(10**9)), "inputs must not be mutated"
    with pytest.raises(ConfigError, match="more than one step"):                       # a rule file cannot deduct it twice
        load_rules(raw_rules(steps=[*raw_rules()["steps"][:3], {"id": "again", "op": "subtract", "from": "applicable_deductible"},
                                    *raw_rules()["steps"][3:]]))


@pytest.mark.parametrize("amount, places, expected", [
    ("100.4", 0, "100"), ("100.5", 0, "101"), ("0.5", 0, "1"), ("0.49", 0, "0"), ("100.005", 2, "100.01"), ("100.004", 2, "100"),
])
def test_rounding_is_half_up_and_traced(amount, places, expected):
    rules = load_rules(raw_rules(rounding={"mode": "half_up", "places": places}))
    r = calculate(rules, inp(assessed_covered_loss=D(amount), non_covered_items_total=D(0), applicable_deductible=D(0)))
    assert r.payable == D(expected) and r.payable.as_tuple().exponent == -places
    assert (r.steps[-1].rule_id, r.steps[-1].amount_before, r.steps[-1].amount_after) == ("rounding", D(amount), D(expected))


def test_rounding_happens_once_at_the_end_not_per_step():
    rules = load_rules(raw_rules(rounding={"mode": "half_up", "places": 0}))
    r = calculate(rules, inp(assessed_covered_loss=D("100.4"), non_covered_items_total=D("0.4"), applicable_deductible=D("0.4")))
    assert r.payable == D(100) and r.steps[2].amount_after == D("99.6")               # exact intermediates, one final rounding


def test_depreciation_is_expressible_in_rules_without_code_changes(monkeypatch):
    from claimtower.core import config
    monkeypatch.setattr(config, "SETTLEMENT_INPUTS", config.SETTLEMENT_INPUTS | {"depreciation_total"})
    steps = raw_rules()["steps"]
    steps.insert(2, {"id": "less_depreciation", "op": "subtract", "from": "depreciation_total"})
    rules = load_rules(raw_rules(version="dep-1", steps=steps))
    values = dict(assessed_covered_loss=D(100000), non_covered_items_total=D(0), applicable_deductible=D(10000),
                  remaining_policy_limit=D(10**9), prior_payments_total=D(0), depreciation_total=D(20000))
    assert calculate_values(rules, values).payable == D(70000)
    assert calculate_values(rules, {**values, "depreciation_total": None}).missing_inputs == ["depreciation_total"]
    # the shipped inputs also cover it: the agent can fold depreciation into non-covered items (docs/assumptions.md Q-S3)
    assert payable(V1, assessed_covered_loss=D(100000), non_covered_items_total=D(20000), applicable_deductible=D(10000),
                   remaining_policy_limit=D(10**9)) == D(70000)


def test_cancelled_policy_without_a_determinable_limit_cannot_be_calculated_but_reinstated_can():
    cancelled = calculate(V1, inp(remaining_policy_limit=None, unavailable_inputs=["remaining_policy_limit"]))
    assert cancelled.status == "cannot_calculate" and cancelled.missing_inputs == ["remaining_policy_limit"] and cancelled.payable is None
    reinstated = calculate(V1, inp(remaining_policy_limit=D(150000)))                 # limit restored on reinstatement
    assert reinstated.status == "calculated" and reinstated.payable == D(150000)


# ------------------------------------------------------------------ cannot_calculate (never fabricate a number)
def test_missing_inputs_are_listed_sorted_and_no_amount_is_returned():
    r = calculate(V1, SettlementInputs(applicable_deductible=D(10000), remaining_policy_limit=D(150000)))
    assert r.status == "cannot_calculate" and r.payable is None and r.steps == []
    assert r.missing_inputs == ["assessed_covered_loss", "non_covered_items_total", "prior_payments_total"]
    assert r.rule_version == "sample-1.0"


def test_unavailable_sublimit_blocks_calculation_even_though_the_step_is_optional():
    """S-05: the schedule references END-01 but its wording is missing - the payable must not silently ignore the sublimit."""
    r = calculate(V1, inp(unavailable_inputs=["water_damage_sublimit"]))
    assert r.status == "cannot_calculate" and r.missing_inputs == ["water_damage_sublimit"] and r.payable is None
    assert calculate(V1, inp()).status == "calculated", "absent optional sublimit with nothing unavailable still calculates"


def test_cannot_calculate_still_reports_the_assessed_loss_when_known():
    r = calculate(V1, inp(applicable_deductible=None))
    assert r.status == "cannot_calculate" and r.assessed_loss == D(200000) and r.missing_inputs == ["applicable_deductible"]


# ------------------------------------------------------------------ determinism
def test_identical_inputs_give_identical_output_regardless_of_dict_order():
    values = dict(assessed_covered_loss=D(200000), non_covered_items_total=D(15000), applicable_deductible=D(10000),
                  remaining_policy_limit=D(150000), prior_payments_total=D(0), water_damage_sublimit=None)
    reference = calculate_values(V1, values)
    for _ in range(25):
        assert calculate_values(V1, dict(reversed(list(values.items())))) == reference
    assert reference.model_dump_json() == calculate_values(V1, values).model_dump_json()


# ------------------------------------------------------------------ rule-file invariants (fail at load, not at payout)
@pytest.mark.parametrize("mutate, match", [
    (lambda d: d["steps"].__setitem__(0, {"id": "start", "op": "subtract", "from": "assessed_covered_loss"}), "first step must be 'set'"),
    (lambda d: d["steps"].append({"id": "again", "op": "set", "from": "prior_payments_total"}), "first step must be 'set'|last step"),
    (lambda d: d["steps"][0].update(optional=True), "cannot be optional"),
    (lambda d: d["steps"][2].update(**{"from": "applicable_deducible"}), "unknown input name"),
    (lambda d: d["steps"].pop(), "last step must be a floor at 0"),
    (lambda d: d["steps"][-1].update(value=5), "floor at 0"),
    (lambda d: d["steps"].pop(4), "must cap at remaining_policy_limit"),
    (lambda d: d["steps"][4].update(optional=True), "must cap at remaining_policy_limit"),
    (lambda d: d.update(rounding={"mode": "half_up", "places": 9}), "rounding must be"),
    (lambda d: d.update(rounding={"mode": "banker", "places": 0}), "rounding must be"),
    (lambda d: d["steps"].append({"id": "start", "op": "floor", "value": 0}), "duplicate step ids"),
])
def test_invalid_rule_files_are_rejected(mutate, match):
    d = raw_rules()
    mutate(d)
    with pytest.raises(ConfigError, match=match):
        load_rules(d)


def test_pack_with_a_broken_calculation_file_fails_at_startup(tmp_path):
    import shutil
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    shutil.copytree(ROOT / "packs", tmp_path / "packs")
    f = tmp_path / "packs/property_water_damage/calculation/settlement_v2_concept.yaml"
    f.write_text(f.read_text(encoding="utf-8").replace("from: remaining_policy_limit", "from: remaining_limit"), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown input name"):
        load_settings(tmp_path)


def test_rules_loader_file_and_version_handling(tmp_path):
    assert load_rules(ROOT / "packs/property_water_damage/calculation/settlement_v1.yaml").version == "sample-1.0"
    with pytest.raises(ConfigError, match="file not found"):
        load_rules(tmp_path / "missing.yaml")
    (tmp_path / "bad.yaml").write_text("steps: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_rules(tmp_path / "bad.yaml")
    (tmp_path / "list.yaml").write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_rules(tmp_path / "list.yaml")
    with pytest.raises(ConfigError, match="duplicate rules version"):
        rules_by_version({"a": raw_rules(), "b": raw_rules()})
    assert isinstance(load_rules(raw_rules()), CalculationRules)


# ------------------------------------------------------------------ hygiene (T5.3: no float, no LLM, deterministic)
SOURCES = sorted((ROOT / "src/claimtower/settlement").glob("*.py"))
FORBIDDEN_IMPORTS = ("openai", "anthropic", "langchain", "langgraph", "random", "time", "datetime",
                     "claimtower.core.llm", "claimtower.agents")


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_settlement_code_has_no_float_no_llm_and_no_nondeterminism(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            pytest.fail(f"{path.name}:{node.lineno} float literal")
        if isinstance(node, ast.Name) and node.id == "float" and isinstance(node.ctx, ast.Load):
            # allowed only as the rejected type in money.py's isinstance check
            assert path.name == "money.py", f"{path.name}:{node.lineno} uses float"
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for n in names:
            assert not any(n == f or n.startswith(f + ".") for f in FORBIDDEN_IMPORTS), f"{path.name}: forbidden import {n}"
