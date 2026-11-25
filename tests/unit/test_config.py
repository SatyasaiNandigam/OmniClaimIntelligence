import shutil
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from claimtower.core.config import load_pack, load_settings
from claimtower.core.errors import ConfigError, RegistryError
from claimtower.core.registry import ADAPTERS, Registry, get_pack, list_packs, resolve_adapter
from claimtower.core.schemas import SettlementInputs
from claimtower.settlement.engine import SettlementCalculator
from claimtower.settlement.rules_loader import rules_by_version

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def root(tmp_path):
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    shutil.copytree(REPO / "packs", tmp_path / "packs")
    return tmp_path


def edit(path: Path, fn):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    fn(data)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_repo_configs_load():
    s = load_settings()
    assert s.pack.meta.name == "property_water_damage"
    assert s.routing.tiers.complex.default
    assert "other" in s.pack.document_types.types
    assert s.autonomy.operations["release_payment"].roles == ["supervisor", "finance"]


def test_loaded_calculation_rules_drive_engine():
    s = load_settings()
    calc = SettlementCalculator(rules_by_version(s.pack.calculation))
    inputs = SettlementInputs(assessed_covered_loss=Decimal(200000), non_covered_items_total=Decimal(15000),
                              applicable_deductible=Decimal(10000), remaining_policy_limit=Decimal(150000), prior_payments_total=Decimal(0))
    assert calc.compute_settlement(inputs, "sample-1.0").payable == Decimal("150000")


def test_editing_routing_yaml_changes_loaded_rules_without_code_change(root):
    edit(root / "configs/routing_rules.yaml", lambda d: d["tiers"]["simple"]["all_of"].__setitem__(
        -1, {"feature": "exposure_inr", "lte": 1, "code": "EXPOSURE_WITHIN_FRONTLINE"}))
    simple = load_settings(root).routing.tiers.simple.all_of
    assert simple[-1].lte == 1


def test_invalid_yaml_syntax_reports_file_and_line(root):
    (root / "configs/app.yaml").write_text("llm: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"app\.yaml: invalid YAML \(line"):
        load_settings(root)


def test_all_problems_reported_together(root):
    edit(root / "configs/app.yaml", lambda d: d["llm"].__setitem__("temperature", 9))
    edit(root / "configs/rbac.yaml", lambda d: d.__setitem__("roles", ["intake"]))
    with pytest.raises(ConfigError) as e:
        load_settings(root)
    msg = str(e.value)
    assert "app.yaml: llm.temperature" in msg
    assert "unknown roles" in msg and "release_payment" in msg


def test_unknown_key_rejected(root):
    edit(root / "configs/app.yaml", lambda d: d.__setitem__("typo_key", 1))
    with pytest.raises(ConfigError, match="typo_key"):
        load_settings(root)


def test_missing_pack_and_missing_file(root):
    edit(root / "configs/app.yaml", lambda d: d.__setitem__("active_pack", "nope"))
    with pytest.raises(ConfigError, match="pack folder not found"):
        load_settings(root)
    (root / "configs/rbac.yaml").unlink()
    with pytest.raises(ConfigError, match="rbac.yaml: file not found"):
        load_settings(root)


def test_extraction_confidence_banned_as_routing_feature(root):
    edit(root / "configs/routing_rules.yaml",
         lambda d: d["tiers"]["medium"]["any_of"].append({"feature": "extraction_confidence", "lt": 0.5, "code": "LOW_CONF"}))
    with pytest.raises(ConfigError, match="extraction confidence"):
        load_settings(root)


@pytest.mark.parametrize("bad, match", [
    ({"feature": "evidence_completness", "is": "complete", "code": "X"}, "unknown feature 'evidence_completness'"),
    ({"feature": "evidence_completeness", "is": "completee", "code": "X"}, "not allowed for 'evidence_completeness'"),
    ({"feature": "exposure_inr", "is": "high", "code": "X"}, "numeric feature"),
    ({"feature": "specialist_required", "is": "yes", "code": "X"}, "boolean feature"),
    ({"feature": "exposure_inr", "code": "X"}, "needs one of"),
    ({"feature": "exposure_inr", "gt": 1, "code": "lower_case"}, "code"),
])
def test_routing_condition_typos_fail_at_startup(root, bad, match):
    edit(root / "configs/routing_rules.yaml", lambda d: d["tiers"]["medium"]["any_of"].append(bad))
    with pytest.raises(ConfigError, match=match):
        load_settings(root)


def test_routing_approver_role_must_exist_in_rbac(root):
    edit(root / "configs/routing_rules.yaml", lambda d: d["approver_roles"].__setitem__("complex", "chief_wizard"))
    with pytest.raises(ConfigError, match="approver_roles use unknown roles"):
        load_settings(root)


def test_bad_calculation_step_rejected(root):
    p = root / "packs/property_water_damage/calculation/settlement_v1.yaml"
    edit(p, lambda d: d["steps"].append({"id": "x", "op": "multiply", "from": "a"}))
    with pytest.raises(ConfigError, match="settlement_v1.yaml"):
        load_settings(root)


def test_env_overrides_mock_flags(monkeypatch):
    monkeypatch.setenv("USE_MOCK_PAS", "false")
    assert load_settings().app.flags.use_mock_pas is False


def test_registry_and_adapter_resolution(monkeypatch):
    r = Registry("thing")
    r.register("a", 1)
    with pytest.raises(RegistryError, match="already registered"):
        r.register("a", 2)
    with pytest.raises(RegistryError, match="registered: \\['a'\\]"):
        r.get("b")

    ADAPTERS.clear()
    ADAPTERS.register("pas:mock", lambda: "mock")
    ADAPTERS.register("pas:live", lambda: "live")
    assert resolve_adapter("pas", load_settings())() == "mock"
    monkeypatch.setenv("USE_MOCK_PAS", "false")
    assert resolve_adapter("pas", load_settings())() == "live"
    ADAPTERS.clear()


def test_pack_discovery():
    s = load_settings()
    assert list_packs(s) == ["property_water_damage"]
    assert get_pack("property_water_damage", s) is s.pack
    with pytest.raises(ConfigError):
        load_pack("nope")
