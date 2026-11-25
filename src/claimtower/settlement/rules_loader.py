"""Load and validate calculation rule files (YAML) into CalculationRules. Business rules live in packs/*/calculation/, not in code."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from claimtower.core.config import CalculationRules
from claimtower.core.errors import ConfigError


def _validate(raw: Mapping[str, Any], where: str) -> CalculationRules:
    try:
        return CalculationRules.model_validate(dict(raw))
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(str(x) for x in err['loc']) or '<root>'}: {err['msg']}" for err in e.errors())
        raise ConfigError(f"{where}: invalid calculation rules - {problems}") from e


def load_rules(source: str | Path | Mapping[str, Any]) -> CalculationRules:
    """From a YAML file path or an already-parsed mapping."""
    if isinstance(source, Mapping):
        return _validate(source, "<mapping>")
    path = Path(source)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"{path.name}: file not found") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"{path.name}: invalid YAML - {e}") from e
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path.name}: expected a mapping at the top level")
    return _validate(raw, path.name)


def rules_by_version(raw_by_name: Mapping[str, Mapping[str, Any]]) -> dict[str, CalculationRules]:
    """Pack calculation files ({file stem: raw rules}) -> {rules version: validated rules}. Versions must be unique."""
    out: dict[str, CalculationRules] = {}
    for name, raw in raw_by_name.items():
        rules = _validate(raw, f"{name}.yaml")
        if rules.version in out:
            raise ConfigError(f"{name}.yaml: duplicate rules version '{rules.version}'")
        out[rules.version] = rules
    return out
