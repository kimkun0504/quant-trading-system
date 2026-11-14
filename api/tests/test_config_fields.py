"""The config fields added for the research tools: ``universe.rank_offset`` (an integer
with bounds) and ``alpha.disabled_features`` (a multi-select whose choices follow
``alpha.feature_set``), in the schema and as launch overrides."""

from __future__ import annotations

import pytest
from qts.core.config import SystemConfig

from qts_api import schemas as S
from qts_api.services.config_schema import ConfigError, config_json_schema, leaf_index, validate_overrides

BASE = {
    "kind": "synthetic",
    "ml": False,
    "phase": 1,
    "stocks": 50,
    "days": 1000,
    "seed": 1,
    "signal_lag_days": 1,
    "config_overrides": {},
}


def test_schema_leaves():
    s = S.JsonSchemaObject.model_validate(config_json_schema())
    leaf = leaf_index()["alpha.disabled_features"]
    assert leaf["type"] == "array" and leaf["multiple"] is True and leaf["uniqueItems"] is True
    assert leaf["choices_key"] == "alpha.feature_set" and leaf["default"] == []
    assert (
        set(leaf["choices"]) == {"v1", "v2"} and len(leaf["choices"]["v1"]) == 18 and len(leaf["choices"]["v2"]) == 26
    )
    assert set(leaf["choices"]["v1"]) < set(leaf["choices"]["v2"])
    assert "renormalise" in leaf["description"]
    arr = s.properties["alpha"].properties["disabled_features"]  # type: ignore[union-attr]
    assert isinstance(arr, S.JsonSchemaArray)
    ro = leaf_index()["universe.rank_offset"]
    assert ro["type"] == "integer" and ro["default"] == 0 and ro["minimum"] == 0 and ro["maximum"] == 10_000


def test_disabled_features_overrides():
    cfg = validate_overrides({"alpha.disabled_features": "strev_21, mom_change"})
    assert cfg.alpha.disabled_features == ("mom_change", "strev_21")
    assert cfg.hash() == validate_overrides({"alpha.disabled_features": "mom_change,strev_21"}).hash()
    assert cfg.hash() != SystemConfig().hash()
    assert validate_overrides({"alpha.disabled_features": ""}).hash() == SystemConfig().hash()
    # the feature set and its toggles in either order
    a = validate_overrides({"alpha.disabled_features": "sue", "alpha.feature_set": "v2"})
    b = validate_overrides({"alpha.feature_set": "v2", "alpha.disabled_features": "sue"})
    assert a.alpha.disabled_features == ("sue",) and a.hash() == b.hash()
    with pytest.raises(ConfigError) as e:
        validate_overrides({"alpha.disabled_features": "sue"})  # a v2 feature under v1
    assert e.value.issues[0][0] == ["body", "config_overrides", "alpha.disabled_features"]
    assert "not features of feature_set 'v1'" in e.value.issues[0][1]
    with pytest.raises(ConfigError, match="unknown name"):
        validate_overrides({"alpha.disabled_features": "nope"})
    with pytest.raises(ConfigError):
        validate_overrides({"alpha.disabled_features": 3})
    every = ",".join(leaf_index()["alpha.disabled_features"]["choices"]["v1"])
    with pytest.raises(ConfigError, match="every feature"):
        validate_overrides({"alpha.disabled_features": every})


def test_rank_offset_overrides():
    assert validate_overrides({"universe.rank_offset": 1000}).universe.rank_offset == 1000
    for bad in (-1, 10_001, 1.5, True):
        with pytest.raises(ConfigError):
            validate_overrides({"universe.rank_offset": bad})


def test_validate_and_command_through_the_api(client):
    body = {**BASE, "config_overrides": {"alpha.disabled_features": "mom_change,strev_21"}}
    v = client.post("/api/config/validate", json=body).json()
    want = validate_overrides({"alpha.disabled_features": "mom_change,strev_21"})
    want = want.with_overrides({"seed": 1, "universe.top_n_by_mcap": 50})  # a synthetic market is clamped
    assert v["valid"] and v["config_hash"] == want.hash()
    bad = client.post("/api/config/validate", json={**BASE, "config_overrides": {"alpha.disabled_features": "sue"}})
    e = bad.json()["errors"][0]
    assert e["loc"] == ["body", "config_overrides", "alpha.disabled_features"]
    from qts_api.services.jobs import cli_command

    assert cli_command(body, None).endswith("--set alpha.disabled_features=mom_change,strev_21")
