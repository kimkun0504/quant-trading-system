"""SystemConfig serialisation, strictness, hashing and the generated default YAML."""

from __future__ import annotations

import dataclasses
import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from qts.core.config import (
    AlphaConfig,
    LGBMParams,
    MLPParams,
    SystemConfig,
    ValidationConfig,
    apply_overrides,
    dump_default_yaml,
)

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ("universe", "alpha", "risk", "cost", "portfolio", "execution", "validation")


def test_round_trip_dict_and_hash():
    cfg = SystemConfig()
    d = cfg.to_dict()
    again = SystemConfig.from_dict(d)
    assert again == cfg
    assert again.hash() == cfg.hash()
    # every sub-config round-trips on its own
    for name in SECTIONS:
        sub = getattr(cfg, name)
        assert type(sub).from_dict(sub.to_dict()) == sub, name
    # tuples come back as tuples (dataclass equality would fail otherwise)
    assert isinstance(again.alpha.momentum_vol_scalar_bounds, tuple)
    assert isinstance(again.alpha.mlp.hidden, tuple)
    assert isinstance(again.alpha.lgbm, LGBMParams) and isinstance(again.alpha.mlp, MLPParams)


def test_from_dict_is_strict_and_partial():
    with pytest.raises(KeyError, match="bogus"):
        SystemConfig.from_dict({"alpha": {"bogus": 1}})
    with pytest.raises(KeyError, match="nope"):
        SystemConfig.from_dict({"nope": {}})
    with pytest.raises(KeyError, match="lgbm"):
        SystemConfig.from_dict({"alpha": {"lgbm": {"num_leaf": 3}}})
    # missing keys take defaults; nested partial dicts are merged onto defaults
    c = SystemConfig.from_dict({"alpha": {"lgbm": {"num_leaves": 7}}, "seed": 3})
    assert c.alpha.lgbm.num_leaves == 7 and c.alpha.lgbm.n_estimators == LGBMParams().n_estimators
    assert c.seed == 3 and c.alpha.horizon_days == AlphaConfig().horizon_days
    # schema_version is accepted and ignored
    assert SystemConfig.from_dict({"schema_version": 1}) == SystemConfig()


def test_yaml_round_trip(tmp_path):
    cfg = SystemConfig().with_overrides(
        {
            "portfolio.risk_aversion": 7.5,
            "alpha.secondary_horizons": [10],
            "pipeline_pbo_configs": [{"portfolio.risk_aversion": 5.0}],
        }
    )
    p = tmp_path / "c.yaml"
    p.write_text(cfg.to_yaml(header="test"))
    back = SystemConfig.from_yaml(p)
    assert back == cfg and back.hash() == cfg.hash()
    assert p.read_text().startswith("# test")


def test_hash_sensitive_and_stable_across_processes():
    a, b = SystemConfig(), SystemConfig()
    assert a.hash() == b.hash() and len(a.hash()) == 16
    b = b.with_overrides({"alpha.ic_shrink": 0.4})
    assert a.hash() != b.hash()
    code = "from qts.core.config import SystemConfig; print(SystemConfig().hash())"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT)
    assert out.stdout.strip() == a.hash()


def test_overrides():
    cfg = SystemConfig()
    c2 = cfg.with_overrides({"portfolio.risk_aversion": 3.0, "alpha.lgbm.num_leaves": 9, "seed": 4})
    assert c2.portfolio.risk_aversion == 3.0 and c2.alpha.lgbm.num_leaves == 9 and c2.seed == 4
    assert cfg.portfolio.risk_aversion != 3.0 and cfg.seed == 0  # original untouched
    with pytest.raises(KeyError):
        apply_overrides(cfg, {"portfolio.risk_aversio": 1})
    with pytest.raises(KeyError):
        apply_overrides(cfg, {"nosection.x": 1})


def test_every_section_serialisable():
    cfg = SystemConfig()
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        if dataclasses.is_dataclass(v):
            assert hasattr(v, "to_dict") and hasattr(type(v), "from_dict"), f.name
    assert cfg.validate() is cfg
    assert ValidationConfig.from_dict({"n_splits": 4}).n_splits == 4


def test_committed_default_yaml_matches_code():
    """configs/default.yaml is generated (``python -m qts.cli config dump > configs/default.yaml``);
    regenerate it when a config default or field changes."""
    committed = (ROOT / "configs" / "default.yaml").read_text()
    assert committed == dump_default_yaml(), (
        "configs/default.yaml is stale: run `python -m qts.cli config dump > configs/default.yaml`"
    )
    assert SystemConfig.from_yaml(ROOT / "configs" / "default.yaml") == SystemConfig()


def test_cli_config_dump_and_validate(tmp_path):
    from qts.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main(["config", "dump"]) == 0
    assert buf.getvalue() == dump_default_yaml()
    good = tmp_path / "good.yaml"
    good.write_text(buf.getvalue())
    with redirect_stdout(io.StringIO()):
        assert main(["config", "validate", str(good)]) == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text("alpha:\n  horizon_dayz: 5\n")
    assert main(["config", "validate", str(bad)]) == 1
