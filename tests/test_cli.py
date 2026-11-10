"""CLI argument handling that does not need a backtest: ``--set KEY=VALUE``
parsing and application (``qts demo`` / ``qts run``).  The end-to-end CLI runs
live in tests/test_pipeline.py (demo) and tests/test_sharadar.py (run --store)."""

from __future__ import annotations

import io
from contextlib import redirect_stderr
from unittest import mock

import pytest

from qts.cli import ConfigOverrideError, _apply_set_overrides, build_parser, main, parse_set_overrides
from qts.core.config import SystemConfig


def test_parse_set_overrides_yaml_scalars():
    ov = parse_set_overrides(
        [
            "portfolio.risk_aversion=0",
            "portfolio.turnover_penalty=0.0",
            "portfolio.beta_neutral=true",
            "portfolio.target_vol_annual=null",
            "portfolio.risk_aversion_mode=target_vol",
            "alpha.lgbm.num_leaves = 15 ",
            'portfolio.solver="1"',
        ]
    )
    assert ov["portfolio.risk_aversion"] == 0 and isinstance(ov["portfolio.risk_aversion"], int)
    assert ov["portfolio.turnover_penalty"] == 0.0 and isinstance(ov["portfolio.turnover_penalty"], float)
    assert ov["portfolio.beta_neutral"] is True
    assert ov["portfolio.target_vol_annual"] is None
    assert ov["portfolio.risk_aversion_mode"] == "target_vol"
    assert ov["alpha.lgbm.num_leaves"] == 15
    assert ov["portfolio.solver"] == "1"  # quoted -> string
    assert parse_set_overrides(None) == {} and parse_set_overrides([]) == {}


@pytest.mark.parametrize("bad", ["portfolio.risk_aversion", "=3", "portfolio.x=[1,2]", "a={b: 1}"])
def test_parse_set_overrides_rejects_malformed(bad):
    with pytest.raises(ConfigOverrideError):
        parse_set_overrides([bad])


def test_apply_set_overrides_changes_config_and_hash():
    base = SystemConfig()
    cfg = _apply_set_overrides(
        base,
        ["portfolio.risk_aversion=5", "portfolio.risk_aversion_mode=fixed", "execution.signal_lag_days=0"],
    )
    assert cfg.portfolio.risk_aversion == 5
    assert cfg.portfolio.risk_aversion_mode == "fixed"
    assert cfg.execution.signal_lag_days == 0
    assert cfg.hash() != base.hash()
    assert base.portfolio.risk_aversion == 2.0  # input untouched (frozen, copied)
    assert _apply_set_overrides(base, None) is base


def test_apply_set_overrides_unknown_key_is_clear_error():
    with pytest.raises(ConfigOverrideError, match=r"unknown config field 'portfolio\.risk_averzion'"):
        _apply_set_overrides(SystemConfig(), ["portfolio.risk_averzion=5"])
    with pytest.raises(ConfigOverrideError, match="unknown config section 'portfolo'"):
        _apply_set_overrides(SystemConfig(), ["portfolo.risk_aversion=5"])
    # section validation (ExecutionConfig.__post_init__) failures surface the same way
    with pytest.raises(ConfigOverrideError, match="signal_lag_days"):
        _apply_set_overrides(SystemConfig(), ["execution.signal_lag_days=1.5"])


def test_parser_set_is_repeatable_on_demo_and_run():
    ap = build_parser()
    a = ap.parse_args(["demo", "--set", "a=1", "--set", "b=2"])
    assert a.set_overrides == ["a=1", "b=2"]
    a = ap.parse_args(["run", "--store", "x", "--set", "portfolio.risk_aversion=5"])
    assert a.set_overrides == ["portfolio.risk_aversion=5"]
    assert ap.parse_args(["demo"]).set_overrides is None


def test_cli_demo_unknown_set_key_exits_2_before_running():
    err = io.StringIO()
    with mock.patch("qts.cli._run_pipeline") as run, redirect_stderr(err):
        rc = main(["demo", "--stocks", "20", "--days", "60", "--set", "portfolio.nope=1"])
    assert rc == 2 and "unknown config field 'portfolio.nope'" in err.getvalue()
    run.assert_not_called()


def test_cli_run_unknown_set_key_exits_2_before_loading_data(tmp_path):
    err = io.StringIO()
    with mock.patch("qts.data.store.ParquetStore") as store, redirect_stderr(err):
        rc = main(["run", "--store", str(tmp_path), "--set", "nope=1"])
    assert rc == 2 and "unknown config field 'nope'" in err.getvalue()
    store.assert_not_called()


def test_cli_demo_set_reaches_pipeline_config(tmp_path):
    """``--set`` is applied after ``--config`` and before the pipeline is built."""
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("portfolio:\n  risk_aversion: 3.0\n  max_position_weight: 0.01\n")
    seen = {}

    def fake_run(md, cfg, a, out=None):
        seen["cfg"] = cfg
        return 0

    with mock.patch("qts.cli._run_pipeline", side_effect=fake_run):
        rc = main(
            [
                "demo",
                "--stocks",
                "20",
                "--days",
                "60",
                "--seed",
                "3",
                "--config",
                str(cfg_path),
                "--set",
                "portfolio.risk_aversion=7",
                "--set",
                "portfolio.target_vol_annual=null",
            ]
        )
    assert rc == 0
    cfg = seen["cfg"]
    assert cfg.portfolio.risk_aversion == 7  # --set wins over --config
    assert cfg.portfolio.max_position_weight == 0.01  # --config kept where not overridden
    assert cfg.portfolio.target_vol_annual is None
    assert cfg.seed == 3
