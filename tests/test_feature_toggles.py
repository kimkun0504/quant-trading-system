"""Feature toggles (``alpha.disabled_features``) and the feature registry
(``qts.alpha.feature_registry``, ``qts features list``)."""

from __future__ import annotations

import dataclasses
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from qts.alpha import AlphaModel
from qts.alpha.composite import THEMES, theme_scores_long
from qts.alpha.feature_registry import FEATURE_DESCRIPTIONS, feature_info
from qts.alpha.features import FEATURE_ORDER, FEATURE_SETS
from qts.core.config import AlphaConfig, SystemConfig
from qts.core.roles import FEATURE_SET_NAMES
from qts.data.synthetic import SyntheticSpec, generate

#: the v1 order before the list moved to qts.core.roles (part of the parity hash)
V1_ORDER = [
    "value_bp", "value_ep", "mom_12_1", "mom_6_1", "mom_change", "industry_momentum", "quality_gpa",
    "investment_ag", "beta_252", "idio_vol_252", "max_ret_21", "turnover_63", "amihud_63",
    "log_dollar_vol_63", "log_mcap", "short_interest", "borrow_fee", "strev_21",
]  # fmt: skip


def test_feature_lists_unchanged_and_registry_complete():
    assert FEATURE_ORDER == V1_ORDER and FEATURE_SETS["v1"] == V1_ORDER
    assert list(FEATURE_SET_NAMES["v2"]) == FEATURE_SETS["v2"]
    assert set(FEATURE_DESCRIPTIONS) == set(FEATURE_SET_NAMES["v2"])
    assert all(len(d) > 10 for d in FEATURE_DESCRIPTIONS.values())
    rows = {r.name: r for r in feature_info("v2")}
    assert rows["beta_252"].role == "risk_only_proxy" and rows["beta_252"].theme == "low_risk"
    assert rows["beta_252"].sign == -1.0
    assert rows["mom_12_1"].role == "alpha_theme" and rows["max_ret_21"].role == "model_input"
    assert rows["sue"].theme == "ml_only" and rows["sue"].feature_set == "v2"
    assert len(feature_info("v1")) == 18
    with pytest.raises(ValueError):
        feature_info("v9")


def test_validation_normalisation_and_order_independence():
    a = AlphaConfig(disabled_features=("strev_21", "mom_change", "strev_21"))
    assert a.disabled_features == ("mom_change", "strev_21")
    assert AlphaConfig(disabled_features="strev_21, mom_change").disabled_features == (
        "mom_change",
        "strev_21",
    )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"not features of feature_set 'v1'.*valid names.*value_bp"):
        AlphaConfig(disabled_features=("bogus",))
    with pytest.raises(ValueError, match="not features"):
        AlphaConfig(disabled_features=("sue",))  # a v2 feature under v1
    assert AlphaConfig(feature_set="v2", disabled_features=("sue",)).disabled_features == ("sue",)
    with pytest.raises(ValueError, match="every feature"):
        AlphaConfig(disabled_features=tuple(V1_ORDER))
    c = SystemConfig()
    x = c.with_overrides({"alpha.disabled_features": "sue", "alpha.feature_set": "v2"})
    y = c.with_overrides({"alpha.feature_set": "v2", "alpha.disabled_features": ["sue"]})
    assert x.hash() == y.hash()
    # round trip through YAML / dict; the default is empty and keeps the default hash stable
    assert SystemConfig.from_dict(x.to_dict()).hash() == x.hash()
    assert SystemConfig().alpha.disabled_features == ()
    meta = next(f for f in dataclasses.fields(AlphaConfig) if f.name == "disabled_features").metadata
    assert meta["choices_key"] == "feature_set" and meta["choices"]["v2"] == list(FEATURE_SET_NAMES["v2"])
    assert meta["multiple"] is True


@pytest.fixture(scope="module")
def market():
    return generate(SyntheticSpec(n_stocks=60, n_days=420, seed=4))


def _fit(md, disabled=()):
    cfg = AlphaConfig(disabled_features=tuple(disabled))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AlphaModel(cfg, use_ml=False).fit(md)


def test_disabled_features_leave_the_model_inputs_only(market):
    full = _fit(market)
    off = ("value_bp", "value_ep", "mom_change")  # the whole value theme + one momentum member
    a = _fit(market, off)
    assert [c for c in a.X_.columns if c in off] == []
    assert list(a.X_.columns) == [c for c in full.X_.columns if c not in off]
    # the processed values of the remaining features are unchanged
    pd.testing.assert_frame_equal(a.X_, full.X_[list(a.X_.columns)])
    # raw panels (warm-up label exposures, neutralisation, residual vol) and the label are unchanged
    assert set(off) <= set(a.raw_features_)
    assert a.target_.equals(full.target_) and a.resid_vol_h_.equals(full.resid_vol_h_)
    # themes: value omitted, momentum renormalised over its remaining members
    assert "value" not in a.theme_scores_ and "value" in full.theme_scores_
    th = theme_scores_long(a.X_)
    mom = [f for f, _ in THEMES["momentum"] if f not in off]
    want = sum(a.X_[f] for f in mom) / len(mom)
    np.testing.assert_allclose(th["momentum"].to_numpy(), want.to_numpy())
    assert not a.composite_score_.equals(full.composite_score_)


def test_features_list_cli(capsys):
    from qts.cli import main

    assert (
        main(["features", "list", "--set", "alpha.feature_set=v2", "--set", "alpha.disabled_features=sue"])
        == 0
    )
    out = capsys.readouterr().out
    assert "feature set v2: 26 features, 1 disabled" in out
    line = next(ln for ln in out.splitlines() if ln.startswith("sue "))
    assert " off " in line and "unexpected" in line
    assert main(["features", "list", "--set", "alpha.disabled_features=nope"]) == 2


def test_features_list_json(tmp_path, capsys):
    from qts.cli import main

    p = tmp_path / "f.json"
    assert main(["features", "list", "--json", str(p)]) == 0
    rows = json.loads(p.read_text())
    assert len(rows) == 18 and all(r["enabled"] for r in rows)
