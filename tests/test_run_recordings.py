"""The run-directory recordings the workbench labs read (``qts.runs.artifacts``):
refit feature importance, theme weights, per-sleeve IC, the feature IC table and
the per-rebalance constraint log.  Recording only: the tests check shapes, the
point-in-time property of the refit windows, and that the numbers the pipeline
reports are untouched (the golden file covers the latter in full).
"""

from __future__ import annotations

import json
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qts.alpha.alpha_model import AlphaModel, RefitRecord, refit_record
from qts.alpha.composite import THEMES
from qts.alpha.models import EnsembleModel, LGBMModel, RidgeModel
from qts.alpha.trainer import FoldInfo
from qts.core.config import AlphaConfig
from qts.data.synthetic import SyntheticSpec, generate
from qts.runs import artifacts as A

warnings.filterwarnings("ignore", category=RuntimeWarning)


# --------------------------------------------------------------------------- unit: refit records
class _NoImportance:
    def fit(self, X, y, valid_mask=None):
        return self

    def predict(self, X):
        return pd.Series(0.0, index=X.index)

    def feature_importance(self):
        return None


class _MLPModel(_NoImportance):
    """Named like the real MLP member so the note is triggered."""


def _fold() -> FoldInfo:
    d = pd.bdate_range("2020-01-01", periods=400)
    return FoldInfo(
        pred_start=d[300],
        pred_end=d[399],
        train_start=d[0],
        train_end=d[279],
        valid_start=d[200],
        n_train_rows=1000,
        n_valid_rows=200,
    )


def test_refit_record_collects_members_and_blend_weights():
    X = pd.DataFrame(
        np.random.default_rng(0).normal(size=(300, 3)),
        columns=["a", "b", "c"],
        index=pd.MultiIndex.from_product(
            [pd.bdate_range("2020-01-01", periods=30), range(10)], names=["date", "ticker"]
        ),
    )
    y = pd.Series(X["a"].to_numpy() * 0.5 + np.random.default_rng(1).normal(size=300), index=X.index)
    ens = EnsembleModel([RidgeModel(), _MLPModel()], weights=[0.7, 0.3]).fit(X, y)
    # the ensemble names its members by class; the fake MLP keeps the real one's name
    _MLPModel.__name__ = "MLPModel"
    rec = refit_record(_fold(), ens)
    assert isinstance(rec, RefitRecord)
    assert rec.refit_date == _fold().pred_start and rec.n_train == 1000
    assert set(rec.importance) == {"ridge", "ensemble"}
    assert list(rec.importance["ridge"].index) == ["a", "b", "c"] or set(rec.importance["ridge"].index) == {
        "a",
        "b",
        "c",
    }
    assert (rec.importance["ridge"] >= 0).all()
    assert rec.blend_weights == {"ridge": 0.7, "mlp": 0.3}
    assert any(n.startswith("mlp:") for n in rec.notes)


def test_refit_record_of_a_bare_model():
    X = pd.DataFrame(
        np.random.default_rng(0).normal(size=(200, 2)),
        columns=["a", "b"],
        index=pd.MultiIndex.from_product(
            [pd.bdate_range("2020-01-01", periods=20), range(10)], names=["date", "ticker"]
        ),
    )
    y = pd.Series(X["b"].to_numpy(), index=X.index)
    rec = refit_record(_fold(), RidgeModel().fit(X, y))
    assert set(rec.importance) == {"ridge"} and rec.blend_weights == {} and rec.notes == []


# --------------------------------------------------------------------------- unit: constraints frame
def _fake_result(log_rows, dates):
    trades = pd.DataFrame(0.0, index=dates, columns=["A", "B", "C"])
    trades.loc[dates[0], ["A", "B"]] = [0.01, -0.02]
    trades.loc[dates[2], "C"] = 0.03
    tc = pd.Series([0.8, np.nan, 0.6], index=dates, name="transfer_coefficient")
    bt = SimpleNamespace(diagnostics={"solve_log": log_rows}, trades=trades, transfer_coefficient=tc)
    return SimpleNamespace(backtest=bt)


def test_constraints_frame_widens_binding_counts():
    dates = pd.bdate_range("2021-03-01", periods=3)
    rows = [
        {
            "date": dates[0],
            "signal_date": dates[0] - pd.offsets.BDay(1),
            "status": "optimal",
            "binding": {"gross": 1, "position": 12},
            "vol_target_binding": "target",
            "vol_scalar": 1.0,
            "n_solves": 4,
        },
        {
            "date": dates[2],
            "signal_date": None,
            "status": "optimal",
            "binding": {"participation": 3},
            "vol_target_binding": "caps",
        },
        "not a mapping",
    ]
    df = A.constraints_frame(_fake_result(rows, dates))
    assert list(df["date"]) == [dates[0], dates[2]]
    assert {c for c in df.columns if c.startswith("binding_")} == {
        "binding_gross",
        "binding_position",
        "binding_participation",
    }
    assert df.loc[0, "binding_position"] == 12 and df.loc[1, "binding_position"] == 0
    assert df.loc[1, "binding_participation"] == 3 and df.loc[0, "binding_participation"] == 0
    assert list(df["vol_target_binding"]) == ["target", "caps"]
    assert list(df["n_traded"]) == [2, 1]
    assert df.loc[0, "transfer_coefficient"] == pytest.approx(0.8) and df.loc[
        1, "transfer_coefficient"
    ] == pytest.approx(0.6)
    assert pd.isna(df.loc[1, "signal_date"])


def test_constraints_frame_empty_without_solve_log():
    dates = pd.bdate_range("2021-03-01", periods=3)
    assert A.constraints_frame(_fake_result([], dates)).empty


# --------------------------------------------------------------------------- alpha model recordings
@pytest.fixture(scope="module")
def md():
    return generate(SyntheticSpec(n_stocks=120, n_days=1000, seed=3, ic_true=0.08))


@pytest.fixture(scope="module")
def composite_alpha(md):
    return AlphaModel(AlphaConfig(model_seeds=1), use_ml=False).fit(md)


def test_theme_weights_are_recorded_per_date(composite_alpha, md):
    W = composite_alpha.composite.theme_weights_
    assert W is not None
    assert set(W.columns) <= set(THEMES)
    assert len(W) == len(composite_alpha.composite_score_.index)
    assert np.isfinite(W.to_numpy()).all()
    # dynamic weights move; the static ones are the config's
    if composite_alpha.dynamic_weights:
        assert W.std().max() > 0
    else:
        for th in W.columns:
            assert (W[th] == composite_alpha.cfg.theme_weights[th]).all()


def test_sleeve_ic_series_matches_oos_ic(composite_alpha):
    df = composite_alpha.sleeve_ic_series()
    assert {"combined", "composite"} <= set(df.columns) and "ml" not in df.columns
    assert all(c.startswith("theme_") for c in df.columns if c not in ("combined", "composite"))
    pd.testing.assert_series_equal(
        df["combined"], composite_alpha.oos_ic_.rename("combined"), check_names=True
    )
    assert df.index.name == "date"


def test_feature_ic_table_uses_the_composite_sign(composite_alpha):
    t = composite_alpha.feature_ic_table()
    assert list(t.columns) == ["feature", "sign", "theme", "mean_ic", "t_stat", "n"]
    assert set(t["feature"]) == set(composite_alpha.X_.columns)
    signs = {f: s for members in THEMES.values() for f, s in members}
    for r in t.itertuples(index=False):
        assert r.sign == signs.get(r.feature, 1.0)
        assert (r.theme == "") == (r.feature not in signs)
    assert (t["n"] > 0).all() and np.isfinite(t["mean_ic"]).all()
    # the synthetic market plants a positive value / momentum signal: signed IC is positive
    planted = t[t["feature"].isin(["value_bp", "mom_12_1"])]
    assert (planted["mean_ic"] > 0).all()


@pytest.mark.slow
def test_refit_records_are_point_in_time(md):
    cfg = AlphaConfig(train_min_days=504, refit_frequency_days=126, model_seeds=1)
    am = AlphaModel(
        cfg,
        ml_factory=lambda: EnsembleModel(
            [LGBMModel(n_seeds=1, n_estimators=30, n_jobs=2), RidgeModel()], weights=[0.8, 0.2]
        ),
        signal_lag_days=1,
    ).fit(md)
    recs = am.refit_records_
    assert len(recs) == len(am.trainer.folds_) >= 2
    dates = am.combined_score_.index
    pos = pd.Series(np.arange(len(dates)), index=dates)
    ml = am.ml_score_
    assert ml is not None
    for r in recs:
        # labels end strictly before the prediction block, by the embargo (h + L)
        assert pos[r.train_end] + am.trainer.embargo < pos[r.refit_date]
        assert r.train_start <= r.train_end < r.refit_date <= r.pred_end
        # the refit date is never after the first date its predictions are used for
        block = ml.loc[r.refit_date : r.pred_end]
        first_used = block.index[block.notna().any(axis=1)]
        if len(first_used):
            assert r.refit_date <= first_used[0]
        assert set(r.importance) == {"lgbm", "ridge", "ensemble"}
        for s in r.importance.values():
            assert set(s.index) <= set(am.X_.columns) and (s >= 0).all()
        assert r.importance["lgbm"].sum() == pytest.approx(1.0, rel=1e-6)
        assert r.blend_weights == pytest.approx({"lgbm": 0.8, "ridge": 0.2})
    # refit dates strictly increase and follow the schedule
    rd = [r.refit_date for r in recs]
    assert rd == sorted(rd) and len(set(rd)) == len(rd)
    assert "ml" in am.sleeve_ic_series().columns

    pipeline = SimpleNamespace(alpha=am, cfg=SimpleNamespace(alpha=cfg))
    long, meta = A.alpha_refits_frames(pipeline)  # type: ignore[arg-type]
    assert list(long.columns) == ["refit_date", "sleeve", "feature", "importance"]
    assert set(long["sleeve"]) == {"lgbm", "ridge", "ensemble"}
    assert len(meta["refits"]) == len(recs) and meta["embargo_days"] == am.trainer.embargo
    assert meta["refits"][0]["blend_weights"] == pytest.approx({"lgbm": 0.8, "ridge": 0.2})
    assert json.dumps(meta)  # JSON-safe


# --------------------------------------------------------------------------- pipeline: files written
@pytest.mark.slow
def test_pipeline_writes_the_lab_recordings(golden_pipeline_run, tmp_path):
    from qts.pipeline import save_run

    pipe, res = golden_pipeline_run
    run_dir = save_run(res, pipe.cfg, tmp_path)
    out = A.write_extra_artifacts(run_dir, res, pipe)
    assert out is not None
    extras = json.loads(out.read_text())
    files = extras["files"]
    # composite-only run: theme weights, sleeve IC, constraints and the feature IC table
    # exist; the refit table does not (no ML sleeve) but its metadata file does
    assert {"theme_weights", "alpha_ic_by_sleeve", "constraints", "alpha_refits_meta"} <= set(files)
    assert "alpha_refits" not in files and not (run_dir / "alpha_refits.parquet").exists()
    meta = json.loads((run_dir / "alpha_refits.json").read_text())
    assert meta["refits"] == [] and len(meta["feature_ic"]) == len(pipe.alpha.X_.columns)

    tw = pd.read_parquet(run_dir / "theme_weights.parquet")
    assert tw.index.name == "date" and set(tw.columns) <= set(THEMES)

    si = pd.read_parquet(run_dir / "alpha_ic_by_sleeve.parquet")
    assert {"combined", "composite"} <= set(si.columns)
    # the combined sleeve's IC at the stored formation dates is the pipeline's own oos_ic_
    pd.testing.assert_series_equal(si["combined"], pipe.alpha.oos_ic_.rename("combined"), check_freq=False)

    cs = pd.read_parquet(run_dir / "constraints.parquet")
    n_solves = res.backtest.diagnostics["n_rebalance_solves"]
    assert len(cs) == n_solves
    assert {"date", "status", "vol_target_binding", "transfer_coefficient", "n_traded"} <= set(cs.columns)
    assert cs["date"].is_monotonic_increasing
    assert set(cs["date"]) <= set(res.backtest.weights.index)
    # the mean transfer coefficient the summary reports is the mean of the recorded column
    summary_tc = res.backtest.summary()["transfer_coefficient_mean"]
    assert cs["transfer_coefficient"].mean() == pytest.approx(summary_tc, rel=1e-9)
