"""The lab endpoints (redesign steps 4-5): ``/alpha/refits``, ``/alpha/theme-weights``,
``/alpha/ic-by-year`` and ``/constraints`` against the fixture run (composite only, so
the refit table is empty), a run directory given a synthetic refit table, and the
"old" run without any of the recordings."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from qts_api import schemas as S
from qts_api.main import create_app
from qts_api.settings import Settings
from tests.conftest import LOCAL, OLD_RUN_ID

ML_RUN_ID = "20200102T000000000000Z_m1refits"


@pytest.fixture(scope="module")
def ml_root(tmp_path_factory, run_dir):
    """A separate runs root holding a copy of the fixture run with a hand-written refit
    table: three refits, two sleeves plus the ensemble, four features (one missing from
    the ridge sleeve)."""
    root = tmp_path_factory.mktemp("ml_runs")
    d = root / ML_RUN_ID
    shutil.copytree(run_dir, d)
    m = json.loads((d / "manifest.json").read_text())
    m["run_id"] = ML_RUN_ID
    m["created_at"] = "2020-01-02T00:00:00+00:00"
    (d / "manifest.json").write_text(json.dumps(m))
    dates = ["2021-01-04", "2022-01-03", "2023-01-02"]
    feats = ["value_bp", "mom_12_1", "quality_gpa", "idio_vol_252"]
    rows = []
    for i, rd in enumerate(dates):
        for j, f in enumerate(feats):
            gain = 0.4 - 0.1 * j + 0.05 * i * (j - 1.5)
            rows.append((rd, "lgbm", f, max(gain, 0.01)))
            rows.append((rd, "ensemble", f, max(gain, 0.01) * 0.5 + 0.1))
            if f != "idio_vol_252":
                rows.append((rd, "ridge", f, 0.02 * (j + 1) * (i + 1)))
    pd.DataFrame(rows, columns=["refit_date", "sleeve", "feature", "importance"]).assign(
        refit_date=lambda x: pd.to_datetime(x["refit_date"])
    ).to_parquet(d / "alpha_refits.parquet", index=False)
    meta = json.loads((d / "alpha_refits.json").read_text())
    meta["refits"] = [
        {
            "refit_date": rd,
            "train_start": "2018-01-02",
            "train_end": str(pd.Timestamp(rd) - pd.offsets.BDay(22))[:10],
            "valid_start": str(pd.Timestamp(rd) - pd.offsets.BDay(300))[:10],
            "pred_end": str(pd.Timestamp(rd) + pd.offsets.BDay(251))[:10],
            "n_train": 10_000 * (i + 1),
            "n_valid": 2_000,
            "blend_weights": {"lgbm": 0.7, "mlp": 0.3},
        }
        for i, rd in enumerate(dates)
    ]
    meta["notes"] = ["mlp: no feature importance recorded (permutation importance is not computed)"]
    meta["embargo_days"] = 21
    (d / "alpha_refits.json").write_text(json.dumps(meta))
    return root


@pytest.fixture()
def ml_client(ml_root):
    app = create_app(Settings(runs_root=ml_root), start_jobs=False)
    with TestClient(app, base_url=LOCAL) as c:
        yield c


# ----------------------------------------------------------------------------- refits
def test_refits_of_the_composite_run_have_no_table_but_the_feature_ic(client, run_id, run_dir):
    r = client.get(f"/api/runs/{run_id}/alpha/refits")
    assert r.status_code == 200
    a = S.AlphaRefits.model_validate(r.json())
    assert a.recorded and a.refits == [] and a.sleeves == [] and a.importance == {}
    meta = json.loads((run_dir / "alpha_refits.json").read_text())
    assert len(a.feature_ic) == len(meta["feature_ic"]) > 10
    by = {f.feature: f for f in a.feature_ic}
    assert by["value_bp"].sign == 1.0 and by["value_bp"].theme == "value"
    assert by["beta_252"].sign == -1.0 and by["beta_252"].theme == "low_risk"
    assert all(f.n > 0 and f.mean_ic is not None for f in a.feature_ic)
    assert a.refit_frequency_days == 252 and a.embargo_days == 21


def test_refits_table_is_pivoted_per_sleeve(ml_client):
    a = S.AlphaRefits.model_validate(ml_client.get(f"/api/runs/{ML_RUN_ID}/alpha/refits").json())
    assert a.recorded
    assert [r.refit_date for r in a.refits] == ["2021-01-04", "2022-01-03", "2023-01-02"]
    assert a.sleeves == ["lgbm", "ridge", "ensemble"]  # display order, mlp absent (not recorded)
    assert a.features == sorted(["value_bp", "mom_12_1", "quality_gpa", "idio_vol_252"])
    assert set(a.importance) == {"lgbm", "ridge", "ensemble"}
    assert len(a.importance["lgbm"]["value_bp"]) == 3
    assert a.importance["ridge"]["idio_vol_252"] == [None, None, None]  # never recorded for ridge
    assert a.importance["ridge"]["mom_12_1"] == pytest.approx([0.04, 0.08, 0.12])
    assert a.refits[0].blend_weights == {"lgbm": 0.7, "mlp": 0.3} and a.refits[2].n_train == 30_000
    assert a.notes and a.notes[0].startswith("mlp:")
    # the refit windows are point-in-time: labels end before the refit date
    for w in a.refits:
        assert w.train_end < w.refit_date <= w.pred_end


def test_refits_of_an_old_run_are_not_recorded(client):
    a = S.AlphaRefits.model_validate(client.get(f"/api/runs/{OLD_RUN_ID}/alpha/refits").json())
    assert not a.recorded and a.refits == [] and a.feature_ic == [] and a.importance == {}
    assert a.refit_frequency_days == 252  # from the config


# ----------------------------------------------------------------------------- theme weights
def test_theme_weights(client, run_id, run_dir):
    t = S.AlphaThemeWeights.model_validate(client.get(f"/api/runs/{run_id}/alpha/theme-weights").json())
    W = pd.read_parquet(run_dir / "theme_weights.parquet")
    assert t.themes == list(W.columns) and set(t.series) == set(t.themes)
    assert len(t.dates) == len(W) == len(t.series["value"]) and not t.downsampled
    assert t.dates[0] == str(W.index[0].date()) and t.dates[-1] == str(W.index[-1].date())
    assert t.static["value"] > 0 and isinstance(t.dynamic, bool)
    assert t.series["momentum"][-1] == pytest.approx(float(W["momentum"].iloc[-1]))


def test_theme_weights_of_an_old_run_are_empty(client):
    t = S.AlphaThemeWeights.model_validate(client.get(f"/api/runs/{OLD_RUN_ID}/alpha/theme-weights").json())
    assert t.dates == [] and t.series == {} and t.themes == [] and t.static["momentum"] > 0


# ----------------------------------------------------------------------------- IC by year
def test_ic_by_year_from_the_daily_sleeves(client, run_id, run_dir):
    a = S.AlphaIcByYear.model_validate(client.get(f"/api/runs/{run_id}/alpha/ic-by-year").json())
    assert a.source == "daily_sleeves" and a.overlap_days == 21  # horizon 20 + lag 1
    si = pd.read_parquet(run_dir / "alpha_ic_by_sleeve.parquet")
    assert a.years == sorted({int(y) for y in pd.DatetimeIndex(si.index).year})
    sleeves = [r.sleeve for r in a.rows]
    assert sleeves[:2] == ["combined", "composite"] and "ml" not in sleeves
    assert all(s.startswith("theme_") for s in sleeves[2:]) and sleeves[2:] == sorted(sleeves[2:])
    for row in a.rows:
        assert [c.year for c in row.cells] == a.years
        for c in row.cells:
            v = si[row.sleeve][pd.DatetimeIndex(si.index).year == c.year].dropna()
            assert c.n == len(v)
            if c.n:
                assert c.mean_ic == pytest.approx(float(v.mean()))
            if c.n >= 3:
                assert c.t_stat is not None and np.isfinite(c.t_stat)
    # every year has at least one observation on the combined row
    assert all(c.n > 0 for c in a.rows[0].cells)


def test_ic_by_year_falls_back_to_the_sampled_ic(client, run_dir, summary):
    a = S.AlphaIcByYear.model_validate(client.get(f"/api/runs/{OLD_RUN_ID}/alpha/ic-by-year").json())
    assert a.source == "sampled_combined" and a.overlap_days == 1
    assert [r.sleeve for r in a.rows] == ["combined"]
    assert sum(c.n for c in a.rows[0].cells) == summary["ic"]["n"]


# ----------------------------------------------------------------------------- constraints
def test_constraints(client, run_id, run_dir, summary):
    c = S.RunConstraints.model_validate(client.get(f"/api/runs/{run_id}/constraints").json())
    cs = pd.read_parquet(run_dir / "constraints.parquet")
    assert c.n_rebalances == len(cs) == len(c.dates) == summary["backtest"]["n_days"]
    assert c.dates == sorted(c.dates)
    assert c.names == sorted(c.names) and set(c.binding) == set(c.names)
    for nm in c.names:
        assert len(c.binding[nm]) == c.n_rebalances
        assert c.share_binding[nm] == pytest.approx(float((cs[f"binding_{nm}"] > 0).mean()))
        assert 0 <= c.share_binding[nm] <= 1
    assert len(c.transfer_coefficient) == len(c.n_traded) == len(c.status) == c.n_rebalances
    tc = [v for v in c.transfer_coefficient if v is not None]
    assert np.mean(tc) == pytest.approx(summary["backtest"]["transfer_coefficient_mean"], rel=1e-6)
    assert all(s == "optimal" or s for s in c.status)
    assert set(c.vol_target_reasons) == {v for v in c.vol_target_binding if v}
    assert all(0 <= n <= 60 for n in c.n_traded)


def test_constraints_of_an_old_run_are_empty(client):
    c = S.RunConstraints.model_validate(client.get(f"/api/runs/{OLD_RUN_ID}/constraints").json())
    assert c.dates == [] and c.names == [] and c.n_rebalances == 0 and c.share_binding == {}


@pytest.mark.parametrize("suffix", ["/alpha/refits", "/alpha/theme-weights", "/alpha/ic-by-year", "/constraints"])
def test_lab_endpoints_404_on_unknown_run(client, suffix):
    r = client.get(f"/api/runs/nope{suffix}")
    assert r.status_code == 404 and r.headers["content-type"].startswith("application/problem+json")


def test_lab_files_are_listed_with_descriptions(client, run_id):
    files = S.RunFiles.model_validate(client.get(f"/api/runs/{run_id}/files").json())
    by = {f.name: f for f in files.files}
    for name in ("theme_weights.parquet", "alpha_ic_by_sleeve.parquet", "constraints.parquet", "alpha_refits.json"):
        assert name in by and by[name].description
