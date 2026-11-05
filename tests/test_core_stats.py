"""qts.core.stats against brute-force references (scipy, numpy lstsq, explicit loops)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, spearmanr

from qts.core import crosssection, stats


@pytest.fixture(scope="module")
def panels():
    rng = np.random.default_rng(0)
    T, N = 40, 60
    dates = pd.bdate_range("2020-01-01", periods=T)
    cols = pd.Index([f"T{i:02d}" for i in range(N)], name="ticker")
    a = pd.DataFrame(rng.standard_normal((T, N)), index=dates, columns=cols)
    b = 0.3 * a + pd.DataFrame(rng.standard_normal((T, N)), index=dates, columns=cols)
    a = a.mask(rng.random((T, N)) < 0.15)
    b = b.mask(rng.random((T, N)) < 0.15)
    a.iloc[3, :50] = np.nan                     # a thin cross-section (< 20 pairs)
    a.iloc[5, :10] = 1.0                        # ties
    return a, b


# --------------------------------------------------------------------------- winsorize
def test_winsorize_rows_matches_rowwise_quantile_clip(panels):
    a, _ = panels
    a = a.copy()
    a.iloc[0, 0] = 50.0
    a.iloc[1, 1] = np.inf
    got = stats.winsorize_rows(a, 0.05)
    for d, row in a.iterrows():
        x = row[np.isfinite(row)]
        if len(x) < 10:
            exp = row.where(np.isfinite(row))
        else:
            exp = row.where(np.isfinite(row)).clip(x.quantile(0.05), x.quantile(0.95))
        np.testing.assert_allclose(got.loc[d].to_numpy(), exp.to_numpy(), equal_nan=True)
    assert got.iloc[0, 0] < 50.0 and np.isnan(got.iloc[1, 1])
    # ndarray in -> ndarray out; pct=0 is the identity
    arr = stats.winsorize_rows(a.to_numpy(), 0.05)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_allclose(arr, got.to_numpy(), equal_nan=True)
    assert stats.winsorize_rows(a, 0.0) is a
    # the legacy wrapper delegates
    pd.testing.assert_frame_equal(crosssection.winsorize(a, 0.05), got)


@pytest.mark.parametrize("seed", range(5))
def test_winsorize_rows_sort_based_equals_nanpercentile_exactly(seed):
    """The sort-based row quantile reproduces np.nanpercentile(method="linear")
    bit for bit (NaNs, +-inf, ties, thin rows, empty rows, extreme pct)."""
    rng = np.random.default_rng(seed)
    for _ in range(40):
        T, N = int(rng.integers(1, 30)), int(rng.integers(1, 80))
        a = rng.standard_t(3, (T, N)) * float(rng.choice([1e-4, 1.0, 1e5]))
        a[rng.random((T, N)) < rng.random()] = np.nan
        a[rng.random((T, N)) < 0.03] = np.inf
        a[rng.random((T, N)) < 0.03] = -np.inf
        if rng.random() < 0.3:
            a = np.round(a, 1)                              # ties
        pct = float(rng.choice([1e-3, 0.01, 0.025, 0.05, 0.2, 0.499]))
        got = stats.winsorize_rows(a, pct, min_obs=1)
        work = np.where(np.isfinite(a), a, np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN rows
            lo = np.nanpercentile(work, 100 * pct, axis=1, method="linear")
            hi = np.nanpercentile(work, 100 * (1 - pct), axis=1, method="linear")
        exp = np.clip(work, lo[:, None], hi[:, None])
        np.testing.assert_array_equal(got, exp)             # exact, NaN == NaN


# --------------------------------------------------------------------------- rank-gauss
def test_rank_gauss_reference_and_bound(panels):
    a, _ = panels
    got = stats.rank_gauss(a)
    for d, row in a.iterrows():
        x = row.dropna()
        n = len(x)
        if n == 0:
            continue
        exp = norm.ppf(np.clip((x.rank() - 0.5) / n, 1e-4, 1 - 1e-4))
        np.testing.assert_allclose(got.loc[d, x.index].to_numpy(), exp, rtol=1e-12, atol=1e-12)
        assert got.loc[d].abs().max() <= norm.ppf(1 - 0.5 / n) + 1e-12
    assert got.isna().equals(a.isna())
    big = pd.DataFrame(np.arange(20000.0)[None, :])
    assert stats.rank_gauss(big).abs().max().max() == pytest.approx(norm.ppf(1 - 1e-4))   # ~3.719
    pd.testing.assert_frame_equal(crosssection.rank_gauss(a), got)


# --------------------------------------------------------------------------- rank IC
def test_rank_ic_series_matches_scipy_spearman(panels):
    a, b = panels
    got = stats.rank_ic_series(a, b)
    for d in a.index:
        m = a.loc[d].notna() & b.loc[d].notna()
        if m.sum() < 20:
            assert np.isnan(got.loc[d])
            continue
        exp = spearmanr(a.loc[d][m], b.loc[d][m]).statistic
        assert got.loc[d] == pytest.approx(exp, abs=1e-12)
    assert np.isnan(got.iloc[3])
    # misaligned inputs: intersection of dates and tickers
    sub = stats.rank_ic_series(a.iloc[5:], b.iloc[:, 10:])
    assert sub.index.equals(a.index[5:])
    # the legacy loop-free wrappers agree
    pd.testing.assert_series_equal(crosssection.ic_series(a, b), got.rename(None))
    assert crosssection.rank_ic(a.iloc[0], b.iloc[0]) == pytest.approx(got.iloc[0])


def test_row_corr_degenerate_rows():
    x = np.array([[1.0] * 25, list(range(25))])
    y = np.array([list(range(25)), list(range(25))])
    r = stats.row_corr(x, y)
    assert np.isnan(r[0]) and r[1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- Newey-West
def _nw_brute(x, lags):
    x = np.asarray(x, float)
    n = len(x)
    xb = x.mean()
    s = 0.0
    for L in range(-lags, lags + 1):
        w = 1 - abs(L) / (lags + 1)
        g = sum((x[t] - xb) * (x[t - abs(L)] - xb) for t in range(abs(L), n)) / n
        s += w * g
    return np.sqrt(s / n)


@pytest.mark.parametrize("lags", [0, 1, 5, 12])
def test_newey_west_matches_explicit_double_loop(lags):
    rng = np.random.default_rng(lags)
    e = rng.standard_normal(300)
    x = 0.1 + np.convolve(e, [1.0, 0.6, 0.3], mode="same")      # MA(2): autocorrelated
    assert stats.newey_west_se(x, lags) == pytest.approx(_nw_brute(x, lags), rel=1e-12)
    assert stats.newey_west_tstat(x, lags) == pytest.approx(x.mean() / _nw_brute(x, lags), rel=1e-12)


def test_newey_west_defaults_and_edge_cases():
    x = np.random.default_rng(1).standard_normal(250) + 0.05
    assert stats.default_nw_lags(250) == 4                       # floor(4 * 2.5^(2/9))
    assert stats.newey_west_se(x) == pytest.approx(_nw_brute(x, 4), rel=1e-12)
    # lags 0 = iid standard error with ddof=0
    assert stats.newey_west_se(x, 0) == pytest.approx(x.std(ddof=0) / np.sqrt(len(x)))
    xs = pd.Series(np.r_[x, np.nan, np.inf])                     # non-finite dropped
    assert stats.newey_west_tstat(xs, 3) == pytest.approx(stats.newey_west_tstat(x, 3))
    assert np.isnan(stats.newey_west_tstat([1.0, 2.0], 1))
    assert np.isnan(stats.newey_west_tstat(np.ones(30), 2))      # zero variance


# --------------------------------------------------------------------------- rolling beta
def test_rolling_beta_matches_lstsq():
    rng = np.random.default_rng(3)
    T, N, W = 120, 4, 30
    dates = pd.bdate_range("2021-01-01", periods=T)
    m = pd.Series(0.01 * rng.standard_normal(T), index=dates)
    true_b = np.array([0.5, 1.0, 1.5, 2.0])
    r = pd.DataFrame(m.to_numpy()[:, None] * true_b + 0.01 * rng.standard_normal((T, N)), index=dates)
    r.iloc[50:55, 2] = np.nan
    res = stats.rolling_beta(r, m, window=W, min_periods=W)
    for t in [W - 1, 60, T - 1]:
        for j in range(N):
            y = r.iloc[t - W + 1:t + 1, j]
            x = m.iloc[t - W + 1:t + 1]
            ok = y.notna()
            if ok.sum() < W:
                assert np.isnan(res.beta.iloc[t, j])
                continue
            X = np.column_stack([np.ones(ok.sum()), x[ok]])
            coef, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
            resid = y[ok] - X @ coef
            assert res.beta.iloc[t, j] == pytest.approx(coef[1], rel=1e-9)
            assert res.resid_var.iloc[t, j] == pytest.approx((resid ** 2).mean(), rel=1e-6, abs=1e-14)
    assert np.isnan(res.beta.iloc[W - 2]).all()
    np.testing.assert_allclose(res.resid_vol.to_numpy(), np.sqrt(res.resid_var.to_numpy()), equal_nan=True)
    with pytest.raises(ValueError):
        stats.rolling_beta(r, m, window=10, min_periods=20)


# --------------------------------------------------------------------------- industry dummies
def test_industry_dummies():
    lab = pd.Series({"A": "Tech", "B": "Banks", "C": "Tech", "D": None})
    d = stats.industry_dummies(lab, pd.Index(["C", "B", "A", "D", "E"]))
    assert list(d.columns) == ["Banks", "Tech"] and list(d.index) == ["C", "B", "A", "D", "E"]
    exp = pd.get_dummies(lab.reindex(d.index), dtype=float).reindex(index=d.index, columns=d.columns)
    pd.testing.assert_frame_equal(d, exp, check_names=False)
    assert d.loc[["D", "E"]].sum().sum() == 0.0
    d1 = stats.industry_dummies(lab, drop_first=True)
    assert list(d1.columns) == ["Tech"]
