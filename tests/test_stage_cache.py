"""Stage cache (``qts.runs.cache`` + ``cfg.cache``): hit / miss, invalidation on
config / data / code changes, point-in-time safety (never served for another
data snapshot or as-of span) and bitwise equality of cached vs uncached runs.

Also pins the two behaviour-preserving speed-ups of the specific-risk path
(``covariance._vra_cap_weights`` and ``covariance._column_quantiles``) to the
pandas expressions they replace, bit for bit (values *and* memory layout,
because downstream row sums are layout-sensitive in the last bit).
"""

from __future__ import annotations

import dataclasses
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from qts.core.config import SystemConfig
from qts.data.synthetic import SyntheticSpec, generate
from qts.pipeline import ResearchPipeline
from qts.risk import covariance as C
from qts.runs.cache import StageCache, source_hash

SPEC = {"n_stocks": 60, "n_days": 640, "seed": 5}
TOP_N = 50
MAX_DAYS = 4


@pytest.fixture(scope="module")
def market():
    return generate(SyntheticSpec(**SPEC))


def _cfg(root, **ov) -> SystemConfig:
    base = {"universe.top_n_by_mcap": TOP_N}
    if root is not None:
        base.update({"cache.enabled": True, "cache.root": str(root)})
    return SystemConfig().with_overrides({**base, **ov})


def _run(md, cfg, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pipe = ResearchPipeline(cfg, use_ml=False, **kw)
        res = pipe.run(md, max_days=MAX_DAYS)
    return pipe, res


def _panels(pipe, res) -> dict:
    bt = res.backtest
    d = dict(bt.diagnostics)
    d.pop("run_time_sec", None)
    for e in d.get("solve_log", []):
        e.pop("solve_time", None)
    a = pipe.alpha
    return {
        "nav": bt.nav,
        "weights": bt.weights,
        "trades": bt.trades,
        "costs": bt.costs,
        "exante_var": bt.exante_var,
        "diag": repr(d),
        "summary": repr(dict(bt.summary())),
        "hurdles": repr(res.hurdles.to_dict()),
        "fr": pipe.risk.factor_returns,
        "sr": pipe.risk.specific_returns,
        "score": a.combined_score_,
        "X": a.X_,
    }


def _assert_bitwise(p, q):
    for k in p:
        x, y = p[k], q[k]
        if isinstance(x, (pd.DataFrame, pd.Series)):
            assert x.equals(y) and x.index.equals(y.index), k
        else:
            assert x == y, k


@pytest.fixture(scope="module")
def uncached(market):
    pipe, res = _run(market, _cfg(None))
    return _panels(pipe, res), res


# --------------------------------------------------------------------------- config
def test_cache_is_off_by_default_and_outside_the_config_hash(tmp_path):
    cfg = SystemConfig()
    assert cfg.cache.enabled is False and cfg.cache.root == ""
    on = cfg.with_overrides(
        {"cache.enabled": True, "cache.root": str(tmp_path), "cache.stages": ["risk_fit"]}
    )
    assert on.hash() == cfg.hash()  # operational: no hash / ledger / golden change
    assert SystemConfig.from_dict(on.to_dict()).cache == on.cache  # round-trips
    with pytest.raises(ValueError, match="unknown stage"):
        cfg.with_overrides({"cache.stages": ["backtest"]})


# --------------------------------------------------------------------------- hit / miss + bitwise
def test_cold_then_warm_run_is_bitwise_identical_to_uncached(market, uncached, tmp_path):
    ref, ref_res = uncached
    p1, r1 = _run(market, _cfg(tmp_path))
    assert p1.cache_info_ == {"risk_fit": "stored", "alpha_fit": "stored"}
    _assert_bitwise(ref, _panels(p1, r1))
    p2, r2 = _run(market, _cfg(tmp_path))
    assert p2.cache_info_ == {"risk_fit": "hit", "alpha_fit": "hit"}
    _assert_bitwise(ref, _panels(p2, r2))
    assert r2.config_hash == ref_res.config_hash and r2.data_hash == ref_res.data_hash
    # the loaded models point at this run's data objects (references, not copies)
    assert p2.alpha.md_ is p2.md_u
    assert len(StageCache(tmp_path).entries("risk_fit")) == 1
    assert len(StageCache(tmp_path).entries("alpha_fit")) == 1


def test_invalidation_follows_the_config_sections_each_stage_reads(market, tmp_path):
    _run(market, _cfg(tmp_path))
    # portfolio / cost knobs are read after the fits: both stages hit
    p, _ = _run(market, _cfg(tmp_path, **{"portfolio.trade_aversion": 2.0}))
    assert p.cache_info_ == {"risk_fit": "hit", "alpha_fit": "hit"}
    # alpha section: the risk fit is reused, the alpha fit is not
    p, _ = _run(market, _cfg(tmp_path, **{"alpha.ic_shrink": 0.4}))
    assert p.cache_info_ == {"risk_fit": "hit", "alpha_fit": "stored"}
    # the signal lag changes the labels (alpha) but not the risk model
    p, _ = _run(market, _cfg(tmp_path, **{"execution.signal_lag_days": 0}))
    assert p.cache_info_ == {"risk_fit": "hit", "alpha_fit": "stored"}
    # risk section: both stages are rebuilt (labels are residualised on the risk model)
    p, _ = _run(market, _cfg(tmp_path, **{"risk.specific_half_life": 63}))
    assert p.cache_info_ == {"risk_fit": "stored", "alpha_fit": "stored"}
    # universe section changes the data the fits see
    p, _ = _run(market, _cfg(tmp_path, **{"universe.top_n_by_mcap": TOP_N - 5}))
    assert p.cache_info_ == {"risk_fit": "stored", "alpha_fit": "stored"}
    # constructor knobs read by the alpha fit
    p, _ = _run(market, _cfg(tmp_path), composite_weight=0.3)
    assert p.cache_info_ == {"risk_fit": "hit", "alpha_fit": "stored"}
    # stage subset
    p, _ = _run(market, _cfg(tmp_path, **{"cache.stages": ["alpha_fit"]}))
    assert p.cache_info_ == {"alpha_fit": "hit"}


# --------------------------------------------------------------------------- point in time
def test_entry_is_never_served_for_another_snapshot_or_as_of_span(market, tmp_path):
    p_full, _ = _run(market, _cfg(tmp_path))
    assert p_full.cache_info_["risk_fit"] == "stored"
    # same data through an earlier as-of date: a new key, never the full-span entry
    earlier = market.slice_until(market.dates[-2])
    p_short, r_short = _run(earlier, _cfg(tmp_path))
    assert p_short.cache_info_ == {"risk_fit": "stored", "alpha_fit": "stored"}
    assert p_short.risk.factor_returns.index[-1] <= earlier.dates[-1]
    # ... and its result equals an uncached run on that snapshot
    p_ref, r_ref = _run(earlier, _cfg(None))
    _assert_bitwise(_panels(p_ref, r_ref), _panels(p_short, r_short))
    # a revised historical price (same span, same shape) is another snapshot
    close = market.close.copy()
    close.iloc[300, 3] *= 1.01
    revised = dataclasses.replace(market, close=close)
    p_rev, _ = _run(revised, _cfg(tmp_path))
    assert p_rev.cache_info_ == {"risk_fit": "stored", "alpha_fit": "stored"}


def test_metadata_mismatch_is_a_miss_even_for_a_matching_key(tmp_path):
    cache = StageCache(tmp_path)
    expect = {"data_hash": "a" * 64, "data_span": ["2020-01-01", "2021-01-01", 260, 10]}
    key = cache.key("risk_fit", {"x": 1})
    cache.store("risk_fit", key, {"v": np.arange(3)}, expect)
    assert cache.load("risk_fit", key, expect)["v"].tolist() == [0, 1, 2]
    other = dict(expect, data_span=["2020-01-01", "2021-01-04", 261, 10])
    assert cache.load("risk_fit", key, other) is None  # never served for another as-of span
    assert cache.load("risk_fit", key, dict(expect, data_hash="b" * 64)) is None


# --------------------------------------------------------------------------- StageCache unit
def test_key_depends_on_stage_parts_and_code():
    k = StageCache.key("risk_fit", {"a": 1, "b": [1, 2]})
    assert k == StageCache.key("risk_fit", {"b": [1, 2], "a": 1})  # canonical JSON
    assert k != StageCache.key("alpha_fit", {"a": 1, "b": [1, 2]})
    assert k != StageCache.key("risk_fit", {"a": 2, "b": [1, 2]})
    assert len(source_hash()) == 64


def test_key_changes_with_the_code(monkeypatch):
    import qts.runs.cache as cache_mod

    k = StageCache.key("risk_fit", {"a": 1})
    monkeypatch.setattr(cache_mod, "source_hash", lambda: "0" * 64)
    assert StageCache.key("risk_fit", {"a": 1}) != k


def test_references_are_reattached_not_pickled(tmp_path):
    cache = StageCache(tmp_path)
    big = pd.DataFrame(np.ones((200, 50)))
    obj = {"data": big, "n": 3}
    exp = {"data_hash": "h"}
    key = cache.key("alpha_fit", {})
    path = cache.store("alpha_fit", key, obj, exp, refs={"md": big})
    assert path is not None and path.stat().st_size < 1000
    other = pd.DataFrame(np.zeros((2, 2)))
    got = cache.load("alpha_fit", key, exp, refs={"md": other})
    assert got["data"] is other and got["n"] == 3
    assert cache.load("alpha_fit", key, exp, refs={}) is None  # missing reference -> miss
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["refs"] == ["md"] and meta["stage"] == "alpha_fit"


def test_corrupt_or_partial_entries_are_misses(tmp_path):
    cache = StageCache(tmp_path)
    exp = {"data_hash": "h"}
    key = cache.key("risk_fit", {"k": 1})
    path = cache.store("risk_fit", key, [1, 2, 3], exp)
    assert path is not None
    path.write_bytes(b"not a pickle")
    assert cache.load("risk_fit", key, exp) is None
    path.unlink()
    assert cache.load("risk_fit", key, exp) is None
    assert cache.misses == 2


def test_unpicklable_object_does_not_fail_the_run(tmp_path):
    cache = StageCache(tmp_path)
    assert cache.store("risk_fit", cache.key("risk_fit", {}), lambda: 1, {}) is None
    assert cache.entries() == []


# --------------------------------------------------------------------------- speed-ups pinned to pandas
def _old_specific_vra_multiplier(U, market_cap, half_life, vra_half_life, min_periods=20):
    """The pre-2026-11-05 body of ``specific_vra_multiplier`` (pandas weights)."""
    A = U.values.astype(float)
    T = A.shape[0]
    if min_periods + 1 >= T:
        return 1.0
    u2 = pd.DataFrame(A**2)
    s2_prev = u2.ewm(halflife=half_life, min_periods=min_periods, ignore_na=True).mean().shift(1).values
    z2 = A**2 / np.where(s2_prev > 0, s2_prev, np.nan)
    latest = market_cap.reindex(columns=U.columns).ffill().iloc[-1] if len(market_cap) else None
    panel = market_cap.reindex(columns=U.columns).shift(1).reindex(U.index)
    if latest is not None:
        panel = panel.fillna(latest)
    w = panel.fillna(0.0).clip(lower=0.0).values
    ok = np.isfinite(z2)
    wsum = (ok * w).sum(axis=1)
    B2 = np.where(wsum > 0, (np.where(ok, z2, 0.0) * w).sum(axis=1) / np.where(wsum > 0, wsum, 1.0), np.nan)
    B2 = B2[np.isfinite(B2)]
    if len(B2) < min_periods:
        return 1.0
    lam2 = float(C.ewma_weights(len(B2), vra_half_life) @ B2)
    return float(np.sqrt(np.clip(lam2, 1e-6, None)))


def _random_case(rng):
    T, N = int(rng.integers(0, 60)), int(rng.integers(1, 9))
    idx = pd.bdate_range("2020-01-01", periods=max(T, 1))[:T]
    cols = [f"c{i}" for i in range(N)]
    M = rng.lognormal(3, 1, (T, N))
    M[rng.random((T, N)) < 0.3] = np.nan
    M[rng.random((T, N)) < 0.05] = -1.0
    M[rng.random((T, N)) < 0.05] = -0.0
    mc = pd.DataFrame(M, index=idx, columns=cols)
    ucols = list(rng.permutation([*cols, "new"]))[: int(rng.integers(1, N + 2))]
    pool = pd.bdate_range("2019-12-20", periods=T + 15)
    uidx = pool[np.sort(rng.choice(len(pool), size=int(rng.integers(0, len(pool))), replace=False))]
    U = pd.DataFrame(rng.standard_normal((len(uidx), len(ucols))) * 0.02, index=uidx, columns=ucols)
    U = U.mask(rng.random(U.shape) < 0.2)
    return mc, U


def test_vra_cap_weights_match_pandas_bit_for_bit_including_layout():
    rng = np.random.default_rng(7)
    for _ in range(400):
        mc, U = _random_case(rng)
        a, b = C._vra_cap_weights(mc, U), C._vra_cap_weights_pandas(mc, U)
        assert a.shape == b.shape
        # same memory layout as DataFrame.values (row sums are layout-sensitive)
        assert (a.flags.c_contiguous, a.flags.f_contiguous) == (b.flags.c_contiguous, b.flags.f_contiguous)
        assert np.array_equal(a, b) and np.array_equal(np.signbit(a), np.signbit(b))


def test_specific_vra_multiplier_matches_the_pandas_version():
    rng = np.random.default_rng(11)
    for _ in range(150):
        mc, U = _random_case(rng)
        if len(U) < 25:
            continue
        new = C.specific_vra_multiplier(U, mc, half_life=10, vra_half_life=8, min_periods=5)
        old = _old_specific_vra_multiplier(U, mc, half_life=10, vra_half_life=8, min_periods=5)
        assert new == old or (np.isnan(new) and np.isnan(old))


def test_column_quantiles_match_dataframe_quantile():
    rng = np.random.default_rng(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # inf - inf inside np.quantile (both paths)
        for _ in range(400):
            n, m = int(rng.integers(0, 40)), int(rng.integers(1, 12))
            A = rng.standard_t(3, (n, m))
            A[rng.random((n, m)) < rng.random() * 0.7] = np.nan
            if rng.random() < 0.1:
                A[rng.random((n, m)) < 0.1] = np.inf
            if rng.random() < 0.3:
                A = np.round(A, 1)
            U = pd.DataFrame(A, columns=[f"t{i}" for i in range(m)])
            for q in (0.25, 0.75):
                a, b = C._column_quantiles(U, q), U.quantile(q)
                assert a.index.equals(b.index) and np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True)
