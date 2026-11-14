"""Single-signal tests (``qts.research.signals``, ``qts signal list|test``).

Hand computations on tiny inputs (Holm, the paper book's P&L and turnover, the
neutral rank weights, the NW t), then one baseline fit on a synthetic market against
which the test's IC, incremental IC and processing are recomputed independently
(scipy's Spearman per date, numpy least squares), the trial accounting, the
custom-signal registry and the CLI.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from qts.core.config import SystemConfig
from qts.data.synthetic import SyntheticSpec, generate
from qts.research import signals as SG
from qts.validation import TrialLedger


# --------------------------------------------------------------------------- by hand
def test_holm_by_hand():
    p = [0.01, 0.04, 0.03, 0.005, float("nan")]
    # sorted 0.005 (x4 = 0.02), 0.01 (x3 = 0.03), 0.03 (x2 = 0.06), 0.04 (x1 -> max(0.06, 0.04))
    np.testing.assert_allclose(SG.holm_adjust(p)[:4], [0.03, 0.06, 0.06, 0.02])
    assert math.isnan(SG.holm_adjust(p)[4])
    assert SG.holm_adjust([0.6, 0.7]).tolist() == [1.0, 1.0]
    assert SG.two_sided_p(1.959963984540054) == pytest.approx(0.05)


def test_ic_stats_newey_west_by_hand():
    rng = np.random.default_rng(0)
    x = pd.Series(0.02 + 0.05 * rng.standard_normal(200))
    r = SG.ic_stats(x, overlap=12)
    L = r["nw_lags"]
    assert L == 12  # max(overlap, floor(4 (200/100)^(2/9)) = 4)
    v = x.to_numpy() - x.mean()
    n = len(v)
    s = v @ v / n + 2 * sum((1 - k / (L + 1)) * (v[k:] @ v[:-k]) / n for k in range(1, L + 1))
    assert r["t_nw"] == pytest.approx(x.mean() / math.sqrt(s / n), rel=1e-6)
    assert r["n"] == 200 and r["mean_ic"] == pytest.approx(x.mean())


def test_neutral_rank_weights_are_neutral_and_scaled():
    rng = np.random.default_rng(1)
    names = pd.Index([f"N{i}" for i in range(40)])
    X = pd.DataFrame(
        {"ind_a": (np.arange(40) % 2).astype(float), "beta": rng.standard_normal(40), "const": 1.0},
        index=names,
    )
    score = pd.Series(rng.standard_normal(40), index=names)
    score.iloc[3] = np.nan
    w = SG.neutral_rank_weights(score, X)
    assert np.abs(w).sum() == pytest.approx(2.0)
    assert w.sum() == pytest.approx(0.0, abs=1e-12)
    assert (X[["ind_a", "beta"]].T @ w).abs().max() < 1e-12
    assert SG.neutral_rank_weights(pd.Series(np.nan, index=names), X).abs().sum() == 0.0


def test_paper_book_pnl_and_turnover_by_hand():
    dates = pd.bdate_range("2021-01-04", periods=10)
    R = pd.DataFrame({"A": np.arange(10) * 0.001, "B": -np.arange(10) * 0.002, "C": 0.01}, index=dates)
    w1 = pd.Series({"A": 1.0, "B": -1.0})
    w2 = pd.Series({"A": 0.5, "C": -0.5, "B": -1.0})
    book = SG.paper_book({dates[0]: w1, dates[3]: w2}, R, lag=1, step=3, cost_bps=10.0)
    # w1 set on d0, executed at the close of d1, earns d2, d3, d4; w2 set on d3 earns d5, d6, d7
    assert list(book.gross.index) == list(dates[2:8])
    for k in (2, 3, 4):
        assert book.gross[dates[k]] == pytest.approx(1.0 * 0.001 * k - (-0.002 * k))
    for k in (5, 6, 7):
        assert book.gross[dates[k]] == pytest.approx(0.5 * 0.001 * k + 0.002 * k - 0.5 * 0.01)
    assert book.turnover[dates[2]] == pytest.approx(2.0)  # from flat
    assert book.turnover[dates[5]] == pytest.approx(0.5 + 0.5)  # A -0.5, C -0.5, B unchanged
    assert book.turnover[[dates[3], dates[4], dates[6]]].sum() == 0.0
    np.testing.assert_allclose(book.net.to_numpy(), (book.gross - 1e-3 * book.turnover).to_numpy())
    st = book.stats()
    assert st["turnover_per_day"] == pytest.approx(3.0 / 6) and st["n_days"] == 6


# --------------------------------------------------------------------------- on a fitted baseline
@pytest.fixture(scope="module")
def fitted():
    from qts.pipeline import ResearchPipeline

    md = generate(SyntheticSpec(n_stocks=90, n_days=900, seed=3, ic_true=0.06))
    cfg = SystemConfig.from_yaml("configs/baseline.yaml").with_overrides({"universe.top_n_by_mcap": 80})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = ResearchPipeline(cfg, use_ml=False, ledger_path=None, runs_root=None).fit(md)
    return md, cfg, pipe


def _run(fitted, tmp_path, feature="mom_12_1", **kw):
    md, cfg, pipe = fitted
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SG.run_signal_test(
            feature, md=md, config=cfg, pipe=pipe, use_ml=False, ledger_path=str(tmp_path / "l.jsonl"),
            runs_root=str(tmp_path / "runs"), cost_bps=3.0, **kw,
        )  # fmt: skip


def _spearman_mean(a: pd.DataFrame, b: pd.DataFrame) -> float:
    vals = []
    for d in a.index:
        x, y = a.loc[d], b.loc[d]
        m = x.notna() & y.notna()
        if m.sum() >= 20:
            vals.append(spearmanr(x[m], y[m]).statistic)
    return float(np.mean(vals))


def test_signal_test_matches_an_independent_computation(fitted, tmp_path):
    _md, cfg, pipe = fitted
    r = _run(fitted, tmp_path, horizons=(20, 60))
    assert r.primary_horizon == 60 and r.horizons == [20, 60]
    # processing: the model's own processing on the covered cells
    raw = pipe.alpha.raw_features_["mom_12_1"]
    proc = SG.process_signal(raw, pipe.md_u, cfg.alpha.winsor_pct, False, "mom_12_1")
    X = pipe.alpha.X_["mom_12_1"].unstack("ticker").reindex_like(proc)
    have = proc.notna()
    np.testing.assert_allclose(proc[have].stack().to_numpy(), X[have].stack().to_numpy())
    assert (X[~have & pipe.md_u.eligibility()].fillna(0.0) == 0.0).all().all()
    # primary IC: per-date Spearman against the baseline's own training label
    sample = pipe.backtest_dates()[:: r.step]
    S = proc.reindex(index=sample)
    want = _spearman_mean(S, pipe.alpha.target_.reindex(index=sample, columns=S.columns))
    assert r.ic_by_horizon["60"]["mean_ic"] == pytest.approx(want, abs=1e-12)
    assert r.ic_by_horizon["60"]["nw_lags"] >= math.ceil(60 / r.step)
    # incremental: OLS residual on [1, exposures, traded score] by numpy, then Spearman
    rows = {}
    for d in sample:
        names = pipe.md_u.eligibility().loc[d]
        names = names.index[names.to_numpy()]
        sg, tb = S.loc[d].reindex(names), pipe.alpha.traded_score_.loc[d].reindex(names)
        m = sg.notna() & tb.notna()
        if m.sum() < 30:
            continue
        Xe = pipe._exposures_fn(d).reindex(names[m]).fillna(0.0).to_numpy(dtype=float)
        Xe = Xe[:, Xe.std(axis=0) > 1e-12]
        A = np.column_stack([np.ones(int(m.sum())), Xe, tb[m].to_numpy()])
        beta = np.linalg.lstsq(A, sg[m].to_numpy(), rcond=None)[0]
        res = sg[m].to_numpy() - A @ beta
        assert abs(res @ tb[m].to_numpy()) < 1e-8  # orthogonal to the traded score
        rows[d] = pd.Series(res, index=names[m])
    Rr = pd.DataFrame(rows).T.reindex(columns=S.columns)
    want_inc = _spearman_mean(Rr, pipe.alpha.target_.reindex(index=Rr.index, columns=S.columns))
    assert r.incremental["ic_by_horizon"]["60"]["mean_ic"] == pytest.approx(want_inc, abs=1e-12)
    # everything is JSON
    d = json.loads(open(r.json_path).read())
    assert d["signal"]["name"] == "mom_12_1" and d["trial"]["family"].startswith("signal:")
    assert set(d["ic_by_horizon"]) == {"20", "60"} and len(d["ic_by_fold"]) >= 1
    assert d["paper"]["cost_bps_per_side"] == 3.0 and "_traded_score" in d["correlations"]
    assert "mom_12_1" not in d["correlations"] and d["correlations"]["mom_6_1"] > 0.3
    assert "Signal test: mom_12_1" in r.format()


def test_a_signal_that_is_the_traded_score_has_no_incremental_ic(fitted, tmp_path):
    _md, _cfg, pipe = fitted
    trad = pipe.alpha.traded_score_

    SG.register_signal(
        "traded_copy", lambda m: trad.reindex(index=m.dates, columns=m.tickers), description="test"
    )
    try:
        r = _run(fitted, tmp_path, feature="traded_copy", horizons=(60,))
    finally:
        SG._CUSTOM.pop("traded_copy", None)
    assert r.correlations["_traded_score"] > 0.99
    # the rank-Gaussian copy is almost linear in the score: next to nothing is left over
    assert r.incremental["resid_var_share"] < 0.05
    # the blend of the baseline with itself is the baseline book
    assert abs(r.incremental["delta_sharpe_net"]) < 0.05


def test_trial_accounting_and_holm_across_the_family(fitted, tmp_path):
    a = _run(fitted, tmp_path, feature="mom_12_1", horizons=(60,))
    b = _run(fitted, tmp_path, feature="strev_21", horizons=(60,))
    led = TrialLedger(str(tmp_path / "l.jsonl"))
    fam = a.trial["family"]
    assert b.trial["family"] == fam and led.n_trials(fam) == 2 and b.trial["n_tests"] == 2
    rec = led.records(fam)[-1]
    assert rec["test"] == "signal" and rec["config"]["signal"] == "strev_21" and rec["kind"] == "trial"
    ts = [x["t"] for x in b.trial["tests"]]
    np.testing.assert_allclose(
        [x["p_holm"] for x in b.trial["tests"]], SG.holm_adjust([SG.two_sided_p(t) for t in ts])
    )
    assert b.trial["passes_hurdle"] == (abs(b.trial["t"]) >= 3.0)
    # --no-log: reported against the family, not counted
    c = _run(fitted, tmp_path, feature="beta_252", horizons=(60,), log_trial=False)
    assert c.trial["logged"] is False and c.trial["n_tests"] == 3 and led.n_trials(fam) == 2
    # the signal family is not a pipeline family: it never enters a strategy's DSR
    assert not any(f["family"].startswith("pipeline_") for f in led.families())


def test_registry_and_custom_module(tmp_path):
    with pytest.raises(ValueError, match="built-in"):
        SG.register_signal("mom_12_1", lambda m: m.close)
    mod = tmp_path / "my_signals.py"
    mod.write_text(
        "from qts.research.signals import register_signal\n"
        "@register_signal('my_vol_5d', description='5-day realised vol', sign=-1.0)\n"
        "def f(md):\n    return md.returns.rolling(5).std()\n"
    )
    names = {s.name: s for s in SG.list_signals(str(mod))}
    assert {"rev_5d", "my_vol_5d", "sue", "value_bp"} <= set(names)
    assert names["my_vol_5d"].sign == -1.0 and names["my_vol_5d"].source == "custom"
    assert names["value_bp"].industry_demean and not names["mom_12_1"].industry_demean
    SG._CUSTOM.pop("my_vol_5d", None)


def test_signal_cli_on_a_store(tmp_path, capsys):
    from qts.cli import main
    from qts.data.store import ParquetStore

    md = generate(SyntheticSpec(n_stocks=60, n_days=640, seed=5))
    ParquetStore(str(tmp_path / "store")).write_market_data(md)
    cfg = SystemConfig.from_yaml("configs/baseline.yaml").with_overrides({"universe.top_n_by_mcap": 50})
    (tmp_path / "b.yaml").write_text(cfg.to_yaml())
    assert main(["signal", "list"]) == 0
    assert "rev_5d" in capsys.readouterr().out
    argv = [
        "signal", "test", "rev_5d", "--store", str(tmp_path / "store"), "--baseline", str(tmp_path / "b.yaml"),
        "--horizons", "5,20", "--no-ml", "--ledger", str(tmp_path / "l.jsonl"), "--runs-root", str(tmp_path / "runs"),
        "--json", str(tmp_path / "s.json"),
    ]  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert main(argv) == 0
        out = capsys.readouterr().out
        assert "Signal test: rev_5d" in out and "Harvey-Liu-Zhu" in out
        assert json.loads((tmp_path / "s.json").read_text())["primary_horizon"] == 5
        # the second test re-uses the cached baseline fit
        assert main([*argv[:2], "mom_6_1", *argv[3:]]) == 0
        out = capsys.readouterr().out
        assert "stage cache {'risk_fit': 'hit', 'alpha_fit': 'hit'}" in out
    assert main(["signal", "test", "nope", "--store", str(tmp_path / "store")]) == 2


def test_signal_test_reuses_the_fit_of_a_qts_run_with_the_same_flags(tmp_path, capsys):
    from qts.cli import main
    from qts.data.store import ParquetStore

    md = generate(SyntheticSpec(n_stocks=60, n_days=640, seed=6))
    ParquetStore(str(tmp_path / "store")).write_market_data(md)
    cfg = SystemConfig.from_yaml("configs/baseline.yaml").with_overrides({"universe.top_n_by_mcap": 50})
    (tmp_path / "b.yaml").write_text(cfg.to_yaml())
    flags = ["--store", str(tmp_path / "store"), "--max-names", "60", "--runs-root", str(tmp_path / "runs"),
             "--ledger", str(tmp_path / "l.jsonl")]  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert main(["run", *flags, "--config", str(tmp_path / "b.yaml"), "--cache", "--max-days", "5"]) == 0
        capsys.readouterr()
        assert main(["signal", "test", "rev_5d", *flags, "--baseline", str(tmp_path / "b.yaml"), "--no-ml",
                     "--horizons", "5"]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "stage cache {'risk_fit': 'hit', 'alpha_fit': 'hit'}" in out
    assert "cost per side" in out and "from run" in out  # the run directory's cost per side
