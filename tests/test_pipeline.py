"""Production-path smoke test: ``ResearchPipeline`` end to end on the synthetic
market (universe -> hybrid risk -> alpha -> cost -> optimizer -> simulator ->
hurdles), plus determinism and the eligibility contract."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from qts.core.config import SystemConfig
from qts.data.synthetic import SyntheticSpec, generate
from qts.pipeline import ResearchPipeline, format_result

warnings.filterwarnings("ignore", category=RuntimeWarning)

GATES = {
    "t_stat",
    "ic_sign",
    "incremental_ic",
    "net_sharpe_delta",
    "breakeven_cost",
    "dsr",
    "pbo",
    "phase_sharpe",
    "suspect",
}
CORE = (
    "sharpe_gross",
    "sharpe_net",
    "cagr_net",
    "ann_vol_net",
    "max_drawdown_net",
    "avg_daily_turnover",
    "avg_cost_bps_per_side",
    "avg_gross_exposure",
    "avg_net_exposure",
)


@pytest.fixture(scope="module")
def md():
    return generate(SyntheticSpec(n_stocks=120, n_days=900, seed=7, ic_true=0.08))


@pytest.fixture(scope="module")
def cfg():
    c = SystemConfig()
    c.universe.top_n_by_mcap = 100  # time-varying universe inside the 120 names
    return c


@pytest.fixture(scope="module")
def result(md, cfg):
    pipe = ResearchPipeline(cfg, use_ml=False)
    res = pipe.run(md, max_days=120)
    return pipe, res


@pytest.mark.slow
def test_pipeline_smoke(result, md):
    pipe, res = result
    bt = res.backtest
    assert len(bt.returns) == 120 and len(bt.nav) == 121
    for k in CORE:
        assert k in res.summary and np.isfinite(res.summary[k]), k
    assert np.isfinite(res.dsr) and 0.0 <= res.dsr <= 1.0
    assert res.decile is not None and np.isfinite(res.decile.t_stat)
    assert np.isfinite(res.ic_report["mean_ic"]) and res.ic_report["n"] >= 5
    # ICIR annualised at the sampled frequency (every H days), not 252
    assert res.ic_report["periods_per_year"] == pytest.approx(252 / pipe.cfg.alpha.horizon_days, rel=0.1)
    assert res.config_hash and res.data_hash
    assert not res.cost_sensitivity.isna().any().any()
    assert bt.diagnostics["n_failed_solves"] == 0
    assert "OVERALL" in format_result(res)


@pytest.mark.slow
def test_pipeline_hurdle_report_shape(result):
    _, res = result
    h = res.hurdles
    assert set(h["status"]) == GATES
    assert set(h["status"].values()) <= {"PASS", "FAIL", "skipped"}
    # standalone run without a config grid: PBO and incremental IC are skipped, not failed
    assert h["status"]["pbo"] == "skipped" and h["status"]["incremental_ic"] == "skipped"
    assert set(h["required"]) == {"phase_sharpe", "suspect", "dsr"}
    for g in h["required"]:
        assert h["status"][g] in ("PASS", "FAIL")
    assert "pbo" not in h["failed"]


@pytest.mark.slow
def test_pipeline_held_names_are_eligible(result):
    """Every *new* position is opened in a name eligible on its decision date
    (positions in names that later leave the universe may be unwound gradually)."""
    pipe, res = result
    elig = pipe.md_u.eligibility()
    W = res.backtest.weights
    prev = pd.Series(0.0, index=W.columns)
    for d, w in W.iterrows():
        opened = (w.abs() > 1e-8) & (prev.abs() <= 1e-8)
        bad = [t for t in W.columns[opened.to_numpy()] if not elig.loc[d, t]]
        assert not bad, (d, bad[:5])
        prev = w
    # alpha emitted only for eligible names at each decision date
    d = W.index[len(W) // 2]
    out = pipe.alpha.as_of(d)
    ok = elig.loc[d]
    assert set(out.score.index) <= set(ok[ok].index)
    # universe genuinely varies inside the sample (the test is not vacuous)
    assert elig.loc[W.index].sum(axis=1).max() < len(W.columns)


@pytest.mark.slow
def test_pipeline_is_deterministic(md, cfg, result):
    _, res1 = result
    res2 = ResearchPipeline(cfg, use_ml=False).run(md, max_days=120)
    np.testing.assert_array_equal(res1.backtest.nav.to_numpy(), res2.backtest.nav.to_numpy())
    assert res1.config_hash == res2.config_hash and res1.data_hash == res2.data_hash


@pytest.mark.slow
def test_pipeline_ledger_dedupes_reruns(md, cfg, tmp_path, monkeypatch):
    import qts.core.manifest as manifest

    # de-duplication needs a clean code version (a dirty tree always appends)
    monkeypatch.setattr(manifest, "code_version", lambda info=None: "cleansha")
    path = str(tmp_path / "ledger.jsonl")
    r1 = ResearchPipeline(cfg, use_ml=False, ledger_path=path).run(md, max_days=60)
    r2 = ResearchPipeline(cfg, use_ml=False, ledger_path=path).run(md, max_days=60)
    assert (
        r1.dsr_inputs["n_trials_raw"] == 1 and r2.dsr_inputs["n_trials_raw"] == 1
    )  # identical re-run: 1 trial
    assert r1.dsr == pytest.approx(r2.dsr)
    cfg2 = SystemConfig()
    cfg2.universe.top_n_by_mcap = 100
    cfg2.portfolio.risk_aversion = cfg.portfolio.risk_aversion * 2
    r3 = ResearchPipeline(cfg2, use_ml=False, ledger_path=path).run(md, max_days=60)
    assert r3.dsr_inputs["n_trials_raw"] == 2 and r3.dsr_inputs["var_trials_sr_annual"] > 0


# --------------------------------------------------------------------------- typed results, folds, decay (1.5, #27, #9)
@pytest.mark.slow
def test_pipeline_result_is_typed(result):
    from qts.alpha.evaluation import ICReport
    from qts.validation import HurdleResult

    pipe, res = result
    assert isinstance(res.ic_report, ICReport) and isinstance(res.hurdles, HurdleResult)
    assert res.ic_report.n == res.ic_report["n"]
    # IC decay at the primary + secondary horizons
    hs = sorted({pipe.cfg.alpha.horizon_days, *pipe.cfg.alpha.secondary_horizons})
    assert list(res.ic_report.decay.index) == hs
    # folds: PurgedWalkForward test blocks over the 120 return days (6 x 20 days)
    v = pipe.cfg.validation
    assert len(res.folds) == min(v.n_splits, 120 // v.min_fold_days) and res.folds["n_days"].sum() == 120
    assert list(res.fold_sharpes.index) == list(pd.DatetimeIndex(res.folds["start"]))
    assert res.manifest is not None and res.manifest.config_hash == res.config_hash
    assert res.manifest.data_hash == res.data_hash and res.git_sha == res.manifest.git_sha
    assert res.decile.nw_lags >= 1


@pytest.mark.slow
def test_pipeline_neutralises_exactly_the_label_set(result):
    """D1: the optimizer's neutral styles and the label's risk-only exposures are one set."""
    from qts.core.roles import ALPHA_THEME_STYLES, RISK_ONLY_STYLES

    pipe, _ = result
    assert set(pipe.neutral_styles()) == set(RISK_ONLY_STYLES) & set(pipe.cfg.risk.styles)
    d = pipe.backtest_dates()[-1]
    cols = set(pipe._exposures_fn(d).columns)
    assert not cols & set(ALPHA_THEME_STYLES)
    assert set(pipe.neutral_styles()) <= cols


# --------------------------------------------------------------------------- run registry + CLI
def test_cli_demo_writes_loadable_run_dir(tmp_path, capsys):
    from qts.cli import main
    from qts.core.config import SystemConfig
    from qts.pipeline import list_runs, load_run

    root, ledger = tmp_path / "runs", tmp_path / "ledger.jsonl"
    rc = main(
        [
            "demo",
            "--stocks",
            "60",
            "--days",
            "650",
            "--seed",
            "5",
            "--max-days",
            "40",
            "--runs-root",
            str(root),
            "--ledger",
            str(ledger),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "OVERALL" in out and "run directory" in out
    runs = list_runs(root)
    assert len(runs) == 1
    d = runs[0]
    for f in (
        "manifest.json",
        "config.yaml",
        "summary.json",
        "nav.parquet",
        "weights.parquet",
        "trades.parquet",
        "costs.parquet",
        "ic.parquet",
        "hurdles.json",
        "returns.parquet",
    ):
        assert (d / f).exists(), f
    rec = load_run(d)
    assert rec.run_id == d.name and isinstance(rec.config, SystemConfig)
    assert rec.config.hash() == rec.manifest.config_hash
    assert rec.summary["config_hash"] == rec.manifest.config_hash
    assert len(rec.nav) == 41 and rec.weights.shape[0] == 40
    assert set(rec.hurdles.status) == GATES
    assert np.isfinite(rec.summary["backtest"]["sharpe_net"])
    # ledger carries config hash, data hash and code version
    import json

    recs = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    assert (
        recs[-1]["config_hash"] == rec.manifest.config_hash
        and recs[-1]["data_hash"] == rec.manifest.data_hash
    )
    assert recs[-1]["code_version"].startswith(rec.manifest.git_sha)
    # runs list / show / ledger show
    assert main(["runs", "list", "--root", str(root)]) == 0
    assert d.name in capsys.readouterr().out
    assert main(["runs", "show", d.name, "--root", str(root)]) == 0
    assert "OVERALL" in capsys.readouterr().out
    assert main(["ledger", "show", "--path", str(ledger)]) == 0
    assert rec.manifest.config_hash in capsys.readouterr().out


@pytest.mark.slow
def test_signal_lag_is_forwarded_when_supported(md, cfg):
    """D4 wiring: cfg.execution.signal_lag_days reaches DailyBarSimulator.run when it has the kwarg."""
    import inspect

    from qts.backtest import DailyBarSimulator

    if "signal_lag_days" not in inspect.signature(DailyBarSimulator.run).parameters or not hasattr(
        cfg.execution, "signal_lag_days"
    ):
        pytest.skip("simulator / ExecutionConfig without signal_lag_days (D4 not landed yet)")
    from qts.cli import compare_lag_report

    c = cfg.with_overrides({"execution.signal_lag_days": 0})
    pipe = ResearchPipeline(c, use_ml=False)
    r0 = pipe.run(md, max_days=40)
    pipe.cfg.execution.signal_lag_days = 1
    r1 = pipe.run(md, max_days=40)
    assert r0.signal_lag_days == 0 and r1.signal_lag_days == 1
    assert r0.config_hash != r1.config_hash
    rep = compare_lag_report(r0, r1)
    assert "lag 0" in rep and "sharpe_net" in rep


@pytest.mark.slow
def test_backtest_dates_leave_room_for_the_signal_lag(result, md):
    """Lag investigation 2026-11-05: with signal_lag_days = L the first decision
    date reads risk/alpha L trading days earlier, which must be out of the risk
    warm-up.  Before the fix ``risk.as_of`` raised for L >= 2 on the first date."""
    from qts.backtest import DailyBarSimulator

    pipe, _ = result
    old = pipe.cfg.execution.signal_lag_days
    try:
        pipe.cfg.execution.signal_lag_days = 0
        d0 = pipe.backtest_dates()
        pipe.cfg.execution.signal_lag_days = 2
        d2 = pipe.backtest_dates()
        assert d2[0] == d0[2] and d2[-1] == d0[-1]
        risk = pipe.risk
        bt = DailyBarSimulator(pipe.cfg, pipe.cost).run(
            pipe.md_u,
            d2[:4],
            pipe._eligible_alpha_fn(),
            lambda d, a: risk.as_of(d, alpha=a),
            pipe.optimizer,
            nav0=pipe.nav0,
            neutral_styles=pipe.neutral_styles(),
            signal_lag_days=2,
        )
        assert bt.diagnostics["n_no_signal_days"] == 0
        assert [c["signal_date"] for c in bt.diagnostics["solve_log"]] == list(d0[:3])
    finally:
        pipe.cfg.execution.signal_lag_days = old


# --------------------------------------------------------------------------- planted oracle through the pipeline (#45)
@pytest.mark.slow
def test_planted_oracle_through_pipeline_is_flagged_suspect():
    """Inject the *future* residualised return (the label itself) as an extra alpha
    component of a fitted ``ResearchPipeline`` and backtest: the hurdle ``suspect``
    gate must fire (DSR alone cannot detect leakage)."""
    from qts.alpha.evaluation import rank_ic_series
    from qts.core.crosssection import zscore

    md_o = generate(SyntheticSpec(n_stocks=60, n_days=650, seed=9))
    c = SystemConfig()
    c.universe.top_n_by_mcap = 50
    pipe = ResearchPipeline(c, use_ml=False).fit(md_o)
    am = pipe.alpha
    H = c.alpha.horizon_days
    oracle = zscore(am.target_)  # future residualised H-day return at t
    am.ml_score_ = oracle  # the extra component (ML sleeve slot)
    am.composite_weight = 0.0
    am.combined_score_ = am._combine_history().where(am.eligible_.reindex_like(am.composite_score_))
    am.oos_ic_ = rank_ic_series(am.combined_score_, am.target_)
    am._half_life_cache = {}
    dates = pipe.backtest_dates()
    dates = dates[dates <= md_o.dates[-H - 2]][-80:]  # oracle exists only where the label is realised
    res = pipe.run(md_o, dates=dates)
    assert res.summary["sharpe_net"] > 2.0
    assert res.hurdles.gate("suspect").status == "FAIL"
    assert not res.hurdles.overall
    assert res.ic_report.mean_ic > 0.5


# --------------------------------------------------------------------------- audit 2 (#2, #4, #11, #16, #17)
@pytest.mark.slow
def test_pipeline_ledger_key_and_family(md, cfg, tmp_path, monkeypatch):
    import qts.core.manifest as manifest
    from qts.validation import TrialLedger

    path = str(tmp_path / "ledger.jsonl")
    monkeypatch.setattr(manifest, "code_version", lambda info=None: "sha_A")
    p1 = ResearchPipeline(cfg, use_ml=False, ledger_path=path)
    r1 = p1.run(md, max_days=40)
    assert p1.ledger_family() == f"pipeline_phase1:{r1.data_hash[:12]}"
    # a knob outside SystemConfig (rebalance_every) is a new trial
    ResearchPipeline(cfg, use_ml=False, ledger_path=path, rebalance_every=2).run(md, max_days=40)
    # new code version with everything else identical: new trial
    monkeypatch.setattr(manifest, "code_version", lambda info=None: "sha_B")
    ResearchPipeline(cfg, use_ml=False, ledger_path=path).run(md, max_days=40)
    # dirty tree: never de-duplicated
    monkeypatch.setattr(manifest, "code_version", lambda info=None: "sha_B+dirty")
    ResearchPipeline(cfg, use_ml=False, ledger_path=path).run(md, max_days=40)
    r5 = ResearchPipeline(cfg, use_ml=False, ledger_path=path).run(md, max_days=40)
    led = TrialLedger(path)
    fam = p1.ledger_family()
    assert led.n_trials(fam) == 5 and r5.dsr_inputs["n_trials_raw"] == 5
    specs = [r["run_spec"] for r in led.records(fam)]
    assert {s["rebalance_every"] for s in specs} == {1, 2}
    assert all(s["signal_lag_days"] == cfg.execution.signal_lag_days for s in specs)
    # other data -> other family
    md2 = generate(SyntheticSpec(n_stocks=120, n_days=900, seed=8, ic_true=0.08))
    p6 = ResearchPipeline(cfg, use_ml=False, ledger_path=path)
    r6 = p6.run(md2, max_days=40)
    assert p6.ledger_family() != fam and r6.dsr_inputs["n_trials_raw"] == 1


@pytest.mark.slow
def test_pbo_grid_configs_are_logged_as_trials(md, tmp_path):
    c = SystemConfig()
    c.universe.top_n_by_mcap = 100
    c.pipeline_pbo_configs = [{"portfolio.risk_aversion": 5.0}, {"portfolio.risk_aversion": 40.0}]
    path = str(tmp_path / "ledger.jsonl")
    pipe = ResearchPipeline(c, use_ml=False, ledger_path=path)
    res = pipe.run(md, max_days=60)
    # base + 2 grid configurations, all in the same family, counted before the DSR
    assert res.dsr_inputs["n_trials_raw"] == 3
    from qts.validation import TrialLedger

    assert TrialLedger(path).n_trials(pipe.ledger_family()) == 3


@pytest.mark.slow
@pytest.mark.parametrize("lag", [0, 1])
def test_evaluate_scores_the_traded_signal(md, cfg, lag):
    from qts.core import stats

    c = cfg.with_overrides({"execution.signal_lag_days": lag})
    pipe = ResearchPipeline(c, use_ml=False)
    res = pipe.run(md, max_days=100)
    assert pipe.alpha.signal_lag_days == lag
    ic = res.ic_report.ic_series.dropna()
    exec_dates = res.backtest.weights.index[:: c.alpha.horizon_days]
    assert set(ic.index) <= set(exec_dates)
    dates = pipe.md_u.dates
    t = ic.index[0]
    s = dates[dates.get_loc(t) - lag]
    score = pipe.alpha.as_of(s).score
    label = pipe.alpha.target_.loc[s].reindex(score.index)
    idx = pd.DatetimeIndex([s])
    want = stats.rank_ic_series(
        pd.DataFrame([score.to_numpy()], index=idx, columns=score.index),
        pd.DataFrame([label.to_numpy()], index=idx, columns=score.index),
    ).iloc[0]
    assert ic.loc[t] == pytest.approx(want, rel=1e-10)
    # decile Sharpe annualised at the IC report's inferred frequency (#15)
    assert res.decile is not None and res.ic_report.periods_per_year == pytest.approx(
        252 / c.alpha.horizon_days, rel=0.1
    )
    # ic_sign gate evaluated on the purged fold blocks (#16)
    by_fold = ResearchPipeline._ic_by_fold(
        res.ic_report.ic_series,
        [
            pd.DatetimeIndex(pipe.md_u.dates[(pipe.md_u.dates >= r.start) & (pipe.md_u.dates <= r.end)])
            for r in res.folds.itertuples()
        ],
    )
    assert len(by_fold) <= len(res.folds)


def test_ic_by_fold_groups_sampled_ic_by_block():
    d = pd.bdate_range("2024-01-01", periods=100)
    ic = pd.Series([0.1, -0.2, 0.3, 0.05, np.nan], index=d[[0, 10, 40, 60, 90]])
    blocks = [d[0:30], d[30:70], d[70:100]]
    out = ResearchPipeline._ic_by_fold(ic, blocks)
    assert list(out.index) == [d[0], d[30]]  # last block has no finite IC -> dropped
    assert out.iloc[0] == pytest.approx(-0.05) and out.iloc[1] == pytest.approx(0.175)
