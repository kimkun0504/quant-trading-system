"""Capacity curve (``qts capacity``) and the risk-forecast calibration report
(``qts risk calibrate``): both re-run only the backtest of one fitted baseline."""

from __future__ import annotations

import itertools
import json
import math
import warnings

import numpy as np
import pytest

from qts.core.config import SystemConfig
from qts.data.synthetic import SyntheticSpec, generate
from qts.research import capacity as CP
from qts.research import risk_calibration as RC
from qts.research.baseline import fit_baseline, variant_run
from qts.runs.context import RunContext
from qts.validation import TrialLedger

MAX_DAYS = 120


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    md = generate(SyntheticSpec(n_stocks=80, n_days=800, seed=9))
    cfg = SystemConfig().with_overrides({"universe.top_n_by_mcap": 70})
    root = tmp_path_factory.mktemp("cap")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bf = fit_baseline(md=md, config=cfg, use_ml=False, runs_root=str(root / "runs"))
    return md, cfg, bf, root


def test_half_sharpe_nav_by_hand():
    # 1.0 -> 0.8 -> 0.3: below 0.5 between 5e7 and 2.5e8; w = (0.8 - 0.5) / (0.8 - 0.3) = 0.6
    got = CP.half_sharpe_nav([5e6, 5e7, 2.5e8], [1.0, 0.8, 0.3])
    assert got == pytest.approx(math.exp(math.log(5e7) + 0.6 * (math.log(2.5e8) - math.log(5e7))))
    assert CP.half_sharpe_nav([5e6, 5e7], [1.0, 0.6]) is None
    assert CP.half_sharpe_nav([5e6, 5e7], [-0.2, -0.5]) is None


def test_cost_model_cost_per_side_rises_with_nav_for_the_same_book():
    import pandas as pd

    from qts.core.config import CostConfig, ExecutionConfig
    from qts.core.costs import LiquiditySnapshot
    from qts.tcost import TransactionCostModel

    t = pd.Index(["A", "B", "C"])
    snap = LiquiditySnapshot(
        date=pd.Timestamp("2021-03-01"), tickers=t, price=pd.Series([50.0, 20.0, 100.0], index=t),
        adv_usd=pd.Series([5e7, 1e7, 2e8], index=t), adv_shares=pd.Series([1e6, 5e5, 2e6], index=t),
        sigma_daily=pd.Series([0.02, 0.03, 0.015], index=t), half_spread=pd.Series(2e-4, index=t),
        borrow_fee=pd.Series(0.005, index=t),
    )  # fmt: skip
    tcm = TransactionCostModel(CostConfig(), ExecutionConfig())
    dw = pd.Series([0.01, -0.005, 0.002], index=t)
    impact = []
    for nav in (5e6, 5e7, 2.5e8, 1e9):
        est = tcm.estimate(dw * nav, snap, venue="routed").frame
        impact.append(float(est["impact_usd"].sum()) / float((dw.abs() * nav).sum()) * 1e4)
    assert all(a < b for a, b in itertools.pairwise(impact))
    # square root law: x200 the size -> about sqrt(200) the impact per dollar
    assert impact[-1] / impact[0] == pytest.approx(math.sqrt(200), rel=0.35)
    cap = [tcm.participation_cap(snap, nav=n) for n in (5e6, 1e9)]
    assert (cap[1] * 200 - cap[0]).abs().max() < 1e-9  # the ADV cap in NAV units shrinks as 1 / NAV


def test_nav_is_not_part_of_the_fit_cache_key(baseline):
    from qts.pipeline import ResearchPipeline

    md, cfg, bf, root = baseline
    c = cfg.with_overrides({"cache.enabled": True, "cache.root": str(root / "runs" / "_cache")})
    a = ResearchPipeline(c, use_ml=False, nav0=5e6)
    b = ResearchPipeline(c, use_ml=False, nav0=1e9)
    from qts.core.manifest import hash_market_data
    from qts.runs.cache import STAGE_EXCLUDED_DATA

    full = hash_market_data(md, n_chars=64, exclude=STAGE_EXCLUDED_DATA)
    cache = a._stage_cache()
    for stage in ("risk_fit", "alpha_fit"):
        assert a._cache_key(cache, stage, md, full) == b._cache_key(cache, stage, md, full)
    assert bf.cache_info == {"risk_fit": "stored", "alpha_fit": "stored"}


def test_capacity_curve_costs_rise_with_nav_and_runs_are_not_trials(baseline, tmp_path):
    md, cfg, _bf, root = baseline
    led = str(tmp_path / "l.jsonl")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = CP.run_capacity(
            navs=[1e9, 5e6, 2e8], config=cfg, md=md, use_ml=False, ledger_path=led, runs_root=str(root / "runs"),
            max_days=MAX_DAYS,
        )  # fmt: skip
    navs = [p.nav for p in r.points]
    assert navs == [5e6, 2e8, 1e9]  # sorted
    # the optimizer sees costs in NAV units: a larger NAV trades less ...
    to = [p.turnover_per_day for p in r.points]
    assert to[0] > to[1] > to[2]
    # ... and still pays more impact per dollar than the smallest book; fixed per-order fees weigh less
    impact = [p.impact_bps_per_side for p in r.points]
    assert impact[0] < min(impact[1], impact[2])
    fees = [p.fees_bps_per_side for p in r.points]
    assert fees[0] > fees[2]
    assert r.points[0].sharpe_net > r.points[-1].sharpe_net
    for p in r.points:
        assert p.cost_bps_per_side == pytest.approx(
            p.impact_bps_per_side + p.spread_bps_per_side + p.fees_bps_per_side
        )
    assert r.points[-1].traded_at_adv_cap >= r.points[0].traded_at_adv_cap
    assert r.cache == {"risk_fit": "hit", "alpha_fit": "hit"}  # the fit was reused, not refitted
    recs = TrialLedger(led).raw_records()
    assert len(recs) == 3 and {x["kind"] for x in recs} == {"capacity"} and recs[0]["family"] == r.family
    assert r.family.startswith("capacity:") and TrialLedger(led).n_trials(r.family) == 0
    assert sorted(x["nav"] for x in recs) == navs
    d = json.loads(open(r.json_path).read())
    assert len(d["points"]) == 3 and "half_sharpe_nav" in d
    assert "Capacity curve" in r.format()


def test_nu_variant_at_the_configured_value_is_the_baseline_backtest(baseline, tmp_path):
    _md, cfg, bf, _root = baseline
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = variant_run(
            bf, nav=5e6, context=RunContext(), ledger_path=None, runs_root=None, max_days=MAX_DAYS
        )
        b = variant_run(
            bf, nav=5e6, context=RunContext(), ledger_path=None, runs_root=None, max_days=MAX_DAYS,
            nu_scale=cfg.risk.alignment_nu_scale,
        )  # fmt: skip
        c = variant_run(
            bf,
            nav=5e6,
            context=RunContext(),
            ledger_path=None,
            runs_root=None,
            max_days=MAX_DAYS,
            nu_scale=4.0,
        )
    assert a.backtest.returns.equals(b.backtest.returns)
    assert a.config_hash == b.config_hash
    assert c.config_hash != a.config_hash and not c.backtest.weights.equals(a.backtest.weights)
    assert bf.pipe.cfg.risk.alignment_nu_scale == cfg.risk.alignment_nu_scale  # the baseline is untouched


def test_risk_calibration_report(baseline, tmp_path):
    md, cfg, bf, root = baseline
    led = str(tmp_path / "l.jsonl")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = RC.run_risk_calibration(
            nu_grid=[0.0, 0.05, 1.0], config=cfg, md=md, use_ml=False, ledger_path=led, runs_root=str(root / "runs"),
            max_days=MAX_DAYS,
        )  # fmt: skip
    assert [x.nu_scale for x in r.results] == [0.0, 0.05, 1.0]
    assert r.recommended_nu_scale == RC.recommend(r.results)
    best = min(abs(math.log(x.overall["ratio"])) for x in r.results)
    assert (
        abs(math.log(next(x for x in r.results if x.nu_scale == r.recommended_nu_scale).overall["ratio"]))
        <= best + 0.01
    )
    recs = TrialLedger(led).raw_records()
    assert {x["kind"] for x in recs} == {"calibration"} and [x["alignment_nu_scale"] for x in recs] == [
        0.0,
        0.05,
        1.0,
    ]
    assert r.cache == {"risk_fit": "hit", "alpha_fit": "hit"}
    assert "ONE logged trial" in r.format()
    # by-year table agrees with the backtest's own whole-period statistic
    res = variant_run(bf, nav=5e6, context=RunContext(), ledger_path=None, runs_root=None, max_days=MAX_DAYS)
    _, overall = RC.vol_by_year(res.backtest)
    assert overall["ratio"] == pytest.approx(res.backtest.realized_vs_exante(min_days=20)["ratio"])


def test_recommend_rule():
    mk = lambda nu, ratio: RC.NuResult(nu, {"ratio": ratio}, {}, 0.0)  # noqa: E731
    assert RC.recommend([mk(0.0, 1.3), mk(0.25, 1.05), mk(1.0, 0.97)]) == 1.0
    assert RC.recommend([mk(0.0, 1.0), mk(0.25, 1.005)]) == 0.0  # tie -> smaller nu
    assert RC.recommend([mk(0.0, float("nan"))]) is None
    assert np.isfinite(RC.TIE_TOL)
