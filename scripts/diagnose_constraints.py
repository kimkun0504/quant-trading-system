"""Constraint diagnostics: why is the transfer coefficient low and the vol target capped?

Fits the pipeline exactly as ``qts run`` does (stage cache reused with ``--cache``),
then -- without running the backtest -- replays ONE optimizer solve on each of
``--n-dates`` rebalance dates sampled evenly over the out-of-sample window, built
from the simulator's own context (signal-date alpha / risk / liquidity snapshot,
execution-date auction share, eligibility, no-trade list), twice per date:

* ``flat``: from an empty book (what the first day of a backtest sees);
* ``chain``: from the previous sampled date's ``chain`` solution (a stale book,
  so the trade caps and cost terms bite as they do mid-backtest).

Per date it prints the solve status, the gamma the vol-target search chose, the
ex-ante vol against the target, gross exposure, the transfer coefficient (TC),
the ex-ante IR (annualised alpha'w / ex-ante vol), ``vol_target_binding`` and the
binding-entry counts the optimizer exposes (``OptimizationResult.binding_constraints``:
``max_position``, ``participation``, ``gross``, ``net_hi/lo``, ``beta_neutral``,
``industry_band``, ``style_band``, ``no_trade``, ``force_to_zero``).  The optimizer
does not carry dual values (shadow prices) into its result -- the compiled cvxpy
problem is discarded after the solve -- so the alpha each constraint family costs
is measured by **relaxation solves** from the flat book: the position cap x2, the
ADV participation cap x2, both, and (``--relax-bands``) the industry / style /
beta bands x2.  The table reports what each buys in TC, ex-ante IR and expected
alpha, and a summary averages the deltas over the sampled dates.

Usage (same data flags as ``qts run`` / scripts/diagnose_signal.py)::

    python scripts/diagnose_constraints.py --store data/store --start 2000-01-01 \\
        --max-names 3000 --min-mcap-ever 3e8 --cache --ml --config configs/ml_h60.yaml
    python scripts/diagnose_constraints.py --synthetic            # smoke test

Runtime: ``n_dates x (2 + relaxations)`` solves, each a vol-target gamma search of
a few convex solves -- a few minutes at 1,000 names for the default 24 dates.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_signal import add_data_args, load_config, load_market
from qts.backtest.simulator import DailyBarSimulator, _BookState, _EventLog
from qts.backtest.spec import BacktestSpec
from qts.pipeline import ResearchPipeline
from qts.portfolio import SinglePeriodOptimizer
from qts.portfolio.optimizer import OK_STATUSES, OptimizationResult

TRADING_DAYS = 252
BINDING_KEYS = (
    "max_position",
    "participation",
    "gross",
    "net_hi",
    "net_lo",
    "beta_neutral",
    "industry_band",
    "style_band",
    "no_trade",
    "force_to_zero",
    "vol_cap",
)
RELAXATIONS: dict[str, dict[str, float]] = {
    "pos x2": {"max_position_weight": 2.0},
    "part x2": {"max_adv_participation": 2.0},
    "pos+part x2": {"max_position_weight": 2.0, "max_adv_participation": 2.0},
}
BAND_RELAXATIONS: dict[str, dict[str, float]] = {
    "bands x2": {"industry_band": 2.0, "style_band_sd": 2.0, "beta_band": 2.0},
}


@dataclass
class _Inputs:
    """Everything one optimizer solve needs, as the simulator would pass it."""

    date: pd.Timestamp
    sig_date: pd.Timestamp
    alpha: pd.Series
    risk: Any
    snapshot: Any
    no_trade: list[str]
    force_caps: dict[str, float]
    horizon_days: int
    half_life: float | None
    n_eligible: int


class SolveReplayer:
    """Rebuilds the simulator's per-day optimizer inputs for chosen dates."""

    def __init__(self, pipe: ResearchPipeline, dates: pd.DatetimeIndex, nav0: float):
        assert pipe.md_u is not None and pipe.risk is not None and pipe.optimizer is not None
        self.pipe = pipe
        self.cfg = pipe.cfg
        self.sim = DailyBarSimulator(pipe.cfg, pipe.cost)
        self.optimizer = pipe.optimizer
        elig = pipe.md_u.eligibility()
        spec = BacktestSpec.create(
            dates,
            nav0=nav0,
            rebalance_every=pipe.rebalance_every,
            venue=pipe.venue,
            neutral_styles=pipe.neutral_styles(),
            eligible_fn=lambda d: elig.columns[elig.loc[d].to_numpy()],
            signal_lag_days=pipe.signal_lag_days(),
        )
        self.spec = spec
        self.ctx = self.sim._context(pipe.md_u, spec, pipe.optimizer)
        self.alpha_fn = pipe._eligible_alpha_fn()
        risk = pipe.risk
        cfg = pipe.cfg

        def risk_fn(d: pd.Timestamp, alpha: pd.Series | None):
            return risk.as_of(d, alpha=alpha if cfg.risk.alignment else None)

        self.risk_fn = risk_fn
        self.nav0 = float(nav0)

    def inputs(self, i: int, w_pre: np.ndarray) -> _Inputs | None:
        """The optimizer inputs of decision index ``i`` given the pre-trade book
        (mirrors ``DailyBarSimulator._universe_step`` + ``_target_step``)."""
        ctx = self.ctx
        day = self.sim._day(ctx, i)
        if day.sig_date is None or i % ctx.rebalance_every != 0:
            return None
        state = _BookState(
            nav=self.nav0,
            w_pre=np.asarray(w_pre, dtype=float),
            streak=np.zeros(ctx.n_names),
            streak_usd=np.zeros(ctx.n_names),
            last_sign=np.zeros(ctx.n_names),
        )
        inel, no_trade, force_caps = self.sim._universe_step(ctx, day, state, _EventLog())
        sig_date = day.sig_date
        a = self.alpha_fn(sig_date)
        exp_ret = pd.Series(a.expected_return, dtype=float).reindex(ctx.tickers)
        exp_ret[~day.alive] = 0.0
        exp_ret[inel] = 0.0
        exp_ret = exp_ret.fillna(0.0)
        risk = self.risk_fn(sig_date, exp_ret)
        snap = self.sim._optimizer_snapshot(ctx.snap_at(sig_date), day.snap_exec)
        hl = getattr(a, "half_life_days", None)
        return _Inputs(
            date=day.t,
            sig_date=sig_date,
            alpha=exp_ret,
            risk=risk,
            snapshot=snap,
            no_trade=no_trade,
            force_caps=force_caps,
            horizon_days=int(getattr(a, "horizon_days", None) or self.optimizer.horizon_days),
            half_life=None if hl is None else float(hl),
            n_eligible=int((day.alive & ~inel).sum()),
        )

    def solve(self, inp: _Inputs, w_pre: np.ndarray, optimizer: SinglePeriodOptimizer) -> OptimizationResult:
        ctx = self.ctx
        kw: dict[str, Any] = {}
        if ctx.solve_accepts_hl and inp.half_life is not None:
            kw["alpha_half_life_days"] = inp.half_life
        return optimizer.solve(
            alpha=inp.alpha,
            risk=inp.risk,
            w_current=pd.Series(np.asarray(w_pre, dtype=float), index=ctx.tickers),
            nav=self.nav0,
            snapshot=inp.snapshot,
            venue=ctx.venue,
            neutral_styles=ctx.neutral_styles,
            no_trade=inp.no_trade,
            horizon_days=inp.horizon_days,
            force_to_zero=inp.force_caps or None,
            **kw,
        )

    def relaxed_optimizer(self, factors: dict[str, float]) -> SinglePeriodOptimizer:
        """A copy of the pipeline's optimizer with the named ``PortfolioConfig``
        caps multiplied by ``factors`` (``beta_band`` is read from the optimizer)."""
        ov = {}
        for k, f in factors.items():
            ov[f"portfolio.{k}"] = float(getattr(self.cfg.portfolio, k)) * f
        cfg = self.cfg.with_overrides(ov)
        return SinglePeriodOptimizer(
            cfg.portfolio,
            self.pipe.cost,
            horizon_days=self.optimizer.horizon_days,
            exec_cfg=self.cfg.execution,
            beta_band=cfg.portfolio.beta_band,
        )


# ----------------------------------------------------------------------------- reporting
def _exante_ir(res: OptimizationResult, horizon_days: int) -> float:
    """Annualised alpha'w / annualised ex-ante vol of the returned book."""
    if not np.isfinite(res.expected_alpha) or not np.isfinite(res.expected_risk) or res.expected_risk <= 0:
        return float("nan")
    return float(res.expected_alpha * TRADING_DAYS / max(horizon_days, 1) / res.expected_risk)


def _row(res: OptimizationResult, inp: _Inputs, target: float | None) -> dict[str, Any]:
    w = res.w_target.to_numpy(dtype=float)
    b = res.binding_constraints or {}
    row: dict[str, Any] = {
        "status": res.status,
        "gamma": res.risk_aversion,
        "exante_vol": res.expected_risk,
        "target": float("nan") if target is None else target,
        "gross": float(np.abs(w).sum()),
        "n_held": int((np.abs(w) > 1e-6).sum()),
        "TC": res.transfer_coefficient,
        "exIR": _exante_ir(res, inp.horizon_days),
        "alpha_ann": res.expected_alpha * TRADING_DAYS / max(inp.horizon_days, 1),
        "turnover": res.turnover,
        "vt": res.vol_target_binding or "",
        "n_solves": res.n_solves,
    }
    for k in BINDING_KEYS:
        row[f"b_{k}"] = int(b.get(k, 0))
    extra = sorted(set(b) - set(BINDING_KEYS))
    row["b_other"] = ",".join(f"{k}={b[k]}" for k in extra)
    return row


def _fmt_main(df: pd.DataFrame) -> str:
    cols = ["status", "gamma", "exante_vol", "target", "gross", "n_held", "TC", "exIR", "vt", "n_solves"]
    bcols = [c for c in df.columns if c.startswith("b_") and c != "b_other" and df[c].abs().sum() > 0]
    out = df[cols + bcols].copy()
    out.columns = [c.replace("b_", "") for c in out.columns]
    with pd.option_context(
        "display.width", 250, "display.max_columns", 60, "display.float_format", "{:.3f}".format
    ):
        return out.to_string()


def _fmt_relax(df: pd.DataFrame) -> str:
    with pd.option_context(
        "display.width", 250, "display.max_columns", 60, "display.float_format", "{:+.3f}".format
    ):
        return df.to_string()


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    ap.add_argument("--n-dates", type=int, default=24, help="rebalance dates sampled evenly (default 24)")
    ap.add_argument("--nav", type=float, default=5e6, help="NAV of the replayed solves (qts run default)")
    ap.add_argument(
        "--rebalance-every", type=int, default=None, help="default: portfolio.rebalance_every_days"
    )
    ap.add_argument("--relax-bands", action="store_true", help="also relax industry/style/beta bands x2")
    ap.add_argument("--no-relax", action="store_true", help="skip the relaxation solves (fastest)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )

    cfg = load_config(a)
    md = load_market(a)
    if md is None:
        print("give --store or --synthetic", file=sys.stderr)
        return 2
    print(
        f"loaded: {md.close.shape[1]} ids x {md.close.shape[0]} days ({md.dates[0].date()}..{md.dates[-1].date()})"
    )
    t0 = time.perf_counter()
    pipe = ResearchPipeline(cfg, use_ml=a.ml, runs_root=None, rebalance_every=a.rebalance_every, nav0=a.nav)
    pipe.fit(md)
    print(f"fitted in {time.perf_counter() - t0:.1f}s  (cache {'on' if cfg.cache.enabled else 'off'})")

    pc = cfg.portfolio
    print(
        f"portfolio: mode={pc.risk_aversion_mode} target_vol={pc.target_vol_annual} gross_max={pc.gross_leverage_max} "
        f"max_pos={pc.max_position_weight} adv_part={pc.max_adv_participation} "
        f"(from_urgency={pc.participation_from_urgency}) ind_band={pc.industry_band} "
        f"style_band={pc.style_band_sd} beta_band={pc.beta_band} trade_av={pc.trade_aversion}"
    )

    dates = pipe.backtest_dates()
    rep = SolveReplayer(pipe, dates, nav0=a.nav)
    n_dec = len(dates) - 1
    n = max(1, min(a.n_dates, n_dec))
    idx = np.unique(np.linspace(0, n_dec - 1, n).round().astype(int))
    # only solve days are replayed (portfolio.rebalance_every_days): snap to the solve before
    k = max(int(rep.ctx.rebalance_every), 1)
    idx = np.unique(idx // k * k)
    target = pc.target_vol_annual if pc.risk_aversion_mode == "target_vol" else None
    relax = dict(RELAXATIONS)
    if a.relax_bands:
        relax.update(BAND_RELAXATIONS)
    if a.no_relax:
        relax = {}
    relaxed = {name: rep.relaxed_optimizer(f) for name, f in relax.items()}
    print(
        f"replaying {len(idx)} of {n_dec} decision dates ({dates[idx[0]].date()}..{dates[idx[-1]].date()}), "
        f"solves per date: 2 + {len(relaxed)} relaxations"
    )

    flat_rows: dict[pd.Timestamp, dict[str, Any]] = {}
    chain_rows: dict[pd.Timestamp, dict[str, Any]] = {}
    relax_rows: dict[pd.Timestamp, dict[str, float]] = {}
    w_chain = np.zeros(rep.ctx.n_names)
    t_start = time.perf_counter()
    for k, i in enumerate(idx):
        inp = rep.inputs(int(i), np.zeros(rep.ctx.n_names))
        if inp is None:
            continue
        flat = rep.solve(inp, np.zeros(rep.ctx.n_names), rep.optimizer)
        flat_rows[inp.date] = {"n_elig": inp.n_eligible, **_row(flat, inp, target)}
        inp_c = rep.inputs(int(i), w_chain) or inp
        chain = rep.solve(inp_c, w_chain, rep.optimizer)
        chain_rows[inp.date] = _row(chain, inp_c, target)
        if chain.status in OK_STATUSES:
            w_chain = chain.w_target.reindex(rep.ctx.tickers).fillna(0.0).to_numpy(dtype=float)
        rr: dict[str, float] = {}
        for name, opt in relaxed.items():
            r = rep.solve(inp, np.zeros(rep.ctx.n_names), opt)
            rr[f"dTC {name}"] = r.transfer_coefficient - flat.transfer_coefficient
            rr[f"dIR {name}"] = _exante_ir(r, inp.horizon_days) - _exante_ir(flat, inp.horizon_days)
            rr[f"dAlpha% {name}"] = (
                100 * (r.expected_alpha - flat.expected_alpha) * TRADING_DAYS / inp.horizon_days
            )
            rr[f"vol {name}"] = r.expected_risk
        if rr:
            relax_rows[inp.date] = rr
        if a.verbose or k == 0:
            el = time.perf_counter() - t_start
            print(
                f"  [{k + 1}/{len(idx)}] {inp.date.date()} flat {flat.status} TC {flat.transfer_coefficient:.2f}  ({el:.0f}s)"
            )

    if not flat_rows:
        print("no rebalance dates could be replayed (signal lag longer than the window?)", file=sys.stderr)
        return 1
    fdf = pd.DataFrame.from_dict(flat_rows, orient="index")
    cdf = pd.DataFrame.from_dict(chain_rows, orient="index")
    fdf.index.name = cdf.index.name = "date"

    print("\n== solve from a FLAT book (binding = entries with a nonzero dual; vt = vol_target_binding) ==")
    print(_fmt_main(fdf))
    print("\n== solve from the PREVIOUS sampled solution (stale book) ==")
    print(_fmt_main(cdf))

    def _summ(df: pd.DataFrame, label: str) -> None:
        ok = df["status"].isin(OK_STATUSES)
        n_ok = int(ok.sum())
        vt = df["vt"].value_counts()
        print(
            f"{label}: {n_ok}/{len(df)} ok; mean TC {df.loc[ok, 'TC'].mean():.3f}  mean exIR {df.loc[ok, 'exIR'].mean():.2f}  "
            f"mean ex-ante vol {df.loc[ok, 'exante_vol'].mean():.3%} (target {target if target is not None else 'n/a'})  "
            f"mean gross {df.loc[ok, 'gross'].mean():.2f}  vol_target_binding: "
            + ", ".join(f"{k or 'none'}={v}" for k, v in vt.items())
        )
        bcols = [c for c in df.columns if c.startswith("b_") and c != "b_other"]
        share = (df.loc[ok, bcols] > 0).mean().sort_values(ascending=False)
        share = share[share > 0]
        if len(share):
            print(
                "  share of solves where each constraint binds: "
                + ", ".join(
                    f"{c[2:]} {v:.0%} (mean entries {df.loc[ok, c].mean():.0f})" for c, v in share.items()
                )
            )

    print("\n== summary ==")
    _summ(fdf, "flat ")
    _summ(cdf, "chain")

    if relax_rows:
        rdf = pd.DataFrame.from_dict(relax_rows, orient="index")
        rdf.index.name = "date"
        print("\n== relaxation solves from the FLAT book: what each cap buys (delta vs the base solve) ==")
        print(_fmt_relax(rdf))
        print("\nmean over dates:")
        for name in relaxed:
            print(
                f"  {name:12s} dTC {rdf[f'dTC {name}'].mean():+.3f}  dexIR {rdf[f'dIR {name}'].mean():+.3f}  "
                f"dAlpha {rdf[f'dAlpha% {name}'].mean():+.2f}% NAV/yr  ex-ante vol {rdf[f'vol {name}'].mean():.2%}"
            )
        print(
            "\nreading: a large dTC/dIR from 'pos x2' means the per-name cap is what the vol-target search is "
            "capped on ('caps'); from 'part x2' the ADV cap throttles the trade into the book (and the chain "
            "solve's TC should be well below the flat one).  Shadow prices are not exposed by the optimizer; "
            "these deltas are the measured substitute."
        )
    print(f"\ntotal {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
