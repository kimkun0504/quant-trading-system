"""Per-industry breakdown of a finished run: is the edge broad across industries or
concentrated in a few?  (``qts breakdown industries RUN``; handbook 12 "Research
tools", docs/notes/industry_breakdown_2026-11-14.md.)

A **diagnostic**: it reads a run directory, reloads the run's market through the one
research loader (:func:`qts.research.data.load_research_market`: the run's store
flags, the holdout seal, the load-coverage guard) and splits the book the run actually
held by industry.  Nothing is re-optimised, nothing is logged to the trial ledger and no
number of the strategy changes.  The result is written to
``<run_dir>/industry_breakdown.json`` (:data:`OUTPUT_FILE`, shape below) and printed as
a text report (:meth:`IndustryBreakdown.format`).

Industries
----------
The industry of a name on decision date ``t`` is the risk model's point-in-time label:
``MarketData.industry_as_of(t)`` (the last ``industry_history`` row dated ``<= t``,
unclassified names filled from the static ``industry``; the same rule as
:func:`qts.risk.exposures.industry_panel`, which builds the risk model's industry
dummies).  On the real store these are the Fama-French 48 industries.  The roll-up is the
FF12 sector the risk model's thin-industry shrinkage uses
(:func:`qts.risk.factor_returns.industry_parent`, ``qts.data.loaders.FF48_TO_FF12``);
labels outside that table (the synthetic ``IND00`` ...) have no roll-up.

What is computed, per industry and per FF12 sector
--------------------------------------------------
1. **Positioning**: mean long, short (negative), net and gross weight (fractions of NAV),
   mean number of names held long / short (``|w| > HELD_EPS``), share of the book's
   gross (sum over days of the industry's gross / sum of the book's).
2. **P&L attribution**: the daily contribution ``sum_{i in industry} w_{i,t} r_{i,t+1}``
   with the run's post-trade weights (``weights.parquet``) and the per-name returns the
   simulator booked (:func:`booked_returns`: the cleaned close, a halt's move on the
   resumption day, a held delisting at the return recorded in ``delistings.parquet``).
   The contributions sum to the run's recorded gross return; the residual is reported
   (:data:`RECON_TOL` per day).  Costs: ``costs.parquet`` holds USD per *day*, not per
   name, so each day's trading cost (spread + impact + fees) is allocated to names in
   proportion to their traded value (``trades.parquet``) and the day's borrow cost to
   their short value; the allocation sums to the book's cost exactly but is an
   approximation per industry (impact is convex in the trade, and fees / borrow rates
   differ by name).  Also: share of the total gross P&L, annualised Sharpe of the
   industry's daily P&L stream, its Newey-West t-statistic, and the monthly hit rate
   (share of calendar months with positive P&L among the months it was held).
3. **Signal quality**: rank IC of the traded score (``AlphaModel.traded_score_``, what
   ``as_of`` scales and the book traded) against the residualised forward target
   (``AlphaModel.target_``), inside each industry's cross-section, on the run's own IC
   dates (every ``alpha.horizon_days``-th decision date, score and label of the signal
   date ``L`` days earlier: ``ResearchPipeline._signal_quality``).  Industries with fewer
   than :data:`IC_MIN_NAMES` scored names on a date are skipped on that date; the
   coverage is reported.  The scores come from re-fitting the pipeline through the stage
   cache exactly as ``qts run`` does (nothing re-optimised; a warm cache loads the cached
   fits); a cold cache is announced before the refit, and ``refuse_refit`` /
   ``--no-refit`` refuses it.  The re-fitted scores are checked against the run's
   recorded ``ic.parquet`` (``ic.score_check``).
4. **Breadth** (industry level, active industries only: :data:`ACTIVE_MIN_GROSS_SHARE`
   of the book's gross or more): number / share of industries with positive P&L and the
   one-sided binomial p-value against a fair coin; the P&L share of the
   :data:`TOP_K` best industries and of the rest; the effective number of industries
   contributing (inverse Herfindahl of the absolute P&L shares); the number with
   ``|t| >= 2`` against ``5 % x count`` expected by chance, and the Holm-adjusted list;
   the book without its top 5 industries; and a verdict (:func:`breadth_verdict`) under
   thresholds fixed in this module before any real result was seen.

The seal
--------
A run made on the sealed window (``manifest.json`` ``sealed_end`` set) is broken down
on the sealed window only: with no seal in force the breakdown refuses before loading,
and a sealed load that does not end on the run's ``sealed_end`` (the seal moved, or
other flags) is refused too.  A full-history run (made before the seal, ``--unseal``, or
the exempt mid-cap check) loaded while a seal is in force is broken down on the
pre-holdout dates only (``window: "truncated"``): a diagnostic never reads the holdout.

Store flags
-----------
From ``manifest.json`` ``command`` (the ``qts run --store ...`` line; ledger records do
not store the loader flags), or given explicitly (``spec``; ``--store`` / ``--start`` /
``--end`` / ``--max-rank`` / ``--max-names`` / ``--min-mcap-ever``; explicit flags
replace the command's).  The loaded close panel must hash to the run's ``market_hash``.

JSON shape (``schema: "qts.industry_breakdown/1"``; NaN -> ``null``)
----------------------------------------------------------------
::

    schema, created_utc, run_id, run_dir, config_hash, use_ml, signal_lag_days, horizon_days
    data:      store_spec{...}, window ("sealed" | "full" | "truncated"), sealed_end,
               holdout_start, market_hash, load_coverage{...} | null
    period:    start, end, n_days, n_months, trading_days_per_year
    book:      gross_pnl_total, gross_pnl_ann, net_pnl_ann, cost_ann, sharpe_gross, t_gross,
               mean_gross, mean_net
    reconciliation: book_gross_total, sum_contributions, residual_total,
               max_abs_daily_residual, tolerance, ok, delistings ("delistings.parquet" | "none"),
               net_max_abs_daily_residual | null
    costs:     available, method, unallocated_ann, note
    ic:        available, reason, n_eval_dates, min_names, cache{risk_fit, alpha_fit},
               score_check{n_dates, n_run_dates, max_abs_diff, ok} | null, book_mean_ic, book_ic_t,
               within_industry_mean_ic, within_industry_ic_t
    levels:    industry: {label, rows[...], breadth{...}, verdict{label, rule, text}}
               ff12:     {label, rows[...], breadth{...}, verdict: null} | null
    thresholds{...}, help{statistic: sentence}, notes[...], seconds

    row:       industry, parent, active, long, short, net, gross, gross_share, n_long, n_short,
               pnl_total, pnl_ann, pnl_share, sharpe, t, hit_rate, n_months, cost_ann,
               net_pnl_ann, ic_mean, ic_t, ic_n_dates, ic_coverage, ic_mean_names
               (rows sorted by pnl_total, largest first; "(unclassified)" last)
    breadth:   n_held, n_active, n_positive, positive_share, binom_p, top, top_share,
               rest_share, top_share_of_positive, n_eff, n_eff_ratio, n_t2, n_t2_chance,
               holm[{industry, t, p, p_holm}], ex_top{pnl_ann, t}, book_t

Weights, P&L and costs are fractions of NAV (``pnl_ann`` = 252 x the mean daily
contribution); the text report prints them in percent.
"""

from __future__ import annotations

import argparse
import math
import shlex
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from qts.research.common import json_safe, resolve_repo_path, utc_stamp, write_json
from qts.research.data import Printer, StoreSpec, _quiet

if TYPE_CHECKING:
    from qts.core.config import SystemConfig
    from qts.core.types import MarketData
    from qts.pipeline import ResearchPipeline

SCHEMA = "qts.industry_breakdown/1"
OUTPUT_FILE = "industry_breakdown.json"
UNCLASSIFIED = "(unclassified)"
UNMAPPED = "(unmapped)"

#: |weight| above this counts as a held name
HELD_EPS = 1e-8
#: per-day |sum of contributions - recorded gross| that still reconciles
RECON_TOL = 1e-8
#: fewer scored names of an industry on an IC date -> no IC for it that date
IC_MIN_NAMES = 5
#: rank IC of the whole book on a date needs this many names (qts.core.stats.IC_MIN_OBS)
BOOK_IC_MIN_NAMES = 20
#: re-fitted book IC vs the run's ic.parquet: max |difference| that counts as the same scores
SCORE_CHECK_TOL = 1e-6

# --------------------------------------------------------------------------- breadth thresholds
# Fixed 2026-11-14 in code and in docs/notes/industry_breakdown_2026-11-14.md, before the
# breakdown was run on any real result (only the synthetic market existed when they were
# set).  Changing them after looking at a real breakdown would be fitting the verdict.

#: an industry is *active* (counted in the breadth statistics) when it held at least this
#: share of the book's gross over the run
ACTIVE_MIN_GROSS_SHARE = 0.0025
#: "the best few industries": the top-K by total P&L
TOP_K = 5
#: below this many active industries the verdict is "undetermined"
VERDICT_MIN_ACTIVE = 5
#: the book's gross P&L must have a Newey-West t of at least this, else "undetermined"
#: (no edge large enough to say where it comes from)
VERDICT_MIN_BOOK_T = 2.0
#: broad: at least this share of the total gross P&L from outside the top K ...
BROAD_MIN_REST_SHARE = 0.40
#: ... and at least this share of the active industries with positive P&L ...
BROAD_MIN_POSITIVE_SHARE = 0.60
#: ... and an effective number of contributing industries of at least this x n_active
BROAD_MIN_NEFF_RATIO = 0.25
#: concentrated: less than this share of the total gross P&L from outside the top K ...
CONCENTRATED_MAX_REST_SHARE = 0.15
#: ... or an effective number of contributing industries below max(2, this x n_active)
CONCENTRATED_MAX_NEFF_RATIO = 0.10
#: |t| of an industry's P&L that "clears" in the count, the chance rate it is compared
#: with (5 % x count, as specified) and the family-wise level of the Holm list
T_CLEAR = 2.0
CHANCE_RATE = 0.05
HOLM_ALPHA = 0.05

THRESHOLDS: dict[str, float] = {
    "active_min_gross_share": ACTIVE_MIN_GROSS_SHARE,
    "top_k": TOP_K,
    "verdict_min_active": VERDICT_MIN_ACTIVE,
    "verdict_min_book_t": VERDICT_MIN_BOOK_T,
    "broad_min_rest_share": BROAD_MIN_REST_SHARE,
    "broad_min_positive_share": BROAD_MIN_POSITIVE_SHARE,
    "broad_min_neff_ratio": BROAD_MIN_NEFF_RATIO,
    "concentrated_max_rest_share": CONCENTRATED_MAX_REST_SHARE,
    "concentrated_max_neff_ratio": CONCENTRATED_MAX_NEFF_RATIO,
    "t_clear": T_CLEAR,
    "chance_rate": CHANCE_RATE,
    "holm_alpha": HOLM_ALPHA,
    "ic_min_names": IC_MIN_NAMES,
}

#: one sentence of plain-English help per statistic (printed under the report, in the JSON)
HELP: dict[str, str] = {
    "long": "Average weight of the industry's long positions, in % of NAV.",
    "short": "Average weight of its short positions (negative), in % of NAV.",
    "net": "Long plus short: the industry bet the book keeps (the optimizer holds it inside its "
    "industry band, so it is small).",
    "gross": "Long minus short: how much capital the book puts into the industry, in % of NAV.",
    "gross_share": "The industry's share of the book's total gross over the run.",
    "n_long": "Average number of names held long on a day (#S: short).",
    "pnl_ann": "Annualised gross P&L contribution (sum of weight x return, before costs), in % of NAV "
    "per year.",
    "pnl_share": "The industry's share of the book's total gross P&L (negative: it lost money; the "
    "shares sum to 100 %; not shown when the book as a whole lost money).",
    "sharpe": "Annualised Sharpe ratio of the industry's daily P&L stream (mean / standard deviation "
    "x sqrt(252)).",
    "t": "Newey-West t-statistic of the industry's mean daily P&L; |t| >= 2 is roughly 'unlikely "
    "to be luck' for one industry taken alone.",
    "hit_rate": "Share of calendar months with positive P&L, among the months the industry was held.",
    "cost_ann": "Trading and borrow costs allocated to the industry (pro rata to traded value and to "
    "short value within each day), in % of NAV per year.",
    "net_pnl_ann": "Gross P&L minus the allocated costs, in % of NAV per year.",
    "ic_mean": "Mean rank correlation, inside the industry, between the traded score and the "
    "residualised forward return, on the run's own IC dates.",
    "ic_t": "Newey-West t-statistic of that mean IC.",
    "ic_n_dates": "Number of IC dates on which the industry had enough scored names "
    f"(at least {IC_MIN_NAMES}).",
    "ic_coverage": "Share of the run's IC dates on which the industry had enough scored names.",
    "reconciliation": "Sum of the industry contributions against the run's recorded gross return; "
    "the residual should be about zero.",
    "positive": "Active industries (holding at least 0.25 % of the book's gross) with positive "
    "total P&L, and the chance of seeing that many or more if each were a coin flip "
    "(one-sided binomial p).",
    "top_share": "Share of the book's gross P&L earned by its five best industries; 'rest' is what "
    "all the other industries add.",
    "n_eff": "Effective number of industries contributing: 1 / sum of squared shares of the "
    "absolute P&L (k industries contributing equally give k).",
    "n_t2": "How many industries clear |t| >= 2, against the 5 % x count expected by chance; the "
    "Holm list keeps those still significant after correcting for testing every industry.",
    "ex_top": "The book's P&L without its five best industries: a deliberately pessimistic check, "
    "since it removes the luckiest ones.",
    "book_t": "Newey-West t-statistic of the whole book's mean daily gross P&L; below 2 there is not "
    "enough edge to say where it comes from.",
    "within_ic": "Name-weighted average of the industry ICs on each date: the signal's skill at "
    "ranking stocks against their own industry peers.",
    "score_check": "Whether the re-fitted scores reproduce the run's recorded IC (then they are the "
    "scores the book traded).",
}


class BreakdownError(ValueError):
    """The breakdown cannot be made (missing files, a refused load, a mismatched market)."""


# ============================================================================ the run directory
@dataclass
class RunBook:
    """A run directory's book, aligned on its decision dates (rows) and tickers (columns)."""

    run_dir: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    cfg: SystemConfig
    dec: pd.DatetimeIndex  #: decision dates t (the post-trade book is held t -> t+1)
    ret_dates: pd.DatetimeIndex  #: realisation dates t+1
    tickers: pd.Index
    W: np.ndarray  #: (n, N) post-trade weights
    X: np.ndarray | None  #: (n, N) trades, None without trades.parquet
    gross: np.ndarray  #: (n,) recorded gross return realised at t+1
    net: np.ndarray | None
    costs: pd.DataFrame | None  #: USD by decision date (spread, impact, fees, borrow, total)
    nav: np.ndarray | None  #: (n,) pre-trade NAV at t
    delistings: pd.DataFrame | None
    ic: pd.Series  #: the run's sampled rank IC (ic.parquet), indexed by execution date
    ic_dates_full: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))

    @property
    def run_id(self) -> str:
        return str(self.manifest.get("run_id") or self.run_dir.name)

    @property
    def sealed_end(self) -> str:
        return str(self.manifest.get("sealed_end") or self.summary.get("sealed_end") or "")

    @property
    def holdout_start(self) -> str:
        return str(self.manifest.get("holdout_start") or self.summary.get("holdout_start") or "")

    @property
    def unsealed(self) -> bool:
        return bool(self.manifest.get("unsealed") or self.summary.get("unsealed"))

    @property
    def use_ml(self) -> bool:
        return bool(self.summary.get("use_ml", False))

    @property
    def run_dates(self) -> pd.DatetimeIndex:
        """The simulator's dates: every decision date plus the last realisation date."""
        return pd.DatetimeIndex([*self.dec, self.ret_dates[-1]]) if len(self.dec) else self.dec

    def drop_tickers(self, drop: pd.Index) -> RunBook:
        """The book without the columns ``drop`` (names it never held)."""
        keep = ~self.tickers.isin(drop)
        return replace(
            self,
            tickers=self.tickers[keep],
            W=self.W[:, keep],
            X=None if self.X is None else self.X[:, keep],
        )

    def truncate(self, last: pd.Timestamp) -> RunBook:
        """The rows whose return is realised on or before ``last``."""
        k = int((self.ret_dates <= last).sum())
        if k == len(self.dec):
            return self

        def cut(a: np.ndarray | None) -> np.ndarray | None:
            return None if a is None else a[:k]

        return replace(
            self,
            dec=self.dec[:k],
            ret_dates=self.ret_dates[:k],
            W=self.W[:k],
            X=cut(self.X),
            gross=self.gross[:k],
            net=cut(self.net),
            costs=None if self.costs is None else self.costs.iloc[:k],
            nav=cut(self.nav),
            ic=self.ic[self.ic.index <= last],
        )


def _read_json(p: Path) -> dict[str, Any]:
    import json

    return json.loads(p.read_text()) if p.exists() else {}


def read_run_book(run_dir: str | Path) -> RunBook:
    """Read a run directory written by :func:`qts.runs.registry.save_run`."""
    from qts.core.config import SystemConfig
    from qts.runs.registry import read_delistings

    d = Path(run_dir)
    for name in ("manifest.json", "config.yaml", "weights.parquet", "returns.parquet"):
        if not (d / name).exists():
            raise BreakdownError(f"{d} is not a finished run directory (no {name})")
    w = pd.read_parquet(d / "weights.parquet")
    w.columns = pd.Index([str(c) for c in w.columns])
    ret = pd.read_parquet(d / "returns.parquet")
    n = min(len(w), len(ret))
    if n < 2:
        raise BreakdownError(f"{d}: fewer than two backtest days")
    dec = pd.DatetimeIndex(w.index[:n])
    ret_dates = pd.DatetimeIndex(ret.index[:n])
    if not ret_dates[:-1].equals(dec[1:]):
        raise BreakdownError(f"{d}: returns.parquet is not dated on the day after each weights row")
    tickers = pd.Index(w.columns)

    def grid(f: str) -> np.ndarray | None:
        p = d / f
        if not p.exists():
            return None
        x = pd.read_parquet(p)
        x.columns = pd.Index([str(c) for c in x.columns])
        return x.reindex(index=dec, columns=tickers).fillna(0.0).to_numpy(dtype=float)

    costs = (
        pd.read_parquet(d / "costs.parquet").reindex(dec).fillna(0.0)
        if (d / "costs.parquet").exists()
        else None
    )
    nav = None
    if (d / "nav.parquet").exists():
        nav = pd.read_parquet(d / "nav.parquet")["nav"].reindex(dec).to_numpy(dtype=float)
    ic = pd.Series(dtype=float, name="rank_ic")
    if (d / "ic.parquet").exists():
        ic_df = pd.read_parquet(d / "ic.parquet")
        if "rank_ic" in ic_df:
            ic = ic_df["rank_ic"].astype(float)
            ic.index = pd.DatetimeIndex(ic.index)
    return RunBook(
        run_dir=d,
        manifest=_read_json(d / "manifest.json"),
        summary=_read_json(d / "summary.json"),
        cfg=SystemConfig.from_yaml(d / "config.yaml"),
        dec=dec,
        ret_dates=ret_dates,
        tickers=tickers,
        W=w.reindex(index=dec).fillna(0.0).to_numpy(dtype=float),
        X=grid("trades.parquet"),
        gross=ret["gross"].to_numpy(dtype=float)[:n],
        net=ret["net"].to_numpy(dtype=float)[:n] if "net" in ret else None,
        costs=costs,
        nav=nav,
        delistings=read_delistings(d),
        ic=ic,
        ic_dates_full=dec,
    )


def store_spec_from_command(command: str) -> StoreSpec | None:
    """The store flags of a ``qts run --store ...`` command line (``manifest.json``
    ``command``); ``None`` when it is not a store run or does not parse."""
    try:
        toks = shlex.split(str(command or ""))
    except ValueError:
        return None
    if "run" not in toks:
        return None
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    ap.add_argument("--store")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--max-names", type=int)
    ap.add_argument("--min-mcap-ever", type=float)
    ap.add_argument("--max-rank", type=int)
    try:
        a, _ = ap.parse_known_args(toks[toks.index("run") + 1 :])
    except (argparse.ArgumentError, SystemExit, ValueError):
        return None
    if not a.store:
        return None
    try:
        return StoreSpec.from_args(a)
    except ValueError:
        return None


# ============================================================================ market inputs
def booked_returns(
    close: pd.DataFrame,
    run_dates: pd.DatetimeIndex,
    tickers: pd.Index,
    cfg: SystemConfig,
    delistings: pd.DataFrame | None,
) -> np.ndarray:
    """``(n, N)`` per-name return ``t -> t+1`` as :class:`~qts.backtest.simulator.DailyBarSimulator`
    booked it (``_context`` / ``_mark_to_market_step`` / ``_liquidate_delisted``): the
    close with data-error prints blanked (``cfg.risk.clean_returns``; the cleaning is
    causal and per name, so the full history of the run's names gives the simulator's
    values), the return measured from the last close *inside the run's dates* (a halt's
    move lands on the resumption day), 0 without a close, and the booked return of every
    held delisting from ``delistings.parquet`` (``haircut_return``).  Row ``k`` is the
    return realised on ``run_dates[k + 1]``."""
    from qts.core.returns import flag_bad_prices

    c = close.reindex(columns=tickers)
    rc = cfg.risk
    if getattr(rc, "clean_returns", False):
        c, _ = flag_bad_prices(c, cap_up=float(rc.return_abs_cap_up), cap_down=float(rc.return_abs_cap_down))
    c = c.reindex(run_dates)
    r = (c / c.ffill().shift(1) - 1.0).where(c.notna()).to_numpy(dtype=float)[1:]
    r = np.where(np.isfinite(r), r, 0.0)
    if delistings is not None and len(delistings):
        ret_dates = run_dates[1:]
        rows = ret_dates.get_indexer(pd.DatetimeIndex(delistings["date"]))
        cols = tickers.get_indexer(pd.Index(delistings["ticker"].astype(str)))
        ok = (rows >= 0) & (cols >= 0)
        r[rows[ok], cols[ok]] = delistings["haircut_return"].to_numpy(dtype=float)[ok]
    return r


def industry_labels(md: MarketData, dates: pd.DatetimeIndex, tickers: pd.Index) -> np.ndarray:
    """``(len(dates), len(tickers))`` object array of point-in-time industry labels
    (``None`` = unclassified): row ``t`` is ``md.industry_as_of(t)`` restricted to
    ``tickers`` -- the rule of :func:`qts.risk.exposures.industry_panel`, computed on the
    requested columns only."""
    static = md.industry.reindex(tickers)
    if md.industry_history is None:
        lab = np.broadcast_to(static.to_numpy(dtype=object), (len(dates), len(tickers))).copy()
    else:
        ih = md.industry_history.reindex(columns=tickers)
        grid = ih.index.union(dates)
        panel = ih.reindex(grid).ffill().reindex(dates)
        stat = np.broadcast_to(static.to_numpy(dtype=object), panel.shape)
        lab = np.where(panel.notna().to_numpy(), panel.to_numpy(dtype=object), stat)
    out = np.asarray(lab, dtype=object)
    out[pd.isna(out)] = None
    return out


@dataclass
class ICInputs:
    """Scores and targets on the run's IC dates (rows: execution dates; values of the
    signal date ``L`` days earlier), with the industry labels of that signal date."""

    dates: pd.DatetimeIndex
    score: np.ndarray  #: (m, M) traded score, NaN where not eligible at the signal date
    target: np.ndarray  #: (m, M) residualised forward target of the signal date
    industry: np.ndarray  #: (m, M) labels at the signal date
    run_ic: pd.Series | None = None  #: the run's ic.parquet, for the score check
    cache: dict[str, str] = field(default_factory=dict)


def ic_inputs_from_pipeline(
    pipe: ResearchPipeline, eval_dates: pd.DatetimeIndex, run_ic: pd.Series | None
) -> ICInputs:
    """:class:`ICInputs` from a fitted pipeline on the run's IC dates (exactly the
    alignment of ``ResearchPipeline._signal_quality``)."""
    md_u, alpha = pipe.md_u, pipe.alpha
    assert md_u is not None and alpha is not None
    L = pipe.signal_lag_days()
    pos = md_u.dates.get_indexer(eval_dates)
    keep = pos >= max(L, 0)
    exec_dates = pd.DatetimeIndex(eval_dates[keep])
    sig = md_u.dates[pos[keep] - L]
    tick = md_u.tickers
    elig = md_u.eligibility().reindex(index=sig, columns=tick).fillna(False).astype(bool).to_numpy()
    traded = getattr(alpha, "traded_score_", alpha.combined_score_)
    score = traded.reindex(index=sig, columns=tick).to_numpy(dtype=float)
    score = np.where(elig, score, np.nan)
    target = alpha.target_.reindex(index=sig, columns=tick).to_numpy(dtype=float)
    lab = industry_labels(md_u, pd.DatetimeIndex(sig), tick)
    return ICInputs(
        dates=exec_dates,
        score=score,
        target=target,
        industry=lab,
        run_ic=run_ic,
        cache=dict(pipe.cache_info_),
    )


# ============================================================================ statistics
def _nw_t(x: np.ndarray, lags: int | None = None) -> float:
    from qts.core.stats import newey_west_tstat

    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3 or float(np.std(v)) == 0.0:
        return float("nan")
    return float(newey_west_tstat(v, lags=lags))


def _sharpe(x: np.ndarray, tdy: int) -> float:
    v = np.asarray(x, dtype=float)
    if len(v) < 3:
        return float("nan")
    sd = float(v.std(ddof=1))
    return float(v.mean() / sd * math.sqrt(tdy)) if sd > 0 else float("nan")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata

    ra, rb = rankdata(a), rankdata(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


def _encode(labels: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Integer codes of an object label array into ``names``; ``len(names)`` for
    unclassified / unknown labels (the extra bucket)."""
    flat = pd.Categorical(pd.Series(labels.ravel(), dtype=object), categories=list(names)).codes
    codes = np.asarray(flat, dtype=np.int64).reshape(labels.shape)
    return np.where(codes < 0, len(names), codes)


def _group_sum(values: np.ndarray, idx: np.ndarray, n: int, G: int) -> np.ndarray:
    return np.bincount(idx, weights=np.asarray(values, dtype=float).ravel(), minlength=n * G).reshape(n, G)


@dataclass
class _Level:
    """One grouping (industries or FF12 sectors): ``names`` (last = unclassified)."""

    key: str
    label: str
    names: list[str]
    parents: list[str | None]
    code_map: np.ndarray  #: industry code -> group code (unclassified -> last)


def _levels(industries: list[str], parent_of: Callable[[str], str | None] | None) -> list[_Level]:
    K = len(industries)
    lv = [
        _Level(
            key="industry",
            label="risk-model industry (Fama-French 48 on the real store)",
            names=[*industries, UNCLASSIFIED],
            parents=[parent_of(s) if parent_of else None for s in industries] + [None],
            code_map=np.arange(K + 1),
        )
    ]
    if parent_of is None:
        return lv
    par = [parent_of(s) for s in industries]
    if not any(p is not None for p in par):
        return lv
    groups = sorted({p for p in par if p is not None})
    if any(p is None for p in par):
        groups.append(UNMAPPED)
    gi = {g: i for i, g in enumerate(groups)}
    cmap = np.array([gi[p if p is not None else UNMAPPED] for p in par] + [len(groups)], dtype=np.int64)
    lv[0].parents = [*par, None]
    lv.append(
        _Level(
            key="ff12",
            label="FF12 sector (the parent of the risk model's thin-industry shrinkage)",
            names=[*groups, UNCLASSIFIED],
            parents=[None] * (len(groups) + 1),
            code_map=cmap,
        )
    )
    return lv


def default_parent(label: str) -> str | None:
    """FF12 parent of a Fama-French 48 label (``None`` for any other label)."""
    from qts.risk.factor_returns import ALL_INDUSTRIES, industry_parent

    p = industry_parent(label)
    return None if p == ALL_INDUSTRIES else p


def _ic_by_group(
    ic: ICInputs, codes: np.ndarray, G: int, min_names: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ic (m, G), names (m, G), book_ic (m,))``: rank IC inside each group per date
    (NaN below ``min_names``), the names it used, and the book-level rank IC."""
    m = len(ic.dates)
    out = np.full((m, G), np.nan)
    cnt = np.zeros((m, G))
    book = np.full(m, np.nan)
    for i in range(m):
        s, y, g = ic.score[i], ic.target[i], codes[i]
        ok = np.isfinite(s) & np.isfinite(y)
        if int(ok.sum()) >= BOOK_IC_MIN_NAMES:
            book[i] = _spearman(s[ok], y[ok])
        gi = g[ok]
        si, yi = s[ok], y[ok]
        for k in np.unique(gi):
            sel = gi == k
            nk = int(sel.sum())
            cnt[i, k] = nk
            if nk >= min_names:
                out[i, k] = _spearman(si[sel], yi[sel])
    return out, cnt, book


def _level_rows(
    lv: _Level,
    codes_ind: np.ndarray,
    W: np.ndarray,
    R: np.ndarray,
    cost_name: np.ndarray | None,
    ret_dates: pd.DatetimeIndex,
    tdy: int,
    ic: ICInputs | None,
    ic_codes_ind: np.ndarray | None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Rows of one level and its daily streams (``pnl``, ``gross`` per group)."""
    from qts.alpha.evaluation import newey_west_lags

    n, _ = W.shape
    G = len(lv.names)
    codes = lv.code_map[codes_ind]
    idx = (np.arange(n)[:, None] * G + codes).ravel()
    pnl = _group_sum(W * R, idx, n, G)
    lng = _group_sum(np.maximum(W, 0.0), idx, n, G)
    sht = _group_sum(np.minimum(W, 0.0), idx, n, G)
    nl = _group_sum(W > HELD_EPS, idx, n, G)
    ns = _group_sum(W < -HELD_EPS, idx, n, G)
    gross = lng - sht
    cost = _group_sum(cost_name, idx, n, G) if cost_name is not None else None
    book_gross_sum = float(gross.sum())
    book_pnl = float(pnl.sum())
    months = ret_dates.to_period("M")
    pm = pd.DataFrame(pnl).groupby(months).sum().to_numpy()
    gm = pd.DataFrame(gross).groupby(months).sum().to_numpy()

    ic_g = cnt_g = None
    if ic is not None and ic_codes_ind is not None:
        ic_g, cnt_g, _ = _ic_by_group(ic, lv.code_map[ic_codes_ind], G, IC_MIN_NAMES)
    rows: list[dict[str, Any]] = []
    for g, name in enumerate(lv.names):
        held = bool(gross[:, g].sum() > 0)
        has_ic = ic_g is not None and bool(np.isfinite(ic_g[:, g]).any())
        if not held and not has_ic:
            continue
        p = pnl[:, g]
        share = float(gross[:, g].sum() / book_gross_sum) if book_gross_sum > 0 else float("nan")
        active_months = gm[:, g] > 0
        row: dict[str, Any] = {
            "industry": name,
            "parent": lv.parents[g],
            "active": bool(name != UNCLASSIFIED and share >= ACTIVE_MIN_GROSS_SHARE),
            "long": float(lng[:, g].mean()),
            "short": float(sht[:, g].mean()),
            "net": float((lng[:, g] + sht[:, g]).mean()),
            "gross": float(gross[:, g].mean()),
            "gross_share": share,
            "n_long": float(nl[:, g].mean()),
            "n_short": float(ns[:, g].mean()),
            "pnl_total": float(p.sum()),
            "pnl_ann": float(tdy * p.mean()),
            "pnl_share": float(p.sum() / book_pnl) if book_pnl > 0 else float("nan"),
            "sharpe": _sharpe(p, tdy) if held else float("nan"),
            "t": _nw_t(p) if held else float("nan"),
            "hit_rate": float((pm[active_months, g] > 0).mean()) if active_months.any() else float("nan"),
            "n_months": int(active_months.sum()),
            "cost_ann": float(tdy * cost[:, g].mean()) if cost is not None else float("nan"),
            "net_pnl_ann": float(tdy * (p - cost[:, g]).mean()) if cost is not None else float("nan"),
            "ic_mean": float("nan"),
            "ic_t": float("nan"),
            "ic_n_dates": 0,
            "ic_coverage": float("nan"),
            "ic_mean_names": float("nan"),
        }
        if ic_g is not None and cnt_g is not None:
            v = ic_g[:, g]
            ok = np.isfinite(v)
            m = len(v)
            row["ic_n_dates"] = int(ok.sum())
            row["ic_coverage"] = float(ok.mean()) if m else float("nan")
            if ok.any():
                row["ic_mean"] = float(v[ok].mean())
                row["ic_t"] = _nw_t(v[ok], lags=newey_west_lags(int(ok.sum()), 1))
                row["ic_mean_names"] = float(cnt_g[ok, g].mean())
        rows.append(row)
    rows.sort(key=lambda r: (r["industry"] == UNCLASSIFIED, -r["pnl_total"]))
    return rows, {"pnl": pnl, "gross": gross}


def breadth_stats(
    rows: list[dict[str, Any]], names: list[str], pnl: np.ndarray, book_gross: np.ndarray, tdy: int
) -> dict[str, Any]:
    """Breadth statistics over the *active* rows (module docstring, item 4)."""
    from scipy.stats import binomtest

    from qts.research.signals import holm_adjust, two_sided_p

    act = [r for r in rows if r["active"]]
    n_act = len(act)
    total = float(np.sum(book_gross))
    n_pos = sum(1 for r in act if r["pnl_total"] > 0)
    top = sorted(act, key=lambda r: -r["pnl_total"])[:TOP_K]
    top_sum = float(sum(r["pnl_total"] for r in top))
    pos_sum = float(sum(r["pnl_total"] for r in act if r["pnl_total"] > 0))
    absp = np.array([abs(r["pnl_total"]) for r in act], dtype=float)
    s = absp / absp.sum() if absp.sum() > 0 else absp
    n_eff = float(1.0 / np.sum(s**2)) if absp.sum() > 0 else float("nan")
    ts = np.array([r["t"] for r in act], dtype=float)
    p = np.array([two_sided_p(float(t)) for t in ts], dtype=float)
    p_holm = holm_adjust(p.tolist()) if n_act else np.array([])
    holm = [
        {"industry": r["industry"], "t": float(t), "p": float(pi), "p_holm": float(ph)}
        for r, t, pi, ph in zip(act, ts, p, p_holm, strict=True)
        if np.isfinite(ph) and ph <= HOLM_ALPHA
    ]
    holm.sort(key=lambda h: h["p_holm"])
    gi = {nm: i for i, nm in enumerate(names)}
    rest = np.asarray(book_gross, dtype=float).copy()
    for r in top:
        rest = rest - pnl[:, gi[r["industry"]]]
    return {
        "n_held": sum(1 for r in rows if r["industry"] != UNCLASSIFIED and r["gross"] > 0),
        "n_active": n_act,
        "n_positive": n_pos,
        "positive_share": n_pos / n_act if n_act else float("nan"),
        "binom_p": float(binomtest(n_pos, n_act, 0.5, alternative="greater").pvalue)
        if n_act
        else float("nan"),
        "top": [r["industry"] for r in top],
        "top_share": top_sum / total if total > 0 else float("nan"),
        "rest_share": 1.0 - top_sum / total if total > 0 else float("nan"),
        "top_share_of_positive": top_sum / pos_sum if pos_sum > 0 else float("nan"),
        "n_eff": n_eff,
        "n_eff_ratio": n_eff / n_act if n_act and np.isfinite(n_eff) else float("nan"),
        "n_t2": int(np.sum(np.abs(ts[np.isfinite(ts)]) >= T_CLEAR)),
        "n_t2_chance": CHANCE_RATE * n_act,
        "holm": holm,
        "ex_top": {"pnl_ann": float(tdy * rest.mean()), "t": _nw_t(rest)},
        "book_t": _nw_t(np.asarray(book_gross, dtype=float)),
    }


def _pct(x: Any, d: int = 0) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return f"{100 * v:.{d}f} %" if math.isfinite(v) else "n/a"


def _num(x: Any, spec: str = ".2f") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return format(v, spec) if math.isfinite(v) else "n/a"


def _plural(k: int, one: str, many: str) -> str:
    return f"{k} {one if k == 1 else many}"


def breadth_verdict(b: Mapping[str, Any], gross_pnl_total: float) -> dict[str, Any]:
    """``{label, rule, text}`` under the fixed thresholds of this module."""

    def f(k: str) -> float:
        v = b.get(k)
        return float(v) if v is not None else float("nan")

    n_act = int(b.get("n_active") or 0)
    book_t, rest, pos, neff, ratio = (
        f("book_t"),
        f("rest_share"),
        f("positive_share"),
        f("n_eff"),
        f("n_eff_ratio"),
    )
    top = ", ".join(b.get("top") or [])
    holm = [h["industry"] for h in b.get("holm") or []]
    ex = b.get("ex_top") or {}
    facts = (
        f"{int(b.get('n_positive') or 0)} of {n_act} active industries made money ({_pct(pos)}; a coin flip "
        f"would give that many or more with probability {_num(b.get('binom_p'), '.2g')}). The five best "
        f"({top}) earned {_pct(b.get('top_share'))} of the gross P&L, so {_pct(rest)} came from the rest; the "
        f"P&L is as spread out as {_num(neff, '.1f')} equally contributing industries. "
        f"{_plural(int(b.get('n_t2') or 0), 'industry clears', 'industries clear')} |t| >= 2 where about "
        f"{_num(b.get('n_t2_chance'), '.1f')} would by chance; after the Holm correction "
        + (f"{_plural(len(holm), 'remains', 'remain')} ({', '.join(holm)})." if holm else "none remains.")
        + f" Without its five best industries the book earns {_pct(ex.get('pnl_ann'), 2)} a year gross "
        f"(t = {_num(ex.get('t'))})."
    )
    if n_act < VERDICT_MIN_ACTIVE:
        return {
            "label": "undetermined",
            "rule": f"n_active {n_act} < {VERDICT_MIN_ACTIVE}",
            "text": f"Undetermined: only {n_act} industries held a meaningful share of the book, too few to "
            "judge breadth.",
        }
    if not (gross_pnl_total > 0) or not (book_t >= VERDICT_MIN_BOOK_T):
        return {
            "label": "undetermined",
            "rule": f"book gross P&L {'<= 0' if not gross_pnl_total > 0 else 'ok'}, book t {_num(book_t)} "
            f"(need > 0 and >= {VERDICT_MIN_BOOK_T:g})",
            "text": f"Undetermined: the whole book's gross P&L has t = {_num(book_t)} (below "
            f"{VERDICT_MIN_BOOK_T:g}{', and it lost money' if not gross_pnl_total > 0 else ''}), so there is "
            "not enough edge to say where it comes from. For the record: " + facts,
        }
    if rest < CONCENTRATED_MAX_REST_SHARE or neff < max(2.0, CONCENTRATED_MAX_NEFF_RATIO * n_act):
        return {
            "label": "concentrated",
            "rule": f"rest_share {_num(rest)} < {CONCENTRATED_MAX_REST_SHARE:g} or n_eff {_num(neff, '.1f')} < "
            f"max(2, {CONCENTRATED_MAX_NEFF_RATIO:g} x {n_act})",
            "text": "Concentrated: the edge comes from a few industries. " + facts,
        }
    if rest >= BROAD_MIN_REST_SHARE and pos >= BROAD_MIN_POSITIVE_SHARE and ratio >= BROAD_MIN_NEFF_RATIO:
        return {
            "label": "broad",
            "rule": f"rest_share {_num(rest)} >= {BROAD_MIN_REST_SHARE:g}, positive_share {_num(pos)} >= "
            f"{BROAD_MIN_POSITIVE_SHARE:g}, n_eff_ratio {_num(ratio)} >= {BROAD_MIN_NEFF_RATIO:g}",
            "text": "Broad: the edge is spread across many industries. " + facts,
        }
    return {
        "label": "moderately concentrated",
        "rule": "neither the broad nor the concentrated rule holds",
        "text": "Moderately concentrated: many industries contribute, but the P&L leans on a subset. "
        + facts,
    }


def compute_breakdown(
    *,
    ret_dates: pd.DatetimeIndex,
    W: np.ndarray,
    R: np.ndarray,
    gross: np.ndarray,
    industry: np.ndarray,
    X: np.ndarray | None = None,
    net: np.ndarray | None = None,
    trading_cost: np.ndarray | None = None,
    borrow_cost: np.ndarray | None = None,
    parent_of: Callable[[str], str | None] | None = default_parent,
    tdy: int = 252,
    ic: ICInputs | None = None,
    has_delistings: bool = True,
) -> dict[str, Any]:
    """The breakdown of one book (pure: arrays in, a dict out).

    ``W`` / ``X`` / ``R`` / ``industry``: ``(n, N)`` post-trade weights at ``t``, trades at
    ``t``, booked returns ``t -> t+1`` and industry labels at ``t``; ``gross`` / ``net``:
    the recorded returns realised on ``ret_dates``; ``trading_cost`` / ``borrow_cost``:
    ``(n,)`` the day's costs as fractions of the pre-trade NAV at ``t`` (``None``: no
    cost attribution).  Returns the ``book`` / ``reconciliation`` / ``costs`` / ``ic`` /
    ``levels`` parts of the JSON (module docstring)."""
    n = W.shape[0]
    gross = np.asarray(gross, dtype=float)
    ind_labels = {str(v) for v in pd.unique(pd.Series(industry.ravel(), dtype=object).dropna())}
    if ic is not None:
        ind_labels |= {str(v) for v in pd.unique(pd.Series(ic.industry.ravel(), dtype=object).dropna())}
    industries = sorted(ind_labels)
    codes = _encode(industry, industries)
    ic_codes = _encode(ic.industry, industries) if ic is not None else None

    # ---- reconciliation: industry contributions vs the recorded gross
    contrib = (W * R).sum(axis=1)
    resid = gross - contrib
    recon: dict[str, Any] = {
        "book_gross_total": float(gross.sum()),
        "sum_contributions": float(contrib.sum()),
        "residual_total": float(resid.sum()),
        "max_abs_daily_residual": float(np.max(np.abs(resid))) if n else 0.0,
        "tolerance": RECON_TOL,
        "ok": bool(np.max(np.abs(resid)) <= RECON_TOL) if n else True,
        "delistings": "delistings.parquet" if has_delistings else "none",
        "net_max_abs_daily_residual": None,
    }

    # ---- costs: the day's trading cost pro rata to traded value, borrow pro rata to short value
    cost_name = None
    costs: dict[str, Any] = {"available": False, "method": None, "unallocated_ann": None, "note": ""}
    if trading_cost is not None and borrow_cost is not None and X is not None:
        absx = np.abs(X)
        sx = absx.sum(axis=1)
        shortv = np.abs(np.minimum(W, 0.0))
        ss = shortv.sum(axis=1)
        tc, bc = np.asarray(trading_cost, dtype=float), np.asarray(borrow_cost, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            cost_name = np.where(sx[:, None] > 0, absx / sx[:, None], 0.0) * tc[:, None]
            cost_name = cost_name + np.where(ss[:, None] > 0, shortv / ss[:, None], 0.0) * bc[:, None]
        unalloc = tc * (sx <= 0) + bc * (ss <= 0)
        costs = {
            "available": True,
            "method": "pro_rata",
            "unallocated_ann": float(tdy * unalloc.mean()),
            "note": "costs.parquet records costs per day, not per name: each day's trading cost (spread + "
            "impact + fees) is allocated to names pro rata to traded value and its borrow cost pro rata to "
            "short value. The allocation sums to the book's cost; per industry it is an approximation "
            "(impact is convex in the trade size, fees and borrow rates differ by name).",
        }
        if net is not None:
            nres = np.asarray(net, dtype=float) - (contrib - cost_name.sum(axis=1) - unalloc)
            recon["net_max_abs_daily_residual"] = float(np.max(np.abs(nres)))
    else:
        costs["note"] = (
            "no cost attribution: the run directory lacks trades.parquet, costs.parquet or nav.parquet"
        )

    tcost = (
        np.asarray(trading_cost, dtype=float) + np.asarray(borrow_cost, dtype=float)
        if trading_cost is not None and borrow_cost is not None
        else None
    )
    book = {
        "gross_pnl_total": float(gross.sum()),
        "gross_pnl_ann": float(tdy * gross.mean()),
        "net_pnl_ann": float(tdy * np.asarray(net, dtype=float).mean()) if net is not None else float("nan"),
        "cost_ann": float(tdy * tcost.mean()) if tcost is not None else float("nan"),
        "sharpe_gross": _sharpe(gross, tdy),
        "t_gross": _nw_t(gross),
        "mean_gross": float(np.abs(W).sum(axis=1).mean()),
        "mean_net": float(W.sum(axis=1).mean()),
    }

    levels: dict[str, Any] = {"industry": None, "ff12": None}
    for lv in _levels(industries, parent_of):
        rows, streams = _level_rows(lv, codes, W, R, cost_name, ret_dates, tdy, ic, ic_codes)
        br = breadth_stats(rows, lv.names, streams["pnl"], gross, tdy)
        levels[lv.key] = {
            "label": lv.label,
            "rows": rows,
            "breadth": br,
            "verdict": breadth_verdict(br, float(gross.sum())) if lv.key == "industry" else None,
        }

    ic_part: dict[str, Any] = {
        "available": ic is not None,
        "reason": "" if ic is not None else "not computed",
        "n_eval_dates": 0,
        "min_names": IC_MIN_NAMES,
        "cache": {},
        "score_check": None,
        "book_mean_ic": float("nan"),
        "book_ic_t": float("nan"),
        "within_industry_mean_ic": float("nan"),
        "within_industry_ic_t": float("nan"),
    }
    if ic is not None and ic_codes is not None:
        from qts.alpha.evaluation import newey_west_lags

        G = len(industries) + 1
        icg, cnt, book_ic = _ic_by_group(ic, ic_codes, G, IC_MIN_NAMES)
        w = np.where(np.isfinite(icg), cnt, 0.0)
        w[:, -1] = 0.0  # the unclassified bucket is not an industry
        with np.errstate(invalid="ignore", divide="ignore"):
            within = np.where(
                w.sum(axis=1) > 0, np.nansum(np.nan_to_num(icg) * w, axis=1) / w.sum(axis=1), np.nan
            )
        ok_b, ok_w = np.isfinite(book_ic), np.isfinite(within)
        ic_part.update(
            n_eval_dates=len(ic.dates),
            cache=dict(ic.cache),
            book_mean_ic=float(book_ic[ok_b].mean()) if ok_b.any() else float("nan"),
            book_ic_t=_nw_t(book_ic[ok_b], lags=newey_west_lags(int(ok_b.sum()), 1))
            if ok_b.any()
            else float("nan"),
            within_industry_mean_ic=float(within[ok_w].mean()) if ok_w.any() else float("nan"),
            within_industry_ic_t=_nw_t(within[ok_w], lags=newey_west_lags(int(ok_w.sum()), 1))
            if ok_w.any()
            else float("nan"),
        )
        if ic.run_ic is not None and len(ic.run_ic):
            mine = pd.Series(book_ic, index=ic.dates).dropna()
            common = mine.index.intersection(ic.run_ic.index)
            diff = (mine.loc[common] - ic.run_ic.loc[common]).abs()
            mx = float(diff.max()) if len(diff) else float("nan")
            ic_part["score_check"] = {
                "n_dates": len(common),
                "n_run_dates": len(ic.run_ic),
                "max_abs_diff": mx,
                "ok": bool(len(common) > 0 and mx <= SCORE_CHECK_TOL),
            }
    return {"book": book, "reconciliation": recon, "costs": costs, "ic": ic_part, "levels": levels}


# ============================================================================ the result
@dataclass
class IndustryBreakdown:
    """A finished breakdown: ``data`` is the JSON document (module docstring)."""

    data: dict[str, Any]
    json_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self.data)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> IndustryBreakdown:
        if d.get("schema") != SCHEMA:
            raise BreakdownError(
                f"not an industry breakdown (schema {d.get('schema')!r}, expected {SCHEMA!r})"
            )
        return cls(data=dict(d))

    @classmethod
    def read(cls, path: str | Path) -> IndustryBreakdown:
        import json

        return cls.from_dict(json.loads(Path(path).read_text()))

    @property
    def verdict(self) -> dict[str, Any]:
        return dict(((self.data.get("levels") or {}).get("industry") or {}).get("verdict") or {})

    def format(self) -> str:
        return format_breakdown(self.to_dict(), json_path=self.json_path)


_COLUMNS: list[tuple[str, str, str]] = [
    # header, key, kind (pct1: x100 with 1 decimal; pct2; num1; num2; int)
    ("long%", "long", "pct2"),
    ("short%", "short", "pct2"),
    ("net%", "net", "pct2"),
    ("gross%", "gross", "pct2"),
    ("share%", "gross_share", "pct1"),
    ("#L", "n_long", "num1"),
    ("#S", "n_short", "num1"),
    ("P&L%/yr", "pnl_ann", "pct2"),
    ("P&L sh%", "pnl_share", "pct1"),
    ("Sharpe", "sharpe", "num2"),
    ("t", "t", "num2"),
    ("hit%", "hit_rate", "pct0"),
    ("cost%/yr", "cost_ann", "pct2"),
    ("net%/yr", "net_pnl_ann", "pct2"),
    ("IC", "ic_mean", "num3"),
    ("IC t", "ic_t", "num2"),
    ("IC n", "ic_n_dates", "int"),
    ("cov%", "ic_coverage", "pct0"),
]


def _cell(v: Any, kind: str) -> str:
    if v is None:
        return "-"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(x):
        return "-"
    if kind.startswith("pct"):
        return f"{100 * x:.{int(kind[3:])}f}"
    if kind.startswith("num"):
        return f"{x:.{int(kind[3:])}f}"
    return f"{round(x)}"


def _table(rows: list[dict[str, Any]], with_parent: bool) -> str:
    head = ["industry", *(["FF12"] if with_parent else []), "act", *[c[0] for c in _COLUMNS]]
    body = []
    for r in rows:
        line = [str(r["industry"])]
        if with_parent:
            line.append(str(r.get("parent") or ""))
        line.append("*" if r.get("active") else "")
        line += [_cell(r.get(k), kind) for _, k, kind in _COLUMNS]
        body.append(line)
    widths = [max(len(h), *(len(b[i]) for b in body)) if body else len(h) for i, h in enumerate(head)]
    n_text = 3 if with_parent else 2

    def fmt(cells: list[str]) -> str:
        return " ".join(
            c.ljust(w) if i < n_text else c.rjust(w)
            for i, (c, w) in enumerate(zip(cells, widths, strict=True))
        )

    return "\n".join([fmt(head), fmt(["-" * w for w in widths]), *(fmt(b) for b in body)])


def _breadth_lines(b: Mapping[str, Any]) -> list[str]:
    ex = b.get("ex_top") or {}
    holm = b.get("holm") or []
    return [
        f"  positive P&L       {b.get('n_positive')} of {b.get('n_active')} active industries "
        f"({_pct(b.get('positive_share'))}); one-sided binomial p vs a coin flip {_num(b.get('binom_p'), '.3g')}",
        f"  top {TOP_K} share        {_pct(b.get('top_share'))} of the gross P&L ({', '.join(b.get('top') or [])}); "
        f"rest {_pct(b.get('rest_share'))}; top {TOP_K} = {_pct(b.get('top_share_of_positive'))} of all positive P&L",
        f"  effective number   {_num(b.get('n_eff'), '.1f')} industries contributing "
        f"({_num(b.get('n_eff_ratio'))} x the active count)",
        f"  |t| >= {T_CLEAR:g}           {b.get('n_t2')} industries (by chance about {_num(b.get('n_t2_chance'), '.1f')}); "
        "Holm at 5 %: "
        + (
            ", ".join(f"{h['industry']} (t {_num(h['t'])}, p_holm {_num(h['p_holm'], '.3f')})" for h in holm)
            or "none"
        ),
        f"  without the top {TOP_K}  {_pct(ex.get('pnl_ann'), 2)} a year gross, t {_num(ex.get('t'))}; whole book t "
        f"{_num(b.get('book_t'))}",
    ]


def _wrap(text: str, width: int = 100, indent: str = "") -> str:
    import textwrap

    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def format_breakdown(d: Mapping[str, Any], json_path: str | None = None) -> str:
    """The text report of a breakdown document (``IndustryBreakdown.to_dict()``)."""
    data, per, book = d.get("data") or {}, d.get("period") or {}, d.get("book") or {}
    rec, costs, ic = d.get("reconciliation") or {}, d.get("costs") or {}, d.get("ic") or {}
    levels = d.get("levels") or {}
    out = [
        f"=== Industry breakdown of run {d.get('run_id')} (config {d.get('config_hash')}, "
        f"{'ML' if d.get('use_ml') else 'composite'}, lag {d.get('signal_lag_days')}) ===",
        f"period {per.get('start')} .. {per.get('end')} ({per.get('n_days')} days, {per.get('n_months')} months); "
        f"window {data.get('window')}"
        + (f" (data through {data.get('sealed_end')})" if data.get("sealed_end") else ""),
        f"book: gross {_pct(book.get('gross_pnl_ann'), 2)}/yr, costs {_pct(book.get('cost_ann'), 2)}/yr, net "
        f"{_pct(book.get('net_pnl_ann'), 2)}/yr; gross Sharpe {_num(book.get('sharpe_gross'))} (t "
        f"{_num(book.get('t_gross'))}); mean gross {_pct(book.get('mean_gross'), 0)} of NAV, net "
        f"{_pct(book.get('mean_net'), 1)}",
        f"reconciliation: industry contributions {_pct(rec.get('sum_contributions'), 3)} vs recorded gross "
        f"{_pct(rec.get('book_gross_total'), 3)} (summed over days); residual {_num(rec.get('residual_total'), '.2e')}, "
        f"max daily {_num(rec.get('max_abs_daily_residual'), '.1e')} -> "
        f"{'OK' if rec.get('ok') else 'NOT RECONCILED'} (delistings from {rec.get('delistings')})",
        "costs: "
        + ("allocated pro rata (see the note below)" if costs.get("available") else str(costs.get("note"))),
    ]
    if ic.get("available"):
        sc = ic.get("score_check") or {}
        out.append(
            f"signal: {ic.get('n_eval_dates')} IC dates; book rank IC {_num(ic.get('book_mean_ic'), '.4f')} "
            f"(t {_num(ic.get('book_ic_t'))}); within-industry IC {_num(ic.get('within_industry_mean_ic'), '.4f')} "
            f"(t {_num(ic.get('within_industry_ic_t'))}); fit {ic.get('cache') or 'off'}; score check "
            + (
                f"{'OK' if sc.get('ok') else 'MISMATCH'} on {sc.get('n_dates')}/{sc.get('n_run_dates')} dates "
                f"(max |diff| {_num(sc.get('max_abs_diff'), '.1e')})"
                if sc
                else "n/a (no ic.parquet)"
            )
        )
    else:
        out.append(f"signal: within-industry IC not computed ({ic.get('reason')})")
    for key, title in (("industry", "BY INDUSTRY"), ("ff12", "BY FF12 SECTOR")):
        lv = levels.get(key)
        if not lv:
            if key == "ff12":
                out.append("\n(no FF12 roll-up: the industry labels are not Fama-French 48)")
            continue
        out.append(f"\n--- {title}: {lv.get('label')}; sorted by P&L contribution; act * = active ---")
        out.append(_table(list(lv.get("rows") or []), with_parent=key == "industry"))
        out.append("breadth (active industries):")
        out += _breadth_lines(lv.get("breadth") or {})
    v = ((levels.get("industry") or {}).get("verdict")) or {}
    out.append(f"\nVERDICT: {str(v.get('label', 'n/a')).upper()}")
    out.append(_wrap(str(v.get("text", "")), indent="  "))
    out.append(f"  (rule: {v.get('rule')}; thresholds fixed in qts/research/industry_breakdown.py)")
    out.append("\nWhat the numbers mean:")
    hl = d.get("help") or {}
    labels = {
        "long": "long%", "short": "short%", "net": "net%", "gross": "gross%", "gross_share": "share%",
        "n_long": "#L / #S", "pnl_ann": "P&L%/yr", "pnl_share": "P&L sh%", "sharpe": "Sharpe", "t": "t",
        "hit_rate": "hit%", "cost_ann": "cost%/yr", "net_pnl_ann": "net%/yr", "ic_mean": "IC", "ic_t": "IC t",
        "ic_n_dates": "IC n", "ic_coverage": "cov%", "reconciliation": "reconciliation",
        "positive": "positive P&L", "top_share": "top 5 share", "n_eff": "effective number",
        "n_t2": "|t| >= 2", "ex_top": "without the top 5", "book_t": "whole book t",
        "within_ic": "within-industry IC", "score_check": "score check",
    }  # fmt: skip
    for k, lab in labels.items():
        if k in hl:
            out.append(_wrap(f"{lab}: {hl[k]}", indent="  "))
    if costs.get("available") and costs.get("note"):
        out.append(_wrap("cost note: " + str(costs["note"]), indent="  "))
    for note in d.get("notes") or []:
        out.append(_wrap("NOTE: " + str(note)))
    if json_path:
        out.append(f"written {json_path}")
    return "\n".join(out)


# ============================================================================ orchestration
def _resolve_run_dir(run: str | Path, runs_root: str | Path | None) -> Path:
    p = Path(run)
    if not p.exists() and runs_root is not None and (Path(runs_root) / str(run)).exists():
        p = Path(runs_root) / str(run)
    if not p.exists():
        raise BreakdownError(f"run {run} not found (give a run directory or a run id under --runs-root)")
    return p


def _check_seal(book: RunBook, seal: Any) -> None:
    """Refuse a sealed run without the seal that made it (module docstring "The seal")."""
    import os

    from qts.research.seal import SEAL_FILE_ENV

    if not book.sealed_end:
        return
    if seal is None:
        env = os.environ.get(SEAL_FILE_ENV)
        why = f"${SEAL_FILE_ENV}={env!r}" if env is not None else "no research/seal.yaml"
        raise BreakdownError(
            f"run {book.run_id} was made on the sealed window (data through {book.sealed_end}); no seal is in "
            f"force now ({why}), so the store would load the full history, holdout included. A sealed run is "
            "broken down on the sealed window only: restore the seal (unset "
            f"{SEAL_FILE_ENV} / put back research/seal.yaml) and run the breakdown again."
        )
    if book.holdout_start and seal.holdout_start_iso != book.holdout_start:
        raise BreakdownError(
            f"run {book.run_id} was made under a seal with holdout from {book.holdout_start}; the seal in force "
            f"starts the holdout on {seal.holdout_start_iso}. The breakdown would not load the run's window; "
            "refused."
        )


def predict_stage_cache(pipe: ResearchPipeline, md: MarketData) -> tuple[dict[str, str], str]:
    """``({stage: "warm" | "cold" | "off"}, data_hash)`` of ``pipe.fit(md)`` without
    fitting: the stage-cache keys ``fit`` would compute (``ResearchPipeline._cache_key``)
    and whether their entries exist.  Hashes the panels once (the fit hashes them again)."""
    from qts.core.manifest import chain_reference_hash, hash_market_data
    from qts.runs.cache import STAGE_EXCLUDED_DATA

    cache = pipe._stage_cache()
    full = hash_market_data(md, n_chars=64, exclude=STAGE_EXCLUDED_DATA)
    data_hash = chain_reference_hash(full, md)[:16]
    if cache is None:
        return {"risk_fit": "off", "alpha_fit": "off"}, data_hash
    rk = pipe._cache_key(cache, "risk_fit", md, full)
    ak = pipe._cache_key(cache, "alpha_fit", md, full, parent=rk)
    out: dict[str, str] = {}
    for stage, key in (("risk_fit", rk), ("alpha_fit", ak)):
        pkl, meta = cache._paths(stage, key)
        out[stage] = "warm" if stage in pipe.cfg.cache.stages and pkl.exists() and meta.exists() else "cold"
    return out, data_hash


def run_industry_breakdown(
    run: str | Path,
    *,
    spec: StoreSpec | None = None,
    runs_root: str | Path | None = None,
    cache_root: str | None = None,
    with_ic: bool = True,
    refuse_refit: bool = False,
    allow_partial_coverage: bool = False,
    md: MarketData | None = None,
    write: bool = True,
    log: Printer = _quiet,
) -> IndustryBreakdown:
    """Break down the finished run ``run`` (a directory, or an id under ``runs_root``).

    ``spec``: the store flags (default: parsed from ``manifest.json`` ``command``);
    ``cache_root``: the stage cache (default: the run config's ``cache.root`` when it
    exists, else ``<run_dir>/../_cache``); ``with_ic=False``: positioning, P&L and costs
    only (no fit); ``refuse_refit``: raise instead of re-fitting on a cold cache;
    ``md``: an already loaded market (tests; skips the loader, the seal and the
    coverage guard -- the library path for the CLI always loads).  Writes
    ``<run_dir>/industry_breakdown.json`` unless ``write`` is False.  Logs nothing to
    the trial ledger."""
    from qts.core.manifest import hash_market
    from qts.research.common import pipeline_config
    from qts.research.data import load_research_market
    from qts.research.seal import load_seal

    t0 = time.perf_counter()
    run_dir = _resolve_run_dir(run, runs_root)
    book = read_run_book(run_dir)
    cfg = book.cfg
    notes: list[str] = []
    if book.manifest.get("config_hash") and cfg.hash() != book.manifest.get("config_hash"):
        notes.append(
            f"config.yaml hashes to {cfg.hash()}, the run recorded {book.manifest.get('config_hash')}: the config "
            "schema changed since the run, so the re-fitted scores may not be the ones it traded"
        )
    window = "full"
    load_cov = None
    sealed_end = ""
    if md is None:
        if spec is None:
            spec = store_spec_from_command(str(book.manifest.get("command") or ""))
            if spec is None:
                raise BreakdownError(
                    f"run {book.run_id}: manifest.json's command is not a `qts run --store ...` line "
                    f"({book.manifest.get('command')!r}); give the run's store flags (--store, --start, "
                    "--max-rank, ...)"
                )
            log(f"store flags from the run's command: {spec.cli_flags()}")
        if not Path(spec.store).exists():  # a relative store path of a run launched from the repo root
            spec = replace(spec, store=str(resolve_repo_path(spec.store)))
        seal = load_seal()
        _check_seal(book, seal)
        root = run_dir.parent  # the run's own runs root (its full-history market-hash cache)
        ld = load_research_market(
            spec,
            seal=seal,
            hash_cache_dir=str(root / "_cache" / "market_hash"),
            log=log,
            universe=cfg,
            allow_partial_coverage=allow_partial_coverage,
        )
        md = ld.md
        load_cov = None if ld.coverage is None else ld.coverage.to_dict()
        last = md.dates[-1]
        if book.sealed_end:
            if not ld.sealed or str(last.date()) != book.sealed_end:
                raise BreakdownError(
                    f"run {book.run_id} was made on data through {book.sealed_end}; this load ends on "
                    f"{last.date()}{'' if ld.sealed else ' (not sealed)'}: other flags or another seal; refused"
                )
            window, sealed_end = "sealed", book.sealed_end
        elif ld.sealed:
            if book.ret_dates[-1] > last:
                window = "truncated"
                sealed_end = str(last.date())
                notes.append(
                    f"run {book.run_id} used the full history (before the seal, --unseal or an exempt check); the "
                    f"seal in force loads data through {sealed_end}, so this breakdown covers the run's dates up "
                    "to then only (a diagnostic never reads the holdout)"
                )
                book = book.truncate(last)
                if len(book.dec) < 2:
                    raise BreakdownError(f"run {book.run_id} has fewer than two days before the holdout")
            else:
                window, sealed_end = "sealed", str(last.date())
        spec_dict: dict[str, Any] | None = spec.to_dict()
    else:
        spec_dict = None if spec is None else spec.to_dict()
    missing = book.tickers.difference(md.tickers)
    if len(missing):
        held = np.abs(book.W[:, book.tickers.get_indexer(missing)]).sum(axis=0) > 0
        if window != "truncated" or held.any():
            raise BreakdownError(
                f"{len(missing)} names of the run are not in the loaded market (e.g. {list(missing[:5])}): the "
                "store flags do not match the run's"
            )
        # truncated: names that enter the universe only after the seal date, never held before it
        book = book.drop_tickers(missing)
    mh = hash_market(md)
    if window != "truncated" and book.manifest.get("market_hash") and mh != book.manifest.get("market_hash"):
        raise BreakdownError(
            f"the loaded close panel hashes to {mh}, the run's to {book.manifest.get('market_hash')}: not the "
            "market the run was made on (other store flags or a refreshed store); refused"
        )
    log(
        f"run {book.run_id}: {len(book.dec)} days {book.dec[0].date()}..{book.ret_dates[-1].date()}, window {window}"
    )

    R = booked_returns(md.close, book.run_dates, book.tickers, cfg, book.delistings)
    lab = industry_labels(md, book.dec, book.tickers)
    trading = borrow = None
    if book.costs is not None and book.nav is not None and book.X is not None:
        c = book.costs
        nav = np.where(np.asarray(book.nav, dtype=float) > 0, book.nav, np.nan)
        cols = [k for k in ("spread", "impact", "fees") if k in c.columns]
        trading = np.nan_to_num(c[cols].sum(axis=1).to_numpy(dtype=float) / nav)
        borrow = np.nan_to_num(
            (c["borrow"].to_numpy(dtype=float) if "borrow" in c else np.zeros(len(c))) / nav
        )

    ic_in: ICInputs | None = None
    ic_reason = "--no-ic" if not with_ic else ""
    if with_ic:
        from qts.pipeline import ResearchPipeline

        if cache_root is None:
            cr = str(cfg.cache.root or "")
            cache_root = cr if cr and Path(cr).is_dir() else str(run_dir.parent / "_cache")
        pcfg = pipeline_config(cfg, md, None).with_overrides(
            {"cache.enabled": True, "cache.root": cache_root}
        )
        pipe = ResearchPipeline(pcfg, use_ml=book.use_ml, ledger_path=None, runs_root=None)
        state, data_hash = predict_stage_cache(pipe, md)
        cold = [s for s, v in state.items() if v != "warm"]
        if (
            book.manifest.get("data_hash")
            and data_hash != book.manifest.get("data_hash")
            and window != "truncated"
        ):
            notes.append(
                f"the loaded data hashes to {data_hash}, the run recorded {book.manifest.get('data_hash')} (same "
                "prices, another reference table): the alpha fit can differ from the traded one"
            )
        if cold:
            msg = (
                f"stage cache {cache_root}: {', '.join(f'{s} {state[s]}' for s in state)} -- the pipeline would be "
                "re-fitted for the scores (about 45 min on the reference load with ML; the fit is stored for next "
                "time)"
            )
            if refuse_refit:
                raise BreakdownError(
                    msg + "; refused (--no-refit). Use --no-ic for positioning and P&L only."
                )
            log(f"WARNING: {msg}")
        t1 = time.perf_counter()
        pipe.fit(md)
        log(f"fit: {time.perf_counter() - t1:.0f}s, stage cache {dict(pipe.cache_info_) or 'off'}")
        H = int(cfg.alpha.horizon_days)
        eval_dates = pd.DatetimeIndex(book.ic_dates_full[::H])
        eval_dates = eval_dates[eval_dates <= book.dec[-1]]
        ic_in = ic_inputs_from_pipeline(pipe, eval_dates, book.ic)
        del pipe

    res = compute_breakdown(
        ret_dates=book.ret_dates,
        W=book.W,
        R=R,
        gross=book.gross,
        industry=lab,
        X=book.X,
        net=book.net,
        trading_cost=trading,
        borrow_cost=borrow,
        tdy=int(cfg.trading_days_per_year),
        ic=ic_in,
        has_delistings=book.delistings is not None,
    )
    if ic_in is None:
        res["ic"]["reason"] = ic_reason
    rec = res["reconciliation"]
    if not rec["ok"]:
        notes.append(
            f"the industry contributions do not reconcile with the recorded gross (max daily residual "
            f"{rec['max_abs_daily_residual']:.2e}): "
            + (
                "the run predates delistings.parquet, so held delistings are booked at 0 here"
                if book.delistings is None
                else "the loaded prices differ from the run's"
            )
        )
    sc = res["ic"].get("score_check")
    if sc and not sc["ok"]:
        notes.append(
            "the re-fitted scores do not reproduce the run's recorded IC (max |diff| "
            f"{sc['max_abs_diff']:.2e}): the code under qts/core, data, risk or alpha changed since the run (or "
            "the data did); the IC columns describe the re-fitted scores, not exactly the traded ones"
        )
    data = {
        "schema": SCHEMA,
        "created_utc": utc_stamp(),
        "run_id": book.run_id,
        "run_dir": str(run_dir),
        "config_hash": str(book.manifest.get("config_hash") or cfg.hash()),
        "use_ml": book.use_ml,
        "signal_lag_days": int(cfg.execution.signal_lag_days),
        "horizon_days": int(cfg.alpha.horizon_days),
        "data": {
            "store_spec": spec_dict,
            "window": window,
            "sealed_end": sealed_end or None,
            "holdout_start": book.holdout_start or None,
            "market_hash": mh,
            "load_coverage": load_cov,
        },
        "period": {
            "start": str(book.dec[0].date()),
            "end": str(book.ret_dates[-1].date()),
            "n_days": len(book.dec),
            "n_months": len(book.ret_dates.to_period("M").unique()),
            "trading_days_per_year": int(cfg.trading_days_per_year),
        },
        **res,
        "thresholds": dict(THRESHOLDS),
        "help": dict(HELP),
        "notes": notes,
        "seconds": time.perf_counter() - t0,
    }
    out = IndustryBreakdown(data=data)
    if write:
        p = write_json(run_dir / OUTPUT_FILE, out.to_dict())
        out.json_path = str(p)
    return out


# ============================================================================ CLI
def cmd_breakdown(a: argparse.Namespace) -> int:
    from qts.research.data import StoreLoadError
    from qts.research.seal import SealError

    if a.breakdown_cmd != "industries":
        return 2
    spec = None
    if a.store:
        try:
            spec = StoreSpec.from_args(a)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    elif any(
        getattr(a, k, None) is not None for k in ("start", "end", "max_names", "min_mcap_ever", "max_rank")
    ):
        print(
            "error: --start / --end / --max-rank / --max-names / --min-mcap-ever need --store",
            file=sys.stderr,
        )
        return 2
    try:
        r = run_industry_breakdown(
            a.run,
            spec=spec,
            runs_root=a.runs_root,
            cache_root=a.cache_root,
            with_ic=not a.no_ic,
            refuse_refit=a.no_refit,
            allow_partial_coverage=a.allow_partial_coverage,
            log=lambda m: print(m, flush=True),
        )
    except (BreakdownError, SealError, StoreLoadError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(r.format())
    if a.json:
        write_json(a.json, r.to_dict())
        print(f"written {a.json}")
    return 0


def add_breakdown_parser(sub: Any) -> None:
    """``qts breakdown industries RUN ...`` (registered by :mod:`qts.research.cli`)."""
    from qts.research.cli import DEFAULT_RUNS_ROOT, _store_args

    bd = sub.add_parser(
        "breakdown", help="diagnostics of a finished run (nothing re-optimised, nothing logged to the ledger)"
    )
    bsub = bd.add_subparsers(dest="breakdown_cmd", required=True)
    ind = bsub.add_parser(
        "industries",
        help="positioning, P&L, costs and within-industry IC per industry (FF48) and FF12 sector, and whether "
        "the edge is broad or concentrated; writes <run>/industry_breakdown.json",
    )
    ind.add_argument("run", help="run directory, or a run id under --runs-root")
    _store_args(ind, required=False)
    ind.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT, help="where run ids are looked up")
    ind.add_argument(
        "--cache-root",
        default=None,
        help="stage cache of the fit (default: the run config's cache.root, else <run>/../_cache)",
    )
    ind.add_argument("--no-ic", action="store_true", help="skip the within-industry IC (no fit; P&L only)")
    ind.add_argument(
        "--no-refit", action="store_true", help="refuse instead of re-fitting when the stage cache is cold"
    )
    ind.add_argument(
        "--allow-partial-coverage",
        action="store_true",
        help="load although the store prefilter misses universe names (as `qts run --allow-partial-coverage`)",
    )
    ind.add_argument("--json", default=None, help="also write the result to this path")
    ind.add_argument("-v", "--verbose", action="store_true")
    bd.set_defaults(func=cmd_breakdown)


__all__ = [
    "HELP",
    "OUTPUT_FILE",
    "SCHEMA",
    "THRESHOLDS",
    "BreakdownError",
    "ICInputs",
    "IndustryBreakdown",
    "RunBook",
    "add_breakdown_parser",
    "booked_returns",
    "breadth_stats",
    "breadth_verdict",
    "cmd_breakdown",
    "compute_breakdown",
    "industry_labels",
    "predict_stage_cache",
    "read_run_book",
    "run_industry_breakdown",
    "store_spec_from_command",
]
