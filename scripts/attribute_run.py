"""Run attribution: why does the IC not show up in the P&L, and where does the turnover come from?

Fits the pipeline exactly as ``qts run`` does (stage cache reused with ``--cache``,
so the alpha scores and the risk model are the ones the run traded), reads the
book from a finished run directory (``weights`` / ``trades`` / ``returns`` /
``costs`` / ``nav`` / ``exante_vol`` / ``constraints``) and prints five tables.
Nothing is re-optimised and nothing is tuned: every number is either a paper
portfolio built from the score with no optimizer, or an accounting split of the
book the run actually held.

1. **Signal -> ideal portfolios, no optimizer.**  Daily rebalanced paper books
   from the combined score on the eligible names, traded with the run's signal lag
   (score of ``s`` earns the return ``s+L -> s+L+1``), gross of costs, drift
   ignored: (a) rank-weighted dollar-neutral long/short; (b) industry-neutral
   (rank demeaned within industry); (c) residual of the rank on industries + the
   risk-only styles the optimizer neutralises (beta, size, residual vol,
   liquidity, reversal); (d) the Grinold form of (c), ``w ~ z / sigma`` (weighted
   residual, sigma = the alpha model's 252-day idiosyncratic vol); (e) top /
   bottom decile, equal and cap weighted; (f) (c) inside each size tercile; and
   (c) from a 20-day EWMA of the rank and from the score sampled every 5 / 21
   days.  Sharpe, exposure and turnover at a common 10 % ex-post vol, by fold.
2. **Realised book attribution.**  Long vs short side, by year, by size tercile,
   by sector, and the risk model's factor attribution (``w_t' B_t f_{t+1}`` per
   market / industry / style, specific = total - factors), the book's style
   exposures against their bands, and its correlation with the ideal book (c),
   the market and every style factor return.
3. **Risk-forecast gap.**  Realised vs ex-ante vol by year; the ex-ante variance
   split (factor groups / specific / alignment, on ``--risk-dates`` sampled solves)
   against the realised split; the risk model's forecast for the *ideal* book (c)
   against its realised vol; the regression of the book on (c).
4. **Turnover anatomy.**  Rank autocorrelation of the score (and of each sleeve)
   at 1 / 5 / 21 / 63 days, on ML refit dates vs other days; day-to-day gamma
   changes; forced trades; drift; proportional re-scaling; small trades; turnover
   by size tercile; and the turnover the same sequence of books would need if it
   were only updated every 5 / 21 days.
5. **Cost reality check.**  Gross P&L, cost by kind, breakeven cost per side, and
   the cost per side by size tercile (re-priced with the run's cost model).

Usage (same data flags as ``qts run`` / scripts/diagnose_signal.py)::

    python scripts/attribute_run.py --store data/store --start 2000-01-01 \\
        --max-names 3000 --min-mcap-ever 3e8 --cache --ml \\
        --set alpha.horizon_days=60 --run runs/<run_id>
    python scripts/attribute_run.py --synthetic      # builds a short run in a temp dir

How to read each table: docs/handbook/12_operating_guide.md, "When IC and P&L disagree".
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_signal import add_data_args, clamp_universe, load_config, load_market
from qts.backtest.simulator import DailyBarSimulator
from qts.pipeline import ResearchPipeline, purged_fold_blocks
from qts.risk.factor_returns import industry_parent

TDY = 252
TARGET_VOL = 0.10  # common ex-post vol of the paper books
MIN_NAMES = 20  # fewer scored names on a date -> no paper book that day
EWMA_SPAN = 20
SMALL_TRADE = 5e-4  # 5 bp of NAV
HELD_EPS = 1e-8
GAMMA_JUMP = 0.10  # |dlog gamma| above this is a "gamma moved" day
BAND_TOUCH = 0.98  # |exposure| >= this share of the band counts as "at the band"
AUTOCORR_LAGS = (1, 5, 21, 63)
HOLD_EVERY = (5, 21)
SIZE_NAMES = ("small", "mid", "large")

IDEAL_LABELS: dict[str, str] = {
    "a_rank": "(a) rank L/S",
    "b_ind": "(b) + industry-neutral",
    "c_resid": "(c) + beta / risk-style neutral",
    "d_grinold": "(d) (c) with w ~ z/sigma",
    "e_dec_ew": "(e) decile L/S, equal weight",
    "e_dec_vw": "(e) decile L/S, cap weight",
    "f_large": "(f) (c) in large tercile",
    "f_mid": "(f) (c) in mid tercile",
    "f_small": "(f) (c) in small tercile",
    "c_ewma": f"(c) on {EWMA_SPAN}d-EWMA score",
    "c_hold5": "(c) sampled every 5d",
    "c_hold21": "(c) sampled every 21d",
}


# ----------------------------------------------------------------------------- small helpers
def _sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TDY)) if sd > 0 else float("nan")


def _ann_vol(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    return float(r.std(ddof=1) * np.sqrt(TDY)) if len(r) > 2 else float("nan")


def _nanmean(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    return float(v[np.isfinite(v)].mean()) if np.isfinite(v).any() else float("nan")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """``(alpha, beta, R2)`` of ``y = alpha + beta x``."""
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 3 or x[m].std() == 0:
        return float("nan"), float("nan"), float("nan")
    b = float(np.cov(y[m], x[m], ddof=1)[0, 1] / x[m].var(ddof=1))
    a = float(y[m].mean() - b * x[m].mean())
    r2 = float(np.corrcoef(y[m], x[m])[0, 1] ** 2)
    return a, b, r2


def _table(df: pd.DataFrame, fmt: str = "{:.2f}", formats: dict[str, str] | None = None) -> str:
    out = df.copy()
    for c in out.columns:
        f = (formats or {}).get(str(c), fmt)
        out[c] = [
            v if isinstance(v, str) else ("" if v is None or not np.isfinite(v) else f.format(v))
            for v in out[c]
        ]
    with pd.option_context("display.width", 250, "display.max_columns", 60, "display.max_rows", 500):
        return out.to_string()


def _head(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def _normalise(w: np.ndarray) -> np.ndarray:
    """Scale to gross 2 (100 % long / 100 % short for a dollar-neutral vector)."""
    g = np.abs(w).sum()
    return w * (2.0 / g) if g > 0 else w


def _residual(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Least-squares residual of the columns of ``Y`` on ``X`` (minimum-norm
    solution: an all-zero or collinear column of ``X`` is harmless)."""
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return Y - X @ beta


# ----------------------------------------------------------------------------- inputs
@dataclass
class RunPanels:
    """A run directory on the fitted market's ticker grid (rows: decision dates)."""

    run_dir: Path
    dec: pd.DatetimeIndex  # decision dates t (post-trade book held t -> t+1)
    ret_dates: pd.DatetimeIndex  # realisation dates t+1
    W: np.ndarray  # post-trade weights
    X: np.ndarray  # trades (weight changes)
    gross: np.ndarray
    net: np.ndarray
    gab: np.ndarray  # gross after borrow
    costs: pd.DataFrame  # USD by decision date
    nav: np.ndarray  # pre-trade NAV at t
    exante_vol: np.ndarray  # annualised, of the post-trade book (NaN when not recorded)
    gamma: np.ndarray  # risk aversion of the solve at t (NaN when not recorded)
    summary: dict[str, Any]
    config_hash: str


def load_run_panels(run_dir: Path, tickers: pd.Index) -> RunPanels:
    w = pd.read_parquet(run_dir / "weights.parquet")
    x = pd.read_parquet(run_dir / "trades.parquet")
    ret = pd.read_parquet(run_dir / "returns.parquet")
    costs = pd.read_parquet(run_dir / "costs.parquet")
    nav = pd.read_parquet(run_dir / "nav.parquet")["nav"]
    unknown = [c for c in w.columns if c not in tickers]
    if unknown:
        raise SystemExit(
            f"{len(unknown)} tickers of the run are not in the loaded market (e.g. {unknown[:5]}): "
            "the data flags do not match the run"
        )
    n = min(len(w), len(ret))
    dec = pd.DatetimeIndex(w.index[:n])

    def grid(df: pd.DataFrame) -> np.ndarray:
        return df.reindex(index=dec, columns=tickers).fillna(0.0).to_numpy(dtype=float)

    ev = np.full(n, np.nan)
    f = run_dir / "exante_vol.parquet"
    if f.exists():
        ev = pd.read_parquet(f)["exante_vol"].reindex(dec).to_numpy(dtype=float)
    gamma = np.full(n, np.nan)
    f = run_dir / "constraints.parquet"
    if f.exists():
        cs = pd.read_parquet(f)
        if "risk_aversion" in cs:
            gamma = cs.set_index("date")["risk_aversion"].reindex(dec).to_numpy(dtype=float)
    summary = (
        json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    )
    manifest = (
        json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
    )
    gab = ret["gross_after_borrow"] if "gross_after_borrow" in ret else ret["gross"]
    return RunPanels(
        run_dir=run_dir,
        dec=dec,
        ret_dates=pd.DatetimeIndex(ret.index[:n]),
        W=grid(w),
        X=grid(x),
        gross=ret["gross"].to_numpy(dtype=float)[:n],
        net=ret["net"].to_numpy(dtype=float)[:n],
        gab=gab.to_numpy(dtype=float)[:n],
        costs=costs.reindex(dec).fillna(0.0),
        nav=nav.reindex(dec).to_numpy(dtype=float),
        exante_vol=ev,
        gamma=gamma,
        summary=summary,
        config_hash=str(manifest.get("config_hash", "")),
    )


class Market:
    """The fitted pipeline's panels as arrays on ``md_u.dates x md_u.tickers``."""

    def __init__(self, pipe: ResearchPipeline, run: RunPanels):
        assert pipe.md_u is not None and pipe.risk is not None and pipe.alpha is not None
        assert pipe.cost is not None
        md, risk, alpha, cfg = pipe.md_u, pipe.risk, pipe.alpha, pipe.cfg
        self.pipe, self.cfg, self.md = pipe, cfg, md
        self.dates, self.tickers = md.dates, md.tickers
        self.lag = pipe.signal_lag_days()
        self.sim = DailyBarSimulator(cfg, pipe.cost)
        # the simulator's own return panel (data-error closes blanked; a halt-period
        # move lands on the resumption day)
        close, _ = self.sim._clean_close(md, md.dates)
        R = close / close.ffill().shift(1) - 1.0
        self.R = np.nan_to_num(R.where(close.notna()).to_numpy(dtype=float))
        self.elig = md.eligibility().reindex(index=md.dates, columns=md.tickers).to_numpy(dtype=bool)
        self.mcap = md.market_cap.reindex(index=md.dates, columns=md.tickers).to_numpy(dtype=float)
        self.score = alpha.combined_score_.reindex(index=md.dates, columns=md.tickers).to_numpy(dtype=float)
        self.sigma = alpha.resid_vol_h_.reindex(index=md.dates, columns=md.tickers).to_numpy(dtype=float)
        # exposures (fundamental model): row of the most recent exposure date <= d
        ex = risk.exposures
        self.ex = ex
        self.ex_row = ex.dates.get_indexer(md.dates, method="ffill")
        self.ex_col = ex.tickers.get_indexer(md.tickers)  # -1: ticker unknown to the risk model
        n_static_ind = ex.static.shape[1] - 1
        self.n_ind = ex.n_industries if ex.industry_codes is not None else n_static_ind
        self.factors = [str(c) for c in ex.factors]
        self.ind_names = self.factors[1 : 1 + self.n_ind]
        self.style_names = self.factors[1 + self.n_ind :]
        self._static_codes = None
        if ex.industry_codes is None:
            dummies = ex.static[:, 1:]
            self._static_codes = np.where(dummies.sum(axis=1) > 0, dummies.argmax(axis=1), -1)
        self.neutral = [s for s in pipe.neutral_styles() if s in self.style_names]
        self.neutral_idx = [self.style_names.index(s) for s in self.neutral]
        fr = risk.factor_returns
        assert fr is not None
        self.f = fr.reindex(index=md.dates, columns=self.factors).fillna(0.0).to_numpy(dtype=float)
        # run alignment
        self.pos = md.dates.get_indexer(run.dec)  # decision positions
        if (self.pos < 0).any() or self.pos[-1] + 1 >= len(md.dates):
            raise SystemExit("the run's dates are not inside the loaded market: the data flags do not match")
        nxt = md.dates[self.pos + 1]
        if not nxt.equals(run.ret_dates):
            print("WARNING: the run's return dates are not the next market date of its decision dates")
        self.sig_pos = self.pos - self.lag  # signal date of the solve at t

    def exposures_at(self, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """``(industry code per ticker [-1 none], styles (N, S), valid (N,))`` of the
        exposure date in force on market position ``p``; ``None`` before the first."""
        i = int(self.ex_row[p])
        if i < 0:
            return None
        ex, col = self.ex, self.ex_col
        ok = col >= 0
        c = np.where(ok, col, 0)
        codes = self._static_codes[c] if self._static_codes is not None else ex.industry_codes[i][c]
        valid = ex.valid[i][c] & ok
        sty = np.nan_to_num(np.asarray(ex.styles[i][c], dtype=float))
        sty[~valid] = 0.0
        return np.where(valid, codes, -1), sty, valid

    def size_tercile(self, p: int, members: np.ndarray) -> np.ndarray:
        """0 / 1 / 2 = small / mid / large third of ``members`` by market cap at ``p``
        (-1 outside ``members`` or without a cap)."""
        out = np.full(len(self.tickers), -1, dtype=np.int8)
        mc = self.mcap[p]
        m = members & np.isfinite(mc) & (mc > 0)
        n = int(m.sum())
        if n >= 3:
            r = rankdata(mc[m], method="ordinal") - 1
            out[m] = np.minimum((3 * r) // n, 2).astype(np.int8)
        return out

    def fold_blocks(self, index: pd.DatetimeIndex) -> list[np.ndarray]:
        v = self.cfg.validation
        blocks = purged_fold_blocks(
            index, v.n_splits, v.embargo_days, self.cfg.alpha.horizon_days, getattr(v, "min_fold_days", 20)
        )
        return [index.get_indexer(b) for b in blocks]


# ----------------------------------------------------------------------------- 1. ideal portfolios
@dataclass
class Ideal:
    pnl: dict[str, np.ndarray]  # per run day (aligned with run.ret_dates), book at gross 2
    turnover: dict[str, np.ndarray]  # sum |w_s - w_{s-1}| on the signal date of that day
    w_c: dict[int, np.ndarray]  # signal position -> weights of (c), for the risk-forecast test


def ideal_portfolios(mk: Market, keep_w_at: set[int]) -> Ideal:
    """One pass over the signal dates of the run; every book is rebuilt daily from
    the score of that date only (no optimizer, no costs, no drift)."""
    N, T = len(mk.tickers), len(mk.dates)
    names = list(IDEAL_LABELS)
    sig = mk.sig_pos
    n_run = len(sig)
    start = max(int(sig.min()) - 3 * EWMA_SPAN, 0)
    row_of = {int(p): k for k, p in enumerate(sig)}
    pnl = {k: np.full(n_run, np.nan) for k in names}
    to = {k: np.full(n_run, np.nan) for k in names}
    prev = {k: np.zeros(N) for k in names}
    held: dict[int, np.ndarray] = {h: np.zeros(N) for h in HOLD_EVERY}
    ew = np.full(N, np.nan)
    lam = 1.0 - 2.0 / (EWMA_SPAN + 1.0)
    w_c_keep: dict[int, np.ndarray] = {}
    zeros = np.zeros(N)

    for p in range(start, int(sig.max()) + 1):
        s = mk.score[p]
        m = np.isfinite(s)
        n = int(m.sum())
        x = np.full(N, np.nan)
        if n >= MIN_NAMES:
            x[m] = (rankdata(s[m]) - 0.5) / n - 0.5
        ew = np.where(np.isfinite(x), np.where(np.isfinite(ew), lam * ew + (1.0 - lam) * x, x), ew)
        k = row_of.get(p)
        if k is None and p < int(sig.min()):
            continue  # EWMA warm-up only
        w: dict[str, np.ndarray] = dict.fromkeys(names, zeros)
        expo = mk.exposures_at(p) if n >= MIN_NAMES else None
        if n >= MIN_NAMES:
            w["a_rank"] = _normalise(np.where(m, x, 0.0))
            top, bot = m & (x >= 0.4), m & (x < -0.4)
            if top.any() and bot.any():
                w["e_dec_ew"] = top / top.sum() - bot / bot.sum()
                mc = np.where(np.isfinite(mk.mcap[p]) & (mk.mcap[p] > 0), mk.mcap[p], 0.0)
                if (mc * top).sum() > 0 and (mc * bot).sum() > 0:
                    w["e_dec_vw"] = mc * top / (mc * top).sum() - mc * bot / (mc * bot).sum()
        if expo is not None:
            codes, sty, valid = expo
            M = m & valid
            idx = np.flatnonzero(M)
            if len(idx) >= MIN_NAMES:
                _, inv = np.unique(codes[idx], return_inverse=True)
                n_g = int(inv.max()) + 1
                Xd = np.zeros((len(idx), n_g + len(mk.neutral_idx)))
                Xd[np.arange(len(idx)), inv] = 1.0
                Xd[:, n_g:] = sty[idx][:, mk.neutral_idx]
                xi = x[idx]
                # (b) industry-neutral
                gmean = np.bincount(inv, weights=xi, minlength=n_g) / np.bincount(inv, minlength=n_g)
                full = np.zeros(N)
                full[idx] = xi - gmean[inv]
                w["b_ind"] = _normalise(full)
                # (c) + beta / risk-only styles; same projection for the smoothed score
                xe = np.where(np.isfinite(ew[idx]), ew[idx], 0.0)
                res = _residual(Xd, np.column_stack([xi, xe - xe.mean()]))
                full = np.zeros(N)
                full[idx] = res[:, 0]
                w["c_resid"] = _normalise(full)
                full = np.zeros(N)
                full[idx] = res[:, 1]
                w["c_ewma"] = _normalise(full)
                # (d) Grinold: alpha = sigma z, w = D^-1 (alpha - B b) => weighted residual / sigma
                sg = mk.sigma[p][idx]
                med = np.nanmedian(sg) if np.isfinite(sg).any() else 1.0
                sg = np.where(np.isfinite(sg) & (sg > 0), sg, med)
                lo, hi = np.quantile(sg, [0.05, 0.95])
                sg = np.clip(sg, max(lo, 1e-8), max(hi, 1e-8))
                rd = _residual(Xd / sg[:, None], xi[:, None])[:, 0]
                full = np.zeros(N)
                full[idx] = rd / sg
                w["d_grinold"] = _normalise(full)
                # (f) (c) inside each size tercile of the scored names
                terc = mk.size_tercile(p, m)[idx]
                for g, nm in enumerate(SIZE_NAMES):
                    rows = np.flatnonzero(terc == g)
                    if len(rows) >= MIN_NAMES:
                        full = np.zeros(N)
                        full[idx[rows]] = _residual(Xd[rows], (xi[rows] - xi[rows].mean())[:, None])[:, 0]
                        w[f"f_{nm}"] = _normalise(full)
        for h in HOLD_EVERY:
            if (p - int(sig.min())) % h == 0:
                held[h] = w["c_resid"]
            w[f"c_hold{h}"] = held[h]
        if p in keep_w_at:
            w_c_keep[p] = w["c_resid"].copy()
        q = p + mk.lag + 1  # the score of p trades at the close of p+L and earns p+L -> p+L+1
        if k is not None and q < T:
            r = mk.R[q]
            for name in names:
                pnl[name][k] = float(w[name] @ r)
                to[name][k] = float(np.abs(w[name] - prev[name]).sum())
        for name in names:
            prev[name] = w[name]
    for name in names:  # the first day's "turnover" is the build of the book
        to[name][0] = np.nan
    return Ideal(pnl=pnl, turnover=to, w_c=w_c_keep)


def print_ideal(mk: Market, run: RunPanels, ideal: Ideal) -> None:
    _head(
        "1. SIGNAL -> IDEAL PORTFOLIOS (no optimizer, gross of costs, signal lag "
        f"{mk.lag}d, each scaled to {TARGET_VOL:.0%} ex-post vol)"
    )
    folds = mk.fold_blocks(run.ret_dates)
    rows: dict[str, dict[str, float]] = {}

    def row(label: str, r: np.ndarray, to: np.ndarray, gross: float) -> None:
        vol = _ann_vol(r)
        k = TARGET_VOL / vol if vol and np.isfinite(vol) and vol > 0 else float("nan")
        mt = float(np.nanmean(to))
        d: dict[str, float] = {
            "Sharpe": _sharpe(r),
            "vol@gross2 %": 100 * vol * 2.0 / gross if gross > 0 else float("nan"),
            "gross@10%": k * gross,
            "to/day %NAV@10%": 100 * k * mt,
            "to/day %gross": 100 * mt / gross if gross > 0 else float("nan"),
            "hold d": 2.0 * gross / mt if mt > 0 else float("nan"),
        }
        for j, b in enumerate(folds):
            d[f"f{j + 1}"] = _sharpe(r[b])
        rows[label] = d

    for name, label in IDEAL_LABELS.items():
        row(label, ideal.pnl[name], ideal.turnover[name], 2.0)
    g_book = float(np.abs(run.W).sum(axis=1).mean())
    to_book = np.abs(run.X).sum(axis=1)
    row("REALISED BOOK gross", run.gross, to_book, g_book)
    row("REALISED BOOK net", run.net, to_book, g_book)
    fm = {
        "vol@gross2 %": "{:.1f}",
        "to/day %NAV@10%": "{:.1f}",
        "to/day %gross": "{:.1f}",
        "hold d": "{:.0f}",
    }
    print(_table(pd.DataFrame(rows).T, formats=fm))
    print(
        "\nSharpe / f1..fN: annualised, whole run and by validation fold.  vol@gross2: ex-post vol of the\n"
        "book at 100 % long / 100 % short.  gross@10% and to/day %NAV@10%: gross exposure and two-sided\n"
        "daily turnover the book needs to run at 10 % vol.  hold d = 2 x gross / turnover (the report\n"
        "card's effective holding period).  Paper books ignore drift, costs, caps and liquidity."
    )
    c = ideal.pnl["c_resid"]
    print(
        f"corr(realised gross, ideal (c)) = {_corr(run.gross, c):+.2f}; "
        f"corr(realised gross, ideal (a)) = {_corr(run.gross, ideal.pnl['a_rank']):+.2f}"
    )


# ----------------------------------------------------------------------------- 2. realised attribution
@dataclass
class Attribution:
    pieces: pd.DataFrame  # per run day: market / industry / style_<k> / specific contribution
    expo_attr: np.ndarray  # (n, K) book exposures on B_t (attribution)
    expo_seen: np.ndarray  # (n, K) book exposures on B_s (what the optimizer constrained)
    terc: np.ndarray  # (n, N) size tercile of names eligible at t (-1 otherwise)
    sector: np.ndarray  # (n, N) industry code at t (-1 none)


def book_attribution(mk: Market, run: RunPanels) -> Attribution:
    n, N = run.W.shape
    K = len(mk.factors)
    e_attr, e_seen = np.zeros((n, K)), np.zeros((n, K))
    terc = np.full((n, N), -1, dtype=np.int8)
    sector = np.full((n, N), -1, dtype=np.int16)
    S0 = 1 + mk.n_ind

    def expose(p: int, w: np.ndarray, out: np.ndarray) -> np.ndarray | None:
        expo = mk.exposures_at(p)
        if expo is None:
            return None
        codes, sty, valid = expo
        wv = np.where(valid, w, 0.0)
        out[0] = wv.sum()
        has = codes >= 0
        if mk.n_ind:
            out[1:S0] = np.bincount(codes[has], weights=wv[has], minlength=mk.n_ind)[: mk.n_ind]
        out[S0:] = wv @ sty
        return codes

    for k in range(n):
        p = int(mk.pos[k])
        w = run.W[k]
        codes = expose(p, w, e_attr[k])
        if codes is not None:
            sector[k] = codes
        if mk.sig_pos[k] >= 0:
            expose(int(mk.sig_pos[k]), w, e_seen[k])
        terc[k] = mk.size_tercile(p, mk.elig[p])
    contrib = e_attr * mk.f[mk.pos + 1]  # f_{t+1} was estimated on B_t
    pieces = pd.DataFrame(index=run.ret_dates)
    pieces["market"] = contrib[:, 0]
    pieces["industry"] = contrib[:, 1:S0].sum(axis=1)
    for j, s in enumerate(mk.style_names):
        pieces[f"style_{s}"] = contrib[:, S0 + j]
    pieces["specific"] = run.gross - contrib.sum(axis=1)
    return Attribution(pieces=pieces, expo_attr=e_attr, expo_seen=e_seen, terc=terc, sector=sector)


def _piece_row(r: np.ndarray, total: np.ndarray) -> dict[str, float]:
    var_t = np.nanvar(total, ddof=1)
    return {
        "ann P&L %": 100 * TDY * float(np.nanmean(r)),
        "ann vol %": 100 * _ann_vol(r),
        "Sharpe": _sharpe(r),
        "var share %": 100 * float(np.cov(r, total, ddof=1)[0, 1] / var_t) if var_t > 0 else float("nan"),
        "corr w/ book": _corr(r, total),
    }


def print_attribution(mk: Market, run: RunPanels, ideal: Ideal, att: Attribution) -> None:
    _head("2. REALISED BOOK ATTRIBUTION (the run's weights x the simulator's returns)")
    W = run.W
    Rn = mk.R[mk.pos + 1]
    C = W * Rn
    recon = C.sum(axis=1)
    print(
        f"reconciliation: sum_i w_i r_i vs recorded gross return: corr {_corr(recon, run.gross):.4f}, "
        f"mean |diff| {1e4 * np.nanmean(np.abs(recon - run.gross)):.2f} bp/day, ann. diff "
        f"{100 * TDY * np.nanmean(run.gross - recon):+.2f} % (delisting haircuts land here)"
    )

    # -- long vs short ---------------------------------------------------------
    el = mk.elig[mk.pos]
    cnt = np.maximum(el.sum(axis=1), 1)
    univ = (Rn * el).sum(axis=1) / cnt  # equal-weight eligible universe, t -> t+1
    rows = {}
    for name, side in (("long", W > 0), ("short", W < 0)):
        pnl = (C * side).sum(axis=1)
        expo = (W * side).sum(axis=1)
        excess = pnl - expo * univ
        rows[name] = {
            "avg exposure": float(expo.mean()),
            "ann P&L %": 100 * TDY * float(pnl.mean()),
            "ann excess %": 100 * TDY * float(excess.mean()),
            "excess vol %": 100 * _ann_vol(excess),
            "excess Sharpe": _sharpe(excess),
        }
    rows["total"] = {
        "avg exposure": float(W.sum(axis=1).mean()),
        "ann P&L %": 100 * TDY * float(recon.mean()),
        "ann excess %": 100 * TDY * float((recon - W.sum(axis=1) * univ).mean()),
        "excess vol %": 100 * _ann_vol(recon - W.sum(axis=1) * univ),
        "excess Sharpe": _sharpe(recon - W.sum(axis=1) * univ),
    }
    print(
        "\n-- long side vs short side (excess = side P&L - side exposure x equal-weight universe return) --"
    )
    print(_table(pd.DataFrame(rows).T))

    # -- by year ---------------------------------------------------------------
    yr = run.ret_dates.year.to_numpy()
    to_book = np.abs(run.X).sum(axis=1)
    c = ideal.pnl["c_resid"]
    cost_bp = 1e4 * (run.gross - run.net)
    rows = {}
    for y in np.unique(yr):
        m = yr == y
        rows[int(y)] = {
            "gross %": 100 * float(np.nansum(run.gross[m])),
            "net %": 100 * float(np.nansum(run.net[m])),
            "cost %": float(np.nansum(cost_bp[m])) / 100,
            "Sharpe gross": _sharpe(run.gross[m]),
            "Sharpe net": _sharpe(run.net[m]),
            "ideal(c) Sharpe": _sharpe(c[m]),
            "corr w/ (c)": _corr(run.gross[m], c[m]),
            "to/day %": 100 * float(to_book[m].mean()),
            "gross lev": float(np.abs(W[m]).sum(axis=1).mean()),
        }
    print("\n-- by calendar year (returns summed, not compounded) --")
    print(_table(pd.DataFrame(rows).T))

    # -- by size tercile ---------------------------------------------------------
    absW, absX = np.abs(W), np.abs(run.X)
    rows = {}
    for g, nm in [(2, "large"), (1, "mid"), (0, "small"), (-1, "not eligible")]:
        m = att.terc == g
        pnl = (C * m).sum(axis=1)
        gross = (absW * m).sum(axis=1)
        d = _piece_row(pnl, run.gross)
        rows[nm] = {
            "avg gross": float(gross.mean()),
            "avg net": float((W * m).sum(axis=1).mean()),
            "ann P&L %": d["ann P&L %"],
            "P&L / gross %": d["ann P&L %"] / gross.mean() if gross.mean() > 1e-3 else float("nan"),
            "Sharpe": d["Sharpe"],
            "long P&L %": 100 * TDY * float((C * (m & (W > 0))).sum(axis=1).mean()),
            "short P&L %": 100 * TDY * float((C * (m & (W < 0))).sum(axis=1).mean()),
            "to/day %": 100 * float((absX * m).sum(axis=1).mean()),
        }
    print("\n-- by size tercile of the eligible universe at the decision date --")
    print(_table(pd.DataFrame(rows).T))
    print("(compare with rows (f) of table 1: where the signal works vs where the book holds its gross)")

    # -- by sector ---------------------------------------------------------------
    parents = [industry_parent(nm) for nm in mk.ind_names]
    if len(set(parents)) <= 1:  # labels the FF48 -> FF12 map does not know (synthetic)
        parents = list(mk.ind_names)
    groups = sorted(set(parents))
    gidx = np.array([groups.index(p) for p in parents] + [len(groups)])  # last = unclassified
    sec = gidx[att.sector]  # code -1 -> last entry
    rows = {}
    for j, nm in enumerate([*groups, "(none)"]):
        m = sec == j
        gross = (absW * m).sum(axis=1)
        if gross.mean() <= 1e-6:
            continue
        pnl = (C * m).sum(axis=1)
        rows[nm] = {
            "avg gross": float(gross.mean()),
            "avg net": float((W * m).sum(axis=1).mean()),
            "ann P&L %": 100 * TDY * float(pnl.mean()),
            "Sharpe": _sharpe(pnl),
        }
    print("\n-- by sector (FF12 parent of the risk model's industry; raw industry when unmapped) --")
    print(_table(pd.DataFrame(rows).T.sort_values("ann P&L %", ascending=False)))

    # -- factor attribution ------------------------------------------------------
    P = att.pieces
    factor_total = P.drop(columns="specific").sum(axis=1).to_numpy()
    style_cols = [c for c in P.columns if c.startswith("style_")]
    rows = {"TOTAL gross": _piece_row(run.gross, run.gross)}
    rows["all factors"] = _piece_row(factor_total, run.gross)
    rows["  market"] = _piece_row(P["market"].to_numpy(), run.gross)
    rows["  industry"] = _piece_row(P["industry"].to_numpy(), run.gross)
    rows["  styles (sum)"] = _piece_row(P[style_cols].sum(axis=1).to_numpy(), run.gross)
    for cname in style_cols:
        rows[f"    {cname[6:]}"] = _piece_row(P[cname].to_numpy(), run.gross)
    rows["specific (total - factors)"] = _piece_row(P["specific"].to_numpy(), run.gross)
    print("\n-- factor attribution: w_t' B_t f_{t+1} with the fundamental risk model's factor returns --")
    print(_table(pd.DataFrame(rows).T))
    print(
        "var share = cov(piece, total) / var(total) (sums to 100).  'specific' also carries the hybrid\n"
        "model's statistical factors, names outside the risk model and delisting haircuts."
    )

    # -- exposures vs bands ------------------------------------------------------
    S0 = 1 + mk.n_ind
    pc = mk.cfg.portfolio
    beta_band = float(getattr(mk.pipe.optimizer, "beta_band", getattr(pc, "beta_band", np.nan)))
    seen = att.expo_seen
    rows = {}

    def band_row(e: np.ndarray, band: float | None, role: str) -> dict[str, Any]:
        d: dict[str, Any] = {
            "role": role,
            "mean": float(e.mean()),
            "mean |e|": float(np.abs(e).mean()),
            "p95 |e|": float(np.quantile(np.abs(e), 0.95)),
            "band": band if band is not None else float("nan"),
            "% days at band": 100 * float((np.abs(e) >= BAND_TOUCH * band).mean()) if band else float("nan"),
        }
        return d

    rows["market (net)"] = band_row(seen[:, 0], float(pc.net_exposure_band), "net band")
    if mk.n_ind:
        rows["industry (max |net|)"] = band_row(
            np.abs(seen[:, 1:S0]).max(axis=1), float(pc.industry_band), "industry band"
        )
    for j, s in enumerate(mk.style_names):
        if s == "beta":
            band, role = beta_band, "beta band"
        elif s in mk.neutral:
            band, role = float(pc.style_band_sd), "risk-only"
        else:
            band, role = None, "alpha theme"
        rows[s] = band_row(seen[:, S0 + j], band, role)
    print("\n-- book exposures on the signal-date exposures the optimizer constrained (w_t' B_s) --")
    print(_table(pd.DataFrame(rows).T, fmt="{:.3f}", formats={"% days at band": "{:.0f}"}))
    print(f"(at band = |exposure| >= {BAND_TOUCH:.0%} of the band; alpha themes are not banded)")

    # -- correlations ------------------------------------------------------------
    f_next = mk.f[mk.pos + 1]
    rows = {
        "ideal (c)": {"corr": _corr(run.gross, c)},
        "ideal (a)": {"corr": _corr(run.gross, ideal.pnl["a_rank"])},
        "EW universe return": {"corr": _corr(run.gross, univ)},
        "market factor return": {"corr": _corr(run.gross, f_next[:, 0])},
    }
    for j, s in enumerate(mk.style_names):
        rows[f"factor return: {s}"] = {"corr": _corr(run.gross, f_next[:, S0 + j])}
    for j, s in enumerate(mk.style_names):
        rows[f"ideal (c) vs factor {s}"] = {"corr": _corr(c, f_next[:, S0 + j])}
    print("\n-- correlation of the book's daily gross return (last block: of the ideal book (c)) --")
    print(_table(pd.DataFrame(rows).T, fmt="{:+.2f}"))


# ----------------------------------------------------------------------------- 3. risk-forecast gap
def _exante_split(
    w: np.ndarray,
    B: np.ndarray,
    F: np.ndarray,
    D: np.ndarray,
    y: np.ndarray | None,
    nu: float,
    groups: dict[str, list[int]],
) -> dict[str, float]:
    """Ex-ante daily variance of ``w`` under ``B F B' + D + nu y y'``, split by factor
    group (each group's share of ``f' F f`` is ``f_g' (F f)_g``: additive, the
    cross-group covariances are shared equally), specific and alignment."""
    fexp = B.T @ w
    Ff = F @ fexp
    d = {f"factor:{gname}": float(fexp[ix] @ Ff[ix]) for gname, ix in groups.items() if ix}
    d["specific"] = float((w * w) @ D)
    d["alignment"] = float(nu * (y @ w) ** 2) if y is not None else 0.0
    return d


def print_risk_gap(mk: Market, run: RunPanels, ideal: Ideal, att: Attribution, sample: np.ndarray) -> None:
    _head("3. RISK-FORECAST GAP (realised vs ex-ante)")
    ev = run.exante_vol
    g = run.gross
    if np.isfinite(ev).any():
        yr = run.ret_dates.year.to_numpy()
        z = g / (ev / np.sqrt(TDY))
        rows = {}
        for y in [*np.unique(yr), -1]:
            m = (yr == y) if y != -1 else np.ones(len(g), dtype=bool)
            m = m & np.isfinite(ev) & (ev > 0)
            if m.sum() < 20:
                continue
            exa = float(np.sqrt(np.mean(ev[m] ** 2)))
            rows["ALL" if y == -1 else int(y)] = {
                "realised vol %": 100 * _ann_vol(g[m]),
                "ex-ante vol %": 100 * exa,
                "ratio": _ann_vol(g[m]) / exa if exa > 0 else float("nan"),
                "bias stat": float(np.nanstd(z[m], ddof=1)),
                "mean gamma": float(np.nanmean(run.gamma[m])) if np.isfinite(run.gamma[m]).any() else np.nan,
            }
        print("-- by year (ex-ante = rms of the recorded forecast for the post-trade book) --")
        print(_table(pd.DataFrame(rows).T))
    else:
        print("no exante_vol.parquet in the run directory: by-year table skipped")

    # -- ex-ante decomposition on sampled solves --------------------------------
    risk, alpha = mk.pipe.risk, mk.pipe.alpha
    assert risk is not None and alpha is not None
    use_align = bool(mk.cfg.risk.alignment)
    acc: dict[str, list[float]] = {}
    ideal_var: list[tuple[float, float]] = []  # (ex-ante var of (c) with alignment, without)
    chk: list[tuple[float, float]] = []
    t0 = time.time()
    for k in sample:
        s = mk.dates[int(mk.sig_pos[k])]
        try:
            a_exp = alpha.as_of(s).expected_return if use_align else None
            out = risk.as_of(s, alpha=a_exp)
        except (ValueError, KeyError):
            continue
        B = out.exposures
        col = mk.tickers.get_indexer(B.index)
        ok = col >= 0
        Bn = B.to_numpy(dtype=float)[ok]
        F = out.factor_cov.reindex(index=B.columns, columns=B.columns).to_numpy(dtype=float)
        D = np.nan_to_num(out.specific_var.reindex(B.index).to_numpy(dtype=float))[ok]
        y = None
        if out.alignment_vector is not None and out.alignment_var > 0:
            y = np.nan_to_num(out.alignment_vector.reindex(B.index).to_numpy(dtype=float))[ok]
        groups = {gname: [B.columns.get_loc(c) for c in cols] for gname, cols in out.factor_groups.items()}

        parts = (Bn, F, D, y, float(out.alignment_var), groups)
        d = _exante_split(run.W[k][col[ok]], *parts)
        for name, v in d.items():
            acc.setdefault(name, []).append(v)
        tot = sum(d.values())
        if np.isfinite(ev[k]):
            chk.append((np.sqrt(tot * TDY), ev[k]))
        wc = ideal.w_c.get(int(mk.sig_pos[k]))
        if wc is not None and np.abs(wc).sum() > 0:
            dc = _exante_split(wc[col[ok]], *parts)
            ideal_var.append((sum(dc.values()), sum(dc.values()) - dc["alignment"]))
    if not acc:
        print("\nno risk model output on the sampled dates: decomposition skipped")
        return
    n_s = len(next(iter(acc.values())))
    print(f"\n-- variance split on {n_s} sampled solves ({time.time() - t0:.0f}s) vs the realised split --")
    if chk:
        a, b = np.array(chk).T
        print(
            f"check: recomputed ex-ante vol / recorded = {np.mean(a) / np.mean(b):.3f} "
            f"(mean {100 * np.mean(a):.2f} % vs {100 * np.mean(b):.2f} %)"
        )
    P = att.pieces
    style_cols = [c for c in P.columns if c.startswith("style_")]
    realised = {
        "factor:market": P["market"].to_numpy(),
        "factor:industry": P["industry"].to_numpy(),
        "factor:style": P[style_cols].sum(axis=1).to_numpy(),
        "specific+statistical+alignment": P["specific"].to_numpy(),
    }
    exante = {
        "factor:market": np.mean(acc.get("factor:market", [0.0])),
        "factor:industry": np.mean(acc.get("factor:industry", [0.0])),
        "factor:style": np.mean(acc.get("factor:style", [0.0])),
        "specific+statistical+alignment": np.mean(acc["specific"])
        + np.mean(acc["alignment"])
        + np.mean(acc.get("factor:statistical", [0.0])),
    }
    var_real_tot = float(np.nanvar(g, ddof=1))
    var_exa_tot = float(sum(exante.values()))
    rows = {}
    for name, r in realised.items():
        cov = float(np.cov(r, g, ddof=1)[0, 1])
        rows[name] = {
            "ex-ante var share %": 100 * exante[name] / var_exa_tot,
            "ex-ante vol %": 100 * np.sqrt(max(exante[name], 0.0) * TDY),
            "realised var share %": 100 * cov / var_real_tot,
            "realised vol %": 100 * _ann_vol(r),
            "realised / ex-ante vol": _ann_vol(r) / np.sqrt(exante[name] * TDY)
            if exante[name] > 0
            else float("nan"),
        }
    rows["TOTAL"] = {
        "ex-ante var share %": 100.0,
        "ex-ante vol %": 100 * np.sqrt(var_exa_tot * TDY),
        "realised var share %": 100.0,
        "realised vol %": 100 * _ann_vol(g),
        "realised / ex-ante vol": _ann_vol(g) / np.sqrt(var_exa_tot * TDY),
    }
    print(_table(pd.DataFrame(rows).T))
    sub = {
        "  of which specific": np.mean(acc["specific"]),
        "  of which statistical factors": np.mean(acc.get("factor:statistical", [0.0])),
        "  of which alignment term": np.mean(acc["alignment"]),
    }
    for name, v in sub.items():
        print(
            f"{name:34s} ex-ante vol {100 * np.sqrt(max(v, 0) * TDY):5.2f} %  ({100 * v / var_exa_tot:4.1f} % of variance)"
        )
    print(
        "realised vol is each piece's stand-alone vol (pieces are correlated, so they do not add);\n"
        "realised var share = cov(piece, total) / var(total).  The piece whose realised / ex-ante\n"
        "ratio is far above 1 is where the forecast is short."
    )

    # -- the alpha portfolio as a risk source -----------------------------------
    c = ideal.pnl["c_resid"]
    print("\n-- is the unmodelled risk the alpha itself? --")
    if ideal_var:
        with_a, without = np.sqrt(np.mean(np.array(ideal_var), axis=0) * TDY)
        rc = _ann_vol(c)
        print(
            f"ideal book (c) at gross 2: realised vol {100 * rc:.2f} %; risk-model forecast "
            f"{100 * with_a:.2f} % with the alignment term, {100 * without:.2f} % without  ->  "
            f"realised / ex-ante = {rc / with_a:.2f} (with), {rc / without:.2f} (without)"
        )
    _, b, r2 = _ols(g, c)
    _, b_s, r2_s = _ols(P["specific"].to_numpy(), c)
    print(
        f"book gross return on ideal (c): beta {b:.3f}, R2 {r2:.3f}  ->  the alpha-portfolio component "
        f"of the book has vol {100 * abs(b) * _ann_vol(c):.2f} % of {100 * _ann_vol(g):.2f} %"
    )
    print(
        f"book 'specific' piece on ideal (c): beta {b_s:.3f}, R2 {r2_s:.3f}  (a diagonal specific model "
        "implies R2 ~ 0: what is above is common risk carried by the alpha direction)"
    )


# ----------------------------------------------------------------------------- 4. turnover anatomy
def rank_autocorr(panel: np.ndarray, elig: np.ndarray, positions: np.ndarray, lags: tuple[int, ...]):
    """Mean cross-sectional rank correlation between the row at ``p`` and at
    ``p - lag`` over names scored and eligible on both dates.  Returns
    ``(mean per lag, per-position lag-1 series)``."""
    out = {lag: [] for lag in lags}
    lag1 = np.full(len(positions), np.nan)
    for j, p in enumerate(positions):
        a = panel[p]
        for lag in lags:
            if p - lag < 0:
                continue
            b = panel[p - lag]
            m = np.isfinite(a) & np.isfinite(b) & elig[p] & elig[p - lag]
            if m.sum() < MIN_NAMES:
                continue
            r = float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])
            out[lag].append(r)
            if lag == 1:
                lag1[j] = r
    return {lag: float(np.mean(v)) if v else float("nan") for lag, v in out.items()}, lag1


def print_turnover(mk: Market, run: RunPanels, ideal: Ideal, att: Attribution, sample_every: int) -> None:
    _head("4. TURNOVER ANATOMY")
    alpha = mk.pipe.alpha
    assert alpha is not None
    X, W = run.X, run.W
    absX = np.abs(X)
    to = absX.sum(axis=1)
    gross = np.abs(W).sum(axis=1)
    n = len(to)
    print(
        f"book: turnover {100 * to.mean():.2f} %/day of NAV (two-sided), gross {gross.mean():.2f}, "
        f"effective holding {2 * gross.mean() / to.mean():.0f} d; "
        f"alpha horizon {mk.cfg.alpha.horizon_days} d"
    )

    # -- (a) score persistence ---------------------------------------------------
    sig = mk.sig_pos[mk.sig_pos >= max(AUTOCORR_LAGS)]
    positions = sig[:: max(sample_every, 1)]
    elig = np.isfinite(mk.score)
    refit_pos = set()
    for r in getattr(alpha, "refit_records_", []) or []:
        i = mk.dates.get_indexer([pd.Timestamp(r.refit_date)])[0]
        if i >= 0:
            refit_pos.add(int(i))
    is_refit = np.array([int(p) in refit_pos for p in sig])
    sleeves: dict[str, pd.DataFrame | None] = {"combined": None, "composite": alpha.composite_score_}
    if getattr(alpha, "ml_score_", None) is not None:
        sleeves["ml"] = alpha.ml_score_
    rows = {}
    lag1_combined = None
    for name, df in sleeves.items():
        panel = (
            mk.score if df is None else df.reindex(index=mk.dates, columns=mk.tickers).to_numpy(dtype=float)
        )
        means, _ = rank_autocorr(panel, elig, positions, AUTOCORR_LAGS)
        d: dict[str, float] = {f"lag {lag}": v for lag, v in means.items()}
        # lag 1 on every signal date, split by refit day (cheap: one lag)
        _, lag1 = rank_autocorr(panel, elig, sig, (1,))
        if name == "combined":
            lag1_combined = lag1
        d["lag 1, refit days"] = _nanmean(lag1[is_refit])
        d["lag 1, other days"] = _nanmean(lag1[~is_refit])
        rows[name] = d
    print("\n-- (a) rank autocorrelation of the score across days (names eligible on both dates) --")
    print(_table(pd.DataFrame(rows).T, fmt="{:.3f}"))
    in_sig = np.flatnonzero(mk.sig_pos >= max(AUTOCORR_LAGS))
    not_build = in_sig > 0  # day 0 of the run builds the book
    if (is_refit & not_build).any():
        to_c = ideal.turnover["c_resid"][in_sig]
        r, o = is_refit & not_build, ~is_refit & not_build
        print(
            f"turnover on solves whose signal date is a refit day vs the other days: book "
            f"{100 * to[in_sig][r].mean():.2f} vs {100 * to[in_sig][o].mean():.2f} %/day of NAV; ideal book (c) "
            f"{100 * _nanmean(to_c[r]) / 2:.2f} vs {100 * _nanmean(to_c[o]) / 2:.2f} % of gross"
        )
    print(
        f"refit days in the run: {int(is_refit.sum())} of {len(sig)}.  A rank-weighted book turns over about\n"
        "sqrt(2 (1 - rho_1)) of its gross per day from the score alone; the measured figure is the\n"
        "'to/day %gross' column of table 1."
    )

    # -- (b) gamma ---------------------------------------------------------------
    gam = run.gamma
    dlog = np.full(n, np.nan)
    if np.isfinite(gam).sum() > 10:
        with np.errstate(divide="ignore", invalid="ignore"):
            dlog[1:] = np.diff(np.log(np.where(gam > 0, gam, np.nan)))
        ok = np.isfinite(dlog)
        jump = ok & (np.abs(dlog) > GAMMA_JUMP)
        calm = ok & ~jump
        print("\n-- (b) risk aversion (gamma) from day to day --")
        print(
            f"gamma: mean {np.nanmean(gam):.2f}, median {np.nanmedian(gam):.2f}, p5 {np.nanquantile(gam, 0.05):.2f}, "
            f"p95 {np.nanquantile(gam, 0.95):.2f};  std of daily dlog(gamma) {np.nanstd(dlog):.3f}"
        )
        print(
            f"days with |dlog gamma| > {GAMMA_JUMP:.0%}: {100 * jump.sum() / max(ok.sum(), 1):.0f} % of days; "
            f"turnover on those days {100 * to[jump].mean() if jump.any() else float('nan'):.2f} %/day vs "
            f"{100 * to[calm].mean() if calm.any() else float('nan'):.2f} %/day on the others; "
            f"corr(|dlog gamma|, turnover) = {_corr(np.abs(dlog), to):+.2f}"
        )
    else:
        print("\n-- (b) gamma: no constraints.parquet / risk_aversion in the run directory: skipped --")

    # -- (c)(d) forced trades, drift, re-scaling ---------------------------------
    W_pre = W - X  # pre-trade (drifted) weights
    W_prev = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
    drift = W_pre - W_prev  # what the market did to the book between two closes
    undo = np.where(X * drift < 0, np.minimum(absX, np.abs(drift)), 0.0).sum(axis=1)
    sig_ok = mk.sig_pos >= 0
    el_sig = mk.elig[np.where(sig_ok, mk.sig_pos, 0)] & sig_ok[:, None]
    forced = (absX * ~el_sig).sum(axis=1)
    entered = (el_sig[1:] & ~el_sig[:-1]).sum(axis=1)
    left = (~el_sig[1:] & el_sig[:-1]).sum(axis=1)
    den = (W_pre * W_pre).sum(axis=1)
    lam = np.where(den > 0, (X * W_pre).sum(axis=1) / np.where(den > 0, den, 1.0), 0.0)
    scale_to = np.abs(lam) * np.abs(W_pre).sum(axis=1)
    resid_to = np.abs(X - lam[:, None] * W_pre).sum(axis=1)
    del W_pre
    target_change = np.abs(W - W_prev).sum(axis=1)
    drift_size = np.abs(drift).sum(axis=1)
    del W_prev, drift
    body = slice(1, None)  # day 0 builds the book
    tot = to[body].mean()
    rows = {
        "total turnover": {"%/day": 100 * tot, "% of total": 100.0},
        "post-to-post change sum|w_t - w_{t-1}|": {
            "%/day": 100 * target_change[body].mean(),
            "% of total": 100 * target_change[body].mean() / tot,
        },
        "(c) forced: names not eligible at the signal date": {
            "%/day": 100 * forced[body].mean(),
            "% of total": 100 * forced[body].mean() / tot,
        },
        "(d) drift size sum|w_pre - w_{t-1}|": {
            "%/day": 100 * drift_size[body].mean(),
            "% of total": float("nan"),
        },
        "(d) trades that undo drift": {
            "%/day": 100 * undo[body].mean(),
            "% of total": 100 * undo[body].mean() / tot,
        },
        "(b) proportional re-scaling of the book": {
            "%/day": 100 * scale_to[body].mean(),
            "% of total": 100 * scale_to[body].mean() / tot,
        },
        "trades orthogonal to the book (re-ranking)": {
            "%/day": 100 * resid_to[body].mean(),
            "% of total": 100 * resid_to[body].mean() / tot,
        },
    }
    print(
        "\n-- (c)(d) what the trades are, excluding the build day (rows overlap: an L1 turnover has no "
        "exact additive split) --"
    )
    print(_table(pd.DataFrame(rows).T, formats={"% of total": "{:.0f}"}))
    print(
        f"universe churn: {entered.mean():.1f} names enter and {left.mean():.1f} leave the eligible set per day "
        f"({el_sig.sum(axis=1).mean():.0f} eligible on average)"
    )

    # -- netting over longer update intervals ------------------------------------
    rows = {
        "daily (as run)": {"turnover %/day": 100 * tot, "vs daily": 1.0, "hold d": 2 * gross.mean() / tot}
    }
    for h in HOLD_EVERY:
        nb = (n - 1) // h
        if nb < 2:
            continue
        net = X[1 : 1 + nb * h].reshape(nb, h, -1).sum(axis=1)
        t_h = float(np.abs(net).sum(axis=1).mean() / h)
        rows[f"netted over {h} days"] = {
            "turnover %/day": 100 * t_h,
            "vs daily": t_h / tot,
            "hold d": 2 * gross.mean() / t_h if t_h > 0 else float("nan"),
        }
    print("\n-- turnover if the same sequence of books were only traded every k days (net of round trips) --")
    print(_table(pd.DataFrame(rows).T, formats={"hold d": "{:.0f}"}))
    print(
        "'vs daily' is the share of the daily trading that is NOT reversed within k days.  1/sqrt(k)\n"
        f"(= {1 / np.sqrt(5):.2f} at 5d, {1 / np.sqrt(21):.2f} at 21d) is what pure day-to-day noise in the target gives; a persistent\n"
        "signal trend stays near 1."
    )

    # -- small trades and size terciles ------------------------------------------
    traded = absX > HELD_EPS
    small = traded & (absX < SMALL_TRADE)
    print(
        f"\ntrades smaller than {1e4 * SMALL_TRADE:.0f} bp of NAV: {100 * small.sum() / max(traded.sum(), 1):.0f} % of trades, "
        f"{100 * absX[small].sum() / absX.sum():.1f} % of turnover; names traded per day {traded.sum(axis=1).mean():.0f}, "
        f"median trade {1e4 * np.median(absX[traded]):.1f} bp"
    )
    rows = {}
    absW = np.abs(W)
    for g, nm in [(2, "large"), (1, "mid"), (0, "small"), (-1, "not eligible")]:
        m = att.terc == g
        t_g = (absX * m).sum(axis=1).mean()
        g_g = (absW * m).sum(axis=1).mean()
        rows[nm] = {
            "turnover %/day": 100 * t_g,
            "% of total": 100 * t_g / to.mean(),
            "avg gross": g_g,
            "hold d": 2 * g_g / t_g if t_g > 0 else float("nan"),
        }
    print("\n-- turnover by size tercile --")
    print(_table(pd.DataFrame(rows).T, formats={"% of total": "{:.0f}", "hold d": "{:.0f}"}))

    # -- drivers of the day-to-day variation --------------------------------------
    drivers: dict[str, np.ndarray] = {}
    if np.isfinite(dlog).sum() > 30:
        drivers["|dlog gamma|"] = np.abs(dlog)
    if lag1_combined is not None:
        chg = np.full(n, np.nan)
        chg[np.flatnonzero(mk.sig_pos >= max(AUTOCORR_LAGS))] = 1.0 - lag1_combined
        drivers["score change (1 - rho_1)"] = chg
    drivers["drift size"] = drift_size
    drivers["forced trades"] = forced
    D = np.column_stack(list(drivers.values()))
    ok = np.isfinite(D).all(axis=1) & np.isfinite(to)
    ok[0] = False
    if ok.sum() > 30:
        Z = (D[ok] - D[ok].mean(axis=0)) / np.where(D[ok].std(axis=0) > 0, D[ok].std(axis=0), 1.0)
        y = (to[ok] - to[ok].mean()) / to[ok].std()
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        r2 = 1.0 - float(((y - Z @ beta) ** 2).sum() / (y**2).sum())
        print(
            "\n-- which days trade more: standardised regression of daily turnover on its candidate drivers --"
        )
        print("  ".join(f"{nm}: {b:+.2f}" for nm, b in zip(drivers, beta, strict=True)) + f"   R2 {r2:.2f}")


# ----------------------------------------------------------------------------- 5. costs
def print_costs(mk: Market, run: RunPanels, att: Attribution, cost_every: int) -> None:
    _head("5. COST REALITY CHECK")
    nav = run.nav
    to = np.abs(run.X).sum(axis=1)
    costs = run.costs
    traded_usd = float((to * nav).sum())
    rows = {}
    for kind in ("spread", "impact", "fees", "borrow"):
        if kind not in costs:
            continue
        usd = costs[kind].to_numpy(dtype=float)
        rows[kind] = {
            "bp/day of NAV": 1e4 * float(np.nanmean(usd / nav)),
            "ann % of NAV": 100 * TDY * float(np.nanmean(usd / nav)),
            "bp per side": 1e4 * float(usd.sum()) / traded_usd
            if kind != "borrow" and traded_usd > 0
            else float("nan"),
        }
    trading = sum(costs[k].to_numpy(dtype=float) for k in ("spread", "impact", "fees") if k in costs)
    rows["trading total"] = {
        "bp/day of NAV": 1e4 * float(np.nanmean(trading / nav)),
        "ann % of NAV": 100 * TDY * float(np.nanmean(trading / nav)),
        "bp per side": 1e4 * float(np.sum(trading)) / traded_usd if traded_usd > 0 else float("nan"),
    }
    print(_table(pd.DataFrame(rows).T))
    g_ann, n_ann, gab_ann = (100 * TDY * float(np.nanmean(v)) for v in (run.gross, run.net, run.gab))
    be = 1e4 * float(np.nanmean(run.gab)) / float(to.mean()) if to.mean() > 0 else float("nan")
    print(
        f"\ngross {g_ann:+.2f} %/yr, after borrow {gab_ann:+.2f} %/yr, net {n_ann:+.2f} %/yr; "
        f"turnover {100 * to.mean():.2f} %/day = {TDY * to.mean():.1f}x NAV/yr"
    )
    paid = rows["trading total"]["bp per side"]
    print(
        f"breakeven cost per side = mean(gross after borrow) / mean(turnover) = {be:.2f} bp; paid {paid:.2f} bp"
    )
    if np.isfinite(paid) and paid > 0 and np.nanmean(run.gab) > 0:
        print(
            f"at the paid cost per side the gross P&L covers a turnover of "
            f"{100 * float(np.nanmean(run.gab)) / (paid / 1e4):.2f} %/day (the book trades {100 * to.mean():.2f})"
        )
    else:
        print("gross after borrow is not positive: no turnover level breaks even at any positive cost")

    # -- per-name re-pricing -----------------------------------------------------
    sim, cm = mk.sim, mk.sim._cm
    n, N = run.X.shape
    streak, streak_usd, last_sign = np.zeros(N), np.zeros(N), np.zeros(N)
    usd_cost = np.zeros(3)  # by tercile small / mid / large
    usd_traded = np.zeros(3)
    kinds = np.zeros((3, 3))  # tercile x (spread, impact, fees)
    rec, mine = 0.0, 0.0
    for k in range(n):
        x = run.X[k]
        sgn = np.sign(np.where(np.abs(x) > 0, x, 0.0))
        same = (sgn != 0) & (sgn == last_sign)
        streak = np.where(sgn != 0, np.where(same, streak + 1.0, 1.0), 0.0)
        prior = np.where(same, streak_usd, 0.0)
        streak_usd = np.where(sgn != 0, prior + np.abs(x) * nav[k], 0.0)
        last_sign = sgn
        if k % max(cost_every, 1) or not np.isfinite(nav[k]):
            continue
        kw: dict[str, Any] = {"venue": mk.pipe.venue, "streak": pd.Series(streak, index=mk.tickers)}
        if sim._pass_prior_streak:
            kw["prior_streak_usd"] = pd.Series(prior, index=mk.tickers)
        try:
            est = cm.estimate(pd.Series(x * nav[k], index=mk.tickers), sim._snapshot(mk.md, run.dec[k]), **kw)
        except ValueError:
            continue
        fr = est.frame.reindex(mk.tickers)
        comp = np.nan_to_num(fr[["spread_usd", "impact_usd", "fees_usd"]].to_numpy(dtype=float))
        mine += float(comp.sum())
        rec += float(trading[k])
        for g in range(3):
            m = att.terc[k] == g
            kinds[g] += comp[m].sum(axis=0)
            usd_cost[g] += comp[m].sum()
            usd_traded[g] += float(np.abs(x[m]).sum() * nav[k])
    if mine > 0:
        rows = {}
        for g, nm in [(2, "large"), (1, "mid"), (0, "small")]:
            if usd_traded[g] <= 0:
                continue
            rows[nm] = {
                "bp per side": 1e4 * usd_cost[g] / usd_traded[g],
                "spread": 1e4 * kinds[g, 0] / usd_traded[g],
                "impact": 1e4 * kinds[g, 1] / usd_traded[g],
                "fees": 1e4 * kinds[g, 2] / usd_traded[g],
                "% of traded": 100 * usd_traded[g] / usd_traded.sum(),
                "% of cost": 100 * usd_cost[g] / usd_cost.sum(),
            }
        print(
            f"\n-- cost per side by size tercile (trades re-priced on every {max(cost_every, 1)}th day with the run's cost "
            f"model; re-priced / recorded = {mine / rec if rec > 0 else float('nan'):.3f}) --"
        )
        print(_table(pd.DataFrame(rows).T, formats={"% of traded": "{:.0f}", "% of cost": "{:.0f}"}))


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    ap.add_argument(
        "--run", help="run directory (runs/<run_id>); omitted with --synthetic: a short run is built"
    )
    ap.add_argument(
        "--risk-dates", type=int, default=60, help="solves sampled for the ex-ante variance split"
    )
    ap.add_argument("--cost-every", type=int, default=5, help="re-price trades on every k-th day (table 5)")
    ap.add_argument("--autocorr-every", type=int, default=2, help="score autocorrelation on every k-th day")
    ap.add_argument("--max-days", type=int, default=250, help="length of the run built for --synthetic")
    a = ap.parse_args(argv)

    from qts.cli import ConfigOverrideError

    try:
        cfg = load_config(a)
    except ConfigOverrideError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    md = load_market(a)
    if md is None:
        print("give --store or --synthetic", file=sys.stderr)
        return 2
    if not a.run and not a.synthetic:
        print("give --run runs/<run_id> (only --synthetic builds its own run)", file=sys.stderr)
        return 2
    cfg = clamp_universe(cfg, md)
    print(
        f"loaded: {md.close.shape[1]} ids x {md.close.shape[0]} days ({md.dates[0].date()}..{md.dates[-1].date()})"
    )
    t0 = time.time()
    tmp = None
    if a.run:
        run_dir = Path(a.run)
        if not (run_dir / "weights.parquet").exists():
            print(f"error: {run_dir} is not a run directory (no weights.parquet)", file=sys.stderr)
            return 2
        pipe = ResearchPipeline(cfg, use_ml=a.ml, runs_root=None)
        pipe.fit(md)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="qts_attribute_")
        pipe = ResearchPipeline(cfg, use_ml=a.ml, runs_root=tmp.name)
        res = pipe.run(md, max_days=a.max_days)
        assert res.run_dir is not None
        run_dir = Path(res.run_dir)
    print(f"fit: {time.time() - t0:.0f}s  stage cache: {pipe.cache_info_ or 'off'}")
    try:
        assert pipe.md_u is not None
        run = load_run_panels(run_dir, pipe.md_u.tickers)
        if run.config_hash and run.config_hash != cfg.hash():
            print(
                f"WARNING: config hash {cfg.hash()} differs from the run's {run.config_hash}: the scores and the\n"
                "risk model below may not be the ones the run traded (pass the run's --config / --set flags)"
            )
        s = run.summary.get("backtest", run.summary)
        print(
            f"run {run_dir.name}: {len(run.dec)} days {run.dec[0].date()}..{run.dec[-1].date()}; summary Sharpe gross "
            f"{s.get('sharpe_gross', float('nan')):.2f} / net {s.get('sharpe_net', float('nan')):.2f}, turnover "
            f"{100 * s.get('avg_daily_turnover', float('nan')):.1f} %/day"
        )
        mk = Market(pipe, run)
        # solves sampled for the ex-ante split (their signal dates also keep the ideal book (c))
        usable = np.flatnonzero(mk.sig_pos >= 0)
        step = max(len(usable) // max(a.risk_dates, 1), 1)
        sample = usable[step // 2 :: step][: max(a.risk_dates, 1)]
        t1 = time.time()
        ideal = ideal_portfolios(mk, {int(mk.sig_pos[k]) for k in sample})
        att = book_attribution(mk, run)
        print(f"paper books + attribution: {time.time() - t1:.0f}s")
        print_ideal(mk, run, ideal)
        print_attribution(mk, run, ideal, att)
        print_risk_gap(mk, run, ideal, att, sample)
        print_turnover(mk, run, ideal, att, a.autocorr_every)
        print_costs(mk, run, att, a.cost_every)
        print(f"\ntotal {time.time() - t0:.0f}s")
    finally:
        if tmp is not None:
            tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
