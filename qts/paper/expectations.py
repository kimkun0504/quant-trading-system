"""What the backtest says the paper book should look like: frozen expectation tables.

Built once at ``qts paper init`` from the reference backtest run directory
(``runs/<id>/returns.parquet``, ``summary.json``, ``exante_vol.parquet``,
``ic.parquet``) and frozen under ``<book>/expectations/`` with their SHA-256 in
``book.json``, so the yardstick cannot move after the book starts.

* ``summary.json``: scalar expectations -- daily net mean / sd (simple and log),
  annualised Sharpe and vol, average daily turnover, trading cost per side (bp),
  average gross exposure, the backtest's own realised / ex-ante vol ratio, the mean
  ex-ante vol, the sampled rank IC mean / sd, the label horizon.
* ``bootstrap.parquet``: a moving-block bootstrap (block :data:`BLOCK_DAYS` = 21,
  :data:`N_PATHS` = 2,000 paths, seed 0) of the backtest's daily **net** returns.
  For every horizon ``h`` of :data:`HORIZONS` (1 day to 5 years) and every quantile
  of :data:`QUANTILES` it holds the distribution of three path statistics over the
  first ``h`` days: cumulative log return (``cum_log``), maximum drawdown
  (``max_dd``, a negative number) and annualised Sharpe (``sharpe``).  Blocks keep
  the volatility clustering and autocorrelation of the daily series that an i.i.d.
  draw would lose; :func:`quantile_at` interpolates linearly between horizons.

The analytic cone of the brief (``n mu +- 1.96 sd sqrt(n)`` on daily log returns) is
computed from ``summary.json`` by :mod:`qts.paper.monitor`; the bootstrap is the
version the stop rules use, because the daily returns are fat-tailed and
autocorrelated.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qts.paper.book import PaperError, atomic_write_json, atomic_write_parquet, sha256_bytes

HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 10, 15, 21, 42, 63, 84, 126, 189, 252, 378, 504, 756, 1008, 1260)
QUANTILES: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.975,
    0.99,
    0.995,
)
BLOCK_DAYS = 21
N_PATHS = 2000
SEED = 0
PERIODS_PER_YEAR = 252
STATS = ("cum_log", "max_dd", "sharpe")


@dataclass(frozen=True)
class Expectations:
    summary: dict[str, Any]
    table: pd.DataFrame  #: columns horizon, stat, q, value
    #: the daily net returns the table was built from (in memory only; ``None`` after :func:`load`)
    returns: pd.Series | None = None

    def quantile_at(self, stat: str, q: float, n: float) -> float:
        return quantile_at(self.table, stat, q, n)

    def digest(self) -> str:
        return expectations_digest(self.summary, self.table)


def bootstrap_paths(
    returns: np.ndarray,
    horizon: int = max(HORIZONS),
    block: int = BLOCK_DAYS,
    n_paths: int = N_PATHS,
    seed: int = SEED,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict[str, np.ndarray]:
    """Moving-block bootstrap paths of the daily returns: ``cum_log``, ``max_dd`` (running
    maximum drawdown, <= 0) and ``sharpe`` (annualised, NaN on day 1), each
    ``n_paths x horizon`` (column ``k`` = the statistic over the first ``k + 1`` days)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < max(2 * block, 30):
        raise PaperError(
            f"the reference run has only {len(r)} daily returns; need at least {max(2 * block, 30)}"
        )
    H = int(horizon)
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(H / block)
    starts = rng.integers(0, len(r) - block + 1, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)[:, :H]
    paths = r[idx]
    logp = np.log1p(np.clip(paths, -0.999999, None))
    cum = np.cumsum(logp, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.zeros((n_paths, 1)), cum], axis=1), axis=1)[:, 1:]
    dd = np.minimum.accumulate(np.expm1(cum - peak), axis=1)
    k = np.arange(1, H + 1, dtype=float)
    m = np.cumsum(paths, axis=1) / k
    with np.errstate(divide="ignore", invalid="ignore"):
        var = np.maximum(np.cumsum(paths * paths, axis=1) / k - m * m, 0.0) * k / (k - 1)
        sd = np.sqrt(var)
        sharpe = np.where(sd > 0, m / sd * math.sqrt(periods_per_year), 0.0)
    sharpe[:, 0] = np.nan
    return {"cum_log": cum, "max_dd": dd, "sharpe": sharpe}


def block_bootstrap_table(
    returns: np.ndarray,
    horizons: tuple[int, ...] = HORIZONS,
    quantiles: tuple[float, ...] = QUANTILES,
    block: int = BLOCK_DAYS,
    n_paths: int = N_PATHS,
    seed: int = SEED,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """Moving-block bootstrap quantiles of cumulative log return, maximum drawdown
    and annualised Sharpe at each horizon (module docstring)."""
    P = bootstrap_paths(returns, max(horizons), block, n_paths, seed, periods_per_year)
    rows = []
    qs = np.asarray(quantiles, dtype=float)
    for h in horizons:
        for stat in STATS:
            if stat == "sharpe" and h < 2:
                continue
            qv = np.quantile(P[stat][:, int(h) - 1], qs)
            for q, x in zip(qs, qv, strict=True):
                rows.append({"horizon": int(h), "stat": stat, "q": float(q), "value": float(x)})
    return pd.DataFrame(rows, columns=["horizon", "stat", "q", "value"])


#: paths of the calibration bootstrap (a separate seed from the quantile table)
CALIBRATION_PATHS = 10_000
CALIBRATION_SEED = SEED + 1
#: calibrated rules: name -> (statistic, first day the rule is active, attribute of the
#: target false-alarm probability on StopRules, attribute of the fixed fallback quantile)
CALIBRATED_RULES: dict[str, tuple[str, str, str, str]] = {
    "dd_stop": ("max_dd", "", "dd_stop_false_alarm", "dd_stop_quantile"),
    "dd_watch": ("max_dd", "", "dd_watch_false_alarm", "dd_watch_quantile"),
    "sharpe_stop": ("sharpe", "sharpe_stop_min_days", "sharpe_stop_false_alarm", "sharpe_stop_quantile"),
    "sharpe_watch": ("sharpe", "sharpe_watch_min_days", "sharpe_watch_false_alarm", "sharpe_watch_quantile"),
}


def _chunked_paths(
    returns: np.ndarray, n_paths: int, seed: int, horizon: int, keep: np.ndarray
) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {"max_dd": [], "sharpe": []}
    rng = np.random.default_rng(seed)
    for k in range(0, n_paths, 2000):
        n = min(2000, n_paths - k)
        P = bootstrap_paths(returns, horizon, n_paths=n, seed=int(rng.integers(0, 2**31 - 1)))
        for st in out:
            out[st].append(P[st][:, keep])
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def calibrate(
    returns: np.ndarray, rules: Any, days: int = PERIODS_PER_YEAR
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Quantile level of each drawdown / Sharpe rule calibrated so that a book whose daily
    returns behave like the backtest's (block bootstrap, :data:`CALIBRATION_PATHS` paths)
    triggers it at least once within ``days`` days, checked every day, with the target
    probability of the rule (``StopRules.*_false_alarm``) -- a simultaneous band instead of
    a per-day quantile.  Returns ``(info, rows)``: the calibrated levels with the achieved
    rates and the year-1 false-alarm rates of the fixed per-horizon quantiles for
    comparison, and the boundary rows (stat, horizon, q = the calibrated level, value) to
    append to the quantile table."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    H = int(max(HORIZONS))
    keep = np.unique(np.concatenate([np.arange(days), np.asarray(HORIZONS) - 1]))
    P = _chunked_paths(r, CALIBRATION_PATHS, CALIBRATION_SEED, H, keep)
    col = {int(d): i for i, d in enumerate(keep)}  # day index (0-based) -> column
    info: dict[str, Any] = {
        "paths": CALIBRATION_PATHS,
        "days": days,
        "levels": {},
        "achieved": {},
        "fixed_quantile_rates": {},
    }
    rows = []
    for name, (stat, min_attr, target_attr, fixed_attr) in CALIBRATED_RULES.items():
        first = int(getattr(rules, min_attr)) if min_attr else (2 if stat == "sharpe" else 1)
        active = [col[d] for d in range(first - 1, days)]
        V = P[stat][:, active]
        srt = np.sort(V, axis=0)
        # empirical CDF of each path's value at each day, then the path's minimum over the days
        cdf = np.empty_like(V)
        for k in range(V.shape[1]):
            cdf[:, k] = np.searchsorted(srt[:, k], V[:, k], side="right") / V.shape[0]
        mins = cdf.min(axis=1)
        target = float(getattr(rules, target_attr))
        level = float(np.quantile(mins, target))
        info["levels"][name] = level
        bound = np.quantile(V, level, axis=0)
        info["achieved"][name] = float((bound > V).any(axis=1).mean())
        fixed = float(getattr(rules, fixed_attr))
        info["fixed_quantile_rates"][name] = float((np.quantile(V, fixed, axis=0) > V).any(axis=1).mean())
        for h in HORIZONS:
            if stat == "sharpe" and h < 2:
                continue
            rows.append(
                {
                    "horizon": int(h),
                    "stat": stat,
                    "q": level,
                    "value": float(np.quantile(P[stat][:, col[h - 1]], level)),
                }
            )
    stop = np.zeros(CALIBRATION_PATHS, dtype=bool)
    for name in ("dd_stop", "sharpe_stop"):
        stat, min_attr, _, _ = CALIBRATED_RULES[name]
        first = int(getattr(rules, min_attr)) if min_attr else 1
        active = [col[d] for d in range(first - 1, days)]
        V = P[stat][:, active]
        stop |= (np.quantile(V, info["levels"][name], axis=0) > V).any(axis=1)
    info["achieved"]["any_return_stop"] = float(stop.mean())
    return info, pd.DataFrame(rows, columns=["horizon", "stat", "q", "value"])


def quantile_at(table: pd.DataFrame, stat: str, q: float, n: float) -> float:
    """Quantile ``q`` of ``stat`` at horizon ``n`` days, linear between the table's
    horizons (clamped at both ends)."""
    sub = table[(table["stat"] == stat) & np.isclose(table["q"].to_numpy(dtype=float), float(q))]
    if sub.empty:
        raise PaperError(f"expectations table has no {stat} quantile {q}")
    sub = sub.sort_values("horizon")
    return float(
        np.interp(float(n), sub["horizon"].to_numpy(dtype=float), sub["value"].to_numpy(dtype=float))
    )


def summary_from_returns(net: pd.Series, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    r = pd.Series(net, dtype=float).dropna()
    lg = np.log1p(r.clip(lower=-0.999999))
    sd = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
    out: dict[str, Any] = {
        "n_days": len(r),
        "start": str(pd.Timestamp(r.index[0]).date()) if len(r) else None,
        "end": str(pd.Timestamp(r.index[-1]).date()) if len(r) else None,
        "mean_daily": float(r.mean()),
        "sd_daily": sd,
        "mean_log": float(lg.mean()),
        "sd_log": float(lg.std(ddof=1)) if len(lg) > 1 else float("nan"),
        "sharpe_ann": float(r.mean() / sd * math.sqrt(PERIODS_PER_YEAR)) if sd and sd > 0 else float("nan"),
        "ann_vol": sd * math.sqrt(PERIODS_PER_YEAR) if np.isfinite(sd) else float("nan"),
        "bootstrap": {"block_days": BLOCK_DAYS, "n_paths": N_PATHS, "seed": SEED, "horizons": list(HORIZONS)},
    }
    out.update(dict(extra or {}))
    return out


def _f(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def from_run_dir(run_dir: str | Path) -> Expectations:
    """Expectations from a finished pipeline run directory (``qts run`` output)."""
    p = Path(run_dir)
    ret_f = p / "returns.parquet"
    summ_f = p / "summary.json"
    if not ret_f.exists() or not summ_f.exists():
        raise PaperError(
            f"reference run {p} has no returns.parquet / summary.json (is it a finished `qts run` directory?)"
        )
    returns = pd.read_parquet(ret_f)
    if "net" not in returns:
        raise PaperError(f"{ret_f} has no 'net' column")
    summ = json.loads(summ_f.read_text())
    bt = summ.get("backtest", {}) or {}
    extra: dict[str, Any] = {
        "source": "run_dir",
        "run_id": p.name,
        "run_config_hash": summ.get("config_hash"),
        "run_data_hash": summ.get("data_hash"),
        "use_ml": summ.get("use_ml"),
        "avg_daily_turnover": _f(bt.get("avg_daily_turnover")),
        "avg_cost_bps_per_side": _f(bt.get("avg_cost_bps_per_side")),
        "avg_gross_exposure": _f(bt.get("avg_gross_exposure")),
        "realized_vs_exante_vol_ratio": _f(bt.get("realized_vs_exante_vol_ratio")),
        "sharpe_net_reported": _f(bt.get("sharpe_net")),
        "max_drawdown_net": _f(bt.get("max_drawdown_net")),
    }
    ev = p / "exante_vol.parquet"
    if ev.exists():
        v = pd.read_parquet(ev).iloc[:, 0].astype(float)
        v = v[np.isfinite(v.to_numpy()) & (v.to_numpy() > 0)]
        extra["mean_exante_vol"] = float(v.mean()) if len(v) else float("nan")
    ic = p / "ic.parquet"
    if ic.exists():
        s = pd.read_parquet(ic).iloc[:, 0].astype(float).dropna()
        extra["ic_mean"] = float(s.mean()) if len(s) else float("nan")
        extra["ic_sd"] = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
        extra["ic_n"] = len(s)
    ic_block = summ.get("ic", {}) or {}
    extra["ic_t_nw"] = _f(ic_block.get("t_stat_nw"))
    cfg_f = p / "config.yaml"
    if cfg_f.exists():
        import yaml

        cfg = yaml.safe_load(cfg_f.read_text()) or {}
        extra["horizon_days"] = int(((cfg.get("alpha") or {}).get("horizon_days")) or 0) or None
    net = returns["net"].astype(float)
    return Expectations(
        summary=summary_from_returns(net, extra), table=block_bootstrap_table(net.to_numpy()), returns=net
    )


def from_backtest(bt: Any, horizon_days: int | None = None, ic: pd.Series | None = None) -> Expectations:
    """Expectations from an in-memory :class:`~qts.backtest.BacktestResult` (tests,
    synthetic books)."""
    s = bt.summary()
    net = bt.returns["net"].astype(float)
    exv = np.sqrt(bt.exante_var.astype(float) * PERIODS_PER_YEAR)
    exv = exv[np.isfinite(exv.to_numpy()) & (exv.to_numpy() > 0)]
    extra: dict[str, Any] = {
        "source": "backtest",
        "run_id": None,
        "avg_daily_turnover": _f(s.get("avg_daily_turnover")),
        "avg_cost_bps_per_side": _f(s.get("avg_cost_bps_per_side")),
        "avg_gross_exposure": _f(s.get("avg_gross_exposure")),
        "realized_vs_exante_vol_ratio": _f(s.get("realized_vs_exante_vol_ratio")),
        "mean_exante_vol": float(exv.mean()) if len(exv) else float("nan"),
        "horizon_days": horizon_days,
    }
    if ic is not None:
        ic = ic.dropna()
        extra.update(ic_mean=float(ic.mean()), ic_sd=float(ic.std(ddof=1)), ic_n=len(ic))
    return Expectations(
        summary=summary_from_returns(net, extra), table=block_bootstrap_table(net.to_numpy()), returns=net
    )


def expectations_digest(summary: Mapping[str, Any], table: pd.DataFrame) -> str:
    from qts.paper.book import json_safe

    payload = (
        json.dumps(json_safe(summary), sort_keys=True)
        + "\n"
        + table.to_csv(index=False, float_format="%.12g")
    )
    return sha256_bytes(payload.encode())


def save(exp: Expectations, directory: str | Path) -> str:
    d = Path(directory)
    atomic_write_json(d / "summary.json", exp.summary)
    atomic_write_parquet(d / "bootstrap.parquet", exp.table)
    return exp.digest()


def load(directory: str | Path, expect_digest: str | None = None) -> Expectations:
    d = Path(directory)
    if not (d / "summary.json").exists():
        raise PaperError(f"no expectations under {d}")
    exp = Expectations(
        summary=json.loads((d / "summary.json").read_text()), table=pd.read_parquet(d / "bootstrap.parquet")
    )
    if expect_digest and exp.digest() != expect_digest:
        raise PaperError(
            f"expectation tables under {d} do not match the digest frozen in book.json "
            "(they were edited after init); the stop rules cannot be evaluated against moved goalposts"
        )
    return exp
