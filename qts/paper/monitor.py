"""Live-vs-backtest monitoring and the pre-registered stop rules.

The rules and their numbers are frozen in ``book.json`` at init
(:class:`StopRules`) and pre-registered in
``docs/notes/paper_trading_rules_2026-11-13.md``; the yardstick is the frozen
expectation tables of :mod:`qts.paper.expectations`.  :func:`evaluate` is a pure
function of the journal records, the expectations, the rules and the
acknowledgements, so the status of any past day can be recomputed exactly.

Levels: ``OK`` < ``WATCH`` < ``STOP``.  A STOP freezes trading (no new orders;
positions stay and are marked to market) until ``qts paper ack --note ...``.
An acknowledgement covers the STOP rules active on the acknowledged day: each of
them is reported as WATCH (``acknowledged``) until it clears on some later day;
a rule that clears and fires again, or a different rule, stops the book again.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from qts.paper.expectations import Expectations

LEVELS = ("OK", "WATCH", "STOP")
TRADING_DAYS = 252


@dataclass(frozen=True)
class StopRules:
    """Pre-registered monitoring thresholds (rationale per field in the rules note)."""

    # drawdown and Sharpe: the live statistic vs a boundary of the backtest's bootstrap
    # distribution at the same horizon (days since the first fill).  The boundary's quantile
    # level is *calibrated* at init (qts.paper.expectations.calibrate) so that a book whose
    # returns behave like the backtest's crosses it at least once in its first year, checked
    # daily, with probability *_false_alarm; the fixed *_quantile is the fallback when the
    # expectations carry no calibration
    dd_stop_false_alarm: float = 0.025
    dd_watch_false_alarm: float = 0.10
    sharpe_stop_false_alarm: float = 0.025
    sharpe_watch_false_alarm: float = 0.10
    dd_stop_quantile: float = 0.01  #: fallback: STOP below the 1st percentile at that horizon
    dd_watch_quantile: float = 0.05
    # Sharpe: live annualised net Sharpe vs the bootstrap Sharpe at the same horizon
    sharpe_stop_min_days: int = 126
    sharpe_stop_quantile: float = 0.025
    sharpe_watch_min_days: int = 63
    sharpe_watch_quantile: float = 0.10
    # cumulative return vs the bootstrap cone (WATCH only; the analytic cone is reported)
    cum_watch_quantile: float = 0.025
    cum_watch_min_days: int = 21
    # realised vs ex-ante vol (21-day rolling), relative to the backtest's own ratio
    vol_window: int = 21
    vol_ratio_stop: float = 2.0
    vol_ratio_stop_days: int = 20
    vol_ratio_watch: float = 1.5
    vol_ratio_watch_days: int = 10
    # trading activity vs the backtest (WATCH only: paper costs come from the cost model)
    activity_window: int = 21
    build_days: int = 10  #: first execution days excluded (the initial build)
    turnover_watch_mult: float = 2.0
    cost_watch_mult: float = 2.0
    gross_watch_frac: float = 0.5  #: invested gross below this share of the backtest's mean
    gross_watch_days: int = 5
    gross_stop_mult: float = 1.10  #: gross above 1.1 x the config's gross limit -> STOP
    # optimizer
    failed_solve_stop_consecutive: int = 3
    # realised rank IC of the traded score once labels mature (WATCH only)
    ic_min_matured: int = 20
    ic_watch_z: float = -2.0
    # data quality: share of gross in held names missing from the store or with a stale price
    data_watch_gross: float = 0.02
    data_stop_gross: float = 0.10
    data_stop_days: int = 3

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> StopRules:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in dict(d or {}).items() if k in known})


@dataclass(frozen=True)
class Check:
    rule: str
    level: str
    value: float | None
    threshold: float | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _max_level(levels: Sequence[str]) -> str:
    return max(levels, key=LEVELS.index) if levels else "OK"


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# --------------------------------------------------------------------------- series
def live_frame(records: Sequence[Mapping[str, Any]], first_execution: str) -> pd.DataFrame:
    """One row per journal day from the first execution date on (the series the rules read)."""
    rows = []
    for r in records:
        if str(r["date"]) < first_execution:
            continue
        pnl = r.get("pnl", {}) or {}
        dq = r.get("data_quality", {}) or {}
        solve = r.get("solve") or {}
        rows.append(
            {
                "date": pd.Timestamp(r["date"]),
                "nav": float(r["nav"]),
                "ret": float(r.get("ret", 0.0) or 0.0),
                "gross": _finite(r.get("gross")),
                "net": _finite(r.get("net")),
                "exante_vol": _finite(r.get("exante_vol")),
                "turnover": _finite(r.get("turnover")) or 0.0,
                "traded_usd": float(r.get("traded_usd", 0.0) or 0.0),
                "trading_cost": float(pnl.get("trading_cost", 0.0) or 0.0),
                "solve_ok": None if not solve else bool(solve.get("ok", True)),
                "exante_alert": bool(solve.get("exante_vol_alert", False)) if solve else False,
                "data_gross": float(dq.get("missing_gross", 0.0) or 0.0)
                + float(dq.get("stale_gross", 0.0) or 0.0),
                "n_unfilled": len((r.get("events", {}) or {}).get("unfilled", []) or []),
            }
        )
    cols = [
        "date",
        "nav",
        "ret",
        "gross",
        "net",
        "exante_vol",
        "turnover",
        "traded_usd",
        "trading_cost",
        "solve_ok",
        "exante_alert",
        "data_gross",
        "n_unfilled",
    ]
    return pd.DataFrame(rows, columns=cols).set_index("date") if rows else pd.DataFrame(columns=cols[1:])


def _run_length(flags: Sequence[bool]) -> int:
    n = 0
    for f in reversed(list(flags)):
        if not f:
            break
        n += 1
    return n


def _latest_ic(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    for r in reversed(records):
        ic = (r.get("signal") or {}).get("realised_ic")
        if ic:
            return dict(ic)
    return None


# --------------------------------------------------------------------------- checks
def compute_checks(
    records: Sequence[Mapping[str, Any]],
    exp: Expectations | None,
    rules: StopRules,
    first_execution: str,
    gross_limit: float,
) -> tuple[list[Check], dict[str, Any]]:
    """Every rule's check on the last record, plus the metrics behind them."""
    df = live_frame(records, first_execution)
    checks: list[Check] = []
    m: dict[str, Any] = {"n_days": len(df)}
    if df.empty:
        return [Check("started", "OK", None, None, "no execution day yet")], m
    ret = df["ret"].astype(float)
    n = len(ret)
    nav = df["nav"].astype(float)
    nav0 = float(nav.iloc[0] / (1.0 + ret.iloc[0])) if n else float("nan")
    cum_log = float(np.log1p(ret.clip(lower=-0.999999)).sum())
    peak = np.maximum.accumulate(np.concatenate([[nav0], nav.to_numpy()]))
    dd_series = np.concatenate([[nav0], nav.to_numpy()]) / peak - 1.0
    max_dd = float(dd_series.min())
    cur_dd = float(dd_series[-1])
    sd = float(ret.std(ddof=1)) if n > 1 else float("nan")
    sharpe = float(ret.mean() / sd * math.sqrt(TRADING_DAYS)) if n > 1 and sd > 0 else float("nan")
    m.update(
        cum_return=float(math.expm1(cum_log)),
        cum_log=cum_log,
        max_drawdown=max_dd,
        drawdown=cur_dd,
        sharpe=sharpe,
        ann_vol=sd * math.sqrt(TRADING_DAYS) if n > 1 else float("nan"),
    )
    s = exp.summary if exp is not None else {}
    # ---- analytic cone (reported)
    mu, sg = _finite(s.get("mean_log")), _finite(s.get("sd_log"))
    if mu is not None and sg is not None:
        m["cone"] = {
            "expected": n * mu,
            "lower": n * mu - 1.96 * sg * math.sqrt(n),
            "upper": n * mu + 1.96 * sg * math.sqrt(n),
        }
    # ---- drawdown
    if exp is not None:
        levels = (s.get("calibration") or {}).get("levels") or {}
        lv = {
            "dd_stop": float(levels.get("dd_stop", rules.dd_stop_quantile)),
            "dd_watch": float(levels.get("dd_watch", rules.dd_watch_quantile)),
            "sharpe_stop": float(levels.get("sharpe_stop", rules.sharpe_stop_quantile)),
            "sharpe_watch": float(levels.get("sharpe_watch", rules.sharpe_watch_quantile)),
        }
        m["boundary_levels"] = lv
        q_stop = exp.quantile_at("max_dd", lv["dd_stop"], n)
        q_watch = exp.quantile_at("max_dd", lv["dd_watch"], n)
        lvl = "STOP" if max_dd < q_stop else "WATCH" if max_dd < q_watch else "OK"
        checks.append(
            Check(
                "drawdown",
                lvl,
                max_dd,
                q_stop if lvl == "STOP" else q_watch,
                f"max drawdown {100 * max_dd:.2f}% after {n} days; WATCH / STOP boundaries "
                f"{100 * q_watch:.2f}% / {100 * q_stop:.2f}% (bootstrap quantiles "
                f"{100 * lv['dd_watch']:.3g}% / {100 * lv['dd_stop']:.3g}%)",
            )
        )
        # ---- Sharpe
        if n >= rules.sharpe_watch_min_days and np.isfinite(sharpe):
            qs = exp.quantile_at("sharpe", lv["sharpe_stop"], n)
            qw = exp.quantile_at("sharpe", lv["sharpe_watch"], n)
            lvl = "OK"
            if n >= rules.sharpe_stop_min_days and sharpe < qs:
                lvl = "STOP"
            elif sharpe < qw:
                lvl = "WATCH"
            checks.append(
                Check(
                    "sharpe",
                    lvl,
                    sharpe,
                    qs if lvl == "STOP" else qw,
                    f"live Sharpe {sharpe:.2f} after {n} days; WATCH / STOP boundaries {qw:.2f} / {qs:.2f} "
                    f"(bootstrap quantiles {100 * lv['sharpe_watch']:.3g}% / {100 * lv['sharpe_stop']:.3g}%; "
                    f"STOP only after {rules.sharpe_stop_min_days} days)",
                )
            )
        else:
            checks.append(
                Check(
                    "sharpe",
                    "OK",
                    sharpe,
                    None,
                    f"Sharpe not judged before {rules.sharpe_watch_min_days} days",
                )
            )
        # ---- cumulative return (bootstrap cone)
        if n >= rules.cum_watch_min_days:
            lo = exp.quantile_at("cum_log", rules.cum_watch_quantile, n)
            hi = exp.quantile_at("cum_log", 1.0 - rules.cum_watch_quantile, n)
            m["bootstrap_cone"] = {"lower": lo, "upper": hi, "median": exp.quantile_at("cum_log", 0.5, n)}
            lvl = "WATCH" if cum_log < lo else "OK"
            checks.append(
                Check(
                    "cumulative_return",
                    lvl,
                    cum_log,
                    lo,
                    f"cumulative log return {cum_log:+.4f} vs bootstrap "
                    f"{100 * rules.cum_watch_quantile:g}% quantile {lo:+.4f}",
                )
            )
    # ---- realised vs ex-ante vol
    w = rules.vol_window
    ref_ratio = _finite(s.get("realized_vs_exante_vol_ratio")) or 1.0
    base = max(1.0, ref_ratio)
    rv = ret.rolling(w, min_periods=w).std(ddof=1) * math.sqrt(TRADING_DAYS)
    ev = df["exante_vol"].astype(float).shift(1).rolling(w, min_periods=w).mean()  # vol of the book held
    ratio = (rv / ev).replace([np.inf, -np.inf], np.nan)
    rel = ratio / base
    m["vol_ratio"] = _finite(ratio.iloc[-1]) if len(ratio) else None
    m["vol_ratio_backtest"] = ref_ratio
    stop_run = _run_length((rel > rules.vol_ratio_stop).fillna(False).tolist())
    watch_run = _run_length((rel > rules.vol_ratio_watch).fillna(False).tolist())
    lvl = (
        "STOP"
        if stop_run >= rules.vol_ratio_stop_days
        else "WATCH"
        if watch_run >= rules.vol_ratio_watch_days
        else "OK"
    )
    checks.append(
        Check(
            "vol_ratio",
            lvl,
            m["vol_ratio"],
            rules.vol_ratio_stop * base if lvl == "STOP" else rules.vol_ratio_watch * base,
            f"{w}-day realised / ex-ante vol {m['vol_ratio'] if m['vol_ratio'] is not None else float('nan'):.2f} "
            f"(backtest ratio {ref_ratio:.2f}); above {rules.vol_ratio_stop:g}x the backtest's for {stop_run} "
            f"day(s) (STOP at {rules.vol_ratio_stop_days}), above {rules.vol_ratio_watch:g}x for {watch_run} "
            f"(WATCH at {rules.vol_ratio_watch_days})",
        )
    )
    # ---- activity (after the build)
    act = df.iloc[rules.build_days :].tail(rules.activity_window)
    if len(act) >= rules.activity_window:
        to = float(act["turnover"].mean())
        traded = float(act["traded_usd"].sum())
        cps = float(act["trading_cost"].sum() / traded * 1e4) if traded > 0 else float("nan")
        m.update(turnover=to, cost_bps_per_side=cps)
        ref_to = _finite(s.get("avg_daily_turnover"))
        ref_c = _finite(s.get("avg_cost_bps_per_side"))
        if ref_to:
            lvl = "WATCH" if to > rules.turnover_watch_mult * ref_to else "OK"
            checks.append(
                Check(
                    "turnover",
                    lvl,
                    to,
                    rules.turnover_watch_mult * ref_to,
                    f"{rules.activity_window}-day mean turnover {100 * to:.2f}%/day vs backtest {100 * ref_to:.2f}%",
                )
            )
        if ref_c and np.isfinite(cps):
            lvl = "WATCH" if cps > rules.cost_watch_mult * ref_c else "OK"
            checks.append(
                Check(
                    "cost_per_side",
                    lvl,
                    cps,
                    rules.cost_watch_mult * ref_c,
                    f"{rules.activity_window}-day trading cost {cps:.2f} bp/side vs backtest {ref_c:.2f} bp",
                )
            )
    # ---- invested gross
    g = df["gross"].astype(float)
    m["gross"] = _finite(g.iloc[-1])
    ref_g = _finite(s.get("avg_gross_exposure"))
    after_build = g.iloc[rules.build_days :]
    lvl = "OK"
    msg = f"gross {m['gross'] if m['gross'] is not None else float('nan'):.2f}"
    if m["gross"] is not None and m["gross"] > rules.gross_stop_mult * gross_limit:
        lvl, msg = "STOP", msg + f" above {rules.gross_stop_mult:g} x the gross limit {gross_limit:g}"
    elif ref_g and len(after_build):
        low = _run_length((after_build < rules.gross_watch_frac * ref_g).tolist())
        if low >= rules.gross_watch_days:
            lvl = "WATCH"
        msg += f" (backtest mean {ref_g:.2f}; below {rules.gross_watch_frac:g}x for {low} day(s))"
    checks.append(Check("gross", lvl, m["gross"], gross_limit * rules.gross_stop_mult, msg))
    # ---- optimizer
    solves = [x for x in df["solve_ok"].tolist() if x is not None]
    fails = _run_length([not x for x in solves])
    any_fail = bool(solves) and not solves[-1]
    lvl = "STOP" if fails >= rules.failed_solve_stop_consecutive else "WATCH" if any_fail else "OK"
    if bool(df["exante_alert"].iloc[-1]):
        lvl = _max_level([lvl, "WATCH"])
    checks.append(
        Check(
            "optimizer",
            lvl,
            float(fails),
            float(rules.failed_solve_stop_consecutive),
            f"{fails} consecutive failed rebalance solve(s)"
            + (
                "; risk-model ex-ante vol alert on the last solve"
                if bool(df["exante_alert"].iloc[-1])
                else ""
            ),
        )
    )
    # ---- data quality
    dg = df["data_gross"].astype(float)
    bad_run = _run_length((dg > rules.data_stop_gross).tolist())
    lvl = (
        "STOP"
        if bad_run >= rules.data_stop_days
        else "WATCH"
        if float(dg.iloc[-1]) > rules.data_watch_gross or int(df["n_unfilled"].iloc[-1]) > 0
        else "OK"
    )
    checks.append(
        Check(
            "data_quality",
            lvl,
            float(dg.iloc[-1]),
            rules.data_stop_gross if lvl == "STOP" else rules.data_watch_gross,
            f"{100 * float(dg.iloc[-1]):.2f}% of NAV in held names missing from the store or with a stale "
            f"price; {int(df['n_unfilled'].iloc[-1])} unfilled order(s) today",
        )
    )
    # ---- realised IC (lagged)
    ic = _latest_ic(records)
    if ic is not None:
        m["realised_ic"] = ic
        n_ic = int(ic.get("n", 0) or 0)
        ref_mean, ref_sd = _finite(s.get("ic_mean")), _finite(s.get("ic_sd"))
        h = int(s.get("horizon_days") or 60)
        lvl, z = "OK", None
        if n_ic >= rules.ic_min_matured and ref_mean is not None and ref_sd:
            n_eff = max(1.0, n_ic / max(h, 1))
            z = ((_finite(ic.get("mean")) or 0.0) - ref_mean) / (ref_sd / math.sqrt(n_eff))
            lvl = "WATCH" if z < rules.ic_watch_z else "OK"
        checks.append(
            Check(
                "realised_ic",
                lvl,
                _finite(ic.get("mean")),
                rules.ic_watch_z,
                f"realised rank IC {_finite(ic.get('mean')) or float('nan'):+.4f} over {n_ic} matured date(s)"
                + (f"; z vs backtest {z:+.2f} (overlap-adjusted)" if z is not None else " (not judged yet)"),
            )
        )
    return checks, m


def evaluate(
    records: Sequence[Mapping[str, Any]],
    exp: Expectations | None,
    rules: StopRules,
    first_execution: str,
    gross_limit: float,
    acks: Sequence[Mapping[str, Any]] = (),
    prior_suppressed: Sequence[str] = (),
) -> dict[str, Any]:
    """Status of the last record: level, reasons, checks and metrics.

    ``prior_suppressed``: rules acknowledged earlier that have not cleared since
    (carried in each day's ``status.suppressed``); ``acks``: acknowledgements dated
    on or before the last record's date that are not yet applied."""
    checks, metrics = compute_checks(records, exp, rules, first_execution, gross_limit)
    date = str(records[-1]["date"]) if records else None
    suppressed = set(prior_suppressed)
    stop_rules = {c.rule for c in checks if c.level == "STOP"}
    # a suppressed rule that is no longer at STOP has cleared
    suppressed &= stop_rules
    acked_now = [a for a in acks if date is not None and str(a.get("date")) == date]
    for a in acked_now:
        suppressed |= set(a.get("rules", []) or []) & stop_rules
    out_checks = []
    for c in checks:
        if c.level == "STOP" and c.rule in suppressed:
            out_checks.append(
                dataclasses.replace(c, level="WATCH", message=c.message + " [STOP acknowledged]")
            )
        else:
            out_checks.append(c)
    level = _max_level([c.level for c in out_checks])
    reasons = [f"{c.rule}: {c.message}" for c in out_checks if c.level != "OK"]
    return {
        "date": date,
        "level": level,
        "reasons": reasons,
        "stop_rules": sorted(c.rule for c in out_checks if c.level == "STOP"),
        "suppressed": sorted(suppressed),
        "checks": [c.to_dict() for c in out_checks],
        "metrics": metrics,
    }
