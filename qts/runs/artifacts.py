"""Extra per-run artifacts for the workbench API (``api/``).

:func:`qts.pipeline.save_run` writes the core run directory (manifest, config,
summary, NAV / weights / trades / costs / returns / IC, hurdles).  The UI needs a
few more things that are only available while the fitted pipeline is still in
memory; :func:`write_extra_artifacts` adds them next to the core files::

    extras.json          run facts (data kind, universe size, market days, seed,
                         execution venue, label), DSR inputs (skew, kurtosis, n_obs),
                         mean IC by calendar year, factor roles
    exante_vol.parquet   column "exante_vol": the risk model's annualised ex-ante
                         volatility of the post-trade book, by decision date
    exposures.parquet    decision date x factor: net book exposure (sum_i w_i X_ik)
                         to the market factor and every style factor
    holdings.parquet     long table at month-end decision dates (most recent
                         ``HOLDINGS_MAX_DATES``): date, ticker, weight, alpha
                         (Grinold expected return over the alpha horizon), specific
                         vol (annualised), industry

Everything here is best effort: a failure is logged and never fails the run (the
API degrades gracefully when a file is missing, as it does for runs written before
this module existed).  Readers: :func:`read_extras` and ``qts_api.services.registry``.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from qts.core.roles import ALPHA_THEME_STYLES, RISK_ONLY_STYLES

if TYPE_CHECKING:  # pragma: no cover
    from qts.pipeline import PipelineResult, ResearchPipeline

log = logging.getLogger(__name__)

EXTRAS_FILE = "extras.json"
EXTRAS_VERSION = 1
HOLDINGS_MAX_DATES = 60
MIN_ABS_WEIGHT = 1e-8

#: Plain-language description of each simulator pricing venue (``ResearchPipeline.venue``).
VENUE_DESCRIPTIONS: dict[str, str] = {
    "routed": "closing auction (LOC/MOC) up to the auction participation cap, "
    "remainder in the continuous session",
    "auction": "closing auction (LOC/MOC)",
    "continuous": "continuous session (arrival-price algorithm)",
}


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


def _guard(name: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:  # best effort by design: never fail a finished run
        log.warning("extra artifact %r not written", name, exc_info=True)
        return None


def _role(factor: str) -> str:
    if factor == "market":
        return "market"
    if factor in ALPHA_THEME_STYLES:
        return "alpha_theme"
    return "risk_only"


def run_facts(result: PipelineResult, pipeline: ResearchPipeline) -> dict[str, Any]:
    """Data kind, universe size, market length, seed and execution venue of a run."""
    md = getattr(pipeline, "_md_input", None) or pipeline.md_u
    kind = None
    stocks = days = None
    start = None
    if md is not None:
        kind = "synthetic" if any(str(k).startswith("true_") for k in md.extra) else "store"
        stocks, days = int(md.close.shape[1]), int(md.close.shape[0])
        start = str(pd.Timestamp(md.dates[0]).date()) if len(md.dates) else None
    venue = str(result.backtest.diagnostics.get("venue", getattr(pipeline, "venue", "routed")))
    return {
        "kind": kind,
        "stocks": stocks,
        "days": days,
        "market_start": start,
        "seed": int(pipeline.cfg.seed),
        "venue": venue,
        "execution_venue": VENUE_DESCRIPTIONS.get(venue, venue),
        "rebalance_every": int(getattr(pipeline, "rebalance_every", 1)),
        "nav0": float(getattr(pipeline, "nav0", float("nan"))),
    }


def dsr_moments(result: PipelineResult) -> dict[str, Any]:
    """Skew / (non-excess) kurtosis / n_obs of the net returns, as the pipeline's DSR used them."""
    net = result.backtest.returns["net"].dropna().astype(float)
    excess = _finite(net.kurt())
    return {
        "skew": _finite(net.skew()),
        "kurtosis": None if excess is None else excess + 3.0,
        "n_obs": len(net),
    }


def ic_by_year(result: PipelineResult) -> list[dict[str, Any]]:
    by = result.ic_report.by_year
    if by is None or not len(by):
        return []
    return [
        {"year": int(str(y)), "mean_ic": _finite(r["mean"]), "n": int(r["count"])} for y, r in by.iterrows()
    ]


def exante_vol(result: PipelineResult, periods_per_year: int) -> pd.DataFrame:
    var = result.backtest.exante_var
    vol = np.sqrt(var.clip(lower=0.0).astype(float).to_numpy() * periods_per_year)
    return pd.DataFrame({"exante_vol": vol}, index=pd.DatetimeIndex(var.index, name="date"))


def factor_exposures(result: PipelineResult, pipeline: ResearchPipeline) -> pd.DataFrame:
    """Net exposure of the post-trade book to the market and each style factor, by decision date."""
    risk = pipeline.risk
    if risk is None:
        return pd.DataFrame()
    styles = list(pipeline.cfg.risk.styles)
    W = result.backtest.weights
    rows: dict[pd.Timestamp, pd.Series] = {}
    for d in W.index:
        try:
            X = risk.exposures_as_of(d)
        except KeyError:
            continue
        cols = [c for c in ["market", *styles] if c in X.columns]
        w = W.loc[d].reindex(X.index).fillna(0.0).to_numpy(dtype=float)
        Xm = X[cols].to_numpy(dtype=float)
        rows[pd.Timestamp(d)] = pd.Series(np.nan_to_num(Xm).T @ w, index=cols)
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index = pd.DatetimeIndex(out.index, name="date")
    return out


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if not len(idx):
        return idx
    periods = idx.to_period("M")
    last = np.r_[periods[1:] != periods[:-1], True]
    return idx[last]


def holdings(result: PipelineResult, pipeline: ResearchPipeline) -> pd.DataFrame:
    """Positions at month-end decision dates with alpha, specific vol and industry."""
    W = result.backtest.weights
    ppy = int(pipeline.cfg.trading_days_per_year)
    dates = month_end_dates(pd.DatetimeIndex(W.index))[-HOLDINGS_MAX_DATES:]
    frames = []
    for d in dates:
        row = W.loc[d]
        assert isinstance(row, pd.Series)
        w = row[row.abs().to_numpy() > MIN_ABS_WEIGHT]
        if not len(w):
            continue
        alpha = pd.Series(np.nan, index=w.index)
        svol = pd.Series(np.nan, index=w.index)
        industry = pd.Series(None, index=w.index, dtype=object)
        if pipeline.alpha is not None:
            try:
                alpha = pipeline.alpha.as_of(d).expected_return.reindex(w.index).astype(float)
            except Exception:
                log.debug("alpha.as_of(%s) failed", d, exc_info=True)
        if pipeline.risk is not None:
            try:
                sv = pipeline.risk.as_of(d).specific_var.reindex(w.index).astype(float)
                svol = np.sqrt(sv.clip(lower=0.0) * ppy)
            except Exception:
                log.debug("risk.as_of(%s) failed", d, exc_info=True)
        if pipeline.md_u is not None:
            try:
                industry = pipeline.md_u.industry_as_of(d).reindex(w.index)
            except Exception:
                log.debug("industry_as_of(%s) failed", d, exc_info=True)
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.Timestamp(d),
                    "ticker": w.index.astype(str),
                    "weight": w.to_numpy(dtype=float),
                    "alpha": alpha.to_numpy(dtype=float),
                    "specific_vol": svol.to_numpy(dtype=float),
                    "industry": industry.astype(object).where(industry.notna(), None).to_numpy(),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "weight", "alpha", "specific_vol", "industry"])
    out = pd.concat(frames, ignore_index=True)
    out["industry"] = out["industry"].astype("string")
    return out


def write_extra_artifacts(
    run_dir: str | Path | None,
    result: PipelineResult,
    pipeline: ResearchPipeline,
    meta: Mapping[str, Any] | None = None,
) -> Path | None:
    """Write ``extras.json`` and the extra Parquet files into ``run_dir``.

    ``meta`` (e.g. ``{"label": ..., "kind": ...}`` from the API job runner) is merged
    into ``extras.json["run"]`` and wins over the inferred facts.  Returns the path of
    ``extras.json`` or ``None`` when nothing could be written; never raises.
    """
    if not run_dir:
        return None
    p = Path(run_dir)
    try:
        ppy = int(pipeline.cfg.trading_days_per_year)
        facts = _guard("run", lambda: run_facts(result, pipeline)) or {}
        facts.update({k: v for k, v in dict(meta or {}).items() if v is not None})
        files: dict[str, str] = {}

        ev = _guard("exante_vol", lambda: exante_vol(result, ppy))
        if ev is not None and len(ev):
            ev.to_parquet(p / "exante_vol.parquet")
            files["exante_vol"] = "exante_vol.parquet"

        fx = _guard("exposures", lambda: factor_exposures(result, pipeline))
        roles: dict[str, str] = {}
        if fx is not None and len(fx):
            fx.columns = [str(c) for c in fx.columns]
            fx.to_parquet(p / "exposures.parquet")
            files["exposures"] = "exposures.parquet"
            roles = {c: _role(c) for c in fx.columns}

        hd = _guard("holdings", lambda: holdings(result, pipeline))
        if hd is not None and len(hd):
            hd.to_parquet(p / "holdings.parquet", index=False)
            files["holdings"] = "holdings.parquet"

        payload = {
            "extras_version": EXTRAS_VERSION,
            "run": facts,
            "dsr_inputs": _guard("dsr_inputs", lambda: dsr_moments(result)) or {},
            "ic_by_year": _guard("ic_by_year", lambda: ic_by_year(result)) or [],
            "exposure_roles": roles,
            "risk_only_styles": list(RISK_ONLY_STYLES),
            "files": files,
        }
        out = p / EXTRAS_FILE
        _write_json(out, payload)
        return out
    except Exception:
        log.warning("extra artifacts not written to %s", p, exc_info=True)
        return None


def read_extras(run_dir: str | Path) -> dict[str, Any]:
    """``extras.json`` of a run directory, or ``{}`` (older runs / unreadable file)."""
    f = Path(run_dir) / EXTRAS_FILE
    if not f.exists():
        return {}
    try:
        d = json.loads(f.read_text())
    except (OSError, ValueError):
        log.warning("unreadable %s", f)
        return {}
    return d if isinstance(d, dict) else {}
