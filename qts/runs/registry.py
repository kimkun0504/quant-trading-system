"""Run registry: write and re-load run directories.

:func:`save_run` writes ``<runs_root>/<run_id>/`` (layout: :mod:`qts.pipeline`
docstring); :func:`load_run` re-loads one as a typed :class:`RunRecord`;
:func:`list_runs` lists them newest first.  ``summary.json`` is
:func:`run_summary_dict` passed through :func:`_json_safe` (non-finite floats
as the strings ``"nan"`` / ``"inf"`` / ``"-inf"``, read back by
:func:`_from_json_scalar`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from qts.core.config import SystemConfig
from qts.core.manifest import RunManifest
from qts.validation import HurdleResult

if TYPE_CHECKING:
    from qts.pipeline import PipelineResult


def _json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return v if math.isfinite(v) else str(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    return v


def _from_json_scalar(v: Any) -> Any:
    if isinstance(v, str) and v in ("nan", "inf", "-inf"):
        return float(v)
    if isinstance(v, dict):
        return {k: _from_json_scalar(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_from_json_scalar(x) for x in v]
    return v


def run_summary_dict(res: PipelineResult) -> dict[str, Any]:
    """The ``summary.json`` payload of a run directory."""
    d = res.decile
    return _json_safe(
        {
            "backtest": res.summary,
            "ic": res.ic_report.scalars(),
            "ic_decay": {str(k): v for k, v in res.ic_report.decay.items()},
            "decile": None
            if d is None
            else {
                "mean": d.mean,
                "sharpe": d.sharpe,
                "t_stat": d.t_stat,
                "horizon": d.horizon,
                "nw_lags": d.nw_lags,
            },
            "dsr": res.dsr,
            "dsr_inputs": res.dsr_inputs,
            "pbo": res.pbo,
            "breakeven_cost_bps": res.breakeven_cost_bps,
            "folds": [
                {
                    "start": str(pd.Timestamp(r.start).date()),
                    "end": str(pd.Timestamp(r.end).date()),
                    "n_days": int(r.n_days),
                    "sharpe": r.sharpe,
                }
                for r in res.folds.itertuples()
            ],
            "cost_sensitivity": {
                str(i): {str(c): res.cost_sensitivity.loc[i, c] for c in res.cost_sensitivity.columns}
                for i in res.cost_sensitivity.index
            },
            "overall": bool(res.hurdles.overall),
            "use_ml": res.use_ml,
            "phase": res.phase,
            "signal_lag_days": res.signal_lag_days,
            "config_hash": res.config_hash,
            "data_hash": res.data_hash,
            "market_hash": res.market_hash,
            "ledger_family": res.ledger_family,
            "git_sha": res.git_sha,
            "timings": res.timings,
            "notes": res.notes,
            # holdout seal / ledger record kind (qts.runs.context.RunContext.summary_fields):
            # sealed_end, unsealed, holdout_start, holdout_already_used, reference, ...
            **_context_fields(res),
        }
    )


def _context_fields(res: PipelineResult) -> dict[str, Any]:
    ctx = getattr(res, "context", None)
    if ctx is None:
        from qts.runs.context import RunContext

        ctx = RunContext()
    return ctx.summary_fields()


def _write_parquet(obj: pd.DataFrame | pd.Series, path: Path, name: str = "value") -> None:
    df = obj.to_frame(name=obj.name or name) if isinstance(obj, pd.Series) else obj.copy()
    df.columns = [str(c) for c in df.columns]
    df.to_parquet(path)


def save_run(res: PipelineResult, cfg: SystemConfig, runs_root: str | Path) -> Path:
    """Write ``res`` to ``<runs_root>/<run_id>/`` (layout: module docstring)."""
    if res.manifest is None:
        ctx = getattr(res, "context", None)
        res.manifest = RunManifest.create(
            config_hash=res.config_hash,
            data_hash=res.data_hash,
            timings=res.timings,
            market_hash=res.market_hash,
            ledger_family=res.ledger_family,
            sealed_end=(ctx.sealed_end if ctx is not None else None) or "",
            unsealed=bool(ctx.unsealed) if ctx is not None else False,
            holdout_start=(ctx.holdout_start if ctx is not None else None) or "",
        )
    run_dir = Path(runs_root) / res.manifest.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    bt = res.backtest
    res.manifest.save(run_dir)
    (run_dir / "config.yaml").write_text(
        cfg.to_yaml(header=f"config of run {res.manifest.run_id} (hash {res.config_hash})")
    )
    (run_dir / "summary.json").write_text(json.dumps(run_summary_dict(res), indent=2, sort_keys=True))
    _write_parquet(bt.nav.rename("nav"), run_dir / "nav.parquet")
    _write_parquet(bt.weights, run_dir / "weights.parquet")
    _write_parquet(bt.trades, run_dir / "trades.parquet")
    _write_parquet(bt.costs, run_dir / "costs.parquet")
    _write_parquet(bt.returns, run_dir / "returns.parquet")
    _write_parquet(res.ic_report.ic_series.rename("rank_ic"), run_dir / "ic.parquet")
    delistings_frame(bt.diagnostics.get("delistings")).to_parquet(run_dir / DELISTINGS_FILE, index=False)
    (run_dir / "hurdles.json").write_text(json.dumps(res.hurdles.to_dict(), indent=2))
    return run_dir


DELISTINGS_FILE = "delistings.parquet"
#: columns of ``delistings.parquet`` (one row per delisting of a held name)
DELISTING_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "weight",
    "reason",
    "side",
    "stock_return",
    "haircut_return",
    "pnl",
    "pnl_usd",
    "liquidation_usd",
    "cost_usd",
)


def delistings_frame(events: object) -> pd.DataFrame:
    """``BacktestResult.diagnostics["delistings"]`` as a frame with
    :data:`DELISTING_COLUMNS` (empty frame, same columns, when there are none)."""
    rows = [e for e in cast(list[Any], events) if isinstance(e, dict)] if isinstance(events, list) else []
    df = pd.DataFrame(rows)
    for c in DELISTING_COLUMNS:
        if c not in df.columns:
            df[c] = pd.Series(dtype=object if c in ("ticker", "reason", "side") else float)
    df = df[list(DELISTING_COLUMNS)]
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    return df.reset_index(drop=True)


def read_delistings(run_dir: str | Path) -> pd.DataFrame | None:
    """``delistings.parquet`` of a run directory; ``None`` for runs written before
    2026-11-12 (the file did not exist)."""
    f = Path(run_dir) / DELISTINGS_FILE
    return pd.read_parquet(f) if f.exists() else None


@dataclass
class RunRecord:
    """A run directory re-loaded (typed)."""

    run_dir: Path
    manifest: RunManifest
    config: SystemConfig
    summary: dict[str, Any]
    hurdles: HurdleResult
    nav: pd.Series
    returns: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    costs: pd.DataFrame
    ic: pd.Series

    @property
    def run_id(self) -> str:
        return self.manifest.run_id


def load_run(run_dir: str | Path) -> RunRecord:
    """Load a run directory written by :func:`save_run`."""
    p = Path(run_dir)
    if not (p / "manifest.json").exists():
        raise FileNotFoundError(f"{p} is not a run directory (no manifest.json)")

    def _pq(name: str) -> pd.DataFrame:
        f = p / name
        return pd.read_parquet(f) if f.exists() else pd.DataFrame()

    nav = _pq("nav.parquet")
    ic = _pq("ic.parquet")
    return RunRecord(
        run_dir=p,
        manifest=RunManifest.load(p),
        config=SystemConfig.from_yaml(p / "config.yaml"),
        summary=_from_json_scalar(json.loads((p / "summary.json").read_text())),
        hurdles=HurdleResult.from_dict(json.loads((p / "hurdles.json").read_text())),
        nav=nav["nav"] if "nav" in nav else pd.Series(dtype=float, name="nav"),
        returns=_pq("returns.parquet"),
        weights=_pq("weights.parquet"),
        trades=_pq("trades.parquet"),
        costs=_pq("costs.parquet"),
        ic=ic["rank_ic"] if "rank_ic" in ic else pd.Series(dtype=float, name="rank_ic"),
    )


def list_runs(runs_root: str | Path) -> list[Path]:
    """Run directories under ``runs_root`` (those with a manifest), newest first."""
    root = Path(runs_root)
    if not root.exists():
        return []
    return sorted(
        (d for d in root.iterdir() if (d / "manifest.json").exists()), key=lambda d: d.name, reverse=True
    )
