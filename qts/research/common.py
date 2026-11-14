"""Small helpers shared by the research tools (no heavy imports at module level)."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from qts.backtest import BacktestResult
    from qts.core.config import SystemConfig
    from qts.core.types import MarketData

#: Default ``configs/baseline.yaml`` (the frozen working model), relative to the repo root.
BASELINE_CONFIG = "configs/baseline.yaml"


def repo_root() -> Path:
    """The checkout that holds the ``qts`` package (``configs/``, ``research/``, ``docs/``)."""
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | os.PathLike[str]) -> Path:
    """``path`` as given when it exists (relative to the working directory), else
    relative to the repo root (so the commands work from any directory)."""
    p = Path(path)
    if p.exists() or p.is_absolute():
        return p
    q = repo_root() / p
    return q if q.exists() else p


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def json_safe(v: Any) -> Any:
    """Recursively JSON-safe: numpy scalars -> python, non-finite floats -> ``None``,
    timestamps -> ISO strings, tuples -> lists, frames / series -> records / dicts."""
    if isinstance(v, Mapping):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        x = float(v)
        return x if math.isfinite(x) else None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    if isinstance(v, pd.DataFrame):
        return [json_safe(r) for r in v.reset_index().to_dict(orient="records")]
    if isinstance(v, pd.Series):
        return {str(k): json_safe(x) for k, x in v.items()}
    return v


def write_json(path: str | os.PathLike[str], obj: Any) -> Path:
    """Atomic JSON write (temp file + ``os.replace``) of ``json_safe(obj)``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(json_safe(obj), indent=1, sort_keys=False))
    os.replace(tmp, p)
    return p


def pipeline_config(cfg: SystemConfig, md: MarketData, cache_root: str | None) -> SystemConfig:
    """What ``qts run`` does to the config before it builds the pipeline: clamp
    ``universe.top_n_by_mcap`` at the number of loaded ids and switch the stage cache
    on (``cache_root``; ``None`` = leave ``cfg.cache`` as it is).  The same steps give
    the same config hash and stage-cache keys as the equivalent ``qts run``."""
    cfg = cfg.with_overrides({"universe.top_n_by_mcap": min(cfg.universe.top_n_by_mcap, md.close.shape[1])})
    if cache_root is not None:
        cfg = cfg.with_overrides({"cache.enabled": True, "cache.root": cfg.cache.root or cache_root})
    return cfg


def adv_cap_share(bt: BacktestResult, eps: float = 1e-10) -> dict[str, float]:
    """How often the ADV participation cap bound, from the simulator's solve log:
    ``traded_at_cap`` = mean over rebalance solves of (names whose participation cap
    binds / names traded on that solve); ``solves_with_cap`` = share of solves with at
    least one name at the cap; ``n_solves``.  NaN without solves."""
    log_ = bt.diagnostics.get("solve_log") or []
    entries = [e for e in log_ if isinstance(e, dict)] if isinstance(log_, list) else []
    if not entries:
        return {"traded_at_cap": float("nan"), "solves_with_cap": float("nan"), "n_solves": 0.0}
    trades = bt.trades
    fracs: list[float] = []
    any_cap = 0
    for e in entries:
        binding = e.get("binding") or {}
        n_cap = int(binding.get("participation", 0)) if isinstance(binding, Mapping) else 0
        d = pd.Timestamp(e.get("date"))
        n_traded = int((trades.loc[d].abs() > eps).sum()) if d in trades.index else 0
        any_cap += int(n_cap > 0)
        if n_traded > 0:
            fracs.append(min(n_cap / n_traded, 1.0))
    return {
        "traded_at_cap": float(np.mean(fracs)) if fracs else float("nan"),
        "solves_with_cap": any_cap / len(entries),
        "n_solves": float(len(entries)),
    }


def fmt_float(v: Any, spec: str = ".2f", na: str = "n/a") -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return na
    return format(x, spec) if math.isfinite(x) else na


__all__ = [
    "BASELINE_CONFIG",
    "adv_cap_share",
    "fmt_float",
    "json_safe",
    "pipeline_config",
    "repo_root",
    "resolve_repo_path",
    "utc_stamp",
    "write_json",
]
