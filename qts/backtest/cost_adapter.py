"""Cost-model plumbing for the simulator.

* :func:`cost_frame` normalises any cost estimate (``CostEstimate``, Series,
  DataFrame, dict) into ``spread`` / ``impact`` / ``fees`` USD per ticker and
  raises on unrecognised columns or non-finite values -- a renamed column must
  never make the backtest silently free.
* :class:`_LegacyCostModel` gives a duck-typed cost model without
  ``snapshot`` (a test double with ``snapshot_from_market`` returning a dict
  and ``estimate`` returning a DataFrame) the :class:`qts.core.costs.CostModel`
  interface.  It is applied once, at simulator construction; there is no
  silent fallback between paths.
"""

from __future__ import annotations

import inspect
from typing import Any, cast

import numpy as np
import pandas as pd

from qts.core.config import SystemConfig
from qts.core.costs import VENUE_ROUTED, CostEstimate
from qts.core.types import MarketData
from qts.portfolio.costs import Snapshot, align_snapshot, snapshot_from_market


def cost_frame(est, tickers: pd.Index) -> pd.DataFrame:
    """Normalise a cost estimate into ``spread``/``impact``/``fees`` USD per ticker.

    Accepted: a :class:`CostEstimate`; ``None`` (zero cost); a Series (booked
    as impact); a DataFrame / dict with columns ``spread``, ``impact``,
    ``fees`` (aliases ``<k>_usd`` / ``<k>_cost``) or a single ``total`` column
    (booked as impact).  Anything else raises ``ValueError`` naming the
    columns, and so does any non-finite cost -- a renamed column or a NaN input
    must never make the backtest silently free.
    """
    out = pd.DataFrame(0.0, index=tickers, columns=["spread", "impact", "fees"])
    if est is None:
        return out
    if isinstance(est, CostEstimate):
        f = est.frame
        for k in ("spread", "impact", "fees"):
            out[k] = f[k + "_usd"].reindex(tickers).fillna(0.0).astype(float)
        return out
    if isinstance(est, pd.Series):
        v = est.reindex(tickers).astype(float)
        missing = v.index[~np.isfinite(v.values)]
        if len(missing):
            raise ValueError(f"cost estimate has non-finite values for {list(missing[:5])}")
        out["impact"] = v
        return out
    if isinstance(est, dict):
        est = pd.DataFrame(est)
    if isinstance(est, pd.DataFrame):
        cols = {c.lower(): c for c in est.columns}
        matched = []
        for k in ("spread", "impact", "fees"):
            for alias in (k, k + "_usd", k + "_cost"):
                if alias in cols:
                    out[k] = est[cols[alias]].reindex(tickers).astype(float)
                    matched.append(cols[alias])
                    break
        if not matched:
            if "total" in cols:
                out["impact"] = est[cols["total"]].reindex(tickers).astype(float)
                matched.append(cols["total"])
            else:
                raise ValueError(
                    "estimate() returned no recognised cost column "
                    f"(expected spread/impact/fees[_usd|_cost] or total); got {list(est.columns)}"
                )
        bad = {
            c: list(out.index[~np.isfinite(out[c].values)][:5])
            for c in out.columns
            if not np.isfinite(out[c].values).all()
        }
        if bad:
            raise ValueError(f"estimate() returned non-finite costs (source columns {matched}): {bad}")
        return out
    raise TypeError(f"unsupported cost estimate type {type(est)}")


class _LegacyCostModel:
    """Adapter giving a duck-typed pre-protocol cost model (no ``snapshot``
    method) the :class:`~qts.core.costs.CostModel` interface the simulator uses.

    * ``snapshot``: the model's ``snapshot_from_market(md, t)`` dict if it has
      one, else the built-in builder; reindexed to ``md.tickers`` with
      :func:`align_snapshot` (dict kept) and ``date`` set.
    * ``estimate``: the model's ``estimate(trades, snap, [venue], [streak])``
      (only the keyword arguments its signature accepts), normalised by
      :func:`cost_frame` into a :class:`CostEstimate`.
    * ``holding_cost_usd``: the model's method if present, else
      ``max(-w, 0) * nav * snapshot["borrow_fee"] / trading_days``.
    """

    def __init__(self, model: Any, cfg: SystemConfig, adv_window: int, vol_window: int):
        self.model = model
        self.cfg = cfg
        self.adv_window = adv_window
        self.vol_window = vol_window
        try:
            self._params = set(inspect.signature(model.estimate).parameters)
        except (TypeError, ValueError):
            self._params = set()

    def snapshot(self, md: MarketData, t: pd.Timestamp) -> Snapshot:
        fn = getattr(self.model, "snapshot_from_market", None)
        raw: Any = (
            fn(md, t)
            if callable(fn)
            else snapshot_from_market(md, t, self.adv_window, self.vol_window, self.cfg.cost).to_dict()
        )
        out = cast(dict[str, Any], align_snapshot(dict(raw), md.tickers, self.cfg.cost))  # dict in, dict out
        out.setdefault("date", pd.Timestamp(t))
        return out

    def estimate(
        self,
        trades_usd: pd.Series,
        snap: Snapshot,
        venue: str | None = None,
        streak: pd.Series | None = None,
        prior_streak_usd: pd.Series | None = None,
    ) -> CostEstimate:
        kw: dict[str, object] = {}
        if "venue" in self._params and venue is not None:
            kw["venue"] = venue
        if "streak" in self._params and streak is not None:
            kw["streak"] = streak
        if "prior_streak_usd" in self._params and prior_streak_usd is not None:
            kw["prior_streak_usd"] = prior_streak_usd
        cf = cost_frame(self.model.estimate(trades_usd, snap, **kw), trades_usd.index)
        return CostEstimate.from_components(
            trades_usd.index,
            venue or VENUE_ROUTED,
            trades_usd,
            cf["spread"].to_numpy(),
            cf["impact"].to_numpy(),
            cf["fees"].to_numpy(),
        )

    def holding_cost_usd(self, w_post: pd.Series, nav: float, snap: Snapshot) -> pd.Series:
        fn = getattr(self.model, "holding_cost_usd", None)
        if callable(fn):
            return pd.Series(cast(Any, fn(w_post, nav, snap)), dtype=float).reindex(w_post.index).fillna(0.0)
        fee = (
            pd.Series(snap["borrow_fee"])
            .reindex(w_post.index)
            .fillna(self.cfg.cost.default_borrow_fee_annual)
        )
        return (-w_post).clip(lower=0.0) * nav * fee / self.cfg.trading_days_per_year
