"""Run inputs of :meth:`qts.backtest.DailyBarSimulator.run`, as one frozen value."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from qts.core.costs import VENUE_ROUTED, VENUES


@dataclass(frozen=True, eq=False)
class BacktestSpec:
    """What to simulate (the simulator itself holds the config and the cost model).

    * ``dates``: decision dates, a subset of ``md.dates`` (at least two; the last
      one is only a realisation date).
    * ``nav0``: starting NAV in USD.
    * ``rebalance_every``: solve on every ``k``-th date (``i % k == 0`` for the
      ``i``-th date of ``dates``) and let the book drift in between; forced exits
      still run daily.  ``None`` = ``cfg.portfolio.rebalance_every_days``.
    * ``venue``: ``"routed"`` (default, design decision D3), ``"auction"`` or
      ``"continuous"``; prices every trade and the optimizer's cost terms.
    * ``neutral_styles``: style bands passed to the optimizer (``None`` =
      its default, ``()`` = none).
    * ``warm_start_weights``: initial book (weights of ``nav0``), default flat.
    * ``eligible_fn``: ``date -> Index`` of investable tickers; ignored when
      ``md.eligible`` is present.
    * ``exit_ineligible_days``: overrides ``cfg.portfolio.exit_ineligible_days``.
    * ``signal_lag_days``: overrides ``cfg.execution.signal_lag_days``.
    * ``progress``: log a progress line every 50 days.

    Construct with :meth:`create` (coerces ``dates`` / ``neutral_styles``) or
    directly; ``__post_init__`` validates.
    """

    dates: pd.DatetimeIndex
    nav0: float = 1e6
    rebalance_every: int | None = None
    venue: str = VENUE_ROUTED
    neutral_styles: tuple[str, ...] | None = None
    warm_start_weights: pd.Series | None = None
    progress: bool = False
    eligible_fn: Callable[[pd.Timestamp], pd.Index] | None = None
    exit_ineligible_days: int | None = None
    signal_lag_days: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dates, pd.DatetimeIndex):
            raise TypeError(f"dates must be a DatetimeIndex, got {type(self.dates).__name__}")
        if not (np.isfinite(self.nav0) and self.nav0 > 0):
            raise ValueError(f"nav0 must be finite and > 0, got {self.nav0}")
        if self.rebalance_every is not None and int(self.rebalance_every) < 1:
            raise ValueError(f"rebalance_every must be >= 1, got {self.rebalance_every}")
        if self.venue not in VENUES:
            raise ValueError(f"venue must be one of {VENUES}, got {self.venue!r}")
        if self.signal_lag_days is not None and int(self.signal_lag_days) < 0:
            raise ValueError("signal_lag_days must be >= 0")
        if self.neutral_styles is not None and not isinstance(self.neutral_styles, tuple):
            raise TypeError("neutral_styles must be a tuple (use BacktestSpec.create to coerce)")

    @classmethod
    def create(
        cls,
        dates: Sequence[Any] | pd.DatetimeIndex,
        *,
        nav0: float = 1e6,
        rebalance_every: int | None = None,
        venue: str = VENUE_ROUTED,
        neutral_styles: Sequence[str] | None = None,
        warm_start_weights: pd.Series | None = None,
        progress: bool = False,
        eligible_fn: Callable[[pd.Timestamp], pd.Index] | None = None,
        exit_ineligible_days: int | None = None,
        signal_lag_days: int | None = None,
    ) -> BacktestSpec:
        return cls(
            dates=pd.DatetimeIndex(cast(Any, dates)),
            nav0=float(nav0),
            rebalance_every=None if rebalance_every is None else int(rebalance_every),
            venue=venue,
            neutral_styles=None if neutral_styles is None else tuple(neutral_styles),
            warm_start_weights=warm_start_weights,
            progress=bool(progress),
            eligible_fn=eligible_fn,
            exit_ineligible_days=exit_ineligible_days,
            signal_lag_days=signal_lag_days,
        )

    def describe(self) -> dict[str, Any]:
        """JSON-safe summary (no callables, no weight vectors)."""
        return {
            "start": str(self.dates[0].date()) if len(self.dates) else "",
            "end": str(self.dates[-1].date()) if len(self.dates) else "",
            "n_dates": len(self.dates),
            "nav0": self.nav0,
            "rebalance_every": self.rebalance_every,
            "venue": self.venue,
            "neutral_styles": None if self.neutral_styles is None else list(self.neutral_styles),
            "warm_start": self.warm_start_weights is not None,
            "eligible_fn": self.eligible_fn is not None,
            "exit_ineligible_days": self.exit_ineligible_days,
            "signal_lag_days": self.signal_lag_days,
        }
