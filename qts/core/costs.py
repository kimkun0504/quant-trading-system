"""The single transaction-cost contract shared by the cost, portfolio and backtest layers.

Three pieces (WORK_PLAN 1.2, audit ``code_quality.md`` refactor #1):

* :class:`LiquiditySnapshot` -- the per-ticker market inputs a cost model needs
  *as of one date* (point-in-time: every field is a function of data
  ``<= date``).  Built by ``CostModel.snapshot(md, date)``; always aligned to
  ``tickers`` and always finite (missing values are filled with the
  cross-sectional median, then floored), so the optimizer's convex terms and
  the simulator's point estimates can never price different inputs.
* :class:`CostEstimate` -- the per-ticker USD cost of one trade vector, with
  fixed columns ``spread_usd, impact_usd, fees_usd, total_usd`` (+ ``venue``,
  ``persistence_mult``).  Non-finite costs raise: a renamed column or a NaN
  input must never make a backtest silently free.
* :class:`CostModel` -- the protocol (``snapshot``, ``estimate``,
  ``cvx_terms``, ``holding_cost_usd``).  ``qts.tcost.TransactionCostModel`` is
  the production implementation; ``qts.portfolio.costs.SimpleCostModel`` is a
  test double.

Units
-----
``price`` USD/share; ``adv_usd`` USD/day; ``adv_shares`` shares/day;
``sigma_daily`` daily return s.d. (fraction); ``half_spread`` **fraction** of
price (``half_spread_bps = 1e4 * half_spread``); ``borrow_fee`` **annual**
fraction; ``auction_share`` = expected closing-auction volume / ADV for the
snapshot date; ``nav`` USD (optional).

Venues
------
``"continuous"`` (in-session algo, V = ADV, pays the half-spread),
``"auction"`` (closing auction, V = auction_share * ADV, no spread) and
``"routed"`` (what the router would do: up to
``ExecutionConfig.auction_max_pct_of_auction_volume`` of the auction at the
auction, the rest in the continuous session -- see
:meth:`qts.tcost.model.TransactionCostModel.estimate`).

Backward compatibility
----------------------
Before this module the snapshot was a ``dict`` of Series.  ``LiquiditySnapshot``
supports read-only mapping access (``snap["adv_usd"]``, ``"date" in snap``,
``snap.get(k)``, ``dict(snap)``) with the old key names, ``to_dict()``, and
:meth:`LiquiditySnapshot.from_mapping` converts an old dict.  Cost models accept
either form in ``cvx_terms`` / ``estimate`` (see :func:`as_snapshot`).
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

#: Anything ``pd.Timestamp`` accepts as a calendar date (``None`` = no date).
DateLike = str | _dt.date | np.datetime64 | None

VENUE_CONTINUOUS = "continuous"
VENUE_AUCTION = "auction"
VENUE_ROUTED = "routed"
VENUES = (VENUE_CONTINUOUS, VENUE_AUCTION, VENUE_ROUTED)

#: default share of ADV printed in the closing auction on a normal day (BMLL ~9%).
#: The configurable single source is ``CostConfig.auction_volume_share_default``;
#: this constant is only the fallback when no config is at hand.
AUCTION_SHARE_NORMAL_DAY = 0.09

# FTSE Russell moved the Russell US indexes to a semi-annual reconstitution
# from 2026 (June + a second review late in the year).  The June date follows
# the last-Friday-of-June rule in ``auction_share_for_date``; the second one is
# published each year and must be added here.  2026: effective after the close
# on Friday 11 December 2026 (rank day 30 October; FTSE Russell press release,
# 1 September 2026).  Share of ADV in the close on those days is a placeholder
# equal to the June figure (~34%).
RUSSELL_RECON_DATES: dict[pd.Timestamp, float] = {
    pd.Timestamp("2026-12-11"): 0.34,
}


def auction_share_for_date(date: DateLike, share_default: float = AUCTION_SHARE_NORMAL_DAY) -> float:
    """Share of daily volume expected in the closing auction on ``date``.

    Defaults (BMLL): ``share_default`` (~9%, pass
    ``CostConfig.auction_volume_share_default``) on a normal day, ~20% on
    quarterly index rebalance / expiration days (third Friday of
    Mar/Jun/Sep/Dec), ~34% on Russell reconstitution day (last Friday of June
    and, from 2026, the dates in :data:`RUSSELL_RECON_DATES`).  ``date=None``
    -> ``share_default``.  The heavy-day figures are floors: a configured
    default above them wins.
    """
    if date is None:
        return float(share_default)
    d = pd.Timestamp(date).normalize()
    if d in RUSSELL_RECON_DATES:
        return max(share_default, RUSSELL_RECON_DATES[d])
    if d.weekday() != 4:
        return float(share_default)
    if d.month == 6:  # Russell: last Friday of June (approximation)
        last_day = d + pd.offsets.MonthEnd(0)
        last_friday = last_day - pd.Timedelta(days=(last_day.weekday() - 4) % 7)
        if d == last_friday.normalize():
            return max(share_default, 0.34)
    if d.month in (3, 6, 9, 12) and 15 <= d.day <= 21:  # third Friday of quarter-end months
        return max(share_default, 0.20)
    return float(share_default)


# --------------------------------------------------------------------------- snapshot
_FILL_DEFAULTS = {
    "price": 100.0,
    "adv_usd": 1e6,
    "sigma_daily": 0.02,
    "half_spread": 2e-4,
    "borrow_fee": 0.005,
}
_FLOORS = {"price": 0.01, "adv_usd": 1e3, "sigma_daily": 1e-4, "half_spread": 0.0, "borrow_fee": 0.0}
_SERIES_FIELDS = ("price", "adv_usd", "adv_shares", "sigma_daily", "half_spread", "borrow_fee")


def _median_fill(s: pd.Series, default: float, floor: float) -> pd.Series:
    s = s.astype(float).replace([np.inf, -np.inf], np.nan)
    if s.isna().any():
        med = float(s.median()) if s.notna().any() else np.nan
        s = s.fillna(med if np.isfinite(med) else default)
    return s.clip(lower=floor)


def _as_series(v: Any, tickers: pd.Index) -> pd.Series:
    if v is None:
        return pd.Series(np.nan, index=tickers, dtype=float)
    if isinstance(v, pd.Series):
        return v.reindex(tickers).astype(float)
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 0:
        return pd.Series(float(arr), index=tickers, dtype=float)
    return pd.Series(arr, index=tickers, dtype=float)


@dataclass(frozen=True, eq=False)
class LiquiditySnapshot:
    """Per-ticker liquidity/cost inputs as of ``date`` (see module docstring for units).

    Construction normalises every per-ticker field: reindexed to ``tickers``;
    NaN/inf filled with the cross-sectional median (fallbacks: price 100,
    ADV $1M, sigma 2%, half-spread 2 bp, borrow 0.5%/yr); floors price >= $0.01,
    ADV >= $1,000, sigma >= 1e-4, spread and borrow >= 0.  Missing
    ``adv_shares`` = ``adv_usd / price``.  A new listing therefore gets median
    impact, never zero impact (audit #10).
    """

    date: pd.Timestamp | None
    tickers: pd.Index
    price: pd.Series
    adv_usd: pd.Series
    adv_shares: pd.Series
    sigma_daily: pd.Series
    half_spread: pd.Series  # fraction of price
    borrow_fee: pd.Series  # annual fraction
    auction_share: float = AUCTION_SHARE_NORMAL_DAY
    borrow_fee_source: str = "default"
    nav: float | None = None

    def __post_init__(self) -> None:
        tickers = pd.Index(self.tickers)
        object.__setattr__(self, "tickers", tickers)
        object.__setattr__(self, "date", None if self.date is None else pd.Timestamp(self.date))
        for f in ("price", "adv_usd", "sigma_daily", "half_spread", "borrow_fee"):
            s = _median_fill(_as_series(getattr(self, f), tickers), _FILL_DEFAULTS[f], _FLOORS[f])
            object.__setattr__(self, f, s.rename(f))
        sh = _as_series(self.adv_shares, tickers).replace([np.inf, -np.inf], np.nan)
        sh = sh.fillna(self.adv_usd / self.price).clip(lower=0.0)
        object.__setattr__(self, "adv_shares", sh.rename("adv_shares"))
        share = float(self.auction_share)
        if not (0.0 < share <= 1.0):
            raise ValueError(f"auction_share must be in (0, 1], got {share}")
        object.__setattr__(self, "auction_share", share)
        if self.nav is not None:
            object.__setattr__(self, "nav", float(self.nav))

    # ------------------------------------------------------------- derived
    @property
    def half_spread_bps(self) -> pd.Series:
        return (self.half_spread * 1e4).rename("half_spread_bps")

    @property
    def auction_volume_usd(self) -> pd.Series:
        return (self.adv_usd * self.auction_share).rename("auction_volume_usd")

    def __len__(self) -> int:
        return len(self.tickers)

    # ------------------------------------------------------------- transforms
    def reindex(self, tickers: pd.Index) -> LiquiditySnapshot:
        """Same snapshot over ``tickers``; names not in ``self.tickers`` get the
        cross-sectional median of each field (via ``__post_init__``)."""
        tickers = pd.Index(tickers)
        if tickers.equals(self.tickers):
            return self
        kw = {f: getattr(self, f).reindex(tickers) for f in _SERIES_FIELDS}
        # median of the *source* cross-section for names that are new
        for f in _SERIES_FIELDS:
            if kw[f].isna().any():
                kw[f] = kw[f].fillna(float(getattr(self, f).median()))
        return replace(self, tickers=tickers, **kw)

    def with_nav(self, nav: float | None) -> LiquiditySnapshot:
        return replace(self, nav=nav)

    def to_dict(self) -> dict[str, Any]:
        """Old dict-of-Series form (keys as consumed before the typed contract)."""
        d: dict[str, Any] = {
            "adv_usd": self.adv_usd,
            "adv_shares": self.adv_shares,
            "sigma_daily": self.sigma_daily,
            "half_spread": self.half_spread,
            "half_spread_bps": self.half_spread_bps,
            "price": self.price,
            "borrow_fee": self.borrow_fee,
            "date": self.date,
            "auction_share": self.auction_share,
            "borrow_fee_source": self.borrow_fee_source,
        }
        if self.nav is not None:
            d["nav"] = self.nav
        return d

    # read-only mapping access with the old key names (optimizer / scripts)
    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        if key not in d:
            raise KeyError(key)
        return d[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self.to_dict()

    def keys(self):
        return self.to_dict().keys()

    def items(self):
        return self.to_dict().items()

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    # ------------------------------------------------------------- adapters
    @classmethod
    def from_mapping(
        cls,
        d: Mapping[str, Any],
        tickers: pd.Index | None = None,
        share_default: float = AUCTION_SHARE_NORMAL_DAY,
        default_half_spread_bps: float = 2.0,
    ) -> LiquiditySnapshot:
        """Convert an old dict snapshot.  ``half_spread_bps`` wins over
        ``half_spread`` (as the old ``estimate`` did); ``auction_share`` defaults
        to :func:`auction_share_for_date` of ``d['date']``.  Per-ticker ``venue``
        keys are ignored: the venue is a call argument of the cost model."""
        if tickers is None:
            first = d.get("adv_usd")
            ref: pd.Series | None = (
                cast(pd.Series, first)
                if isinstance(first, pd.Series)
                else next((cast(pd.Series, v) for v in d.values() if isinstance(v, pd.Series)), None)
            )
            if ref is None:
                raise ValueError("cannot infer tickers from a snapshot with no Series")
            tickers = ref.index
        tickers = pd.Index(tickers)
        if d.get("half_spread_bps") is not None:
            hs = _as_series(d["half_spread_bps"], tickers) / 1e4
        elif d.get("half_spread") is not None:
            hs = _as_series(d["half_spread"], tickers)
        else:
            hs = pd.Series(default_half_spread_bps / 1e4, index=tickers)
        if hs.isna().all():
            hs = pd.Series(default_half_spread_bps / 1e4, index=tickers)
        date = d.get("date")
        share = d.get("auction_share")
        if share is None or not np.isfinite(float(share)):
            share = auction_share_for_date(date, share_default)
        nav = d.get("nav")
        return cls(
            date=date,
            tickers=tickers,
            price=_as_series(d.get("price"), tickers),
            adv_usd=_as_series(d.get("adv_usd"), tickers),
            adv_shares=_as_series(d.get("adv_shares"), tickers),
            sigma_daily=_as_series(d.get("sigma_daily"), tickers),
            half_spread=hs,
            borrow_fee=_as_series(d.get("borrow_fee"), tickers),
            auction_share=float(share),
            borrow_fee_source=str(d.get("borrow_fee_source", "default")),
            nav=None if nav is None else float(nav),
        )


SnapshotLike = LiquiditySnapshot | Mapping[str, Any]


def as_snapshot(
    snap: SnapshotLike,
    tickers: pd.Index | None = None,
    share_default: float = AUCTION_SHARE_NORMAL_DAY,
    default_half_spread_bps: float = 2.0,
) -> LiquiditySnapshot:
    """Adapter: a :class:`LiquiditySnapshot` (reindexed to ``tickers`` if given)
    or an old dict snapshot converted with :meth:`LiquiditySnapshot.from_mapping`."""
    if isinstance(snap, LiquiditySnapshot):
        return snap if tickers is None else snap.reindex(tickers)
    # runtime guard for untyped callers (the annotation already says Mapping)
    if isinstance(cast(object, snap), Mapping):
        return LiquiditySnapshot.from_mapping(snap, tickers, share_default, default_half_spread_bps)
    raise TypeError(f"expected LiquiditySnapshot or dict snapshot, got {type(snap)}")


# --------------------------------------------------------------------------- estimate
COST_USD_COLUMNS = ("spread_usd", "impact_usd", "fees_usd", "total_usd")
COST_ESTIMATE_COLUMNS = (*COST_USD_COLUMNS, "venue", "persistence_mult")


@dataclass(frozen=True, eq=False)
class CostEstimate:
    """USD cost of one trade vector, per ticker.

    ``frame`` columns (fixed): ``spread_usd, impact_usd, fees_usd, total_usd``
    (all >= 0, finite, ``total = spread + impact + fees``), ``venue`` (the
    venue priced) and ``persistence_mult`` (multi-day impact multiplier applied,
    1 when off).  ``trade_usd`` is the signed trade; ``auction_fraction`` (routed
    venue only) is the share of ``|trade|`` priced in the closing auction.
    Holding (borrow) cost is *not* part of a trade estimate: see
    ``CostModel.holding_cost_usd``.
    """

    frame: pd.DataFrame
    venue: str
    trade_usd: pd.Series
    auction_fraction: pd.Series | None = None

    def __post_init__(self) -> None:
        missing = [c for c in COST_ESTIMATE_COLUMNS if c not in self.frame.columns]
        if missing:
            raise ValueError(f"CostEstimate.frame missing columns {missing}")
        num = self.frame[list(COST_USD_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(num).all():
            bad = self.frame.index[np.flatnonzero(~np.isfinite(num).all(axis=1))]
            raise ValueError(f"non-finite cost estimate for {list(bad[:5])}")

    @property
    def index(self) -> pd.Index:
        return self.frame.index

    def __getitem__(self, col: str) -> pd.Series:
        return self.frame[col]

    @property
    def total(self) -> pd.Series:
        return self.frame["total_usd"]

    def totals(self) -> pd.Series:
        """Sums over tickers: ``spread, impact, fees, total`` (USD)."""
        f = self.frame
        return pd.Series(
            {
                "spread": float(f["spread_usd"].sum()),
                "impact": float(f["impact_usd"].sum()),
                "fees": float(f["fees_usd"].sum()),
                "total": float(f["total_usd"].sum()),
            }
        )

    def bps(self) -> pd.DataFrame:
        """Per-ticker cost in bps of ``|trade_usd|`` (0 where no trade)."""
        q = self.trade_usd.abs().reindex(self.frame.index).to_numpy(dtype=float)
        out: dict[str, NDArray[np.float64]] = {}
        for c in COST_USD_COLUMNS:
            v = self.frame[c].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[c.replace("_usd", "_bps")] = np.where(q > 0, v / q * 1e4, 0.0)
        return pd.DataFrame(out, index=self.frame.index)

    @classmethod
    def from_components(
        cls,
        index: pd.Index,
        venue: str,
        trade_usd: pd.Series,
        spread: ArrayLike,
        impact: ArrayLike,
        fees: ArrayLike,
        persistence_mult: ArrayLike = 1.0,
        auction_fraction: pd.Series | None = None,
    ) -> CostEstimate:
        spread = np.asarray(spread, dtype=float) * np.ones(len(index))
        impact = np.asarray(impact, dtype=float) * np.ones(len(index))
        fees = np.asarray(fees, dtype=float) * np.ones(len(index))
        frame = pd.DataFrame(
            {
                "spread_usd": spread,
                "impact_usd": impact,
                "fees_usd": fees,
                "total_usd": spread + impact + fees,
                "venue": venue,
                "persistence_mult": np.asarray(persistence_mult, dtype=float) * np.ones(len(index)),
            },
            index=index,
        )
        return cls(frame=frame, venue=venue, trade_usd=trade_usd, auction_fraction=auction_fraction)

    @classmethod
    def zero(cls, index: pd.Index, venue: str = VENUE_ROUTED) -> CostEstimate:
        z = np.zeros(len(index))
        return cls.from_components(index, venue, pd.Series(z, index=index), z, z, z)


# --------------------------------------------------------------------------- protocol
@runtime_checkable
class CostModel(Protocol):
    """What the portfolio and backtest layers know about a cost model.

    * ``snapshot(md, date)``: point-in-time :class:`LiquiditySnapshot`.
    * ``estimate(trades_usd, snap, venue, streak=None)``: :class:`CostEstimate`
      (USD).  ``streak`` = per-name count of consecutive same-direction trading
      days including today (multi-day impact persistence).
    * ``cvx_terms(x, w_post, nav, snap, venue)``: ``(trade_cost, holding_cost)``
      cvxpy expressions in **fraction of NAV**; ``holding_cost`` is one day.
      ``snap`` may be a :class:`LiquiditySnapshot` or an old dict snapshot.
    * ``holding_cost_usd(w_post, nav, snap)``: one day of borrow, USD per ticker.
    """

    def snapshot(self, md: Any, date: Any) -> LiquiditySnapshot: ...

    def estimate(
        self,
        trades_usd: pd.Series,
        snap: SnapshotLike,
        venue: str | None = None,
        streak: pd.Series | None = None,
    ) -> CostEstimate: ...

    def cvx_terms(
        self, x: Any, w_post: Any, nav: float, snap: SnapshotLike, venue: str | None = None
    ) -> tuple[Any, Any]: ...

    def holding_cost_usd(self, w_post: pd.Series, nav: float, snap: SnapshotLike) -> pd.Series: ...
