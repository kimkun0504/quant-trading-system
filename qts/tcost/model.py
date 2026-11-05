"""``TransactionCostModel``: the production :class:`qts.core.costs.CostModel`.

One model, two forms: :meth:`TransactionCostModel.estimate` (point estimate in
USD per ticker, used by the backtester) and :meth:`TransactionCostModel.cvx_terms`
(convex cvxpy expression in fraction of NAV, used by the optimizer), both
priced on the same :class:`~qts.core.costs.LiquiditySnapshot` built by
:meth:`TransactionCostModel.snapshot` from data ``<= date`` only.

Notation (per name): ``q = |trade|`` USD, ``P`` price, ``A`` = ``adv_usd``,
``phi`` = auction share of ADV on the date, ``sigma`` daily vol, ``h`` half-spread
(fraction), ``e`` = ``cfg.impact_exponent`` (1.5), ``b_c`` / ``b_a`` the
continuous / auction prefactors, and the impact cost of a metaorder of ``Q``
dollars against volume ``V``::

    C_V,b(Q) = b * sigma * Q^e / V^(e-1)

Venues in :meth:`estimate`
--------------------------
* ``continuous``: ``h q + C_A,b_c(q) + fees(q, continuous)``.
* ``auction``:    ``0 + C_{phi A},b_a(q) + fees(q, auction)`` (one clearing price: no spread).
* ``routed`` (design decision D3; simulator default) -- price the trade the way
  :class:`qts.execution.ExecutionRouter` executes it.  The part up to
  ``K = kappa * phi * A`` (``kappa`` = ``ExecutionConfig.auction_max_pct_of_auction_volume``,
  2%) goes to the auction, the remainder ``q_c = q - q_a`` (``q_a = min(q, K)``)
  to the continuous session::

      spread = h * q_c
      impact = C_{phi A},b_a(q_a) + [ C_A,b_c(q) - C_A,b_c(q_a) ]
      fees   = fees(q_a shares, auction) + fees(q_c shares, continuous)   (two orders)

  *Why the continuous leg is an increment, not C(q_c).*  Both legs are one
  day's metaorder in the same name and move the same price.  The square-root
  law is a law of the *whole* metaorder (impact of a child depends on what has
  already been done), so pricing ``q_c`` as an independent order would use
  ``C(q_a) + C(q_c) < C(q)`` (``C`` is convex with ``C(0) = 0``, hence
  superadditive) and make splitting look artificially cheap.  Treating the
  auction fill as the first ``q_a`` dollars of the metaorder and charging the
  continuous remainder the *marginal* cost of extending it from ``q_a`` to
  ``q`` -- exactly the multi-day persistence argument below -- is consistent
  with the square-root law.  With the D2 prior (``b_a/phi^(e-1) = 0.3/sqrt(0.09)
  = 1.0 = b_c``) the auction leg equals ``C_A,b_c(q_a)`` and the routed impact
  collapses to ``C_A,b_c(q)``: the router saves only the half-spread on
  ``q_a`` (and pays the auction exchange fee), never impact.  On heavy days
  (``phi`` = 20% / 34%) the auction leg is cheaper still.

Convex form for the optimizer (:meth:`cvx_terms`)
------------------------------------------------
For ``continuous`` and ``auction`` the expression is the *exact* estimate
(``trade = estimate.total / nav``) whenever no per-order minimum commission or
per-trade cap binds -- tested in ``tests/test_cost_equivalence.py`` (finding #24).
For ``routed`` the optimizer gets, with ``k = K / nav``, ``beta_a = b_a/phi^(e-1)``
(effective auction prefactor per ``(Q/ADV)^(e-1)``) and ``c = sigma/(A/nav)^(e-1)``::

    h * pos(|x| - k)                                           spread on the continuous part
  + c * [ beta_a * |x|^e + (b_c - beta_a) * pos(|x|^e - k^e) ]   if beta_a <= b_c
  + c * beta_a * |x|^e                                           if beta_a >  b_c
  + (f_c + f_auc) * pos/neg(x)                                   side-specific linear fees

Derivation: the routed impact is ``c*[beta_a*min(|x|,k)^e + b_c*(|x|^e - min(|x|,k)^e)]
= c*[beta_a*|x|^e + (b_c - beta_a)*(|x|^e - min(|x|,k)^e)]`` and
``|x|^e - min(|x|,k)^e = pos(|x|^e - k^e)``, which is convex (``pos`` of a convex
function minus a constant).  So when ``beta_a <= b_c`` -- the D2 default
(``beta_a = 1.0 = b_c`` on a normal day, < 1 on heavy days) -- the spread and
impact terms are **exact** and convex.  When ``beta_a > b_c`` the
``(b_c - beta_a)`` coefficient is negative (non-convex) and the surrogate falls
back to pricing the whole trade at ``beta_a``, an upper bound.  The only other
approximation is the fee: the auction exchange fee is charged on the whole
trade (exact: on ``q_a`` only; upper bound) and, as for every venue, the
per-order commission minimum (here possibly twice: two orders) and per-trade
caps are ignored.  The measured relative error vs :meth:`estimate` is reported
by ``tests/test_cost_equivalence.py`` (a few percent, from fees only).

Multi-day impact persistence (``cfg.impact_persistence``)
----------------------------------------------------------
A position built over ``k`` days in the same direction with daily notionals
``q_1..q_k`` is one metaorder of size ``Q_k = sum q_j``; its integrated cost is
``C(Q_k)`` (``V`` = one day's volume -- today's ``V``, the convention under which
``b`` is calibrated).  The cost attributable to day ``k`` is the increment
``C(Q_k) - C(Q_{k-1})``.

*Exact form (simulator default).*  :class:`qts.backtest.DailyBarSimulator` tracks
each name's cumulative same-direction USD ``Q_{k-1}`` of the current streak
(reset when the sign flips or a day has no trade) and passes it as
``prior_streak_usd``.  The impact leg becomes the one-day venue impact plus the
streak surcharge ``S = C(Q_{k-1} + q) - C(Q_{k-1}) - C(q) >= 0`` (``C``
superadditive), so for ``continuous`` it is exactly ``C(Q_k) - C(Q_{k-1})``; for
``auction`` ``C`` is ``C_{phi A},b_a``; for ``routed`` ``S`` uses ``C_A,b_c`` (the
earlier days and the continuous increment share ``V = A``), i.e. impact
``= [C_A,b_c(Q_k) - C_A,b_c(Q_{k-1})] + [C_{phi A},b_a(q_a) - C_A,b_c(q_a)]``:
the exact streak increment plus the one-day auction discount on the first
``q_a`` dollars (0 under D2).  This is exact for any trade path (halving or
growing daily trades).

*Fallback.*  When only the streak length ``k`` is known (``streak``; no
``prior_streak_usd``), equal earlier days are assumed (``Q_{k-1} = (k-1) q_k``),
which gives the multiplier on the one-day impact::

    m_k = k^e - (k-1)^e     (e = 1.5: m_1 = 1, m_2 = 1.828, m_3 = 2.368, m_k ~ 1.5 sqrt(k))

This is **not** a bound: it understates the increment when daily trades shrink
(e.g. 0.79x on a halving path at e = 1.5) and overstates it when they grow.

Both apply to the impact leg only (spread and fees are per-day) and treat impact
within a streak as fully persistent (Frazzini et al. estimate ~70% permanent);
deliberately conservative.  **Asymmetry:** :meth:`cvx_terms`
gives the optimizer the one-day marginal cost (``m = 1``), because the optimizer
does not know how many days it will keep trading in the same direction; the
realised backtest cost is therefore >= the optimizer's expectation during
multi-day builds.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from qts.core.config import CostConfig, ExecutionConfig
from qts.core.costs import (
    VENUE_AUCTION,
    VENUE_CONTINUOUS,
    VENUE_ROUTED,
    VENUES,
    CostEstimate,
    LiquiditySnapshot,
    SnapshotLike,
    as_snapshot,
)
from qts.core.types import MarketData
from qts.tcost.borrow import TRADING_DAYS, borrow_fee_proxy
from qts.tcost.fees import FeeSchedule
from qts.tcost.impact import SquareRootImpact

log = logging.getLogger(__name__)


def borrow_source_label(by_name: pd.Series) -> str:
    """``LiquiditySnapshot.borrow_fee_source`` summary of the per-name sources: the
    sources used by at least one name, in the order data, proxy, default, joined by
    ``"+"`` (``"data"``, ``"proxy+default"``, ...).  ``"data"`` therefore only
    appears when at least one name has a finite datum (an all-NaN ``borrow_fee`` row
    reports ``"proxy"`` / ``"default"``, audit 2 finding 8)."""
    used = set(by_name.astype(str))
    parts = [k for k in ("data", "proxy", "default") if k in used]
    return "+".join(parts) if parts else "default"


def persistence_multiplier(streak, exponent: float = 1.5) -> np.ndarray:
    """``m_k = k^e - (k-1)^e`` for streak length ``k >= 1`` (1 for ``k <= 1`` or non-finite)."""
    k = np.asarray(streak, dtype=float)
    k = np.where(np.isfinite(k), np.maximum(k, 1.0), 1.0)
    return k**exponent - (k - 1.0) ** exponent


class TransactionCostModel:
    """Production cost model (fees + spread + power-law impact + borrow).

    ``exec_cfg`` supplies the routed-venue auction cap
    (``auction_max_pct_of_auction_volume``); pass ``SystemConfig.execution``.
    ``default_venue`` is used when ``estimate`` / ``cvx_terms`` get ``venue=None``.
    """

    def __init__(
        self,
        cfg: CostConfig | None = None,
        exec_cfg: ExecutionConfig | None = None,
        default_venue: str = VENUE_ROUTED,
    ):
        self.cfg = cfg or CostConfig()
        self.exec_cfg = exec_cfg or ExecutionConfig()
        if default_venue not in VENUES:
            raise ValueError(f"unknown venue {default_venue!r}")
        self.default_venue = default_venue
        self.fees = FeeSchedule(self.cfg)
        self.impact = SquareRootImpact(self.cfg)
        self.exponent = self.impact.exponent

    # ------------------------------------------------------------ snapshot
    def snapshot(
        self,
        md: MarketData,
        date,
        adv_window: int = 63,
        vol_window: int | None = None,
        auction_share: float | None = None,
    ) -> LiquiditySnapshot:
        """Point-in-time :class:`LiquiditySnapshot` as of ``date`` (data ``<= date`` only).

        ``adv_usd``: mean dollar volume (``md.dollar_volume`` = as-traded price x
        actual shares) over the last ``adv_window`` days (includes day ``t``,
        audit #18); ``adv_shares``: mean share volume over the same window
        expressed on day ``t``'s share basis (volumes before a split are
        rescaled by the change in ``close_raw / close``); ``price``: as-traded
        price ``md.trade_price()`` at ``t`` (per-share fees, share counts);
        ``sigma_daily``: s.d. of daily simple returns of the *adjusted* close
        over ``vol_window`` (default ``adv_window``);
        ``half_spread`` from ``md.spread_bps`` (default
        ``cfg.default_half_spread_bps``); ``borrow_fee`` data -> short-interest
        proxy -> GC default (``borrow_fee_source`` records which);
        ``auction_share`` = :func:`~qts.core.costs.auction_share_for_date` with
        ``cfg.auction_volume_share_default``.  Missing values are median-filled
        by the snapshot itself (see :class:`LiquiditySnapshot`).
        """
        date = pd.Timestamp(date)
        vol_window = adv_window if vol_window is None else int(vol_window)
        close_all = md.close.loc[:date]
        if close_all.empty:
            raise ValueError(f"no data on or before {date}")
        n_obs = len(close_all)  # md panels share close's index (validate())
        win = slice(max(n_obs - adv_window, 0), n_obs)
        traded = md.trade_price().iloc[win]  # as-traded price (== close when no close_raw)
        vol = md.volume.iloc[win]  # actual shares traded
        dv = (traded * vol).mean(axis=0, skipna=True)  # == md.dollar_volume over the window
        price = traded.iloc[-1]
        if md.close_raw is not None:
            # split ratio (as-traded / adjusted) per day; restate share volume on day t's basis
            ratio = traded / close_all.iloc[win]
            vol = vol * ratio / ratio.ffill().iloc[-1]
        adv_sh = vol.mean(axis=0, skipna=True)
        rets = close_all.tail(vol_window + 1).pct_change(fill_method=None).tail(vol_window)
        sigma = rets.std(axis=0, ddof=1, skipna=True)
        tickers = md.tickers
        if md.spread_bps is not None and len(md.spread_bps.loc[:date]):
            hs_bps = md.spread_bps.loc[:date].iloc[-1].reindex(tickers).astype(float)
            hs_bps = hs_bps.fillna(self.cfg.default_half_spread_bps)
        else:
            hs_bps = pd.Series(self.cfg.default_half_spread_bps, index=tickers, dtype=float)
        bf, bf_source = self._borrow_fee_asof(md, date, tickers, adv_sh.reindex(tickers))
        share = self.impact.auction_share_for_date(date) if auction_share is None else float(auction_share)
        return LiquiditySnapshot(
            date=date,
            tickers=tickers,
            price=price.reindex(tickers),
            adv_usd=dv.reindex(tickers),
            adv_shares=adv_sh.reindex(tickers),
            sigma_daily=sigma.reindex(tickers),
            half_spread=hs_bps / 1e4,
            borrow_fee=bf,
            auction_share=share,
            borrow_fee_source=bf_source,
        )

    def snapshot_from_market(
        self, md: MarketData, date, cfg_adv_window: int = 63, auction_share: float | None = None
    ) -> dict:
        """Deprecated dict form: ``snapshot(md, date).to_dict()``."""
        return self.snapshot(md, date, adv_window=cfg_adv_window, auction_share=auction_share).to_dict()

    def _borrow_fee_asof(
        self, md: MarketData, date, tickers: pd.Index, adv_shares: pd.Series | None = None
    ) -> tuple[pd.Series, str]:
        bf, by_name = self._borrow_fee_detail(md, date, tickers, adv_shares)
        return bf, borrow_source_label(by_name)

    def borrow_fee_sources(self, md: MarketData, date) -> pd.Series:
        """Per-name source of the snapshot's borrow fee as of ``date``: ``"data"``,
        ``"proxy"`` or ``"default"`` (see :meth:`_borrow_fee_detail`)."""
        snap = self.snapshot(md, date)
        return self._borrow_fee_detail(md, pd.Timestamp(date), md.tickers, snap.adv_shares)[1]

    def _borrow_fee_detail(
        self, md: MarketData, date, tickers: pd.Index, adv_shares: pd.Series | None = None
    ) -> tuple[pd.Series, pd.Series]:
        """Per-name annual borrow fee as of ``date`` and its per-name source.

        Order: the ``md.borrow_fee`` datum when finite (``"data"``); else the
        short-interest proxy (:func:`qts.tcost.borrow.borrow_fee_proxy`) when the
        name has a short-interest ratio (``"proxy"``); else the GC default
        (``"default"``).  **Small caps without SI data:** the GC default is still
        multiplied by ``cfg.small_cap_borrow_multiplier`` when the name's market cap
        is known and below ``cfg.small_cap_threshold_usd`` -- the data-poor small
        caps are exactly the ones that should not get the cheapest borrow (audit 2,
        finding 8).  Their source stays ``"default"``."""
        cfg = self.cfg
        default = float(cfg.default_borrow_fee_annual)
        bf = pd.Series(np.nan, index=tickers, dtype=float)
        src = pd.Series("default", index=tickers, dtype=object)
        if md.borrow_fee is not None:
            hist = md.borrow_fee.loc[:date]
            if len(hist):
                bf = hist.iloc[-1].reindex(tickers).astype(float)
                bf = bf.where(np.isfinite(bf))
                src[bf.notna()] = "data"
        if not bf.isna().any():
            return bf, src
        mc = None
        if md.market_cap is not None and len(md.market_cap.loc[:date]):
            mc = md.market_cap.loc[:date].iloc[-1].reindex(tickers).astype(float)
        si = pd.Series(np.nan, index=tickers, dtype=float)
        dtc = None
        if md.short_interest_ratio is not None:
            si_hist = md.short_interest_ratio.loc[:date]
            if len(si_hist):
                si = si_hist.iloc[-1].reindex(tickers).astype(float)
                if adv_shares is not None and md.shares_outstanding is not None:
                    so = md.shares_outstanding.loc[:date]
                    if len(so):
                        dtc = (
                            si
                            * so.iloc[-1].reindex(tickers).astype(float)
                            / adv_shares.reindex(tickers).replace(0.0, np.nan)
                        )
        # NaN SI -> GC default inside the proxy, times the small-cap multiplier when small
        proxy = borrow_fee_proxy(si, days_to_cover=dtc, cfg=cfg, market_cap=mc)
        proxy = pd.Series(proxy, index=tickers, dtype=float).fillna(default)
        fill = bf.isna()
        is_proxy = fill & si.notna()
        if is_proxy.any() and not (src == "data").any() and not getattr(self, "_warned_proxy", False):
            log.warning(
                "md.borrow_fee absent: using the short-interest borrow-fee PROXY "
                "for %d names (placeholder until IBKR SLB history exists)",
                int(is_proxy.sum()),
            )
            self._warned_proxy = True
        src[is_proxy] = "proxy"
        return bf.fillna(proxy), src

    # ------------------------------------------------------------- helpers
    def _snap(self, snap: SnapshotLike, index: pd.Index | None) -> LiquiditySnapshot:
        return as_snapshot(
            snap,
            index,
            share_default=self.cfg.auction_volume_share_default,
            default_half_spread_bps=self.cfg.default_half_spread_bps,
        )

    def _venue(self, venue: str | None) -> str:
        v = self.default_venue if venue is None else str(venue)
        if v not in VENUES:
            raise ValueError(f"unknown venue {v!r}; expected one of {VENUES}")
        return v

    def _impact_usd(self, q: np.ndarray, vol: np.ndarray, sigma: np.ndarray, b) -> np.ndarray:
        """``b * sigma * q^e / V^(e-1)`` (USD), 0 where q == 0."""
        e = self.exponent
        with np.errstate(divide="ignore", invalid="ignore"):
            out = b * sigma * q**e / vol ** (e - 1.0)
        return np.where(q > 0, out, 0.0)

    def auction_cap_usd(self, snap: LiquiditySnapshot) -> np.ndarray:
        """Routed venue: max USD per name sent to the auction,
        ``auction_max_pct_of_auction_volume * auction_share * adv_usd``."""
        return (
            float(self.exec_cfg.auction_max_pct_of_auction_volume)
            * snap.auction_share
            * snap.adv_usd.to_numpy(dtype=float)
        )

    def effective_auction_b(self, auction_share: float) -> float:
        """Auction prefactor per ``(Q/ADV)^(e-1)``: ``b_a / phi^(e-1)``
        (0.3 / sqrt(0.09) = 1.0 at the defaults)."""
        return float(self.cfg.impact_b_auction) / float(auction_share) ** (self.exponent - 1.0)

    # ------------------------------------------------------------ estimate
    def estimate(
        self,
        trades_usd: pd.Series,
        snap: SnapshotLike,
        venue: str | None = None,
        streak: pd.Series | None = None,
        date=None,
        prior_streak_usd: pd.Series | None = None,
    ) -> CostEstimate:
        """Expected USD cost of ``trades_usd`` (signed dollars, indexed by ticker).

        ``venue`` in {"continuous", "auction", "routed"} (default
        ``self.default_venue``) -- formulas in the module docstring.  Multi-day
        impact persistence (``cfg.impact_persistence``; module docstring):
        ``prior_streak_usd`` (per-name USD already traded in the current
        same-direction streak *before* today, ``Q_{k-1}``; 0 / missing = first
        day) prices the exact streak increment ``C(Q_{k-1} + q) - C(Q_{k-1})``;
        without it, ``streak`` (consecutive same-direction days including today)
        falls back to the equal-days multiplier ``k^e - (k-1)^e``.
        ``date`` overrides ``snap.date`` for the dated fee tables.
        Fully vectorised (numpy); ``snap`` may be an old dict snapshot.
        """
        idx = trades_usd.index
        s = self._snap(snap, idx)
        v = self._venue(venue)
        x = trades_usd.to_numpy(dtype=float)
        x = np.where(np.isfinite(x), x, np.nan)
        if np.isnan(x).any():
            raise ValueError(f"non-finite trades for {list(idx[np.isnan(x)][:5])}")
        q = np.abs(x)
        sell = x < 0
        d = s.date if date is None else pd.Timestamp(date)
        P = s.price.to_numpy(dtype=float)
        A = s.adv_usd.to_numpy(dtype=float)
        sig = s.sigma_daily.to_numpy(dtype=float)
        h = s.half_spread.to_numpy(dtype=float)
        phi = s.auction_share
        b_c, b_a = float(self.cfg.impact_b_continuous), float(self.cfg.impact_b_auction)
        auction_fraction = None

        if v == VENUE_CONTINUOUS:
            spread = h * q
            impact = self._impact_usd(q, A, sig, b_c)
            fees = self.fees.explicit_cost_vec(q / P, P, sell, d, is_auction=False)
        elif v == VENUE_AUCTION:
            spread = np.zeros_like(q)
            impact = self._impact_usd(q, phi * A, sig, b_a)
            fees = self.fees.explicit_cost_vec(q / P, P, sell, d, is_auction=True)
        else:  # routed
            qa = np.minimum(q, self.auction_cap_usd(s))
            qc = q - qa
            spread = h * qc
            impact = (
                self._impact_usd(qa, phi * A, sig, b_a)
                + self._impact_usd(q, A, sig, b_c)
                - self._impact_usd(qa, A, sig, b_c)
            )
            fees = self.fees.explicit_cost_vec(
                qa / P, P, sell, d, is_auction=True
            ) + self.fees.explicit_cost_vec(qc / P, P, sell, d, is_auction=False)
            with np.errstate(divide="ignore", invalid="ignore"):
                auction_fraction = pd.Series(np.where(q > 0, qa / q, 0.0), index=idx, name="auction_fraction")

        mult = np.ones_like(q)
        if prior_streak_usd is not None and self.cfg.impact_persistence:
            qp = pd.Series(prior_streak_usd, dtype=float).reindex(idx).fillna(0.0).to_numpy()
            qp = np.where(np.isfinite(qp) & (q > 0), np.maximum(np.abs(qp), 0.0), 0.0)
            # streak surcharge at the venue's metaorder volume / prefactor:
            #   C(Q_prev + q) - C(Q_prev) - C(q)  (>= 0: C is superadditive)
            if v == VENUE_AUCTION:
                vol_s, b_s = phi * A, b_a
            else:  # continuous; routed: earlier days and the continuous increment share V = ADV
                vol_s, b_s = A, b_c
            surcharge = (
                self._impact_usd(qp + q, vol_s, sig, b_s)
                - self._impact_usd(qp, vol_s, sig, b_s)
                - self._impact_usd(q, vol_s, sig, b_s)
            )
            surcharge = np.where(qp > 0, np.maximum(surcharge, 0.0), 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                mult = np.where((q > 0) & (impact > 0), (impact + surcharge) / impact, 1.0)
            impact = impact + surcharge
        elif streak is not None and self.cfg.impact_persistence:
            k = pd.Series(streak, dtype=float).reindex(idx).fillna(1.0).to_numpy()
            mult = np.where(q > 0, persistence_multiplier(k, self.exponent), 1.0)
            impact = impact * mult
        return CostEstimate.from_components(
            idx,
            v,
            trades_usd.astype(float),
            spread,
            impact,
            fees,
            persistence_mult=mult,
            auction_fraction=auction_fraction,
        )

    def total_cost_usd(self, trades_usd: pd.Series, snap: SnapshotLike, **kw) -> float:
        return float(self.estimate(trades_usd, snap, **kw).frame["total_usd"].sum())

    def holding_cost_usd(self, w_post: pd.Series, nav: float, snap: SnapshotLike) -> pd.Series:
        """One day of borrow cost in USD per ticker (shorts only):
        ``max(-w, 0) * nav * borrow_fee / 252``."""
        s = self._snap(snap, w_post.index)
        w = w_post.astype(float).fillna(0.0).to_numpy()
        return pd.Series(
            np.maximum(-w, 0.0) * float(nav) * s.borrow_fee.to_numpy() / TRADING_DAYS,
            index=w_post.index,
            name="borrow_usd",
        )

    # ------------------------------------------------------------ convex
    def cvx_coefficients(
        self, nav: float, snap: SnapshotLike, venue: str | None = None, tickers: pd.Index | None = None
    ) -> dict:
        """Numpy coefficients of :meth:`cvx_terms` (fraction-of-NAV units):
        ``spread`` (per |x|, or per ``pos(|x| - spread_kink)`` for routed),
        ``spread_kink`` (routed auction cap / nav ``k``, else 0), ``impact`` (per
        ``|x|^e``), ``impact_kink`` (routed: per ``pos(|x|^e - k^e)``, else 0), ``fee_buy`` / ``fee_sell`` (per ``pos(x)`` / ``neg(x)``),
        ``borrow`` (per day per ``neg(w_post)``), ``exponent``.  Exposed so a
        parametrised (DPP) optimizer can update them without rebuilding."""
        s = self._snap(snap, tickers)
        v = self._venue(venue)
        nav = float(nav)
        e = self.exponent
        A = s.adv_usd.to_numpy(dtype=float)
        sig = s.sigma_daily.to_numpy(dtype=float)
        h = s.half_spread.to_numpy(dtype=float)
        P = s.price.to_numpy(dtype=float)
        n = len(s.tickers)
        kink = np.zeros(n)
        kink_coef = np.zeros(n)
        if v == VENUE_CONTINUOUS:
            b, vol, spread, auc_fee = self.cfg.impact_b_continuous, A, h, False
        elif v == VENUE_AUCTION:
            b, vol, spread, auc_fee = self.cfg.impact_b_auction, s.auction_share * A, np.zeros(n), True
        else:
            b_c = float(self.cfg.impact_b_continuous)
            beta_a = self.effective_auction_b(s.auction_share)
            vol, spread, auc_fee = A, h, True
            kink = self.auction_cap_usd(s) / nav
            if beta_a <= b_c * (1.0 + 1e-12):  # exact convex form (module docstring)
                b = beta_a
                kink_coef = max(b_c - beta_a, 0.0) * sig / (A / nav) ** (e - 1.0)
            else:  # upper bound: whole trade at the auction prefactor
                b = beta_a
        coef = float(b) * sig / (vol / nav) ** (e - 1.0)
        d = s.date
        fb = self.fees.marginal_fraction_vec(P, False, d, is_auction=auc_fee)
        fs = self.fees.marginal_fraction_vec(P, True, d, is_auction=auc_fee)
        return {
            "spread": spread,
            "spread_kink": kink,
            "impact": coef,
            "impact_kink": kink_coef,
            "fee_buy": fb,
            "fee_sell": fs,
            "borrow": s.borrow_fee.to_numpy(dtype=float) / TRADING_DAYS,
            "exponent": e,
            "venue": v,
        }

    def cvx_terms(
        self,
        x,
        w_post,
        nav: float,
        snap: SnapshotLike,
        venue: str | None = None,
        tickers: pd.Index | None = None,
        include_explicit: bool = True,
    ):
        """``(trade_cost_expr, holding_cost_expr)`` as fractions of NAV, for the optimizer.

        ``x`` and ``w_post`` are cvxpy expressions over ``tickers`` (default:
        the snapshot's tickers).  ``snap`` may be a :class:`LiquiditySnapshot`
        or an old dict snapshot (the optimizer passes a dict today).  Trade cost::

            sum( spread*|x| + impact*|x|^e + fee_buy*pos(x) + fee_sell*neg(x) )

        (routed: ``spread*pos(|x| - k) + impact_kink*pos(|x|^e - k^e)``) -- coefficients from
        :meth:`cvx_coefficients`; exact vs :meth:`estimate` for continuous /
        auction, a documented convex upper-bound surrogate for routed (module
        docstring).  One-day marginal cost: no persistence multiplier.
        Holding cost: ``sum(borrow/252 * neg(w_post))`` (one day).
        """
        import cvxpy as cp

        if tickers is None and not isinstance(snap, LiquiditySnapshot):
            tickers = pd.Series(snap["adv_usd"]).index
        c = self.cvx_coefficients(nav, snap, venue, tickers)
        ax = cp.abs(x)
        if c["venue"] == VENUE_ROUTED:
            spread_term = cp.sum(cp.multiply(c["spread"], cp.pos(ax - c["spread_kink"])))
        else:
            spread_term = cp.sum(cp.multiply(c["spread"], ax))
        pw = cp.power(ax, c["exponent"])
        trade = spread_term + cp.sum(cp.multiply(c["impact"], pw))
        if np.any(c["impact_kink"] > 0):
            trade = trade + cp.sum(
                cp.multiply(c["impact_kink"], cp.pos(pw - c["spread_kink"] ** c["exponent"]))
            )
        if include_explicit:
            trade = (
                trade
                + cp.sum(cp.multiply(c["fee_buy"], cp.pos(x)))
                + cp.sum(cp.multiply(c["fee_sell"], cp.neg(x)))
            )
        hold = cp.sum(cp.multiply(c["borrow"], cp.neg(w_post)))
        return trade, hold

    def participation_cap(self, snap: SnapshotLike, nav: float, max_part: float | None = None) -> pd.Series:
        s = self._snap(snap, None)
        return self.impact.participation_cap(s.adv_usd, nav, max_part)


__all__ = [
    "VENUE_AUCTION",
    "VENUE_CONTINUOUS",
    "VENUE_ROUTED",
    "TransactionCostModel",
    "persistence_multiplier",
]
