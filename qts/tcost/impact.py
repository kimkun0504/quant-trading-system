"""Square-root market impact in point-estimate, dollar and convex (cvxpy) forms.

Model (BUILD_PLAN section 4)
----------------------------
Expected impact of a metaorder of ``Q`` dollars in a name with dollar volume
``V`` over the trading period and daily return volatility ``sigma``::

    I / sigma = b * (Q / V)^(e-1)          (return units; e = 1.5 -> square root)

The *integrated* cost of a trade of ``x`` dollars, used by the optimizer, is
(cvxportfolio form)::

    C(x) = a*|x| + b * sigma * |x|^e / V^(e-1)        (dollars)

where ``a`` is the proportional cost (half-spread + linear explicit fees) and
``e = cfg.impact_exponent`` (1.5; validated ``1 < e <= 2``).  ``b`` is
``cfg.impact_b_continuous`` (1.0) for continuous-session execution and
``cfg.impact_b_auction`` (0.3; a prior to be recalibrated from own fills via
``TCALedger.calibrate_b_by_venue``) for closing-auction fills, where ``V`` must
be the *auction* volume, not full-day ADV.  Because ``V`` is ~9% of ADV on a
normal day, the effective auction coefficient per ``sqrt(Q/ADV)`` is
``0.3/sqrt(0.09) = 1.0`` -- equal to the continuous session (design decision
D2), so an auction fill costs the same impact and saves the half-spread.

Units
-----
* ``sigma_daily``: daily return standard deviation (fraction, e.g. 0.02).
* ``adv_usd`` / ``V``: dollars per day (or per auction).
* ``notional`` / ``x_usd``: dollars (sign = direction, abs value used).
* ``cost_bps`` returns basis points of the trade notional.
* ``total_cost_fraction`` returns **dollars**; divide by ``|x|`` to get the
  cost as a fraction of notional.
* In the cvxpy form ``x`` is a vector of *weight changes* (fraction of NAV)
  and the expression is the cost as a **fraction of NAV**; the dollar form
  divided by ``nav`` with ``X = x*nav`` gives
  ``a|x| + b*sigma*|x|^e / (V / nav)^(e-1)``.

The power term is ``cp.power(cp.abs(x), e)`` which is DCP-convex for ``e >= 1``
(convex, increasing power of a convex non-negative argument).  Clarabel handles
it through the power cone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qts.core.config import CostConfig
from qts.core.costs import (  # noqa: F401 (re-export)
    RUSSELL_RECON_DATES,
    VENUE_AUCTION,
    VENUE_CONTINUOUS,
    VENUE_ROUTED,
    auction_share_for_date,
)

try:  # cvxpy is a hard dependency of the project, but keep import lazy-safe
    import cvxpy as cp
except ImportError:  # pragma: no cover
    cp = None

ArrayLike = float | np.ndarray | pd.Series


def validate_impact_exponent(e: float) -> float:
    e = float(e)
    if not (1.0 < e <= 2.0):
        raise ValueError(f"CostConfig.impact_exponent must satisfy 1 < e <= 2, got {e}")
    return e


class SquareRootImpact:
    def __init__(self, cfg: CostConfig | None = None):
        self.cfg = cfg or CostConfig()
        self.exponent = validate_impact_exponent(getattr(self.cfg, "impact_exponent", 1.5))

    @property
    def share_default(self) -> float:
        """Normal-day auction share of ADV: ``cfg.auction_volume_share_default``."""
        return float(self.cfg.auction_volume_share_default)

    # ---------------------------------------------------------------- params
    def b_for_venue(self, venue: str = VENUE_CONTINUOUS, b: float | None = None) -> float:
        if b is not None:
            return float(b)
        if venue == VENUE_AUCTION:
            return float(self.cfg.impact_b_auction)
        if venue == VENUE_CONTINUOUS:
            return float(self.cfg.impact_b_continuous)
        raise ValueError(f"unknown venue {venue!r}")

    # --------------------------------------------------------- point estimate
    def cost_bps(
        self,
        notional: ArrayLike,
        adv_usd: ArrayLike,
        sigma_daily: ArrayLike,
        b: float | None = None,
        venue: str = VENUE_CONTINUOUS,
    ) -> ArrayLike:
        """Expected impact in **bps of notional**: ``1e4 * b * sigma * (|Q|/V)^(e-1)``
        (``e = cfg.impact_exponent``; square root at the default 1.5).

        Vectorised over numpy arrays / pandas Series.  ``adv_usd`` is the
        volume relevant to the venue (full-day ADV for continuous, auction
        volume for auction fills).
        """
        bb = self.b_for_venue(venue, b)
        q = np.abs(np.asarray(notional, dtype=float))
        v = np.asarray(adv_usd, dtype=float)
        s = np.asarray(sigma_daily, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = 1e4 * bb * s * np.where(v > 0, q / v, np.inf) ** (self.exponent - 1.0)
        out = np.where(q == 0, 0.0, out)
        if isinstance(notional, pd.Series):
            return pd.Series(out, index=notional.index)
        return out if out.ndim else float(out)

    def total_cost_fraction(
        self,
        x_usd: ArrayLike,
        adv_usd: ArrayLike,
        sigma_daily: ArrayLike,
        half_spread_bps: ArrayLike = 0.0,
        venue: str = VENUE_CONTINUOUS,
        b: float | None = None,
        explicit_bps: ArrayLike = 0.0,
    ) -> ArrayLike:
        """Integrated square-root cost of a trade, in **dollars**::

            C = (half_spread + explicit) * |x| + b * sigma * |x|^e / V^(e-1)

        ``half_spread_bps`` and ``explicit_bps`` are proportional costs in bps
        (converted to fractions internally).  Despite the name (kept for API
        compatibility with the plan), the return value is dollars; the cost as
        a *fraction of notional* is ``C / |x|`` and equals
        ``a + b*sigma*(|x|/V)^(e-1)`` -- i.e. the average impact of the
        metaorder is exactly :meth:`cost_bps`.
        """
        bb = self.b_for_venue(venue, b)
        q = np.abs(np.asarray(x_usd, dtype=float))
        v = np.asarray(adv_usd, dtype=float)
        s = np.asarray(sigma_daily, dtype=float)
        a = (np.asarray(half_spread_bps, dtype=float) + np.asarray(explicit_bps, dtype=float)) / 1e4
        with np.errstate(divide="ignore", invalid="ignore"):
            imp = bb * s * q**self.exponent / v ** (self.exponent - 1.0)
        imp = np.where(q == 0, 0.0, imp)
        out = a * q + imp
        if isinstance(x_usd, pd.Series):
            return pd.Series(out, index=x_usd.index)
        return out if np.ndim(out) else float(out)

    # --------------------------------------------------------------- convex
    def cvx_cost_expression(
        self,
        x,
        nav: float,
        adv_usd: np.ndarray,
        sigma: np.ndarray,
        half_spread: np.ndarray,
        venue: str = VENUE_CONTINUOUS,
        b: float | None = None,
        explicit_buy_frac: np.ndarray | None = None,
        explicit_sell_frac: np.ndarray | None = None,
    ):
        """cvxpy expression for trade cost as a **fraction of NAV**.

        Parameters
        ----------
        x : cvxpy.Variable or Expression (n,)
            Weight changes ``w_post - w_pre``.
        nav : float
            Portfolio NAV in dollars.
        adv_usd, sigma, half_spread : arrays (n,)
            Dollar volume for the venue, daily vol, half-spread as a
            *fraction* (not bps).
        explicit_buy_frac / explicit_sell_frac : optional arrays (n,)
            Linear explicit fee per dollar bought / sold (see
            :meth:`FeeSchedule.marginal_fraction`).  Applied to ``pos(x)`` and
            ``neg(x)`` respectively, which keeps the expression convex.

        Returns a DCP-convex scalar expression::

            sum( hs*|x| + b*sigma*|x|^e / (V/nav)^(e-1) + fee_b*pos(x) + fee_s*neg(x) )
        """
        if cp is None:  # pragma: no cover
            raise ImportError("cvxpy is required for cvx_cost_expression")
        bb = self.b_for_venue(venue, b)
        v = np.maximum(np.asarray(adv_usd, dtype=float), 1e-9)
        s = np.asarray(sigma, dtype=float)
        hs = np.asarray(half_spread, dtype=float)
        n = x.shape[0]
        hs = np.broadcast_to(hs, (n,))
        s = np.broadcast_to(s, (n,))
        v = np.broadcast_to(v, (n,))
        coef = bb * s / (v / float(nav)) ** (self.exponent - 1.0)  # multiplies |x|^e
        ax = cp.abs(x)
        expr = cp.sum(cp.multiply(hs, ax)) + cp.sum(cp.multiply(coef, cp.power(ax, self.exponent)))
        if explicit_buy_frac is not None:
            expr = expr + cp.sum(
                cp.multiply(np.broadcast_to(np.asarray(explicit_buy_frac, float), (n,)), cp.pos(x))
            )
        if explicit_sell_frac is not None:
            expr = expr + cp.sum(
                cp.multiply(np.broadcast_to(np.asarray(explicit_sell_frac, float), (n,)), cp.neg(x))
            )
        return expr

    # ---------------------------------------------------------- constraints
    def participation_cap(self, adv_usd: ArrayLike, nav: float, max_part: float | None = None) -> ArrayLike:
        """Maximum absolute weight change per name for one day:
        ``max_part * adv_usd / nav``.  Default ``max_part = cfg.max_participation_adv``."""
        mp = self.cfg.max_participation_adv if max_part is None else float(max_part)
        cap = mp * np.asarray(adv_usd, dtype=float) / float(nav)
        if isinstance(adv_usd, pd.Series):
            return pd.Series(cap, index=adv_usd.index)
        return cap if np.ndim(cap) else float(cap)

    # ------------------------------------------------------------- auctions
    def auction_share_for_date(self, date, share_default: float | None = None) -> float:
        """:func:`qts.core.costs.auction_share_for_date` with the normal-day share
        from ``cfg.auction_volume_share_default`` (the single source)."""
        return auction_share_for_date(date, self.share_default if share_default is None else share_default)

    def auction_volume_estimate(
        self, adv_shares: ArrayLike, date=None, share_default: float | None = None
    ) -> ArrayLike:
        """Expected closing-auction volume (same units as ``adv_shares``)."""
        share = self.auction_share_for_date(date, share_default)
        out = np.asarray(adv_shares, dtype=float) * share
        if isinstance(adv_shares, pd.Series):
            return pd.Series(out, index=adv_shares.index)
        return out if np.ndim(out) else float(out)

    def cost_for_venue_choice(
        self,
        notional: float,
        adv_usd: float,
        sigma_daily: float,
        half_spread_bps: float = 0.0,
        date=None,
        auction_share: float | None = None,
        auction_spread_bps: float = 0.0,
        price: float | None = None,
        auction_max_pct: float | None = None,
    ) -> dict[str, float | str | bool]:
        """Compare expected all-in cost (bps of ``Q = |notional|``) of continuous,
        closing-auction and routed execution.

        With ``A = adv_usd``, ``phi`` the auction share of the day (``auction_share``
        or :meth:`auction_share_for_date`), ``h = half_spread_bps``, ``f_a`` = the
        closing-auction exchange fee in bps (``cfg.auction_exchange_fee_per_share /
        price``; 0 when ``price`` is not given -- then ``fees_included`` is False)
        and ``C_{V,b}(Q) = b * sigma * Q^e / V^(e-1)`` (dollars):

        * continuous: ``h + 1e4 * C_{A,b_c}(Q) / Q``
        * auction:    ``auction_spread + f_a + 1e4 * C_{phi A,b_a}(Q) / Q`` (one clearing
          price, so ``auction_spread_bps`` defaults to 0)
        * routed (as :class:`qts.execution.ExecutionRouter` executes, and the
          ``routed`` venue of :mod:`qts.tcost.model` prices): ``q_a = min(Q, kappa *
          phi * A)`` to the auction, ``q_c = Q - q_a`` continuous, the continuous leg
          charged the metaorder increment::

              1e4 * [ (auction_spread + f_a) q_a / 1e4 + h q_c / 1e4
                      + C_{phi A,b_a}(q_a) + C_{A,b_c}(Q) - C_{A,b_c}(q_a) ] / Q

          with ``kappa = auction_max_pct`` (default
          ``ExecutionConfig().auction_max_pct_of_auction_volume``, 2%).

        Commissions and regulatory fees are the same per share on both venues and
        are left out (they do not change the choice).  With the D2 defaults the
        auction and continuous impact legs are equal on a normal day, so the pure
        auction wins by ``h - f_a``; it is also the venue the 2% cap would forbid for
        a large order, which is what ``routed`` prices.  ``venue`` is the cheapest of
        the three; ``venue_within_cap`` the cheapest the router may use (``auction`` only
        when ``Q`` fits under the cap, else ``routed`` or ``continuous``).  Also returned: ``auction_share``, ``auction_fee_bps``,
        ``routed_auction_fraction`` (``q_a / Q``) and ``fees_included``.
        """
        from qts.core.config import ExecutionConfig

        share = self.auction_share_for_date(date) if auction_share is None else auction_share
        q = abs(float(notional))
        a = float(adv_usd)
        fee_bps = 0.0
        fees_included = price is not None and np.isfinite(float(price)) and float(price) > 0
        if fees_included:
            fee_bps = 1e4 * float(getattr(self.cfg, "auction_exchange_fee_per_share", 0.0)) / float(price)
        kappa = (
            float(ExecutionConfig().auction_max_pct_of_auction_volume)
            if auction_max_pct is None
            else float(auction_max_pct)
        )

        def c_usd(qq: float, v: float, venue: str) -> float:
            return float(self.cost_bps(qq, v, sigma_daily, venue=venue)) * qq / 1e4 if qq > 0 else 0.0

        cont = float(half_spread_bps) + float(self.cost_bps(q, a, sigma_daily, venue=VENUE_CONTINUOUS))
        auc = (
            float(auction_spread_bps)
            + fee_bps
            + float(self.cost_bps(q, a * share, sigma_daily, venue=VENUE_AUCTION))
        )
        qa = min(q, kappa * share * a) if np.isfinite(a) and a > 0 else 0.0
        qc = q - qa
        routed_usd = (
            (float(auction_spread_bps) + fee_bps) * qa / 1e4
            + float(half_spread_bps) * qc / 1e4
            + c_usd(qa, a * share, VENUE_AUCTION)
            + c_usd(q, a, VENUE_CONTINUOUS)
            - c_usd(qa, a, VENUE_CONTINUOUS)
        )
        routed = 1e4 * routed_usd / q if q > 0 else 0.0
        costs = {VENUE_AUCTION: auc, VENUE_ROUTED: routed, VENUE_CONTINUOUS: cont}
        return {
            "continuous_bps": cont,
            "auction_bps": auc,
            "routed_bps": routed,
            "auction_share": share,
            "auction_fee_bps": fee_bps,
            "routed_auction_fraction": qa / q if q > 0 else 0.0,
            "fees_included": bool(fees_included),
            "venue": min(costs, key=lambda k: costs[k]),
            # respecting the router's auction cap: the pure auction only when Q fits under it
            "venue_within_cap": (
                min(costs, key=lambda k: costs[k])
                if qa >= q
                else min((VENUE_ROUTED, VENUE_CONTINUOUS), key=lambda k: costs[k])
            ),
        }
