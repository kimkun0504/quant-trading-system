"""Hybrid risk model: fundamental (USE4-style) + APCA companion on its residuals.

Construction
------------
``HybridRiskModel.fit(md)`` fits the :class:`BarraStyleRiskModel`, then fits a
:class:`StatisticalRiskModel` on the fundamental model's **specific returns**
(the cross-sectional regression residuals).  ``as_of(date)`` returns

    exposures     = [ B_fund | B_stat~ ]
    factor_cov    = blockdiag( F_fund , F_stat )            (see ``cross_block``)
    specific_var  = D_fund * keep,   keep_i = clip(1 - share_i, 0.1, 1)

where ``share_i = b_i' F_stat b_i / (b_i' F_stat b_i + s_i)`` is the
fraction of stock ``i``'s (latent + idiosyncratic) residual variance that
the residual factors explain in the trailing window (``s_i`` the APCA
specific variance, ``b_i`` the loadings chosen by ``loadings``), and
``B_stat~`` is the loading matrix expressed in the fundamental model's
specific-variance units, ``b~_i = b_i * sqrt((1 - keep_i) D_fund_i /
(b_i' F_stat b_i))``.  Hence ``(B~ F B~')_ii = (1 - keep_i) D_fund_i``
(``= share_i D_fund_i`` unless the 10% floor binds, i.e. ``share_i > 0.9``;
finding #17) and the hybrid diagonal ``(1 - keep_i) D_fund_i + keep_i
D_fund_i`` equals the fundamental's **stock by stock**: a diversified random
book keeps its total predicted variance (the added off-diagonal terms
average out), while a book aligned with a residual factor picks up the
covariance the fundamental model cannot see.  The 10% floor keeps every
stock's idiosyncratic variance strictly positive even when a residual
factor explains most of its residual.

Where the residual factor's loadings come from (``loadings``)
--------------------------------------------------------------
The factor *returns* ``F_stat`` are extracted from the residuals.  The
loadings used in the hybrid are, by default (``loadings="total"``), the
time-series regression of each stock's **total** return on ``F_stat`` over
the window; with ``loadings="residual"`` they are the APCA loadings of the
residuals themselves.  The distinction matters because of how a latent
factor with loadings ``g`` enters a cross-sectional regression: each day the
hat matrix ``H`` splits it, ``(I - H) g f_t`` stays in the residuals and
``H g f_t`` leaks into the fundamental factor returns.  Residual loadings
therefore recover only ``(I - H) g``; for a book aligned with ``g`` the
realized return is ``w'g f_t`` and the block model loses the cross term
``2 (w'Hg)(w'(I-H)g)`` -- roughly ``2p/N`` of the aligned variance (p
factors on N stocks; ~35% in the N = 150 test market, a few percent at
N = 1000).  The leak cannot be recovered from factor-return time series
either, because ``H_t`` moves with the style exposures and the leaked
component changes sign over time.  Regressing *total* returns on ``F_stat``
recovers ``g`` directly (the latent return is in the total return in full),
so the aligned book is priced correctly; because the explained share is
computed from the same loadings, the diagonal stays pinned to the
fundamental's either way (the aligned book gives up only the leaked share
through the ``D_fund / (sys + s)`` normalisation, a few percent).

Cross block (``cross_block``)
-----------------------------
With ``loadings="total"`` the factor covariance is block-diagonal: residual
factors are cross-sectionally orthogonal to the exposures by construction,
their time-series correlation with the fundamental factor returns is only
the (sign-varying) leak above, and the total-return loadings already carry
the leaked part, so a cross block would double count.  ``cross_block=True``
estimates the cross block as the weighted correlation of the two factor
return blocks over the window, scaled by the two blocks' own volatilities,
and projects ``F`` to PSD; it is meant for ``loadings="residual"``, where it
recovers the time-stable part of the leak.  Should a residual factor later
become observable, promoting it to a fundamental style removes it from the
residual block automatically.

Factor count (point in time)
----------------------------
``k`` is chosen **at every APCA refit date** by Bai-Ng ``IC_p2`` on that
refit's trailing ``statistical_window`` days of residuals only, floored at 1
and capped at ``cfg.max_statistical_factors`` (``StatisticalRiskModel(select_k=
"bai_ng")``; finding #19 -- the earlier single end-of-sample choice was a
look-ahead).  The output always carries ``cfg.max_statistical_factors``
columns ``stat_01..``; slots beyond the refit's ``k`` have zero exposure and
zero variance, so the ``RiskModelOutput`` shape is stable across refits.
``cfg.statistical_n_factors`` fixes ``k`` instead (output then has exactly
that many columns).  ``n_statistical_factors`` is the number of output
columns; ``factor_count_details["by_refit"]`` holds the per-refit choice.

Alignment factor
----------------
Same Saxena-Stubbs passthrough as the fundamental model, computed against the
hybrid exposures (so alpha spanned by a residual factor is no longer
"unseen" and the alignment term shrinks accordingly).

Missing-factor monitor
----------------------
:func:`missing_factor_monitor` reports, per date, the share of trailing
residual variance carried by the first and first-``k`` principal components
together with the Marchenko-Pastur noise edge, and raises a flag when PC1's
share exceeds ``cfg.missing_factor_alarm_share`` *and* its eigenvalue sits
above the MP edge -- "there is a common factor in the residuals that the
fundamental model does not price".
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from qts.core.config import RiskConfig
from qts.core.types import MarketData, RiskModelOutput
from qts.risk.alignment import alpha_alignment
from qts.risk.covariance import nearest_psd
from qts.risk.model import BarraStyleRiskModel, factor_groups_for
from qts.risk.statistical import (
    StatisticalRiskModel,
    _prepare_panel,
    _spectrum,
    marchenko_pastur_edge,
)

SPECIFIC_FLOOR = 0.10


class HybridRiskModel:
    def __init__(
        self,
        fundamental: BarraStyleRiskModel | None = None,
        cfg: RiskConfig | None = None,
        half_life: float | None = None,
        loadings: str = "total",
        cross_block: bool = False,
        **fundamental_kwargs,
    ):
        """``fundamental`` defaults to ``BarraStyleRiskModel(cfg, **fundamental_kwargs)``.
        ``half_life`` is the EWMA half-life of the statistical block (``None`` =
        equal weights over ``cfg.statistical_window``); ``loadings`` is
        ``"total"`` (regress total returns on the residual factor returns) or
        ``"residual"`` (APCA loadings of the residuals); ``cross_block`` adds
        the estimated fundamental/residual factor cross-covariance (meant for
        ``loadings="residual"``).  See the module docstring."""
        if loadings not in ("total", "residual"):
            raise ValueError("loadings must be 'total' or 'residual'")
        self.cfg = cfg or (fundamental.cfg if fundamental is not None else RiskConfig())
        self.fundamental = (
            fundamental if fundamental is not None else BarraStyleRiskModel(self.cfg, **fundamental_kwargs)
        )
        self.half_life = half_life
        self.loadings = loadings
        self.cross_block = bool(cross_block)
        self.statistical: StatisticalRiskModel | None = None
        self.n_statistical_factors: int = 0
        self.factor_count_details: dict | None = None

    # ------------------------------------------------------------------ passthroughs used by the pipeline
    @property
    def md(self) -> MarketData | None:
        return self.fundamental.md

    @property
    def exposures(self):
        return self.fundamental.exposures

    @property
    def factor_returns(self):
        return self.fundamental.factor_returns

    @property
    def specific_returns(self):
        return self.fundamental.specific_returns

    @property
    def r2(self):
        return self.fundamental.r2

    @property
    def diagnostics(self) -> dict:
        """Fundamental model's cleaning / reference-vol diagnostics."""
        return self.fundamental.diagnostics

    @property
    def cleaning(self):
        return self.fundamental.cleaning

    def exposures_as_of(self, date: pd.Timestamp) -> pd.DataFrame:
        """Fundamental exposures (labels are residualised on these)."""
        return self.fundamental.exposures_as_of(date)

    def residual_covariance_as_of(
        self, date: pd.Timestamp, window: int | None = None, method: str = "qis"
    ) -> pd.DataFrame:
        return self.fundamental.residual_covariance_as_of(date, window, method)

    # ------------------------------------------------------------------ fit
    def fit(self, md: MarketData, dates: Iterable[pd.Timestamp] | None = None) -> HybridRiskModel:
        cfg = self.cfg
        self.fundamental.fit(md, dates)
        assert self.fundamental.specific_returns is not None
        U = self.fundamental.specific_returns.dropna(how="all")
        if cfg.statistical_n_factors is not None:
            k = int(min(max(int(cfg.statistical_n_factors), 1), max(cfg.max_statistical_factors, 1)))
            select_k = None
        else:
            # PIT: k is re-chosen at every refit from that refit's window (finding #19)
            k = max(int(cfg.max_statistical_factors), 1)
            select_k = "bai_ng"
        self.n_statistical_factors = k
        min_obs = min(cfg.statistical_window, max(60, int(0.8 * cfg.statistical_window)))
        self.statistical = StatisticalRiskModel(
            n_factors=k,
            window=cfg.statistical_window,
            half_life=self.half_life,
            min_obs=min_obs,
            method="apca",
            apca_iters=3,
            refit_every=cfg.statistical_refit_every,
            select_k=select_k,
        ).fit(U)
        self.factor_count_details = {
            "method": "fixed" if select_k is None else "bai_ng_per_refit",
            "n_slots": k,
            "by_refit": self.statistical.n_factors_by_refit(),
        }
        return self

    # ------------------------------------------------------------------ as_of
    def statistical_available(self, date: pd.Timestamp) -> bool:
        return self.statistical is not None and self.statistical.available(date)

    def as_of(
        self, date: pd.Timestamp, alpha: pd.Series | None = None, nu_scale: float | None = None
    ) -> RiskModelOutput:
        """Hybrid risk model as of ``date``.  Before the statistical block has
        enough residual history the fundamental output is returned unchanged
        (documented fallback; check :meth:`statistical_available`)."""
        date = pd.Timestamp(date)
        if not self.statistical_available(date):
            return self.fundamental.as_of(date, alpha=alpha, nu_scale=nu_scale)
        assert self.statistical is not None
        fund = self.fundamental.as_of(date)
        st = self.statistical.as_of(date)
        tickers = fund.exposures.index
        cfg = self.cfg

        Bs = st.exposures.reindex(tickers).fillna(0.0).values
        Fs = st.factor_cov.values
        spec_s = st.specific_var.reindex(tickers).values
        has_stat = np.einsum("ik,kl,il->i", Bs, Fs, Bs) > 0
        if self.loadings == "total":
            B_use = np.where(has_stat[:, None], self._total_return_loadings(st, tickers), 0.0)
        else:
            B_use = Bs
        # explained share of the stock's (latent + idiosyncratic) residual variance
        sys_var = np.einsum("ik,kl,il->i", B_use, Fs, B_use)
        tot = sys_var + np.where(np.isfinite(spec_s), spec_s, 0.0)
        share = np.where(tot > 0, sys_var / np.where(tot > 0, tot, 1.0), 0.0)
        keep = np.clip(1.0 - share, SPECIFIC_FLOOR, 1.0)
        D_fund = fund.specific_var.reindex(tickers).values
        D_h = D_fund * keep
        # express the statistical block in the fundamental specific-variance units so
        # that (B~ F B~')_ii = (1 - keep_i) D_fund_i: the hybrid diagonal equals the
        # fundamental's stock by stock, only the off-diagonal structure is added
        scale = np.where(
            sys_var > 0, np.sqrt((1.0 - keep) * D_fund / np.where(sys_var > 0, sys_var, 1.0)), 0.0
        )
        Bs_scaled = B_use * scale[:, None]

        stat_cols = list(st.factor_cov.columns)
        exposures = pd.concat(
            [fund.exposures, pd.DataFrame(Bs_scaled, index=tickers, columns=stat_cols)], axis=1
        )
        exposures.columns.name = "factor"
        K1, K2 = fund.factor_cov.shape[0], len(stat_cols)
        F = np.zeros((K1 + K2, K1 + K2))
        F[:K1, :K1] = fund.factor_cov.values
        F[K1:, K1:] = Fs
        fr = fund.factor_returns.join(st.factor_returns.reindex(fund.factor_returns.index), how="left")
        if self.cross_block:
            F[:K1, K1:] = self._cross_block(fr, fund.factor_cov, st.factor_cov)
            F[K1:, :K1] = F[:K1, K1:].T
            F = nearest_psd(F)
        cols = list(fund.factor_cov.columns) + stat_cols
        factor_cov = pd.DataFrame(F, index=cols, columns=cols)
        spec = pd.Series(D_h, index=tickers, name="specific_var")
        y: pd.Series | None = None
        nu = 0.0
        if alpha is not None and cfg.alignment:
            y, nu = alpha_alignment(
                alpha.reindex(tickers), exposures, spec, nu_scale=nu_scale, factor_cov=factor_cov, cfg=cfg
            )
        return RiskModelOutput(
            date=fund.date,
            exposures=exposures,
            factor_cov=factor_cov,
            specific_var=spec,
            factor_returns=fr,
            alignment_vector=y,
            alignment_var=float(nu),
            factor_groups=factor_groups_for(exposures.columns, cfg, statistical=stat_cols),
        )

    def _total_return_loadings(self, st: RiskModelOutput, tickers: pd.Index) -> np.ndarray:
        """Per-stock OLS of trailing-window *total* returns on the residual
        factor returns (observed days only; stocks with too few observations
        get zero loadings)."""
        from qts.risk.statistical import _prepare_panel, _ts_regress

        assert self.md is not None and self.statistical is not None
        Fh = st.factor_returns.tail(self.cfg.statistical_window)
        rets = self.fundamental.returns_clean  # cleaned panel (md.returns when cleaning is off)
        if rets is None:
            rets = self.md.returns
        Rw = rets.reindex(index=Fh.index, columns=tickers)
        R, mask, _ = _prepare_panel(Rw.values)
        Fv = Fh.values
        active = np.any(Fv != 0, axis=0)  # unused slots (k < n_slots) stay at zero loading
        B = np.zeros((R.shape[1], Fv.shape[1]))
        if active.any():
            B[:, active], _ = _ts_regress(R, mask, Fv[:, active], self.statistical.min_stock_obs)
        return B

    def _cross_block(self, fr: pd.DataFrame, F_fund: pd.DataFrame, F_stat: pd.DataFrame) -> np.ndarray:
        """``diag(sd_fund) rho diag(sd_stat)`` with ``rho`` the weighted
        correlation of the two factor-return blocks over the trailing
        statistical window (same weights as the statistical block)."""
        from qts.risk.statistical import _weights

        X = fr.tail(self.cfg.statistical_window).dropna(how="any")
        K1, K2 = F_fund.shape[0], F_stat.shape[0]
        if len(X) < 20:
            return np.zeros((K1, K2))
        A = X.values
        w = _weights(len(A), self.half_life)
        Ac = A - w @ A
        S = (Ac * w[:, None]).T @ Ac
        sd = np.sqrt(np.clip(np.diag(S), 1e-30, None))
        rho = np.clip(S[:K1, K1:] / np.outer(sd[:K1], sd[K1:]), -1.0, 1.0)
        return rho * np.outer(np.sqrt(np.diag(F_fund.values)), np.sqrt(np.diag(F_stat.values)))

    def explained_share(self, date: pd.Timestamp) -> pd.Series:
        """Per-stock share of fundamental specific variance moved into the residual factors."""
        if self.statistical is None:
            raise RuntimeError("call fit() first")
        return self.statistical.explained_share(date)


# --------------------------------------------------------------------------- monitor
def missing_factor_monitor(
    md: MarketData,
    model,
    lookback: int = 252,
    n_pcs: int | None = None,
    alarm_share: float | None = None,
    step: int = 1,
    min_coverage: float = 0.8,
    standardize: bool = True,
    dates: Iterable[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Per-date "unmodelled common factor" alarm on the fundamental residuals.

    For every ``step``-th date ``t`` (with at least ``lookback // 2`` residual
    days available) take the trailing ``lookback`` days of specific returns of
    stocks observed on >= ``min_coverage`` of those days, demean (and
    standardize by residual vol when ``standardize``), and eigen-decompose the
    sample covariance.  Columns:

    ``n_stocks, n_days, pc1_share, pck_share, lambda1, mp_edge, mp_edge_share,
    n_factors_mp, flag`` where the shares are fractions of total residual
    variance, ``mp_edge`` is the Marchenko-Pastur noise edge with ``sigma^2``
    re-estimated from the bulk, and ``flag = (pc1_share > alarm_share) &
    (lambda1 > mp_edge)``.  ``alarm_share`` defaults to
    ``model.cfg.missing_factor_alarm_share``; ``n_pcs`` to the hybrid's factor
    count (or 3).
    """
    fundamental = getattr(model, "fundamental", model)
    cfg = getattr(model, "cfg", None) or RiskConfig()
    if alarm_share is None:
        alarm_share = cfg.missing_factor_alarm_share
    if n_pcs is None:
        n_pcs = getattr(model, "n_statistical_factors", 0) or 3
    U_all = fundamental.specific_returns.dropna(how="all")
    if dates is None:
        cand = U_all.index[lookback // 2 - 1 :: step] if len(U_all) >= lookback // 2 else U_all.index[:0]
    else:
        cand = pd.DatetimeIndex(list(dates))
    rows = []
    for d in cand:
        U = U_all.loc[:d].tail(lookback)
        if len(U) < lookback // 2:
            continue
        cov_ok = U.notna().mean() >= min_coverage
        U = U.loc[:, cov_ok]
        T, N = U.shape
        if N < 10:
            continue
        R, mask, _ = _prepare_panel(U.values)
        if standardize:
            n = np.maximum(mask.sum(axis=0), 1)
            sd = np.sqrt((R**2).sum(axis=0) / n)
            R = R / np.where(sd > 0, sd, 1.0)
        lam = _spectrum(R)
        total = float(lam.sum())
        k_mp, edge, _ = marchenko_pastur_edge(lam, N, T, max_factors=20)
        pc1 = float(lam[0] / total) if total > 0 else np.nan
        pck = float(lam[:n_pcs].sum() / total) if total > 0 else np.nan
        rows.append(
            {
                "date": d,
                "n_stocks": N,
                "n_days": T,
                "pc1_share": pc1,
                "pck_share": pck,
                "lambda1": float(lam[0]),
                "mp_edge": edge,
                "mp_edge_share": edge / total if total > 0 else np.nan,
                "n_factors_mp": k_mp,
                "flag": bool(pc1 > alarm_share and lam[0] > edge),
            }
        )
    out = pd.DataFrame(rows)
    return out.set_index("date") if len(out) else out
