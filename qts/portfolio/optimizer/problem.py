"""Problem data for one date: the numeric inputs every solve shares.

:func:`prepare_problem` turns the optimizer's inputs (alpha, risk model output,
current book, cost snapshot, no-trade / forced-exit requests) into a
:class:`_ProblemData`: the traded universe (only names in ``risk.tickers`` are
held or traded; held out-of-model names are forced out), factor loadings and
the PSD factor-covariance root, specific variances, the alignment vector,
factor roles (:func:`resolve_factor_roles`) and the per-name trade caps.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from qts.core.config import PortfolioConfig
from qts.core.risk_fill import out_of_model_specific_var
from qts.core.roles import RISK_ONLY_STYLES
from qts.core.types import RiskModelOutput
from qts.portfolio.costs import Snapshot, align_snapshot

log = logging.getLogger("qts.portfolio.optimizer")


def _psd_factor(F: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Return L with L L' = PSD-clipped symmetric part of F."""
    Fs = 0.5 * (F + F.T)
    vals, vecs = np.linalg.eigh(Fs)
    vals = np.clip(vals, floor, None)
    return vecs * np.sqrt(vals)[None, :]


def _snap_get(snapshot: Any, key: str):
    """Read a per-ticker field from a dict snapshot or an attribute-style snapshot object."""
    try:
        return snapshot[key]
    except (TypeError, KeyError):
        return getattr(snapshot, key)


@dataclass
class _ProblemData:
    """Everything the constraint builder and the solves share (numeric, one date)."""

    tickers: pd.Index
    a: np.ndarray
    w0: np.ndarray
    snap: Snapshot
    factors: list[str]
    Bv: np.ndarray
    F: np.ndarray
    L: np.ndarray
    d: np.ndarray
    sd: np.ndarray
    y: np.ndarray | None
    nu: float
    nav: float
    beta_col: str | None
    ind_cols: list[str]
    sty_cols: list[str]
    wb: np.ndarray | None
    no_trade_mask: np.ndarray
    force_mask: np.ndarray
    force_caps: np.ndarray
    part_cap: np.ndarray
    n_out_of_model: int  # held names outside the risk model in the problem (forced exits)
    in_model: np.ndarray | None = None  # (n,) bool: name in risk.tickers
    full_tickers: pd.Index | None = None  # alpha | w_current | risk tickers (result index)
    excluded_alpha: list[str] = field(default_factory=list)  # out-of-model names with alpha, excluded
    _L_ord: np.ndarray | None = field(default=None, repr=False)

    def variance(self, w: np.ndarray) -> float:
        fw = self.Bv.T @ w
        v = float(fw @ self.F @ fw + (w**2 * self.d).sum())
        if self.y is not None:
            v += self.nu * float(self.y @ w) ** 2
        return v

    def alignment_share(self, w: np.ndarray) -> float:
        v = self.variance(w)
        if self.y is None or v <= 0:
            return 0.0 if v > 0 else float("nan")
        return self.nu * float(self.y @ w) ** 2 / v

    # factor layout for the parametrised problem ------------------------------
    static_cols: list[str] = field(default_factory=list)  # market + industry dummies (constant, sparse)
    dynamic_cols: list[str] = field(default_factory=list)  # styles / statistical (parameter, dense)

    def ordered(self) -> list[str]:
        return list(self.static_cols) + list(self.dynamic_cols)

    def static_block(self) -> np.ndarray:
        return self.cols(self.static_cols)

    def L_ordered(self) -> np.ndarray:
        """``L`` with ``L L' = F`` in the ``[static | dynamic]`` factor order (cached)."""
        if self._L_ord is None:
            idx = [self.factors.index(c) for c in self.ordered()]
            self._L_ord = _psd_factor(self.F[np.ix_(idx, idx)])
        return self._L_ord

    def B_ordered(self) -> np.ndarray:
        """Exposures ``(n, K)`` in the ``[static | dynamic]`` factor order."""
        return self.cols(self.ordered())

    def sigma_inv_quad(self, a: np.ndarray) -> float:
        """``a' Sigma^-1 a`` for ``Sigma = B F B' + D + nu y y'`` (daily), in
        factor form (Woodbury with ``U = [B L, sqrt(nu) y]``, ``F = L L'``):
        ``Sigma^-1 = D^-1 - D^-1 U (I + U' D^-1 U)^-1 U' D^-1``; valid for a
        singular (PSD-clipped) ``F``.  O(N K^2)."""
        U = self.Bv @ self.L
        if self.y is not None and self.nu > 0:
            U = np.column_stack([U, np.sqrt(self.nu) * self.y])
        dinv = 1.0 / self.d
        Da = dinv * a
        UtDa = U.T @ Da
        M = np.eye(U.shape[1]) + (U * dinv[:, None]).T @ U
        try:
            corr = float(UtDa @ np.linalg.solve(M, UtDa))
        except np.linalg.LinAlgError:
            corr = float(UtDa @ np.linalg.lstsq(M, UtDa, rcond=None)[0])
        return max(float(a @ Da) - corr, 0.0)

    def y_scaled(self) -> np.ndarray:
        n = len(self.tickers)
        return np.zeros(n) if self.y is None else np.sqrt(self.nu) * self.y

    def cols(self, names: Sequence[str]) -> np.ndarray:
        if not names:
            return np.zeros((len(self.tickers), 0))
        return self.Bv[:, [self.factors.index(c) for c in names]]

    def offsets(self, names: Sequence[str]) -> np.ndarray:
        if self.wb is None or not names:
            return np.zeros(len(names))
        return self.cols(names).T @ self.wb


# --------------------------------------------------------------------------- factor roles
def resolve_factor_roles(
    cfg: PortfolioConfig,
    risk: RiskModelOutput,
    factors: list[str],
    neutral_styles: Sequence[str] | None,
    industry_columns: Sequence[str] | None,
    notes: list[str],
    has_benchmark: bool = False,
) -> tuple[str | None, list[str], list[str]]:
    """``(beta_col, industry_cols, style_band_cols)`` from
    ``risk.factor_groups``; prefix fallback only when the groups are empty.
    ``neutral_styles=None`` with ``cfg.neutralize_risk_only_styles`` means
    ``RISK_ONLY_STYLES`` (those present in the model; no error for absent
    ones) for long/short books or benchmark-relative solves; an explicit
    list must name existing style factors."""
    groups = getattr(risk, "factor_groups", None) or {}
    has_groups = any(len(v) for v in groups.values()) if isinstance(groups, dict) else False
    beta_name = str(getattr(cfg, "beta_factor_name", "beta"))
    # D1 default: only where neutrality is meaningful -- a dollar-neutral book, or
    # active weights against a benchmark (a fully invested long-only book cannot be
    # style-neutral in absolute terms)
    defaulted = (
        neutral_styles is None
        and bool(getattr(cfg, "neutralize_risk_only_styles", False))
        and (cfg.book == "long_short" or has_benchmark)
    )
    if has_groups:
        style = [c for c in groups.get("style", []) if c in factors]
        industry = [c for c in groups.get("industry", []) if c in factors]
        if defaulted:
            neutral_styles = [s for s in RISK_ONLY_STYLES if s in style]
        unknown = [s for s in (neutral_styles or []) if s not in style]
        if unknown:
            raise ValueError(f"neutral_styles {unknown} are not in factor_groups['style'] = {style}")
        beta_col = beta_name if beta_name in style else None
        if cfg.beta_neutral and beta_col is None:
            notes.append(
                f"beta_neutral requested but style '{beta_name}' not in factor_groups['style']; skipped"
            )
        ind_cols = [c for c in industry_columns if c in factors] if industry_columns else industry
    else:
        log.warning(
            "risk model output has no factor_groups; falling back to name-based factor "
            "role detection (beta = exact '%s', industry = 'IND*' prefix)",
            beta_name,
        )
        notes.append("factor_groups missing: prefix fallback for factor roles")
        low = {str(c).lower(): c for c in factors}
        beta_col = low.get(beta_name.lower())
        if cfg.beta_neutral and beta_col is None:
            notes.append(f"beta_neutral requested but no '{beta_name}' factor in risk model; skipped")
        if industry_columns:
            ind_cols = [c for c in industry_columns if c in factors]
        else:
            ind_cols = [c for c in factors if str(c).upper().startswith("IND")]
        if defaulted:
            neutral_styles = [s for s in RISK_ONLY_STYLES if s in factors]
    if cfg.industry_band is not None and cfg.industry_band > 0 and not ind_cols:
        notes.append("industry_band > 0 but no industry factors identified; band not applied")
    if not (cfg.industry_band is not None and cfg.industry_band > 0):
        ind_cols = []
    # beta already has its own (tighter) band when beta_neutral is on
    sty = [c for c in (neutral_styles or []) if c in factors and not (cfg.beta_neutral and c == beta_col)]
    if not (cfg.style_band_sd is not None and cfg.style_band_sd > 0):
        sty = []
    return (beta_col if cfg.beta_neutral else None), ind_cols, sty


def static_columns(risk: RiskModelOutput, factors: list[str], Bv: np.ndarray) -> list[str]:
    """Factor columns entered as a *constant sparse* block of the cached
    problem: the market and industry groups (0/1 dummies that change only on
    reclassification / universe change, which then triggers a rebuild).
    Without ``factor_groups``: any column whose values are all 0 or 1."""
    groups = getattr(risk, "factor_groups", None) or {}
    named = (
        set(groups.get("market", [])) | set(groups.get("industry", [])) if isinstance(groups, dict) else set()
    )
    out = []
    for j, c in enumerate(factors):
        col = Bv[:, j]
        dummy = bool(np.all((col == 0.0) | (col == 1.0)))
        if (c in named and dummy) or (not named and dummy):
            out.append(c)
    return out


# --------------------------------------------------------------------------- preparation
@dataclass(frozen=True, eq=False)
class _Universe:
    full: pd.Index  # alpha | w_current | risk tickers, sorted (the result index)
    tickers: pd.Index  # the problem's names: in the risk model, or held
    in_model: np.ndarray
    a: np.ndarray
    w0: np.ndarray
    excluded_alpha: list[str]


def _universe(alpha: pd.Series, risk: RiskModelOutput, w_current: pd.Series, notes: list[str]) -> _Universe:
    """Second-pass finding 3: the optimizer holds / trades only names the risk
    model covers.  A name outside it has no factor exposure in the model, so it
    would act as a "free" hedge for the net / beta bands while its real market
    beta is ~1.  Not-held names are dropped from the variable set; held ones
    stay only to be exited (forced to zero), alpha 0."""
    full = pd.Index(alpha.index).union(pd.Index(w_current.index)).union(risk.tickers)
    full = pd.Index(sorted(full))
    a_full = pd.Series(alpha, dtype=float).reindex(full).fillna(0.0).to_numpy()
    w0_full = pd.Series(w_current, dtype=float).reindex(full).fillna(0.0).to_numpy()
    in_model_full = full.isin(risk.tickers)
    keep = in_model_full | (w0_full != 0)
    excl = (~keep) & (a_full != 0)
    excluded_alpha = [str(t) for t in full[excl]]
    if excluded_alpha:
        shown = ", ".join(excluded_alpha[:5]) + (" ..." if len(excluded_alpha) > 5 else "")
        notes.append(
            f"{len(excluded_alpha)} names with alpha outside the risk model excluded "
            f"(not held, not traded): {shown}"
        )
    in_model = in_model_full[keep]
    return _Universe(
        full=full,
        tickers=full[keep],
        in_model=in_model,
        a=np.where(in_model, a_full[keep], 0.0),
        w0=w0_full[keep],
        excluded_alpha=excluded_alpha,
    )


def _specific_var(risk: RiskModelOutput, u: _Universe, notes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """``(d, held_out)``: daily specific variances (names outside the risk
    universe -- held names being exited -- get the shared conservative 90th
    percentile fill and zero factor exposure) and the held-out-of-model mask."""
    d_known = pd.Series(risk.specific_var, dtype=float)
    d = d_known.reindex(u.tickers)
    fill = out_of_model_specific_var(d_known)
    d = d.fillna(fill).clip(lower=1e-10).to_numpy()
    held_out = (~u.in_model) & (u.w0 != 0)
    n_out = int(held_out.sum())
    if n_out:
        notes.append(
            f"{n_out} held names outside the risk model universe forced toward zero; "
            f"specific var set to 90th pct ({fill:.2e}), zero factor exposure"
        )
        log.debug("optimizer: %d held names outside risk universe on %s", n_out, getattr(risk, "date", "?"))
    return d, held_out


def _trade_caps(
    u: _Universe,
    snap: Snapshot,
    nav: float,
    rate: float,
    held_out: np.ndarray,
    no_trade: Sequence[str] | None,
    force_to_zero: Mapping[str, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(part_cap, no_trade_mask, force_mask, force_caps)``: the per-name ADV
    participation cap (weight units), the no-trade mask and the forced-exit
    caps (out-of-model holdings always exit; the caller's cap wins if tighter;
    never more than the ADV cap allows in one day, which keeps the problem
    feasible).  A name in both lists is treated as no-trade."""
    tickers, w0 = u.tickers, u.w0
    adv = np.asarray(pd.Series(_snap_get(snap, "adv_usd")).reindex(tickers).values, dtype=float)
    part_cap = rate * adv / max(nav, 1.0)
    nt = [t for t in (no_trade or []) if t in tickers]
    no_trade_mask = np.isin(tickers, nt)
    req_map = dict(force_to_zero or {})
    for t in tickers[held_out]:
        req_map[t] = min(float(req_map.get(t, 0.0)), 0.0)
    force_mask = np.zeros(len(tickers), dtype=bool)
    force_caps = np.zeros(len(tickers))
    if req_map:
        fz = pd.Series(req_map, dtype=float).reindex(tickers)
        force_mask = fz.notna().to_numpy() & ~no_trade_mask
        req = np.nan_to_num(fz.to_numpy(), nan=np.inf)
        force_caps = np.where(force_mask, np.maximum(np.maximum(req, 0.0), np.abs(w0) - part_cap), 0.0)
    return part_cap, no_trade_mask, force_mask, force_caps


def prepare_problem(
    cfg: PortfolioConfig,
    alpha: pd.Series,
    risk: RiskModelOutput,
    w_current: pd.Series,
    nav: float,
    snapshot: Snapshot,
    neutral_styles: Sequence[str] | None,
    industry_columns: Sequence[str] | None,
    no_trade: Sequence[str] | None,
    force_to_zero: Mapping[str, float] | None,
    benchmark_weights: pd.Series | None,
    notes: list[str],
    part_rate: float | None = None,
) -> _ProblemData:
    """All numeric inputs of one date's solves (module docstring)."""
    u = _universe(alpha, risk, w_current, notes)
    tickers = u.tickers
    snap = align_snapshot(snapshot, tickers)
    B = risk.exposures.reindex(tickers).fillna(0.0)
    factors = list(B.columns)
    Bv = B.values
    F = risk.factor_cov.loc[factors, factors].values
    L = _psd_factor(F)
    d, held_out = _specific_var(risk, u, notes)
    y = None
    nu = 0.0
    if risk.alignment_vector is not None and risk.alignment_var > 0:
        y = pd.Series(risk.alignment_vector, dtype=float).reindex(tickers).fillna(0.0).to_numpy()
        nu = float(risk.alignment_var)
    beta_col, ind_cols, sty = resolve_factor_roles(
        cfg,
        risk,
        factors,
        neutral_styles,
        industry_columns,
        notes,
        has_benchmark=benchmark_weights is not None,
    )
    wb = None
    if benchmark_weights is not None:
        wb = pd.Series(benchmark_weights, dtype=float).reindex(tickers).fillna(0.0).to_numpy()
    rate = float(cfg.max_adv_participation if part_rate is None else part_rate)
    part_cap, no_trade_mask, force_mask, force_caps = _trade_caps(
        u, snap, nav, rate, held_out, no_trade, force_to_zero
    )
    static_cols = static_columns(risk, factors, Bv)
    dynamic_cols = [c for c in factors if c not in set(static_cols)]
    return _ProblemData(
        tickers=tickers,
        a=u.a,
        w0=u.w0,
        snap=snap,
        factors=factors,
        Bv=Bv,
        F=F,
        L=L,
        d=d,
        sd=np.sqrt(d),
        y=y,
        nu=nu,
        nav=float(nav),
        beta_col=beta_col,
        ind_cols=ind_cols,
        sty_cols=sty,
        wb=wb,
        no_trade_mask=no_trade_mask,
        force_mask=force_mask,
        force_caps=force_caps,
        part_cap=part_cap,
        n_out_of_model=int(held_out.sum()),
        in_model=u.in_model,
        full_tickers=u.full,
        excluded_alpha=u.excluded_alpha,
        static_cols=static_cols,
        dynamic_cols=dynamic_cols,
    )
