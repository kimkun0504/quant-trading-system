"""Central configuration.  Defaults follow docs/BUILD_PLAN.md and the review
amendments (lower Phase-1 bar, dollar-neutral long/short book at gross 2.0,
auction impact prefactor 0.3 against auction volume (design decision D2),
momentum crash control, capped factor-momentum tilt).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, TypeVar, cast

#: Bumped when the serialised layout changes incompatibly (renamed / removed
#: fields).  A ``schema_version`` key in a dict/YAML is accepted and ignored by ``from_dict``.
CONFIG_SCHEMA_VERSION = 1

_C = TypeVar("_C", bound="_ConfigBase")


# ============================================================================ serialisation
class _ConfigBase:
    """Mixin giving every config dataclass a strict, round-trippable
    serialisation.

    * ``to_dict()``  -> plain nested ``dict`` (tuples become lists, so the
      result is JSON/YAML-safe);
    * ``from_dict(d)`` -> instance; **strict**: an unknown key raises
      ``KeyError`` (a typo in a YAML file must not be silently ignored), a
      missing key takes the dataclass default, nested config dataclasses are
      rebuilt recursively and lists are converted back to tuples where the
      default is a tuple;
    * ``to_yaml()`` / ``from_yaml(path)`` via PyYAML (``safe_dump`` /
      ``safe_load``);
    * ``hash()`` -> first 16 hex chars of SHA-256 of the canonical JSON
      (sorted keys, compact separators) of ``to_dict()`` — stable across
      processes and machines for equal field values.
    """

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))  # type: ignore[call-overload]

    @classmethod
    def from_dict(cls: type[_C], d: Mapping[str, Any] | None) -> _C:
        return _from_dict(cls, d or {}, path=cls.__name__)

    def to_json(self) -> str:
        """Canonical JSON (sorted keys, no whitespace variance) used for hashing."""
        return json.dumps(self.to_dict(), sort_keys=True, default=_json_default, separators=(",", ":"))

    def hash(self) -> str:
        """Short SHA-256 of the canonical JSON: two configs with identical fields
        share the hash, so the trial ledger can de-duplicate identical re-runs."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]

    def to_yaml(self, header: str | None = None) -> str:
        import yaml  # PyYAML (runtime dependency for config files)

        body = yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False, width=100)
        if header:
            body = "".join(f"# {ln}\n" if ln else "#\n" for ln in header.splitlines()) + body
        return body

    @classmethod
    def from_yaml(cls: type[_C], path: str | Path) -> _C:
        import yaml

        with open(path) as f:
            d: Any = yaml.safe_load(f) or {}
        if not isinstance(d, dict):
            raise ValueError(f"{path}: top level of a config file must be a mapping")
        return cls.from_dict(cast(dict[str, Any], d))


def _plain(o: Any) -> Any:
    """Recursively convert tuples -> lists and numpy scalars -> python scalars."""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in cast(dict[Any, Any], o).items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in cast(list[Any], o)]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return o.item()
    return o


def _field_default(f: Any) -> Any:
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    return MISSING


def _from_dict(cls: type[_C], d: Any, path: str) -> _C:
    if not isinstance(d, Mapping):
        raise TypeError(f"{path}: expected a mapping, got {type(d).__name__}")
    d = dict(cast(Mapping[str, Any], d))
    d.pop("schema_version", None)
    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(d) - set(known))
    if unknown:
        raise KeyError(f"{path}: unknown config key(s) {unknown}; known: {sorted(known)}")
    kwargs: dict[str, Any] = {}
    for name, val in d.items():
        default = _field_default(known[name])
        if is_dataclass(default) and isinstance(default, _ConfigBase):
            kwargs[name] = _from_dict(type(default), val or {}, path=f"{path}.{name}")
        elif isinstance(default, tuple) and isinstance(val, list):
            kwargs[name] = tuple(cast(list[Any], val))
        else:
            kwargs[name] = copy.deepcopy(val)
    return cls(**kwargs)


def apply_overrides(cfg: _C, overrides: Mapping[str, Any]) -> _C:
    """Copy of ``cfg`` with dotted-path overrides applied, e.g.
    ``{"portfolio.risk_aversion": 5, "alpha.lgbm.num_leaves": 15, "seed": 1}``.
    Unknown paths raise ``KeyError`` (same strictness as ``from_dict``).  Configs
    are frozen: each override rebuilds the sections on its path with
    ``dataclasses.replace`` (so a section's ``__post_init__`` validation runs
    on the new values); the input is never modified."""
    out = copy.deepcopy(cfg)
    for key, val in overrides.items():
        out = _replace_path(out, key.split("."), val, key)
    return out


def _replace_path(obj: Any, parts: list[str], val: Any, key: str) -> Any:
    if len(parts) > 1:
        if not hasattr(obj, parts[0]):
            raise KeyError(f"unknown config section {parts[0]!r} in {key!r}")
        child = _replace_path(getattr(obj, parts[0]), parts[1:], val, key)
        return replace(obj, **{parts[0]: child})
    if not isinstance(obj, _ConfigBase) or parts[0] not in {f.name for f in fields(cast(Any, obj))}:
        raise KeyError(f"unknown config field {key!r}")
    return replace(cast(Any, obj), **{parts[0]: val})


# ============================================================================ sections
@dataclass(frozen=True)
class UniverseConfig(_ConfigBase):
    top_n_by_mcap: int = 1000
    min_price: float = 5.0
    min_adv_usd: float = 2_000_000.0
    adv_window: int = 63
    exclude_bottom_size_quantile: float = 0.20  # NYSE-breakpoint proxy
    # membership is re-evaluated every ``refresh_days`` trading days and held in
    # between; a member leaves only after ``exit_after_n_fails`` consecutive
    # failed refreshes (turnover hysteresis; qts.data.universe.build_universe_mask)
    refresh_days: int = 21
    exit_after_n_fails: int = 3
    # Refresh grid anchor: "month_start" (default) refreshes on the first trading
    # day of every calendar month (plus the first panel date), so the refresh
    # dates do not depend on the load ``start`` (refresh_days is then unused);
    # "panel_start" is the legacy grid every ``refresh_days`` from the first date.
    refresh_anchor: str = "month_start"


@dataclass(frozen=True)
class LGBMParams(_ConfigBase):
    """LightGBM hyper-parameters of ``qts.alpha.models.LGBMModel`` (sklearn API
    names in parentheses).  Early stopping is on the per-date rank IC of the
    trainer's validation block; with ``refit_on_full`` the model is refit on
    train + validation with the early-stopped round count."""

    n_estimators: int = 600
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_data_in_leaf: int = 200  # (min_child_samples)
    feature_fraction: float = 0.7  # (colsample_bytree)
    bagging_fraction: float = 0.8  # (subsample)
    bagging_freq: int = 1  # (subsample_freq)
    lambda_l2: float = 1.0  # (reg_lambda)
    early_stopping_rounds: int = 100
    refit_on_full: bool = True
    # OpenMP threads (None -> os.cpu_count() at fit time).  Kept at 4: with
    # threads == cores, any competing process stalls LightGBM's OpenMP barriers
    # (measured ~15x slower on a loaded 2-core box), while a mild
    # oversubscription degrades gracefully.  Set to the core count on a
    # dedicated machine.
    n_jobs: int | None = 4


@dataclass(frozen=True)
class MLPParams(_ConfigBase):
    """sklearn ``MLPRegressor`` settings of ``qts.alpha.models.MLPModel``.

    Early stopping is *time-contiguous*: the trainer's (purged) validation
    block when supplied, else the last ``valid_fraction`` of dates.  Training
    runs epoch by epoch (``partial_fit``), keeps the epoch with the best
    validation mean rank IC and stops after ``n_iter_no_change`` epochs
    without improvement; with ``refit_on_full`` a fresh net is then trained on
    train + validation for the selected number of epochs."""

    hidden: tuple[int, ...] = (64, 32)
    alpha: float = 1e-3  # L2 penalty
    max_iter: int = 200  # max epochs
    batch_size: int = 2048
    learning_rate_init: float = 1e-3
    max_train_rows: int | None = 400_000  # random row subsample cap (memory / time)
    n_iter_no_change: int = 10
    valid_fraction: float = 0.1  # fallback validation block (last 10 % of dates)
    refit_on_full: bool = True


@dataclass(frozen=True)
class AlphaConfig(_ConfigBase):
    horizon_days: int = 20
    # Extra label horizons for the IC-decay section of ``ic_report`` (reported
    # alongside ``horizon_days``; diagnostics only, never trained on).
    secondary_horizons: list[int] = field(default_factory=lambda: [5, 60])
    winsor_pct: float = 0.01
    industry_demean: list[str] = field(
        default_factory=lambda: ["value_bp", "value_ep", "quality_gpa", "investment_ag"]
    )
    theme_weights: dict[str, float] = field(
        default_factory=lambda: {
            "value": 1.0,
            "momentum": 1.0,
            "quality": 1.0,
            "investment": 1.0,
            "low_risk": 1.0,
            "short_interest": 1.0,
        }
    )
    factor_momentum_lookback_months: int = 1
    factor_momentum_max_tilt: float = 0.25  # review item 6
    # Momentum sleeve weight x clip(target / realised 63d theme vol, lo, hi):
    # a *bounded* vol scaling (Barroso & Santa-Clara 2015 style), not full vol
    # targeting -- the clip stops the weight exploding in calm regimes and
    # vanishing in turbulent ones (the crash scalar handles the latter).
    momentum_vol_target_annual: float = 0.12
    momentum_vol_scalar_bounds: tuple[float, float] = (0.5, 1.5)
    momentum_vol_window: int = 63
    momentum_crash_market_lookback_days: int = 504  # review item 8
    # Grinold IC (design decision D5): ``ic_shrink`` is the fraction *kept*
    # (0.5 => half).  IC_used = max(0, ic_shrink * IC_hat) where IC_hat is the
    # trailing OOS rank IC; when IC_hat is unavailable (fewer than
    # ``ic_min_obs`` realised dates) IC_hat := ``default_ic``, which is stated
    # *pre-shrink* and shrunk the same way (0.03 x 0.5 = 0.015).
    ic_shrink: float = 0.5
    default_ic: float = 0.03
    ic_window: int = 252  # trailing dates for IC_hat
    ic_min_obs: int = 60  # min realised IC dates for IC_hat
    # combined score = zscore(w * composite + (1 - w) * ML) where both exist
    composite_weight: float = 0.5
    dynamic_weights: bool = True  # factor-momentum / crash / vol theme weights
    half_life_lookback_days: int = 504  # score history for the half-life fit
    model_seeds: int = 3
    valid_days: int = 252  # trailing validation block inside each training window
    # ensemble member weights of ``default_ml_factory`` (rank-averaged, fixed ex ante)
    ensemble_weights: dict[str, float] = field(
        default_factory=lambda: {"lgbm": 0.45, "mlp": 0.35, "ridge": 0.20}
    )
    lgbm: LGBMParams = field(default_factory=LGBMParams)
    mlp: MLPParams = field(default_factory=MLPParams)
    refit_frequency_days: int = 252
    train_min_days: int = 756
    embargo_days: int | None = None  # defaults to horizon_days
    # Causal delisting rule for labels when ``MarketData.delisting_dates`` does not
    # cover a name: a price gap at the label's window end counts as a delisting
    # only once the name has had no close for this many trading days after its
    # last close (decided at that date, never with later prices); a close within
    # the window makes it a temporary halt (qts.alpha.targets.forward_returns).
    delisting_gap_days: int = 63
    # Label treatment of names that delist inside the forward window (t, t+h]
    # uses ``CostConfig.delisting_haircut`` -- the single source of the
    # delisting assumption for labels and backtest accounting (the former
    # ``AlphaConfig.delisting_return`` was removed; ResearchPipeline passes
    # cfg.cost.delisting_haircut to AlphaModel).


@dataclass(frozen=True)
class RiskConfig(_ConfigBase):
    vol_half_life: int = 84
    corr_half_life: int = 504
    vra_half_life: int = 42
    newey_west_lags_vol: int = 5
    newey_west_lags_corr: int = 2
    specific_half_life: int = 84
    specific_shrink_q: float = 0.1
    min_history_days: int = 252
    styles: list[str] = field(
        default_factory=lambda: [
            "beta",
            "size",
            "momentum",
            "value",
            "residual_vol",
            "liquidity",
            "short_term_reversal",
            "quality",
            "investment",
            "short_interest",
        ]
    )
    # (``use_nonlinear_shrinkage`` removed, finding #16: it never affected
    # ``specific_var``, and a QIS-diagonal prior for thin-history specific risk
    # lost to the USE4 structural model out of sample -- QLIKE 0.07 vs 0.15-0.47
    # on synthetic seeds 0-2.  ``shrunk_residual_covariance(method=...)`` picks
    # QIS or Ledoit-Wolf for evaluation use.)
    alignment: bool = True
    # Saxena-Stubbs alignment variance: nu = alignment_nu_scale * N * median style
    # variance (qts.risk.alignment).  Calibrated so the optimized book's bias
    # statistic is ~1: scripts/calibrate_alignment_nu.py,
    # docs/notes/alignment_nu_calibration.md.  Recalibrated 2026-11-05 with the
    # production settings (gamma-search sizing, signal_lag_days=1, no out-of-model
    # names): synthetic seeds 0-2 all at the search floor 0.05 (bias 0.87 / 0.85 /
    # 0.88; already < 1 at nu = 0) -> 0.05.  Recalibrate on real data.
    alignment_nu_scale: float = 0.05
    eigenfactor_adjust: bool = True
    n_eigen_sims: int = 200
    # USE4 empirical scaling of the simulated eigenfactor volatility bias,
    # v_adj = a (v - 1) + 1 (Menchero, Orr & Wang 2011, a = 1.2): the Monte Carlo
    # understates the bias because real factor returns are fat-tailed and
    # non-stationary.  1.0 = raw simulated bias.
    eigen_inflation_a: float = 1.2
    # Fit a parabola v(k) across eigenfactor index before the inflation (USE4
    # smooths the noisy per-eigenfactor bias this way).
    eigen_parabola_smoothing: bool = False
    # --- statistical (APCA) companion model on the fundamental residuals
    statistical_companion: bool = True  # build_risk_model -> HybridRiskModel
    max_statistical_factors: int = 5  # cap on select_n_factors
    statistical_n_factors: int | None = None  # fixed k; None = Bai-Ng per refit, capped (PIT)
    statistical_window: int = 252
    statistical_refit_every: int = 21
    missing_factor_alarm_share: float = 0.10  # residual-variance share of PC1 that raises the flag
    # storage dtype of the (T, N, S) style-exposure tensor (qts.risk.exposures.ExposureHistory);
    # "float32" halves its memory at ~1e-7 relative error in the exposures.
    exposure_dtype: str = "float64"


#: ``(low, high, low_inclusive, high_inclusive)``; ``None`` = unbounded on that side.
FieldBound = tuple[float | None, float | None, bool, bool]


def check_bounds(obj: object, bounds: Mapping[str, FieldBound]) -> list[str]:
    """Messages for every field of ``obj`` in ``bounds`` that is not a finite real
    number inside its interval (empty list = all good).  ``None`` values are skipped
    (optional fields)."""
    errs: list[str] = []
    for name, (lo, hi, lo_inc, hi_inc) in bounds.items():
        v: Any = getattr(obj, name)
        if v is None:
            continue
        if isinstance(v, bool) or (not isinstance(v, (int, float)) and not hasattr(v, "__float__")):
            errs.append(f"{name} must be a number, got {v!r}")
            continue
        x = float(v)
        if not math.isfinite(x):
            errs.append(f"{name} must be finite, got {v!r}")
            continue
        if lo is not None and (x < lo or (not lo_inc and x == lo)):
            errs.append(f"{name} must be {'>=' if lo_inc else '>'} {lo}, got {v!r}")
        elif hi is not None and (x > hi or (not hi_inc and x == hi)):
            errs.append(f"{name} must be {'<=' if hi_inc else '<'} {hi}, got {v!r}")
    return errs


@dataclass(frozen=True)
class CostConfig(_ConfigBase):
    # explicit fees (IBKR Pro tiered, 2026)
    commission_per_share: float = 0.0035
    commission_min_per_order: float = 0.35
    commission_max_pct_value: float = 0.01
    clearing_per_share: float = 0.0002
    sec_fee_per_dollar_sold: float = 20.60 / 1e6  # from 2026-04-04
    # FINRA TAF 2026 rate (SR-FINRA-2024-019 fee-adjustment schedule, kept
    # through 2028 by SR-FINRA-2026-020); dated history in qts.tcost.fees.
    finra_taf_per_share_sold: float = 0.000195
    finra_taf_max_per_trade: float = 9.79
    cat_fee_per_share: float = 0.000003
    # NYSE/Nasdaq closing-auction exchange fee, per share, applied to MOC/LOC
    # (auction-venue) fills only.  Not a regulatory fee; pass-through from IBKR.
    auction_exchange_fee_per_share: float = 0.001
    # implicit
    impact_b_continuous: float = 1.0
    # Auction prefactor (design decision D2).  It applies with V = auction volume
    # (auction_share x ADV, ~9% on a normal day), so the effective coefficient per
    # sqrt(Q/ADV) is b_auction / sqrt(0.09) = 0.3 / 0.3 = 1.0 = impact_b_continuous:
    # the same impact as the continuous session, and the auction then saves the
    # half-spread -- consistent with the router sending small orders to the close.
    # (0.5 gave an effective 1.67, making every auction trade above ~0.02% ADV
    # dearer than continuous, contradicting the router.)  On heavy days (20% / 34%
    # share) the auction is cheaper still.  A prior: recalibrate from own fills via
    # TCALedger.calibrate_b_by_venue.
    impact_b_auction: float = 0.3
    # THE single source for the normal-day closing-auction share of ADV (BMLL ~9%);
    # heavy days via qts.core.costs.auction_share_for_date.  Read by the cost
    # model, router, fill simulator, dry-run broker and TCA ledger.
    auction_volume_share_default: float = 0.09
    # Exponent e of the integrated impact cost C(Q) = b*sigma*Q^e / V^(e-1)
    # (average impact per dollar b*sigma*(Q/V)^(e-1)); 1.5 = square-root law.
    # Must satisfy 1 < e <= 2 (validated by SquareRootImpact).
    impact_exponent: float = 1.5
    default_half_spread_bps: float = 2.0
    default_borrow_fee_annual: float = 0.005  # GC placeholder; use data when present
    max_participation_adv: float = 0.05
    # --- borrow-fee proxy (qts.tcost.borrow.borrow_fee_proxy); placeholder until
    # IBKR SLB history is logged.  GC below `borrow_proxy_si_gc`, rising convexly
    # to `htb_fee_cap_annual` at/above `borrow_proxy_si_htb` short-interest ratio.
    htb_fee_cap_annual: float = 0.25
    borrow_proxy_si_gc: float = 0.05
    borrow_proxy_si_htb: float = 0.20
    small_cap_borrow_multiplier: float = 2.0  # applied when market cap < small_cap_threshold_usd
    small_cap_threshold_usd: float = 2e9
    # --- backtest accounting
    # Return applied on the final day to a held name whose next close is NaN
    # (delisting / permanent halt), against the position's sign.  Conservative:
    # true recoveries range from 0 (bankruptcy) to a premium (cash takeover).
    delisting_haircut: float = -0.30
    # Multi-day impact persistence: day k of a same-direction streak is charged
    # the incremental impact of extending a k-day metaorder (see
    # TransactionCostModel.estimate).  The optimizer still sees the one-day marginal.
    impact_persistence: bool = True
    # TCA reversion diagnostic reference price: "next_close" (BUILD_PLAN
    # convention, default) or "next_open" (TCALedger.decompose).
    tca_reversion_reference: str = "next_close"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> CostConfig:
        """Semantic bounds (:data:`COST_CONFIG_BOUNDS`) plus ``gc < htb`` and the
        reversion enum; raises ``ValueError`` naming every offending field.  Runs on
        construction (so ``from_dict`` / ``SystemConfig.validate()`` check it);
        call it again after mutating fields in place."""
        errs = check_bounds(self, COST_CONFIG_BOUNDS)
        if not errs and self.borrow_proxy_si_gc >= self.borrow_proxy_si_htb:
            errs.append("borrow_proxy_si_gc must be < borrow_proxy_si_htb")
        if self.tca_reversion_reference not in TCA_REVERSION_REFERENCES:
            errs.append(f"tca_reversion_reference must be one of {list(TCA_REVERSION_REFERENCES)}")
        if errs:
            raise ValueError("CostConfig: " + "; ".join(errs))
        return self


TCA_REVERSION_REFERENCES = ("next_close", "next_open")

#: Semantic bounds of :class:`CostConfig` (checked on construction; the API's config
#: schema publishes them as ``minimum`` / ``exclusiveMinimum`` / ...).  Fees, spreads and
#: borrow are non-negative (a negative fee would be booked as a rebate); the upper bounds
#: only reject absurd values (e.g. a typo of 1e12).
COST_CONFIG_BOUNDS: dict[str, FieldBound] = {
    "commission_per_share": (0.0, 1.0, True, True),
    "commission_min_per_order": (0.0, 100.0, True, True),
    "commission_max_pct_value": (0.0, 1.0, True, True),
    "clearing_per_share": (0.0, 1.0, True, True),
    "sec_fee_per_dollar_sold": (0.0, 0.01, True, True),
    "finra_taf_per_share_sold": (0.0, 1.0, True, True),
    "finra_taf_max_per_trade": (0.0, 1000.0, True, True),
    "cat_fee_per_share": (0.0, 1.0, True, True),
    "auction_exchange_fee_per_share": (0.0, 1.0, True, True),
    "impact_b_continuous": (0.0, 10.0, False, True),
    "impact_b_auction": (0.0, 10.0, False, True),
    "auction_volume_share_default": (0.0, 1.0, False, True),
    "impact_exponent": (1.0, 2.0, False, True),
    "default_half_spread_bps": (0.0, 1000.0, True, True),
    "default_borrow_fee_annual": (0.0, 5.0, True, True),
    "max_participation_adv": (0.0, 1.0, False, True),
    "htb_fee_cap_annual": (0.0, 5.0, True, True),
    "borrow_proxy_si_gc": (0.0, 1.0, True, True),
    "borrow_proxy_si_htb": (0.0, 1.0, False, True),
    "small_cap_borrow_multiplier": (0.0, 100.0, True, True),
    "small_cap_threshold_usd": (0.0, None, True, True),
    "delisting_haircut": (-1.0, 1.0, True, True),
}


@dataclass(frozen=True)
class PortfolioConfig(_ConfigBase):
    book: str = "long_short"  # or "long_only", "130_30"
    gross_leverage_max: float = 2.0  # dollar-neutral long/short ~100/100 at gross 2.0; review item 7
    net_exposure_target: float = 0.0
    net_exposure_band: float = 0.05
    max_position_weight: float = 0.02
    # Per-name daily trade cap |x_i| <= p * ADV_i / NAV.  p = max_adv_participation,
    # or -- with participation_from_urgency -- the router's urgency-derived algo
    # participation ExecutionRouter.participation(half_life) (1-5 %, 1.3 % at a 20-day
    # half-life; qts.execution.router), so the optimizer never plans a same-day trade
    # the router would flag as multi-day.  Off by default: it lowers turnover and the
    # transfer coefficient for slow signals (docs/notes/participation_alignment.md).
    max_adv_participation: float = 0.05
    participation_from_urgency: bool = False
    beta_neutral: bool = True
    industry_band: float = 0.03
    style_band_sd: float = 0.10
    # Sizing (second-pass audit finding 1).  ``risk_aversion_mode``:
    #   "target_vol" (default): gamma is chosen per solve so the optimal book's
    #       ex-ante annual vol equals target_vol_annual (subject to every constraint;
    #       search on log gamma from the closed form gamma0 = IR / (2 target), see
    #       SinglePeriodOptimizer); vol_target_mode is not used.  With
    #       target_vol_annual=None it falls back to the fixed risk_aversion.
    #   "fixed": gamma = risk_aversion; the target, if any, is then enforced by
    #       vol_target_mode ("projection" / "constraint").
    risk_aversion_mode: str = "target_vol"
    # Annual risk aversion of the objective alpha'w - gamma*H*w'Sigma_d*w ("fixed"
    # mode; fallback otherwise).  2 ~ IR / (2 x 10 % target) on the measured alpha.
    risk_aversion: float = 2.0
    trade_aversion: float = 1.0
    hold_aversion: float = 1.0
    turnover_penalty: float = 0.0  # extra linear penalty on |x|
    target_vol_annual: float | None = 0.10
    vol_target_scale_up: bool = True  # lever up toward target within hard caps
    effective_holding_days_target: int = 50  # review item 5 (monitored)
    solver: str = "CLARABEL"
    # --- beta neutrality: the *style* factor named ``beta_factor_name`` (resolved
    # through ``RiskModelOutput.factor_groups["style"]``, never the unit-exposure
    # market/country factor) is held inside +-beta_band (sd units of the
    # standardised style; with a dollar-neutral book this bounds raw beta too).
    beta_factor_name: str = "beta"
    beta_band: float = 0.05
    # If the problem is infeasible *with* the ADV participation caps, re-solve
    # without them (opt-in; default off = report the failure and hold the book).
    relax_participation_on_infeasible: bool = False
    # Names that drop out of the eligible universe while held are liquidated
    # linearly over this many trading days (simulator ``force_to_zero`` hook).
    exit_ineligible_days: int = 5
    # --- vol targeting in risk_aversion_mode="fixed" (qts.portfolio.optimizer, finding #28)
    # "projection": solve the alpha problem, then re-solve
    #     min gamma*H*(w - s w*)'Sigma(w - s w*) + vol_target_projection_cost_weight * TC(w - w0)
    #     over the same constraints plus sqrt(w'Sigma w) <= target  (two solves;
    #     risk-aversion-weighted tracking variance, so both terms are NAV fractions);
    # "constraint": one solve of the alpha problem with the SOC risk cap
    #     sqrt(w'Sigma w) <= target added (no scale-up: risk_aversion sets the level
    #     below the cap).
    vol_target_mode: str = "projection"
    vol_target_projection_cost_weight: float = 1.0
    # Garleanu-Pedersen alpha shrinkage g = 1 / (1 + holding_days / half_life) with the
    # cost-implied holding period (qts.portfolio.alpha_scaling); needs the alpha
    # half-life passed to ``SinglePeriodOptimizer.solve(alpha_half_life_days=...)``.
    # Off until validated.
    use_gp_shrink: bool = False
    # Design decision D1: with ``neutral_styles=None`` the optimizer bands every
    # risk-only style of qts.core.roles.RISK_ONLY_STYLES present in the risk model.
    neutralize_risk_only_styles: bool = True
    # Reuse one DPP-parametrised cvxpy problem across solves while the ticker set
    # and constraint structure are unchanged (canonicalise once).  Off by default:
    # with cvxpy 1.9 the DPP compile of this problem (3/2-power cost cones + N-vector
    # parameters) needs O(N^2) memory (0.8 GB at N=300, 2.9 GB at N=600, > 7 GB at
    # N=1000) while canonicalisation is only ~20% of a daily solve; False builds the
    # same parametrised problem and canonicalises it with values as constants.
    reuse_problem: bool = False


@dataclass(frozen=True)
class ExecutionConfig(_ConfigBase):
    auction_max_pct_of_auction_volume: float = 0.02
    algo_max_pct_adv: float = 0.05
    loc_offset_bps: float = 3.0
    # (the auction share of ADV lives in CostConfig.auction_volume_share_default only)
    urgency_half_life_days_threshold: float = 5.0  # faster signals -> arrival algo
    # Execution delay (design decision D4): signals, risk and the optimizer's cost
    # snapshot use data as of the trading day `signal_lag_days` before the
    # execution date t; trades still execute at the close of t and are charged
    # costs on t's liquidity.  0 = close-as-signal (optimistic bound); 1 = default
    # that bounds the close-as-signal bias.  DailyBarSimulator.run(signal_lag_days=) overrides.
    # Default 1 (decision D4, docs/notes/lag_investigation_2026-11-05.md): real MOC/LOC
    # orders must be submitted before the 15:50 ET cutoff, so a strategy cannot use the
    # official close it trades at.  Lag 0 remains available as an optimistic bound.
    signal_lag_days: int = 1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> ExecutionConfig:
        """Semantic bounds (:data:`EXECUTION_CONFIG_BOUNDS`; ``signal_lag_days``
        must be an integer); raises ``ValueError`` naming every offending field."""
        errs = check_bounds(self, EXECUTION_CONFIG_BOUNDS)
        lag: Any = self.signal_lag_days
        if isinstance(lag, bool) or not float(lag).is_integer():
            errs.append(f"signal_lag_days must be an integer, got {lag!r}")
        if errs:
            raise ValueError("ExecutionConfig: " + "; ".join(errs))
        return self


#: Semantic bounds of :class:`ExecutionConfig` (see :data:`COST_CONFIG_BOUNDS`).
EXECUTION_CONFIG_BOUNDS: dict[str, FieldBound] = {
    "auction_max_pct_of_auction_volume": (0.0, 1.0, False, True),
    "algo_max_pct_adv": (0.0, 1.0, False, True),
    "loc_offset_bps": (0.0, 1000.0, True, True),
    "urgency_half_life_days_threshold": (0.0, 1000.0, True, True),
    "signal_lag_days": (0.0, 10.0, True, True),
}


@dataclass(frozen=True)
class ValidationConfig(_ConfigBase):
    # Fold construction for the pipeline's fold-Sharpe gate (``net_sharpe_delta``):
    # ``PurgedWalkForward(n_splits, embargo_days, label_horizon=H)`` test blocks
    # over the backtest's return dates; blocks shorter than ``min_fold_days``
    # are merged away by reducing the number of splits.
    n_splits: int = 6
    embargo_days: int = 20
    min_fold_days: int = 20
    cpcv_n_groups: int = 8
    cpcv_k_test: int = 2
    dsr_threshold: float = 0.95
    pbo_threshold: float = 0.5
    phase1_min_net_sharpe: float = 0.3  # review item 1
    phase2_min_net_sharpe: float = 0.5
    min_fold_pass_fraction: float = 2 / 3
    hurdle_t_stat: float = 3.0
    # ``failed_solves`` gate: FAIL when the share of rebalance solves that did not
    # reach an optimal status (the simulator then holds the book) exceeds this.  A
    # broken optimizer must not produce a passing "hold everything" backtest.
    max_failed_solve_frac: float = 0.02
    # Gates that make ``overall`` FAIL when their input is missing.  Any other
    # gate with a missing input is reported as "skipped" (status, reason) and
    # does not count against ``overall``.
    required_gates: list[str] = field(default_factory=lambda: ["phase_sharpe", "suspect", "dsr"])


#: Stages :class:`CacheConfig` may list (``qts.runs.cache.CACHEABLE_STAGES``).
CACHE_STAGES = ("risk_fit", "alpha_fit")


@dataclass(frozen=True)
class CacheConfig(_ConfigBase):
    """Content-addressed stage cache (:mod:`qts.runs.cache`).  **Operational
    only**: a cache hit returns bitwise the object a fresh fit would build, so
    this section is excluded from :meth:`SystemConfig.hash` (turning the cache
    on or moving it changes no number and no config hash).

    ``enabled``: off by default.  ``root``: cache directory; ``""`` means
    ``<runs_root>/_cache`` (``runs/_cache`` when the pipeline has no runs root).
    ``stages``: which of ``CACHE_STAGES`` to cache."""

    enabled: bool = False
    root: str = ""
    stages: list[str] = field(default_factory=lambda: list(CACHE_STAGES))

    def __post_init__(self) -> None:
        unknown = sorted(set(self.stages) - set(CACHE_STAGES))
        if unknown:
            raise ValueError(f"cache.stages: unknown stage(s) {unknown}; known: {list(CACHE_STAGES)}")


@dataclass(frozen=True)
class SystemConfig(_ConfigBase):
    """Every behavioural knob of a run.  Serialise with ``to_dict``/``to_yaml``,
    rebuild with ``from_dict``/``from_yaml`` (strict), identify with ``hash()``.
    ``configs/default.yaml`` is generated from these defaults by
    ``python -m qts.cli config dump`` and a test fails if it drifts."""

    universe: UniverseConfig = field(default_factory=UniverseConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    trading_days_per_year: int = 252
    seed: int = 0
    # Optional CSCV/PBO grid for ``ResearchPipeline``: a list of {"<section>.<field>": value, ...}
    # overrides (e.g. [{"portfolio.risk_aversion": 5}, {"portfolio.risk_aversion": 20}]).
    # When empty the PBO gate is reported as skipped (a single configuration has no PBO).
    pipeline_pbo_configs: list[dict[str, Any]] = field(default_factory=lambda: cast(list[dict[str, Any]], []))
    # Operational settings that cannot change a number; excluded from hash().
    cache: CacheConfig = field(default_factory=CacheConfig)

    #: Sections left out of :meth:`hash` (operational, not behavioural).
    HASH_EXCLUDED_SECTIONS = ("cache",)

    # ------------------------------------------------------------------ serialisation
    # to_dict / from_dict / to_json / hash / to_yaml / from_yaml: see _ConfigBase.

    def hash(self) -> str:
        """Short SHA-256 of the canonical JSON **without** the operational
        sections (:attr:`HASH_EXCLUDED_SECTIONS`): enabling the stage cache
        changes no result, so it must not change the config hash, the trial
        ledger's de-duplication key or the golden file."""
        d = {k: v for k, v in self.to_dict().items() if k not in self.HASH_EXCLUDED_SECTIONS}
        js = json.dumps(d, sort_keys=True, default=_json_default, separators=(",", ":"))
        return hashlib.sha256(js.encode()).hexdigest()[:16]

    def with_overrides(self, overrides: Mapping[str, Any]) -> SystemConfig:
        """Deep copy with dotted-path overrides (``apply_overrides``)."""
        return apply_overrides(self, overrides)

    def validate(self) -> SystemConfig:
        """Round-trip check: ``from_dict(to_dict())`` must reproduce the hash
        (catches non-serialisable values set programmatically).  Returns self."""
        again = SystemConfig.from_dict(self.to_dict())
        if again.hash() != self.hash():
            raise ValueError("config does not round-trip through to_dict/from_dict")
        return self


DEFAULT_YAML_HEADER = (
    "GENERATED FILE - do not edit by hand.\n"
    "Defaults of qts.core.config.SystemConfig, written by:\n"
    "    python -m qts.cli config dump > configs/default.yaml\n"
    "tests/test_config.py fails when this file differs from the dump."
)


def dump_default_yaml() -> str:
    """The text of ``configs/default.yaml`` (header + ``SystemConfig().to_yaml()``)."""
    return SystemConfig().to_yaml(header=DEFAULT_YAML_HEADER)


def _json_default(o: Any) -> Any:
    if isinstance(o, (set, frozenset)):
        return sorted(cast(Any, o))
    if isinstance(o, tuple):
        return list(cast(tuple[Any, ...], o))
    if hasattr(o, "item"):  # numpy scalars
        return o.item()
    return str(o)
