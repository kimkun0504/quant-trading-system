"""Central configuration.  Defaults follow docs/BUILD_PLAN.md and the review
amendments (lower Phase-1 bar, dollar-neutral long/short book at gross 2.0,
auction impact prefactor 0.3 against auction volume (design decision D2),
momentum crash control, capped factor-momentum tilt).
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
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
    """Deep copy of ``cfg`` with dotted-path overrides applied, e.g.
    ``{"portfolio.risk_aversion": 5, "alpha.lgbm.num_leaves": 15, "seed": 1}``.
    Unknown paths raise ``KeyError`` (same strictness as ``from_dict``)."""
    out = copy.deepcopy(cfg)
    for key, val in overrides.items():
        parts = key.split(".")
        target: Any = out
        for p in parts[:-1]:
            if not hasattr(target, p):
                raise KeyError(f"unknown config section {p!r} in {key!r}")
            target = getattr(target, p)
        if not is_dataclass(target) or parts[-1] not in {f.name for f in fields(target)}:
            raise KeyError(f"unknown config field {key!r}")
        setattr(target, parts[-1], val)
    return out


# ============================================================================ sections
@dataclass
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


@dataclass
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


@dataclass
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


@dataclass
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
    # Label treatment of names that delist inside the forward window (t, t+h]
    # uses ``CostConfig.delisting_haircut`` -- the single source of the
    # delisting assumption for labels and backtest accounting (the former
    # ``AlphaConfig.delisting_return`` was removed; ResearchPipeline passes
    # cfg.cost.delisting_haircut to AlphaModel).


@dataclass
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
    # docs/notes/alignment_nu_calibration.md (synthetic seeds 0-2: calibrated
    # 1.58 / 0.05 / 0.28, median 0.28 -> 0.25; recalibrate on real data).
    alignment_nu_scale: float = 0.25
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


@dataclass
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


@dataclass
class PortfolioConfig(_ConfigBase):
    book: str = "long_short"  # or "long_only", "130_30"
    gross_leverage_max: float = 2.0  # dollar-neutral long/short ~100/100 at gross 2.0; review item 7
    net_exposure_target: float = 0.0
    net_exposure_band: float = 0.05
    max_position_weight: float = 0.02
    max_adv_participation: float = 0.05
    beta_neutral: bool = True
    industry_band: float = 0.03
    style_band_sd: float = 0.10
    risk_aversion: float = 10.0
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
    # --- vol targeting (qts.portfolio.optimizer, finding #28)
    # "projection": solve the alpha problem, then re-solve
    #     min ||w - s w*||^2 + vol_target_projection_cost_weight * TC(w - w0)
    #     over the same constraints plus sqrt(w'Sigma w) <= target  (two solves);
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


@dataclass
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


@dataclass
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
    # Gates that make ``overall`` FAIL when their input is missing.  Any other
    # gate with a missing input is reported as "skipped" (status, reason) and
    # does not count against ``overall``.
    required_gates: list[str] = field(default_factory=lambda: ["phase_sharpe", "suspect", "dsr"])


@dataclass
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

    # ------------------------------------------------------------------ serialisation
    # to_dict / from_dict / to_json / hash / to_yaml / from_yaml: see _ConfigBase.

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
