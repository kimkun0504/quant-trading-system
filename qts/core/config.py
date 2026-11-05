"""Central configuration.  Defaults follow docs/BUILD_PLAN.md and the review
amendments (lower Phase-1 bar, 150/50 default book, auction prefactor 1.5,
momentum crash control, capped factor-momentum tilt).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UniverseConfig:
    top_n_by_mcap: int = 1000
    min_price: float = 5.0
    min_adv_usd: float = 2_000_000.0
    adv_window: int = 63
    exclude_bottom_size_quantile: float = 0.20   # NYSE-breakpoint proxy


@dataclass
class AlphaConfig:
    horizon_days: int = 20
    secondary_horizons: List[int] = field(default_factory=lambda: [5, 60])
    winsor_pct: float = 0.01
    industry_demean: List[str] = field(default_factory=lambda: [
        "value_bp", "value_ep", "quality_gpa", "investment_ag"])
    theme_weights: Dict[str, float] = field(default_factory=lambda: {
        "value": 1.0, "momentum": 1.0, "quality": 1.0, "investment": 1.0,
        "low_risk": 1.0, "short_interest": 1.0})
    factor_momentum_lookback_months: int = 1
    factor_momentum_max_tilt: float = 0.25     # review item 6
    momentum_vol_target_annual: float = 0.12   # vol-scale momentum sleeve only
    momentum_crash_market_lookback_days: int = 504  # review item 8
    ic_shrink: float = 0.5                     # shrink estimated IC by 50%
    default_ic: float = 0.03
    model_seeds: int = 3
    refit_frequency_days: int = 252
    train_min_days: int = 756
    embargo_days: Optional[int] = None         # defaults to horizon_days
    # Label treatment of names that delist inside the forward window (t, t+h]:
    # the forward return is measured to the last available close and then
    # multiplied by (1 + delisting_return).  -0.30 is the conservative
    # CRSP-style assumption for performance-related delistings with a missing
    # delisting return (Shumway 1997); replace with actual delisting returns
    # when the price source carries them.  ``None`` / 0.0 disables the haircut.
    delisting_return: Optional[float] = -0.30


@dataclass
class RiskConfig:
    vol_half_life: int = 84
    corr_half_life: int = 504
    vra_half_life: int = 42
    newey_west_lags_vol: int = 5
    newey_west_lags_corr: int = 2
    specific_half_life: int = 84
    specific_shrink_q: float = 0.1
    min_history_days: int = 252
    styles: List[str] = field(default_factory=lambda: [
        "beta", "size", "momentum", "value", "residual_vol", "liquidity",
        "short_term_reversal", "quality", "investment", "short_interest"])
    use_nonlinear_shrinkage: bool = True
    alignment: bool = True
    # Saxena-Stubbs alignment variance: nu = alignment_nu_scale * N * median style
    # variance (qts.risk.alignment).  Calibrated so the optimized book's bias
    # statistic is ~1: scripts/calibrate_alignment_nu.py,
    # docs/notes/alignment_nu_calibration.md (synthetic seeds 0-2: calibrated
    # 1.58 / 0.05 / 0.28, median 0.28 -> 0.25; recalibrate on real data).
    alignment_nu_scale: float = 0.25
    eigenfactor_adjust: bool = True
    n_eigen_sims: int = 200
    # --- statistical (APCA) companion model on the fundamental residuals
    statistical_companion: bool = True        # build_risk_model -> HybridRiskModel
    max_statistical_factors: int = 5          # cap on select_n_factors
    statistical_n_factors: Optional[int] = None  # fixed k (skips the full-sample count selection)
    statistical_window: int = 252
    statistical_refit_every: int = 21
    missing_factor_alarm_share: float = 0.10  # residual-variance share of PC1 that raises the flag


@dataclass
class CostConfig:
    # explicit fees (IBKR Pro tiered, 2026)
    commission_per_share: float = 0.0035
    commission_min_per_order: float = 0.35
    commission_max_pct_value: float = 0.01
    clearing_per_share: float = 0.0002
    sec_fee_per_dollar_sold: float = 20.60 / 1e6   # from 2026-04-04
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
    # Auction prefactor applies with V = auction volume (~9% of ADV), so the
    # effective impact per dollar vs the continuous session is b_auction/sqrt(0.09)
    # = 3.3x b_auction.  0.5 keeps small auction orders cheaper than crossing the
    # spread while large ones route to in-session algos (plan section 6 table);
    # it is a prior to be recalibrated from own fills via TCALedger.calibrate_b.
    impact_b_auction: float = 0.5
    auction_volume_share_default: float = 0.09    # share of ADV traded in the close
    impact_exponent: float = 1.5                  # integrated sqrt law
    default_half_spread_bps: float = 2.0
    default_borrow_fee_annual: float = 0.005      # GC placeholder; use data when present
    max_participation_adv: float = 0.05
    # --- borrow-fee proxy (qts.tcost.borrow.borrow_fee_proxy); placeholder until
    # IBKR SLB history is logged.  GC below `borrow_proxy_si_gc`, rising convexly
    # to `htb_fee_cap_annual` at/above `borrow_proxy_si_htb` short-interest ratio.
    htb_fee_cap_annual: float = 0.25
    borrow_proxy_si_gc: float = 0.05
    borrow_proxy_si_htb: float = 0.20
    small_cap_borrow_multiplier: float = 2.0      # applied when market cap < small_cap_threshold_usd
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


@dataclass
class PortfolioConfig:
    book: str = "long_short"                      # or "long_only", "130_30"
    gross_leverage_max: float = 2.0               # 150/50 -> 2.0 gross; review item 7
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
    turnover_penalty: float = 0.0                 # extra linear penalty on |x|
    target_vol_annual: Optional[float] = 0.10
    vol_target_scale_up: bool = True              # lever up toward target within hard caps
    effective_holding_days_target: int = 50       # review item 5 (monitored)
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


@dataclass
class ExecutionConfig:
    auction_max_pct_of_auction_volume: float = 0.02
    algo_max_pct_adv: float = 0.05
    loc_offset_bps: float = 3.0
    auction_volume_share_default: float = 0.09   # share of ADV in the close
    urgency_half_life_days_threshold: float = 5.0  # faster signals -> arrival algo


@dataclass
class ValidationConfig:
    n_splits: int = 6
    embargo_days: int = 20
    cpcv_n_groups: int = 8
    cpcv_k_test: int = 2
    dsr_threshold: float = 0.95
    pbo_threshold: float = 0.5
    phase1_min_net_sharpe: float = 0.3            # review item 1
    phase2_min_net_sharpe: float = 0.5
    min_fold_pass_fraction: float = 2 / 3
    hurdle_t_stat: float = 3.0
    # Gates that make ``overall`` FAIL when their input is missing.  Any other
    # gate with a missing input is reported as "skipped" (status, reason) and
    # does not count against ``overall``.
    required_gates: List[str] = field(default_factory=lambda: ["phase_sharpe", "suspect", "dsr"])


@dataclass
class SystemConfig:
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
    pipeline_pbo_configs: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> Dict[str, Any]:
        """Plain nested dict of every field (dataclass ``asdict``)."""
        return asdict(self)

    def to_json(self) -> str:
        """Canonical JSON (sorted keys, no whitespace variance) used for hashing."""
        return json.dumps(self.to_dict(), sort_keys=True, default=_json_default, separators=(",", ":"))

    def hash(self) -> str:
        """Short SHA-256 of the canonical JSON: two configs with identical fields
        share the hash, so the trial ledger can de-duplicate identical re-runs."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


def _json_default(o):
    if isinstance(o, (set, frozenset, tuple)):
        return sorted(o) if isinstance(o, (set, frozenset)) else list(o)
    if hasattr(o, "item"):          # numpy scalars
        return o.item()
    return str(o)
