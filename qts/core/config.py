"""Central configuration.  Defaults follow docs/BUILD_PLAN.md and the review
amendments (lower Phase-1 bar, 150/50 default book, auction prefactor 1.5,
momentum crash control, capped factor-momentum tilt).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
    eigenfactor_adjust: bool = True
    n_eigen_sims: int = 200


@dataclass
class CostConfig:
    # explicit fees (IBKR Pro tiered, 2026)
    commission_per_share: float = 0.0035
    commission_min_per_order: float = 0.35
    commission_max_pct_value: float = 0.01
    clearing_per_share: float = 0.0002
    sec_fee_per_dollar_sold: float = 20.60 / 1e6   # from 2026-04-04
    finra_taf_per_share_sold: float = 0.000166
    finra_taf_max_per_trade: float = 8.30
    cat_fee_per_share: float = 0.000003
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
