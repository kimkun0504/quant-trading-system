"""Validation layer: purged CV, deflated Sharpe, PBO, trial ledger, hurdles, leakage tests."""

from qts.validation.deflated_sharpe import (
    annualize_sharpe,
    deannualize_sharpe,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_annual,
    dsr_from_trial_returns,
    expected_max_sharpe,
    min_backtest_length,
    min_backtest_length_years,
    min_track_record_length,
    moments_from_returns,
    probabilistic_sharpe,
    sharpe_from_returns,
    sharpe_ratio_se,
)
from qts.validation.hurdles import GateResult, HurdleReport, HurdleResult
from qts.validation.leakage import (
    FEATURE_REGISTRY,
    LeakageError,
    feature_registry,
    forward_return,
    label_overlap_check,
    pit_shift_test,
    planted_oracle_test,
    simple_rank_backtest,
)
from qts.validation.ledger import TrialLedger
from qts.validation.pbo import probability_of_backtest_overfitting
from qts.validation.splits import (
    CombinatorialPurgedCV,
    PurgedKFold,
    PurgedWalkForward,
    check_no_overlap,
    label_windows_overlap,
    purge_train,
)
from qts.validation.synthetic_audit import generate_noise_panel, nagel_collapse_test

__all__ = [
    "FEATURE_REGISTRY",
    "CombinatorialPurgedCV",
    "GateResult",
    "HurdleReport",
    "HurdleResult",
    "LeakageError",
    "PurgedKFold",
    "PurgedWalkForward",
    "TrialLedger",
    "annualize_sharpe",
    "check_no_overlap",
    "deannualize_sharpe",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_annual",
    "dsr_from_trial_returns",
    "expected_max_sharpe",
    "feature_registry",
    "forward_return",
    "generate_noise_panel",
    "label_overlap_check",
    "label_windows_overlap",
    "min_backtest_length",
    "min_backtest_length_years",
    "min_track_record_length",
    "moments_from_returns",
    "nagel_collapse_test",
    "pit_shift_test",
    "planted_oracle_test",
    "probabilistic_sharpe",
    "probability_of_backtest_overfitting",
    "purge_train",
    "sharpe_from_returns",
    "sharpe_ratio_se",
    "simple_rank_backtest",
]
