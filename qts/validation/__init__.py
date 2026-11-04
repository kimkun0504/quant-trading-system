"""Validation layer: purged CV, deflated Sharpe, PBO, trial ledger, hurdles, leakage tests."""
from qts.validation.splits import (  # noqa: F401
    CombinatorialPurgedCV, PurgedKFold, PurgedWalkForward, check_no_overlap, label_windows_overlap, purge_train)
from qts.validation.deflated_sharpe import (  # noqa: F401
    annualize_sharpe, deannualize_sharpe, deflated_sharpe_ratio, deflated_sharpe_ratio_annual,
    dsr_from_trial_returns, expected_max_sharpe, min_backtest_length, min_backtest_length_years,
    min_track_record_length, moments_from_returns, probabilistic_sharpe, sharpe_from_returns, sharpe_ratio_se)
from qts.validation.pbo import probability_of_backtest_overfitting  # noqa: F401
from qts.validation.ledger import TrialLedger  # noqa: F401
from qts.validation.hurdles import GateResult, HurdleReport  # noqa: F401
from qts.validation.leakage import (  # noqa: F401
    FEATURE_REGISTRY, LeakageError, feature_registry, forward_return, label_overlap_check,
    pit_shift_test, planted_oracle_test, simple_rank_backtest)
from qts.validation.synthetic_audit import generate_noise_panel, nagel_collapse_test  # noqa: F401

__all__ = [
    "CombinatorialPurgedCV", "PurgedKFold", "PurgedWalkForward", "check_no_overlap", "label_windows_overlap",
    "purge_train", "annualize_sharpe", "deannualize_sharpe", "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_annual", "dsr_from_trial_returns", "expected_max_sharpe", "min_backtest_length",
    "min_backtest_length_years", "min_track_record_length", "moments_from_returns", "probabilistic_sharpe",
    "sharpe_from_returns", "sharpe_ratio_se", "probability_of_backtest_overfitting", "TrialLedger",
    "GateResult", "HurdleReport", "FEATURE_REGISTRY", "LeakageError", "feature_registry", "forward_return",
    "label_overlap_check", "pit_shift_test", "planted_oracle_test", "simple_rank_backtest",
    "generate_noise_panel", "nagel_collapse_test",
]
