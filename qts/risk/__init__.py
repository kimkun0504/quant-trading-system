"""Risk model layer: Barra/USE4-style fundamental factor model.

Public API
----------
``BarraStyleRiskModel``          fit() / as_of() -> RiskModelOutput
``StatisticalRiskModel``         APCA companion model (qts.risk.statistical)
``HybridRiskModel``              fundamental + APCA on residuals (qts.risk.hybrid)
``build_risk_model``             factory keyed on RiskConfig.statistical_companion
``select_n_factors``, ``missing_factor_monitor``
``build_exposures``, ``exposure_history``
``estimate_factor_returns``, ``estimate_factor_return_history``
``ewma_cov``, ``factor_covariance``, ``specific_risk``
``nonlinear_shrinkage`` (QIS), ``linear_shrinkage``, ``shrunk_residual_covariance``
``alpha_alignment``, ``misalignment_gauge``, ``calibrate_alignment_nu``
``bias_statistic``, ``rolling_bias_stats``, ``portfolio_bias_test``,
``mincer_zarnowitz``, ``qlike``, ``min_variance_oos_test``
"""
from qts.risk.alignment import alpha_alignment, calibrate_alignment_nu, misalignment_gauge
from qts.risk.covariance import (
    ewma_cov, factor_covariance, linear_shrinkage, nonlinear_shrinkage,
    shrunk_residual_covariance, specific_risk,
)
from qts.risk.evaluation import (
    bias_band, bias_statistic, mincer_zarnowitz, min_variance_oos_test,
    portfolio_bias_test, qlike, random_long_short_weights, rolling_bias_stats,
)
from qts.risk.exposures import build_exposures, exposure_history
from qts.risk.factor_returns import (
    CrossSectionalFit, estimate_factor_return_history, estimate_factor_returns,
)
from qts.risk.hybrid import HybridRiskModel, missing_factor_monitor
from qts.risk.model import BarraStyleRiskModel, build_risk_model
from qts.risk.statistical import StatisticalRiskModel, apca, select_n_factors

__all__ = [
    "BarraStyleRiskModel", "StatisticalRiskModel", "HybridRiskModel", "build_risk_model",
    "apca", "select_n_factors", "missing_factor_monitor", "build_exposures", "exposure_history",
    "estimate_factor_returns", "estimate_factor_return_history", "CrossSectionalFit",
    "ewma_cov", "factor_covariance", "specific_risk", "nonlinear_shrinkage",
    "linear_shrinkage", "shrunk_residual_covariance",
    "alpha_alignment", "misalignment_gauge", "calibrate_alignment_nu",
    "bias_statistic", "bias_band", "rolling_bias_stats", "portfolio_bias_test",
    "random_long_short_weights", "mincer_zarnowitz", "qlike", "min_variance_oos_test",
]
