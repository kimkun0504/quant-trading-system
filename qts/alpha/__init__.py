"""Alpha model layer: features, targets, composite, ML models, walk-forward
training, evaluation and the ``AlphaModel`` orchestrator."""
from .alpha_model import AlphaModel, default_ml_factory
from .composite import (
                        THEMES,
                        ThemeComposite,
                        factor_momentum_tilt,
                        momentum_crash_scalar,
                        theme_portfolio_returns,
)
from .evaluation import (
                        DecileSpreadResult,
                        ICReport,
                        IncrementalICReport,
                        decile_spread,
                        ic_decay_curve,
                        ic_report,
                        incremental_ic,
                        newey_west_lags,
                        rank_ic_series,
                        signal_half_life,
)
from .features import FEATURE_ORDER, FeatureBuilder, feature_signature
from .models import DateGroupedRankIC, EnsembleModel, LGBMModel, MLPModel, RankCompositeModel, RidgeModel
from .targets import forward_returns, residualize_forward_returns
from .trainer import WalkForwardTrainer, train_live_parity_check

__all__ = [
                        "FEATURE_ORDER",
                        "THEMES",
                        "AlphaModel",
                        "DateGroupedRankIC",
                        "DecileSpreadResult",
                        "EnsembleModel",
                        "FeatureBuilder",
                        "ICReport",
                        "IncrementalICReport",
                        "LGBMModel",
                        "MLPModel",
                        "RankCompositeModel",
                        "RidgeModel",
                        "ThemeComposite",
                        "WalkForwardTrainer",
                        "decile_spread",
                        "default_ml_factory",
                        "factor_momentum_tilt",
                        "feature_signature",
                        "forward_returns",
                        "ic_decay_curve",
                        "ic_report",
                        "incremental_ic",
                        "momentum_crash_scalar",
                        "newey_west_lags",
                        "rank_ic_series",
                        "residualize_forward_returns",
                        "signal_half_life",
                        "theme_portfolio_returns",
                        "train_live_parity_check",
]
