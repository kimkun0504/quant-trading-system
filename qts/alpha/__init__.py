"""Alpha model layer: features, targets, composite, ML models, walk-forward
training, evaluation and the ``AlphaModel`` orchestrator."""
from .alpha_model import AlphaModel, default_ml_factory
from .composite import (THEMES, ThemeComposite, factor_momentum_tilt, momentum_crash_scalar,
                        theme_portfolio_returns)
from .evaluation import (decile_spread, ic_decay_curve, ic_report, incremental_ic, rank_ic_series,
                         signal_half_life)
from .features import FEATURE_ORDER, FeatureBuilder, feature_signature
from .models import EnsembleModel, LGBMModel, MLPModel, RankCompositeModel, RidgeModel
from .targets import forward_returns, residualize_forward_returns
from .trainer import WalkForwardTrainer, train_live_parity_check

__all__ = [
    "AlphaModel", "default_ml_factory", "THEMES", "ThemeComposite", "factor_momentum_tilt",
    "momentum_crash_scalar", "theme_portfolio_returns", "decile_spread", "ic_decay_curve",
    "ic_report", "incremental_ic", "rank_ic_series", "signal_half_life", "FEATURE_ORDER",
    "FeatureBuilder", "feature_signature", "EnsembleModel", "LGBMModel", "MLPModel",
    "RankCompositeModel", "RidgeModel", "forward_returns", "residualize_forward_returns",
    "WalkForwardTrainer", "train_live_parity_check",
]
