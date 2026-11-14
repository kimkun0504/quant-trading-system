"""Single source of truth for which risk-model styles are *risk-only* (hedged) and
which are *alpha themes* (traded), and the alpha-layer features that proxy them.

Design decision D1 (docs/notes/doc_review_findings.md): the training label is
residualised on exactly the set of styles the optimizer neutralises.  Predicting a
premium that the portfolio is then forced to hedge away is wasted model capacity
and, worse, leaks hedged exposure into the ML score.  Consequently the linear
premium of the low-risk theme (beta / residual volatility) is intentionally
*not* part of the ML target: the book is beta-neutral and residual-vol-banded, so
it could not harvest that premium anyway.  The rank composite still uses the
low-risk theme, but the optimizer's beta band and style bands limit how much of
it survives into positions.

Everything that needs these sets — ``qts.alpha.targets``, ``qts.pipeline`` and
``qts.portfolio`` — imports them from here.
"""

from __future__ import annotations

#: Risk-model style factors the optimizer neutralises (beta tightly, the rest
#: within ``PortfolioConfig.style_band_sd``) and the label is residualised on.
RISK_ONLY_STYLES: tuple[str, ...] = ("beta", "size", "residual_vol", "liquidity", "short_term_reversal")

#: Risk-model style factors that are alpha themes (traded, not neutralised).
ALPHA_THEME_STYLES: tuple[str, ...] = ("momentum", "value", "quality", "investment", "short_interest")

#: Alpha-layer feature that proxies each risk-only style when the risk model's
#: own exposures are unavailable (warm-up dates).  Several features can map to
#: one style (liquidity has two proxies).
RISK_ONLY_FEATURE_PROXIES: dict[str, tuple[str, ...]] = {
    "beta": ("beta_252",),
    "size": ("log_mcap",),
    "residual_vol": ("idio_vol_252",),
    "liquidity": ("log_dollar_vol_63", "turnover_63"),
    "short_term_reversal": ("strev_21",),
}

#: Flat tuple of the proxy features, in a stable order.
RISK_ONLY_FEATURES: tuple[str, ...] = tuple(f for s in RISK_ONLY_STYLES for f in RISK_ONLY_FEATURE_PROXIES[s])

#: The 18 Tier-1 features of ``AlphaConfig.feature_set = "v1"`` in canonical model-input
#: order (``qts.alpha.features.FEATURE_ORDER`` is this list; the order is part of the
#: train / live parity hash).  Kept here, with no dependency, so that the config can
#: validate ``AlphaConfig.disabled_features`` without importing the alpha layer.
FEATURE_SET_1_FEATURES: tuple[str, ...] = (
    "value_bp",
    "value_ep",
    "mom_12_1",
    "mom_6_1",
    "mom_change",
    "industry_momentum",
    "quality_gpa",
    "investment_ag",
    "beta_252",
    "idio_vol_252",
    "max_ret_21",
    "turnover_63",
    "amihud_63",
    "log_dollar_vol_63",
    "log_mcap",
    "short_interest",
    "borrow_fee",
    "strev_21",
)

#: The eight features ``AlphaConfig.feature_set = "v2"`` adds, in model-input order
#: (handbook 04 "Feature set 2").  All are alpha features: none proxies a risk-only
#: style, so none is ever used to residualise the label or neutralise the score.
FEATURE_SET_2_FEATURES: tuple[str, ...] = (
    "ear_3d",
    "sue",
    "days_since_earnings",
    "abn_volume_5d",
    "abn_volume_earn",
    "accruals",
    "net_issuance",
    "high_52w",
)

#: ``AlphaConfig.feature_set`` -> its model inputs in canonical order (the names
#: ``AlphaConfig.disabled_features`` may list).  ``v1`` is a prefix of ``v2``.
FEATURE_SET_NAMES: dict[str, tuple[str, ...]] = {
    "v1": FEATURE_SET_1_FEATURES,
    "v2": (*FEATURE_SET_1_FEATURES, *FEATURE_SET_2_FEATURES),
}

#: Alpha-layer features belonging to traded themes; must never be used to
#: residualise the label.
ALPHA_THEME_FEATURES: frozenset[str] = frozenset(
    {
        "value_bp",
        "value_ep",
        "mom_12_1",
        "mom_6_1",
        "mom_change",
        "industry_momentum",
        "quality_gpa",
        "investment_ag",
        "short_interest",
        "borrow_fee",
        # feature set 2 (``AlphaConfig.feature_set = "v2"``; ML sleeve only)
        *FEATURE_SET_2_FEATURES,
    }
)

assert not set(RISK_ONLY_STYLES) & set(ALPHA_THEME_STYLES)
assert not set(RISK_ONLY_FEATURES) & ALPHA_THEME_FEATURES
