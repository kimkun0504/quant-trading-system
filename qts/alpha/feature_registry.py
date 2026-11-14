"""One line per alpha feature: theme, role, feature set and what it measures
(``qts features list``, ``qts signal list``, the API's config schema).

The feature *lists* and their order live in :mod:`qts.core.roles`
(``FEATURE_SET_NAMES``); the composite themes and signs in
:data:`qts.alpha.composite.THEMES`; the code in :mod:`qts.alpha.features` /
:mod:`qts.alpha.features_v2`.  This module only describes them;
``tests/test_feature_toggles.py`` checks that every feature has exactly one entry.

Roles
-----
``risk_only_proxy``  proxies a risk-only style (``roles.RISK_ONLY_FEATURE_PROXIES``):
                     besides being a model input it is the label residualiser's
                     warm-up exposure (disabling the *input* never changes that);
``alpha_theme``      a traded-theme feature (``roles.ALPHA_THEME_FEATURES``); never used
                     to residualise the label;
``model_input``      neither (e.g. ``max_ret_21``, ``amihud_63``): only a model input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from qts.core.roles import (
    ALPHA_THEME_FEATURES,
    FEATURE_SET_1_FEATURES,
    FEATURE_SET_2_FEATURES,
    RISK_ONLY_FEATURES,
)

#: feature -> one-line description (what the raw panel at date t measures; all PIT).
FEATURE_DESCRIPTIONS: dict[str, str] = {
    # ---- feature set 1
    "value_bp": "log book-to-price from the latest filing visible at t (filing date + 1 day)",
    "value_ep": "earnings yield: trailing earnings / market cap from the latest visible filing",
    "mom_12_1": "12-1 month momentum: return from t-252 to t-21 trading days",
    "mom_6_1": "6-1 month momentum: return from t-126 to t-21 trading days",
    "mom_change": "change of 12-1 momentum over the last 63 trading days",
    "industry_momentum": "cap-weighted mean 6-1 momentum of the name's industry (eligible names at t)",
    "quality_gpa": "gross profitability: gross profits / total assets (latest visible filing)",
    "investment_ag": "minus year-over-year total-asset growth (low investment scores high)",
    "beta_252": "252-day rolling beta on the equal-weighted return of the eligible names",
    "idio_vol_252": "252-day residual volatility of that beta regression (daily)",
    "max_ret_21": "largest daily return of the last 21 trading days (lottery demand)",
    "turnover_63": "63-day mean of dollar volume / market cap",
    "amihud_63": "log 63-day mean Amihud illiquidity |return| / dollar volume",
    "log_dollar_vol_63": "log 63-day mean dollar volume (as-traded price x shares traded)",
    "log_mcap": "log market capitalisation",
    "short_interest": "short interest / shares outstanding (where the store has it)",
    "borrow_fee": "annualised stock-borrow fee (where the store has it)",
    "strev_21": "short-term reversal: minus the 21-day return, demeaned within industry",
    # ---- feature set 2 (handbook 04 "Feature set 2")
    "ear_3d": "earnings announcement return over [p-1, p+1] minus the market, visible from p+2",
    "sue": "standardised unexpected EPS (seasonal random walk, trailing 8 quarters), filing + 1 day",
    "days_since_earnings": "trading days since the last visible announcement, capped at 126 (conditioning)",
    "abn_volume_5d": "log of 5-day mean dollar volume over the 252-day mean before it",
    "abn_volume_earn": "log dollar volume around the last announcement over its 252-day base",
    "accruals": "Sloan accruals: (net income - operating cash flow) / average assets (TTM filing)",
    "net_issuance": "minus the 252-day log growth of split-adjusted shares (Daniel-Titman issuance)",
    "high_52w": "close / its 252-day high, in (0, 1] (George-Hwang 52-week high)",
}


@dataclass(frozen=True)
class FeatureInfo:
    """One row of ``qts features list``."""

    name: str
    theme: str  #: composite theme it belongs to, or "ml_only"
    sign: float  #: sign the composite applies (+1 for ML-only features)
    role: str  #: risk_only_proxy / alpha_theme / model_input
    feature_set: str  #: first feature set containing it ("v1" or "v2")
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _theme_of() -> dict[str, tuple[str, float]]:
    from qts.alpha.composite import THEMES

    out: dict[str, tuple[str, float]] = {}
    for theme, members in THEMES.items():
        for f, sgn in members:
            out[f] = (theme, float(sgn))
    return out


def feature_role(name: str) -> str:
    if name in RISK_ONLY_FEATURES:
        return "risk_only_proxy"
    if name in ALPHA_THEME_FEATURES:
        return "alpha_theme"
    return "model_input"


def feature_info(feature_set: str = "v2") -> list[FeatureInfo]:
    """Every feature of ``feature_set`` (canonical order) with theme, sign, role and
    description.  ``v1`` lists the 18 Tier-1 features, ``v2`` those plus eight."""
    from qts.core.roles import FEATURE_SET_NAMES

    if feature_set not in FEATURE_SET_NAMES:
        raise ValueError(f"unknown feature_set {feature_set!r}; known: {sorted(FEATURE_SET_NAMES)}")
    themes = _theme_of()
    rows = []
    for n in FEATURE_SET_NAMES[feature_set]:
        theme, sign = themes.get(n, ("ml_only", 1.0))
        rows.append(
            FeatureInfo(
                name=n,
                theme=theme,
                sign=sign,
                role=feature_role(n),
                feature_set="v1" if n in FEATURE_SET_1_FEATURES else "v2",
                description=FEATURE_DESCRIPTIONS.get(n, ""),
            )
        )
    return rows


def feature_sign(name: str) -> float:
    """Sign the composite applies to ``name`` (+1 for features no theme reads)."""
    return _theme_of().get(name, ("", 1.0))[1]


__all__ = [
    "FEATURE_DESCRIPTIONS",
    "FEATURE_SET_2_FEATURES",
    "FeatureInfo",
    "feature_info",
    "feature_role",
    "feature_sign",
]
