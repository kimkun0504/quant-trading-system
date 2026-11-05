"""Shared fixtures: synthetic market, stand-in risk model (market beta +
size + industry dummies with sample factor covariance and rolling residual
variance) and a stand-in alpha built from the planted latent score.

These stand-ins exist so the portfolio/backtest layers can be tested before
``qts.risk`` / ``qts.alpha`` land; they honour the point-in-time rule (only
data <= date is used for any ``as_of(date)``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qts.core.config import SystemConfig
from qts.core.crosssection import zscore
from qts.core.types import AlphaOutput, MarketData, RiskModelOutput
from qts.data.synthetic import SyntheticSpec, generate
from qts.portfolio.alpha_scaling import grinold_alpha


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="regenerate tests/golden/*.json from the current code instead of comparing "
        "(review the diff before committing: see tests/test_golden.py)",
    )


def pytest_configure(config):
    # registered here because pyproject.toml is owned elsewhere
    config.addinivalue_line(
        "markers",
        "slow: benchmark / long-running test (timings reported, never asserted); deselect with -m 'not slow'",
    )
    config.addinivalue_line(
        "markers",
        "statistical: threshold tuned on a synthetic seed; checked over several seeds with a majority rule",
    )


@pytest.fixture(scope="session")
def update_golden(request) -> bool:
    return bool(request.config.getoption("--update-golden"))


class StandInRiskModel:
    """Minimal Barra-like model: factors = beta, size, industry dummies."""

    def __init__(
        self,
        md: MarketData,
        beta_window: int = 126,
        cov_window: int = 252,
        spec_window: int = 126,
        min_hist: int = 60,
    ):
        self.md = md
        self.cov_window = cov_window
        self.spec_window = spec_window
        R = md.returns
        mkt = R.mean(axis=1)
        cov = R.rolling(beta_window, min_periods=min_hist).cov(mkt)
        var = mkt.rolling(beta_window, min_periods=min_hist).var()
        beta = cov.div(var, axis=0)
        beta = (0.7 * beta + 0.3).clip(0.2, 2.5).fillna(1.0)  # Blume-style shrink to 1
        size = zscore(np.log(md.market_cap)).fillna(0.0)
        self.beta, self.size = beta, size
        inds = sorted(md.industry.unique())
        self.ind_cols = list(inds)
        D = pd.get_dummies(md.industry).reindex(columns=inds).astype(float).reindex(md.tickers)
        self.D = D
        self.factors = ["beta", "size", *inds]
        # cross-sectional regressions: r_t on exposures as of t-1
        dates = md.dates
        fr = np.full((len(dates), len(self.factors)), np.nan)
        resid = np.full((len(dates), len(md.tickers)), np.nan)
        Dv = D.values
        for i in range(1, len(dates)):
            r = R.iloc[i].values
            ok = np.isfinite(r)
            if ok.sum() < 20:
                continue
            B = np.column_stack([beta.iloc[i - 1].values, size.iloc[i - 1].values, Dv])
            f, *_ = np.linalg.lstsq(B[ok], r[ok], rcond=None)
            fr[i] = f
            resid[i, ok] = r[ok] - B[ok] @ f
        self.factor_returns = pd.DataFrame(fr, index=dates, columns=self.factors)
        self.resid = pd.DataFrame(resid, index=dates, columns=md.tickers)

    def exposures(self, date: pd.Timestamp) -> pd.DataFrame:
        B = self.D.copy()
        B.insert(0, "size", self.size.loc[date])
        B.insert(0, "beta", self.beta.loc[date])
        return B[self.factors]

    def as_of(self, date: pd.Timestamp, alpha: pd.Series | None = None) -> RiskModelOutput:
        fr = self.factor_returns.loc[:date].dropna(how="all")
        F = fr.tail(self.cov_window).cov().fillna(0.0)
        F = F.reindex(index=self.factors, columns=self.factors).fillna(0.0)
        sv = self.resid.loc[:date].tail(self.spec_window).var()
        sv = sv.fillna(sv.median()).clip(lower=1e-6)
        return RiskModelOutput(
            date=date, exposures=self.exposures(date), factor_cov=F, specific_var=sv, factor_returns=fr
        )


class StandInAlpha:
    """Alpha from the planted latent score: Grinold-scaled with IC ``ic``."""

    def __init__(
        self,
        md: MarketData,
        risk: StandInRiskModel,
        ic: float = 0.04,
        horizon_days: int = 20,
        half_life_days: float = 60.0,
        scale: float = 1.0,
    ):
        self.md, self.risk, self.ic, self.H = md, risk, ic, horizon_days
        self.hl, self.scale = half_life_days, scale
        self.z = zscore(md.extra["true_latent_score"])

    def as_of(self, date: pd.Timestamp) -> AlphaOutput:
        z = self.z.loc[date]
        sv = self.risk.resid.loc[:date].tail(self.risk.spec_window).var()
        sv = sv.fillna(sv.median()).clip(lower=1e-6)
        er = grinold_alpha(z, np.sqrt(sv * self.H), self.ic) * self.scale
        return AlphaOutput(
            date=date,
            score=z,
            expected_return=er,
            horizon_days=self.H,
            half_life_days=self.hl,
            ic_estimate=self.ic,
        )


@pytest.fixture(scope="session")
def md() -> MarketData:
    # strong planted alpha: these tests check plumbing, not alpha strength
    return generate(SyntheticSpec(n_stocks=120, n_days=600, seed=3, ic_true=0.08))


@pytest.fixture(scope="session")
def risk_model(md) -> StandInRiskModel:
    return StandInRiskModel(md)


@pytest.fixture(scope="session")
def alpha_model(md, risk_model) -> StandInAlpha:
    return StandInAlpha(md, risk_model)


@pytest.fixture(scope="session")
def sys_cfg() -> SystemConfig:
    return SystemConfig()


@pytest.fixture(scope="session")
def test_date(md) -> pd.Timestamp:
    return md.dates[400]


# ----------------------------------------------------------------------------- golden pipeline run
#: Fixed synthetic market + pipeline settings for the golden-file regression
#: (tests/test_golden.py) and the pipeline determinism test (tests/test_determinism.py).
#: Changing any of these invalidates tests/golden/pipeline_summary.json
#: (regenerate with ``pytest tests/test_golden.py --update-golden``).
GOLDEN_SPEC = {"n_stocks": 100, "n_days": 700, "seed": 11, "ic_true": 0.08}
GOLDEN_TOP_N = 80
GOLDEN_MAX_DAYS = 60


def golden_config() -> SystemConfig:
    cfg = SystemConfig()
    cfg.universe.top_n_by_mcap = GOLDEN_TOP_N  # time-varying universe inside the 100 names
    return cfg


@pytest.fixture(scope="session")
def golden_market() -> MarketData:
    return generate(SyntheticSpec(**GOLDEN_SPEC))


@pytest.fixture(scope="session")
def golden_pipeline_run(golden_market):
    """``ResearchPipeline(use_ml=False).run(md, max_days=60)`` on the golden market
    (~25 s; shared by the golden and determinism tests)."""
    import warnings

    from qts.pipeline import ResearchPipeline

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pipe = ResearchPipeline(golden_config(), use_ml=False)
        res = pipe.run(golden_market, max_days=GOLDEN_MAX_DAYS)
    return pipe, res
