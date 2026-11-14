"""Single-signal research: ``qts signal list`` / ``qts signal test <feature>``.

What a test computes (pre-seal period; the baseline's fitted pipeline from the stage
cache: nothing is re-fitted when the baseline is unchanged)
------------------------------------------------------------------------------------
The signal is the feature processed exactly as the model processes it -- per date over
the eligible names: winsorise (``alpha.winsor_pct``) -> industry demean (the
features in ``alpha.industry_demean`` and the v2 set, or ``industry_demean=True`` of
a custom signal) -> rank-Gaussian -- times the sign the composite applies (+1 for a
feature no theme reads; the registered sign of a custom signal).  Missing raw values
stay missing (the model fills them with 0; a test of the signal must not).

Evaluation dates: the baseline's out-of-sample backtest dates (``ResearchPipeline
.backtest_dates()``: after the risk warm-up and the first ML prediction), sampled every
``step`` trading days (default 5 = the baseline's rebalance cadence).  ``s`` is a
signal date; with execution lag ``L`` the label of ``s`` at horizon ``h`` is the
risk-only residualised return ``close[s+L] -> close[s+L+h]`` (the training label of
the baseline at its own horizon; the same construction at the others).

* **coverage by year**: share of eligible (date, name) cells with a raw value;
* **rank IC by horizon** (``--horizons``, default 5, 20, 60, 120): mean per-date
  Spearman IC; Newey-West t with Bartlett lag ``max(ceil(h / step), auto)``, the number
  of consecutive sampled observations whose labels overlap (HAC lag = horizon overlap);
* **IC by year and by fold** at the *primary* horizon (the baseline's
  ``alpha.horizon_days`` when listed, else the first); folds = the purged walk-forward
  test blocks of the pipeline's gates;
* **decay**: IC by horizon (cumulative), IC of the signal lagged ``k`` trading days
  against the primary label (``k`` = 0, 5, 10, 20, 40, 60) and the half-life of the
  signal's rank autocorrelation;
* **paper book**: on every sampled date, centred ranks of the signal projected off an
  intercept + the risk model's industries + market + risk-only styles (equal-weighted
  OLS over the eligible names; the exposures the label is residualised on), scaled to
  gross 2, held for ``step`` days from the execution close ``s + L``.  Daily gross
  return ``sum_i w_i r_{i,t}`` (data-error returns of ``qts.core.returns`` rule (a)
  and missing returns count 0), turnover ``sum_i |w_i - w_i^{prev}|`` at each
  rebalance, net = gross - cost per side x turnover.  Cost per side: the baseline run's
  ``avg_cost_bps_per_side`` (newest run directory with the baseline configuration;
  ``--cost-bps`` overrides; 5 bp if none is found, with a note);
* **correlations**: mean per-date Spearman correlation with every processed baseline
  feature, the composite and the traded score;
* **incremental value**: the per-date OLS residual of the signal on an intercept + the
  baseline's traded score + the same exposures, and its rank IC (all horizons, NW t as
  above); the paper book of ``0.75 z(traded score) + 0.25 z(signal)`` against the book
  of the traded score alone (``w = 0.25`` fixed and pre-declared; never optimised):
  net Sharpe of both, their difference and the Newey-West t of the daily return
  difference.

Trial accounting
----------------
Each test is one trial in the ledger family ``signal:<market_hash[:12]>`` (the
full-history market hash under the seal, like the pipeline families), with the
primary horizon's IC t-statistic as its statistic.  The report applies the
Harvey-Liu-Zhu (2016) hurdle ``|t| >= 3.0`` (``validation.hurdle_t_stat``) and a Holm
(1979) step-down adjustment of the two-sided normal p-values of *every* signal test
logged in that family so far (identical re-runs collapse in the ledger).  The primary
horizon is fixed before the test (the baseline's); the other horizons are reported, not
selected from.

Entry points: :func:`list_signals` -> ``list[SignalDef]``; :func:`run_signal_test` ->
:class:`SignalTestResult` (``to_dict()`` is the ``runs/_signals/<id>.json`` payload);
:func:`register_signal`; :func:`holm_adjust`.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from qts.core.types import MarketData, Wide
from qts.research.common import (
    BASELINE_CONFIG,
    fmt_float,
    json_safe,
    resolve_repo_path,
    utc_stamp,
    write_json,
)
from qts.research.data import Printer, StoreSpec, _quiet

if TYPE_CHECKING:
    from qts.core.config import SystemConfig
    from qts.pipeline import ResearchPipeline

SignalFn = Callable[[MarketData], Wide]

#: Default horizons (trading days) of a test.
DEFAULT_HORIZONS: tuple[int, ...] = (5, 20, 60, 120)
#: Sampling step (trading days) of the evaluation dates and the paper book's holding period.
DEFAULT_STEP = 5
#: Pre-declared weight of the signal in the incremental paper book.
INCREMENTAL_WEIGHT = 0.25
#: Lags (trading days) of the signal-lag decay curve.
DECAY_LAGS: tuple[int, ...] = (0, 5, 10, 20, 40, 60)
#: Cost per side when no baseline run directory is found (bp).
FALLBACK_COST_BPS = 5.0
#: Ledger family prefix.
FAMILY_PREFIX = "signal:"
#: Where results are written (under the runs root).
SIGNALS_DIRNAME = "_signals"


# ----------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class SignalDef:
    """A testable signal: a built-in feature (v1 / v2) or a registered custom one."""

    name: str
    theme: str
    sign: float
    role: str
    feature_set: str  #: "v1", "v2" or "custom"
    description: str
    industry_demean: bool = False
    fn: SignalFn | None = field(default=None, compare=False, repr=False)

    @property
    def source(self) -> str:
        return "custom" if self.fn is not None else "builtin"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("fn")
        d["source"] = self.source
        return d


_CUSTOM: dict[str, SignalDef] = {}
_CUSTOM_LOADED = False


def register_signal(
    name: str,
    fn: SignalFn | None = None,
    *,
    description: str = "",
    sign: float = 1.0,
    industry_demean: bool = False,
    theme: str = "custom",
) -> Any:
    """Register a custom signal ``fn(md) -> Wide`` (point in time; see
    :mod:`qts.research.custom_signals`).  Usable as a decorator.  A name that is a
    built-in feature is refused; re-registering a custom name replaces it."""
    from qts.core.roles import FEATURE_SET_NAMES

    if name in FEATURE_SET_NAMES["v2"]:
        raise ValueError(f"register_signal: {name!r} is a built-in feature")
    if float(sign) not in (1.0, -1.0):
        raise ValueError("register_signal: sign must be +1 or -1")

    def deco(f: SignalFn) -> SignalFn:
        _CUSTOM[name] = SignalDef(
            name=name,
            theme=theme,
            sign=float(sign),
            role="custom",
            feature_set="custom",
            description=description or (inspect.getdoc(f) or "").split("\n")[0],
            industry_demean=bool(industry_demean),
            fn=f,
        )
        return f

    return deco(fn) if fn is not None else deco


def load_signal_module(path: str | None) -> None:
    """Import the built-in custom signals once, and ``path`` (a ``.py`` file or a
    dotted module name) when given, so their ``register_signal`` calls run."""
    global _CUSTOM_LOADED
    if not _CUSTOM_LOADED:
        importlib.import_module("qts.research.custom_signals")
        _CUSTOM_LOADED = True
    if not path:
        return
    p = Path(path)
    if p.suffix == ".py" or p.exists():
        spec = importlib.util.spec_from_file_location(f"qts_user_signals_{p.stem}", p)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import signal module {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        importlib.import_module(path)


def list_signals(signal_module: str | None = None) -> list[SignalDef]:
    """Every testable signal: the v1 and v2 features (``qts features list`` metadata)
    and the registered custom signals."""
    from qts.alpha.feature_registry import feature_info
    from qts.alpha.features import V2_INDUSTRY_DEMEAN
    from qts.core.config import AlphaConfig

    load_signal_module(signal_module)
    demean = set(AlphaConfig().industry_demean) | set(V2_INDUSTRY_DEMEAN)
    out = [
        SignalDef(
            name=i.name,
            theme=i.theme,
            sign=i.sign,
            role=i.role,
            feature_set=i.feature_set,
            description=i.description,
            industry_demean=i.name in demean,
        )
        for i in feature_info("v2")
    ]
    return out + sorted(_CUSTOM.values(), key=lambda d: d.name)


def get_signal(name: str, signal_module: str | None = None, cfg: SystemConfig | None = None) -> SignalDef:
    """The :class:`SignalDef` of ``name`` (built-in industry-demean flags follow
    ``cfg.alpha.industry_demean`` when ``cfg`` is given)."""
    from qts.alpha.features import V2_INDUSTRY_DEMEAN

    sigs = {s.name: s for s in list_signals(signal_module)}
    if name not in sigs:
        raise KeyError(f"unknown signal {name!r}; `qts signal list` shows {len(sigs)} names")
    s = sigs[name]
    if cfg is not None and s.fn is None:
        demean = set(cfg.alpha.industry_demean) | set(V2_INDUSTRY_DEMEAN)
        s = SignalDef(**{**asdict(s), "industry_demean": name in demean})
    return s


# ----------------------------------------------------------------------------- processing
def raw_signal_panel(sig: SignalDef, pipe: ResearchPipeline) -> Wide:
    """The raw panel of ``sig`` on the pipeline's (universe-applied) market: the
    fitted alpha's raw feature when it has it, the v2 builder for a v2 feature under a
    v1 baseline, or the custom function."""
    from qts.alpha.features_v2 import build_v2_features, require_v2_inputs

    assert pipe.md_u is not None and pipe.alpha is not None
    md = pipe.md_u
    if sig.fn is not None:
        out = sig.fn(md)
    elif sig.name in pipe.alpha.raw_features_:
        out = pipe.alpha.raw_features_[sig.name]
    elif sig.feature_set == "v2":
        require_v2_inputs(md)
        market = md.returns.where(md.eligibility()).mean(axis=1)
        out = build_v2_features(md, market)[sig.name]
    else:
        raise ValueError(f"feature {sig.name!r} has no data in this store (the panel it needs is missing)")
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"signal {sig.name!r} must return a date x ticker DataFrame")
    return out.reindex(index=md.dates, columns=md.tickers).astype(float)


def process_signal(
    panel: Wide, md: MarketData, winsor_pct: float, industry_demean: bool, name: str = ""
) -> Wide:
    """Per date over the eligible names: winsorise -> [industry demean] ->
    rank-Gaussian; missing stays missing.  On the cells with a raw value it equals
    ``FeatureBuilder.process`` (which then fills the rest with 0); a conditioning
    feature (``days_since_earnings``) is only scaled, as there."""
    from qts.alpha.features import CONDITIONING_FEATURES, _by_industry, industry_segments
    from qts.core import stats
    from qts.core.crosssection import demean_by_group
    from qts.core.stats import rank_gauss

    valid = md.eligibility()
    p = panel.reindex(index=valid.index, columns=valid.columns).where(valid)
    if name in CONDITIONING_FEATURES:
        divisor, _fill = CONDITIONING_FEATURES[name]
        return p / divisor
    w = stats.winsorize_rows(p, winsor_pct)
    if industry_demean:
        w = _by_industry(demean_by_group, w, industry_segments(md))
    return rank_gauss(w)


# ----------------------------------------------------------------------------- statistics
def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values (same order as the input; NaN stays NaN):
    sort the m finite p ascending, ``adj_(k) = max_{j <= k} min(1, (m - j + 1) p_(j))``."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = len(ok)
    if not m:
        return out
    order = ok[np.argsort(p[ok], kind="mergesort")]
    running = 0.0
    for j, i in enumerate(order):
        running = max(running, min(1.0, (m - j) * p[i]))
        out[i] = running
    return out


def two_sided_p(t: float) -> float:
    from scipy.stats import norm

    return float(2.0 * norm.sf(abs(t))) if math.isfinite(t) else float("nan")


def ic_stats(ic: pd.Series, overlap: int) -> dict[str, float]:
    """Mean, sd, hit rate, n and the Newey-West t (lag = ``newey_west_lags(n, overlap)``)."""
    from qts.alpha.evaluation import newey_west_lags
    from qts.core import stats

    v = pd.Series(ic, dtype=float).dropna()
    n = len(v)
    if n < 3:
        return {
            "mean_ic": float("nan"),
            "sd": float("nan"),
            "t_nw": float("nan"),
            "nw_lags": 0,
            "n": n,
            "hit": float("nan"),
        }
    lags = newey_west_lags(n, max(int(overlap), 1))
    return {
        "mean_ic": float(v.mean()),
        "sd": float(v.std(ddof=1)),
        "t_nw": float(stats.newey_west_tstat(v.to_numpy(dtype=float), lags=int(lags))),
        "nw_lags": int(lags),
        "n": n,
        "hit": float((v > 0).mean()),
    }


def _design(X: pd.DataFrame | None, names: pd.Index) -> np.ndarray:
    """Intercept + the non-constant columns of ``X`` (reindexed to ``names``, NaN -> 0)."""
    ones = np.ones((len(names), 1))
    if X is None or not len(X.columns):
        return ones
    A = X.reindex(names).fillna(0.0).to_numpy(dtype=float)
    keep = A.std(axis=0) > 1e-12
    return np.hstack([ones, A[:, keep]])


def ols_resid(y: np.ndarray, A: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def neutral_rank_weights(score: pd.Series, X: pd.DataFrame | None, gross: float = 2.0) -> pd.Series:
    """Centred ranks of ``score`` (NaN -> 0) projected off ``[1, X]`` by equal-weighted
    OLS and scaled to ``sum |w| = gross``: the rank-weighted, industry- and
    risk-style-neutral paper book (zero weights when the projection leaves nothing)."""
    s = score.astype(float)
    r = s.rank(pct=True) - 0.5
    r = r.fillna(0.0)
    res = ols_resid(r.to_numpy(dtype=float), _design(X, r.index))
    tot = float(np.abs(res).sum())
    w = res * (gross / tot) if tot > 1e-12 else np.zeros_like(res)
    return pd.Series(w, index=r.index)


@dataclass
class PaperBook:
    """Daily returns of a sampled paper book (:func:`paper_book`)."""

    gross: pd.Series
    turnover: pd.Series  #: per return day (the rebalance turnover on its first day)
    cost_bps: float

    @property
    def net(self) -> pd.Series:
        return self.gross - self.cost_bps / 1e4 * self.turnover

    def stats(self) -> dict[str, float]:
        from qts.backtest import metrics as M

        n = len(self.gross)
        to = float(self.turnover.sum() / n) if n else float("nan")
        g, net = M.annualized_sharpe(self.gross), M.annualized_sharpe(self.net)
        mg = float(self.gross.mean()) if n else float("nan")
        return {
            "sharpe_gross": float(g),
            "sharpe_net": float(net),
            "ann_return_gross": mg * 252,
            "ann_vol": float(self.gross.std(ddof=1) * math.sqrt(252)) if n > 2 else float("nan"),
            "turnover_per_day": to,
            "cost_bps_per_side": float(self.cost_bps),
            "breakeven_cost_bps": float(mg / to * 1e4) if to and to > 0 else float("nan"),
            "n_days": float(n),
        }


def paper_book(
    weights: Mapping[pd.Timestamp, pd.Series],
    returns: pd.DataFrame,
    lag: int,
    step: int,
    cost_bps: float,
) -> PaperBook:
    """Hold the weights set on signal date ``s`` for ``step`` return days from the
    execution close ``s + lag`` (return days ``pos(s) + lag + 1 .. + step``)."""
    dates = returns.index
    R = returns.to_numpy(dtype=float)
    col = {t: j for j, t in enumerate(returns.columns)}
    gross = np.zeros(len(dates))
    to = np.zeros(len(dates))
    used = np.zeros(len(dates), dtype=bool)
    prev: pd.Series | None = None
    for s, w in sorted(weights.items()):
        p0 = int(dates.get_loc(s)) + int(lag) + 1
        if p0 >= len(dates):
            break
        idx = np.array([col[t] for t in w.index], dtype=int)
        wv = w.to_numpy(dtype=float)
        p1 = min(p0 + int(step), len(dates))
        gross[p0:p1] = R[p0:p1][:, idx] @ wv
        used[p0:p1] = True
        delta = w.sub(prev, fill_value=0.0) if prev is not None else w
        to[p0] = float(np.abs(delta.to_numpy(dtype=float)).sum())
        prev = w
    keep = pd.DatetimeIndex(dates[used])
    return PaperBook(
        gross=pd.Series(gross[used], index=keep, name="gross"),
        turnover=pd.Series(to[used], index=keep, name="turnover"),
        cost_bps=float(cost_bps),
    )


def clean_daily_returns(md: MarketData, caps: tuple[float, float] | None) -> pd.DataFrame:
    """Daily returns with rule-(a) data errors and missing values set to 0."""
    from qts.core.returns import data_error_mask

    r = md.returns
    if caps is not None:
        r = r.mask(data_error_mask(r, None, cap_up=caps[0], cap_down=caps[1], min_prev_price=None))
    return r.fillna(0.0)


# ----------------------------------------------------------------------------- result
@dataclass
class SignalTestResult:
    """Everything one signal test produced (``to_dict()`` = the JSON the API serves)."""

    id: str
    signal: SignalDef
    horizons: list[int]
    primary_horizon: int
    step: int
    window: list[str]
    baseline: dict[str, Any]
    coverage_by_year: dict[str, float]
    ic_by_horizon: dict[str, dict[str, float]]
    ic_by_year: dict[str, dict[str, float]]
    ic_by_fold: list[dict[str, Any]]
    decay: dict[str, Any]
    paper: dict[str, float]
    correlations: dict[str, float]
    incremental: dict[str, Any]
    trial: dict[str, Any]
    timings: dict[str, float]
    notes: list[str] = field(default_factory=list)
    json_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["signal"] = self.signal.to_dict()
        return json_safe(d)

    # ---- printing
    def format(self) -> str:
        s = self.signal
        L = [
            f"=== Signal test: {s.name} ({s.source}, theme {s.theme}, sign {s.sign:+.0f}) ===",
            f"{s.description}",
            f"window {self.window[0]} .. {self.window[1]} (baseline out-of-sample dates, every {self.step} days); "
            f"primary horizon {self.primary_horizon}d; baseline config {self.baseline.get('config_hash')} "
            f"(fit: {self.baseline.get('cache')})",
            "",
            "coverage by year (share of eligible names with a value):",
            "  " + "  ".join(f"{y}: {100 * v:.0f}%" for y, v in self.coverage_by_year.items()),
            "",
            f"{'horizon':>8} {'mean IC':>8} {'NW t':>7} {'lags':>5} {'hit':>5} {'n':>6}   "
            f"{'incr IC':>8} {'incr t':>7}",
        ]
        for h, r in self.ic_by_horizon.items():
            inc = self.incremental["ic_by_horizon"].get(h, {})
            L.append(
                f"{h + 'd':>8} {fmt_float(r['mean_ic'], '.4f'):>8} {fmt_float(r['t_nw']):>7} {r['nw_lags']:>5} "
                f"{fmt_float(100 * r['hit'], '.0f'):>4}% {r['n']:>6}   {fmt_float(inc.get('mean_ic'), '.4f'):>8} "
                f"{fmt_float(inc.get('t_nw')):>7}"
            )
        L += ["", f"IC by year ({self.primary_horizon}d):"]
        L.append(
            "  " + "  ".join(f"{y}: {fmt_float(v['mean_ic'], '.3f')}" for y, v in self.ic_by_year.items())
        )
        L.append(
            "IC by fold: "
            + "  ".join(f"{f['start']}..{f['end']} {fmt_float(f['mean_ic'], '.4f')}" for f in self.ic_by_fold)
        )
        dec = self.decay
        L.append(
            "decay (signal lagged k days vs the primary label): "
            + "  ".join(f"k={k}: {fmt_float(v, '.4f')}" for k, v in dec["lag_ic"].items())
            + f"   rank-autocorrelation half-life {fmt_float(dec['half_life_days'], '.0f')}d"
        )
        p = self.paper
        L += [
            "",
            f"paper book (rank-weighted, industry + risk-style neutral, gross 2, rebalanced every {self.step}d):",
            f"  Sharpe gross {fmt_float(p['sharpe_gross'])}  net {fmt_float(p['sharpe_net'])} at "
            f"{fmt_float(p['cost_bps_per_side'], '.1f')} bp/side   turnover {fmt_float(100 * p['turnover_per_day'], '.1f')}%/day   "
            f"breakeven {fmt_float(p['breakeven_cost_bps'], '.1f')} bp",
        ]
        top = sorted(self.correlations.items(), key=lambda kv: -abs(kv[1]) if math.isfinite(kv[1]) else 0)[:6]
        L.append(
            "correlation with baseline features (mean per-date Spearman): "
            + "  ".join(f"{k} {v:+.2f}" for k, v in top)
        )
        inc = self.incremental
        L += [
            "",
            f"incremental: residual on the traded score + risk-only exposures (IC table above; "
            f"{fmt_float(100 * inc['resid_var_share'], '.0f')}% of the signal's variance is unexplained); "
            f"baseline + {inc['weight']:.2f} x signal paper book:",
            f"  net Sharpe baseline {fmt_float(inc['baseline_book']['sharpe_net'])} -> with signal "
            f"{fmt_float(inc['blend_book']['sharpe_net'])}  (diff {fmt_float(inc['delta_sharpe_net'], '+.2f')}, "
            f"NW t of the daily difference {fmt_float(inc['delta_t_nw'])})",
        ]
        t = self.trial
        L += [
            "",
            f"trial: family {t['family']} ({t['n_tests']} signal test(s) so far)   primary t {fmt_float(t['t'])}  "
            f"p {fmt_float(t['p'], '.4f')}  Holm-adjusted p {fmt_float(t['p_holm'], '.4f')}",
            f"Harvey-Liu-Zhu hurdle |t| >= {t['hurdle']:.1f}: {'PASS' if t['passes_hurdle'] else 'FAIL'}"
            + ("   (not logged: --no-log)" if not t.get("logged") else ""),
        ]
        if self.notes:
            L += ["", "notes: " + " | ".join(self.notes)]
        if self.json_path:
            L.append(f"written {self.json_path}")
        return "\n".join(L)


# ----------------------------------------------------------------------------- the test
def baseline_cost_bps(
    cfg: SystemConfig, runs_root: str | None, sealed: bool
) -> tuple[float | None, str | None]:
    """``(avg_cost_bps_per_side, run id)`` of the newest run directory whose config
    (re-hashed by the current ``SystemConfig``) equals ``cfg`` -- a run under the seal
    first when ``sealed``; ``(None, None)`` when there is none."""
    import json

    from qts.core.config import SystemConfig
    from qts.pipeline import list_runs

    if not runs_root:
        return None, None
    target = cfg.hash()
    found: list[tuple[bool, float, str]] = []
    for d in list_runs(runs_root)[:500]:
        try:
            if SystemConfig.from_yaml(d / "config.yaml").hash() != target:
                continue
            s = json.loads((d / "summary.json").read_text())
        except (OSError, ValueError, KeyError, TypeError):
            continue
        c = s.get("backtest", {}).get("avg_cost_bps_per_side")
        if isinstance(c, (int, float)) and math.isfinite(c):
            found.append((bool(s.get("sealed_end")) == sealed, float(c), d.name))
    if not found:
        return None, None
    best = next((f for f in found if f[0]), found[0])
    return best[1], best[2]


def _primary(horizons: Sequence[int], baseline_h: int) -> int:
    return int(baseline_h) if int(baseline_h) in horizons else int(horizons[0])


def _source_hash(sig: SignalDef) -> str | None:
    if sig.fn is None:
        return None
    try:
        return hashlib.sha256(inspect.getsource(sig.fn).encode()).hexdigest()[:16]
    except (OSError, TypeError):
        return None


def run_signal_test(
    feature: str,
    spec: StoreSpec | None = None,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    baseline: str | Path = BASELINE_CONFIG,
    config: SystemConfig | None = None,
    step: int = DEFAULT_STEP,
    cost_bps: float | None = None,
    ledger_path: str | None = "runs/trial_ledger.jsonl",
    runs_root: str | None = "runs",
    cache_root: str | None = None,
    use_ml: bool = True,
    signal_module: str | None = None,
    log_trial: bool = True,
    md: MarketData | None = None,
    pipe: ResearchPipeline | None = None,
    log: Printer = _quiet,
) -> SignalTestResult:
    """Run one signal test (module docstring).  The market is loaded from ``spec`` under
    the research seal unless ``md`` is given (tests; then ``md`` is used as is, its own
    market hash keys the family); ``pipe``: an already fitted baseline pipeline on that
    market (tests / repeated calls).  ``config`` overrides ``baseline`` (a YAML path)."""
    import time

    from qts.alpha.evaluation import signal_half_life
    from qts.alpha.targets import residualize_forward_returns
    from qts.core.config import SystemConfig
    from qts.core.manifest import code_version, hash_market
    from qts.core.stats import rank_ic_series
    from qts.pipeline import ResearchPipeline, data_hash_of, purged_fold_blocks
    from qts.research.common import pipeline_config
    from qts.research.data import load_research_market

    t_all = time.perf_counter()
    timings: dict[str, float] = {}
    notes: list[str] = []
    horizons = sorted({int(h) for h in horizons if int(h) > 0})
    if not horizons:
        raise ValueError("signal test: give at least one positive horizon")
    step = int(step)
    if step < 1:
        raise ValueError("signal test: step must be >= 1")
    base_cfg = config if config is not None else SystemConfig.from_yaml(resolve_repo_path(baseline))
    sig = get_signal(feature, signal_module, base_cfg)

    # ---- data + baseline fit (stage cache)
    t0 = time.perf_counter()
    mh_family = ""
    sealed = False
    if md is None:
        if spec is None:
            raise ValueError("signal test: give a store spec or a MarketData")
        load = load_research_market(
            spec,
            hash_cache_dir=f"{runs_root}/_cache/market_hash" if runs_root else None,
            log=log,
            universe=base_cfg,
        )
        md = load.md
        mh_family = load.market_hash_full
        sealed = load.sealed
    mh_family = mh_family or hash_market(md)
    timings["load"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    cache = cache_root if cache_root is not None else (f"{runs_root}/_cache" if runs_root else None)
    cfg = pipeline_config(base_cfg, md, cache)
    if pipe is None:
        pipe = ResearchPipeline(cfg, use_ml=use_ml, ledger_path=None, runs_root=None)
        pipe.fit(md)
    timings["baseline_fit"] = time.perf_counter() - t0
    cache_info = dict(pipe.cache_info_) or {"cache": "off"}
    if cache_info and any(v != "hit" for v in cache_info.values()):
        notes.append(
            f"baseline fit was not served by the stage cache ({cache_info}): run the sealed baseline once "
            "(`qts run ... --config configs/baseline.yaml --ml --cache`) so later tests take minutes"
        )
    log(f"baseline fit: {timings['baseline_fit']:.0f}s  stage cache {cache_info}")
    assert pipe.md_u is not None and pipe.alpha is not None and pipe.risk is not None
    md_u, alpha = pipe.md_u, pipe.alpha
    acfg = pipe.cfg.alpha
    L = pipe.signal_lag_days()
    H = int(acfg.horizon_days)
    primary = _primary(horizons, H)

    # ---- the signal
    t0 = time.perf_counter()
    raw = raw_signal_panel(sig, pipe)
    proc = process_signal(raw, md_u, float(acfg.winsor_pct), sig.industry_demean, sig.name) * float(sig.sign)
    timings["signal"] = time.perf_counter() - t0

    # ---- evaluation dates
    bt = pipe.backtest_dates()
    sample = pd.DatetimeIndex(bt[::step])
    window = [str(bt[0].date()), str(bt[-1].date())]
    elig = md_u.eligibility()
    cov = raw.where(elig).notna().sum(axis=1) / elig.sum(axis=1).replace(0, np.nan)
    by_year = cov.groupby(cov.index.year).mean()
    coverage = {str(int(y)): float(v) for y, v in by_year.items() if math.isfinite(float(v))}
    if coverage and float(np.mean(list(coverage.values()))) < 0.5:
        notes.append(
            "coverage below 50% of eligible names on average: the IC is computed on the covered names only"
        )

    # ---- labels at every horizon, on the sampled dates (exposures cached across horizons)
    t0 = time.perf_counter()
    ex_cache: dict[pd.Timestamp, pd.DataFrame | None] = {}

    def exposures(d: pd.Timestamp) -> pd.DataFrame | None:
        d = pd.Timestamp(d)
        if d not in ex_cache:
            ex_cache[d] = pipe._exposures_fn(d)
        return ex_cache[d]

    caps = (
        (pipe.cfg.risk.return_abs_cap_up, pipe.cfg.risk.return_abs_cap_down)
        if pipe.cfg.risk.clean_returns
        else None
    )
    labels: dict[int, Wide] = {}
    for h in horizons:
        if h == H:
            labels[h] = alpha.target_.reindex(index=sample, columns=md_u.tickers)
            continue
        lab = residualize_forward_returns(
            md_u,
            h,
            exposures_fn=exposures,
            features=alpha.raw_features_,
            winsor_pct=acfg.winsor_pct,
            delisting_return=pipe.cfg.cost.delisting_returns(),
            only_dates=sample,
            label_lag_days=L,
            delisting_gap_days=acfg.delisting_gap_days,
            bad_price_caps=caps,
        )
        labels[h] = lab.reindex(index=sample, columns=md_u.tickers)
    timings["labels"] = time.perf_counter() - t0

    # ---- IC by horizon / year / fold, decay
    t0 = time.perf_counter()
    S = proc.reindex(index=sample, columns=md_u.tickers)
    overlap = {h: max(1, math.ceil(h / step)) for h in horizons}
    ic_series = {h: rank_ic_series(S, labels[h]) for h in horizons}
    ic_h = {str(h): ic_stats(ic_series[h], overlap[h]) for h in horizons}
    icp = ic_series[primary].dropna()
    ic_y = {str(int(y)): {"mean_ic": float(g.mean()), "n": len(g)} for y, g in icp.groupby(icp.index.year)}
    v = pipe.cfg.validation
    blocks = purged_fold_blocks(pd.DatetimeIndex(bt), v.n_splits, v.embargo_days, H, v.min_fold_days)
    folds = []
    for b in blocks:
        sel = icp[(icp.index >= b[0]) & (icp.index <= b[-1])]
        folds.append(
            {
                "start": str(b[0].date()),
                "end": str(b[-1].date()),
                "mean_ic": float(sel.mean()) if len(sel) else None,
                "n": len(sel),
            }
        )
    lag_ic: dict[str, float] = {}
    pos = md_u.dates.get_indexer(sample)
    for k in DECAY_LAGS:
        ok = pos - k >= 0
        Sk = proc.iloc[pos[ok] - k].set_axis(sample[ok], axis=0).reindex(columns=md_u.tickers)
        lag_ic[str(k)] = float(rank_ic_series(Sk, labels[primary].loc[sample[ok]]).mean())
    hl = signal_half_life(proc.loc[bt[0] : bt[-1]].iloc[-504 * 4 :])
    decay = {
        "ic_by_horizon": {str(h): ic_h[str(h)]["mean_ic"] for h in horizons},
        "lag_ic": lag_ic,
        "half_life_days": float(hl),
    }
    timings["ic"] = time.perf_counter() - t0

    # ---- paper books (signal, baseline, blend)
    t0 = time.perf_counter()
    if cost_bps is None:
        c, run_id = baseline_cost_bps(pipe.cfg, runs_root, sealed=sealed)
        if c is None:
            cost_bps = FALLBACK_COST_BPS
            notes.append(
                f"no baseline run directory found for the cost per side: {FALLBACK_COST_BPS:g} bp assumed (--cost-bps)"
            )
        else:
            cost_bps = c
            notes.append(f"cost per side {c:.2f} bp from run {run_id}")
    rets = clean_daily_returns(md_u, caps)
    trad = alpha.traded_score_
    w_sig: dict[pd.Timestamp, pd.Series] = {}
    w_base: dict[pd.Timestamp, pd.Series] = {}
    w_blend: dict[pd.Timestamp, pd.Series] = {}
    resid_rows: dict[pd.Timestamp, pd.Series] = {}
    resid_share: list[float] = []
    w_inc = INCREMENTAL_WEIGHT
    for s in sample:
        names = elig.columns[elig.loc[s].to_numpy()]
        if len(names) < 20:
            continue
        X = exposures(s)
        X = None if X is None else X.reindex(names)
        sg = S.loc[s].reindex(names)
        w_sig[s] = neutral_rank_weights(sg, X)
        tb = trad.loc[s].reindex(names) if s in trad.index else pd.Series(np.nan, index=names)
        w_base[s] = neutral_rank_weights(tb, X)
        zb, zs = _z(tb), _z(sg)
        w_blend[s] = neutral_rank_weights((1.0 - w_inc) * zb + w_inc * zs, X)
        # incremental residual: signal on [1, traded score, exposures], covered names only
        m = sg.notna() & tb.notna()
        if int(m.sum()) >= 30:
            A = _design(X.loc[m[m].index] if X is not None else None, m[m].index)
            A = np.hstack([A, tb[m].to_numpy(dtype=float)[:, None]])
            y = sg[m].to_numpy(dtype=float)
            res = ols_resid(y, A)
            resid_rows[s] = pd.Series(res, index=m[m].index)
            vy = float(np.var(y))
            if vy > 0:
                resid_share.append(float(np.var(res)) / vy)
    book = paper_book(w_sig, rets, L, step, cost_bps)
    book_b = paper_book(w_base, rets, L, step, cost_bps)
    book_x = paper_book(w_blend, rets, L, step, cost_bps)
    timings["paper"] = time.perf_counter() - t0

    # ---- correlations
    t0 = time.perf_counter()
    corr: dict[str, float] = {}
    X_ = alpha.X_
    lvl = X_.index.get_level_values("date")
    Xs = X_[lvl.isin(sample)]
    for f in Xs.columns:
        if f == sig.name:
            continue
        panel = Xs[f].unstack("ticker").reindex(index=sample, columns=md_u.tickers)
        corr[str(f)] = float(rank_ic_series(S, panel).mean())
    corr["_composite"] = float(rank_ic_series(S, alpha.composite_score_.reindex(index=sample)).mean())
    corr["_traded_score"] = float(rank_ic_series(S, trad.reindex(index=sample)).mean())
    timings["correlations"] = time.perf_counter() - t0

    # ---- incremental IC
    R = pd.DataFrame(resid_rows).T.reindex(index=sample, columns=md_u.tickers)
    inc_h = {str(h): ic_stats(rank_ic_series(R, labels[h]), overlap[h]) for h in horizons}
    from qts.alpha.evaluation import newey_west_lags
    from qts.core import stats

    common = book_x.net.index.intersection(book_b.net.index)
    diff = (book_x.net - book_b.net).reindex(common).dropna()
    d_t = (
        float(stats.newey_west_tstat(diff.to_numpy(dtype=float), lags=int(newey_west_lags(len(diff), step))))
        if len(diff) > 10
        else float("nan")
    )
    bs, xs = book_b.stats(), book_x.stats()
    incremental = {
        "weight": w_inc,
        "ic_by_horizon": inc_h,
        # mean per-date share of the signal's cross-sectional variance that the traded
        # score and the risk-only exposures do not explain (1 = all new, 0 = nothing new)
        "resid_var_share": float(np.mean(resid_share)) if resid_share else float("nan"),
        "baseline_book": bs,
        "blend_book": xs,
        "delta_sharpe_net": xs["sharpe_net"] - bs["sharpe_net"],
        "delta_sharpe_gross": xs["sharpe_gross"] - bs["sharpe_gross"],
        "delta_t_nw": d_t,
    }

    # ---- trial accounting
    family = f"{FAMILY_PREFIX}{mh_family[:12]}"
    t_primary = float(ic_h[str(primary)]["t_nw"])
    hurdle = float(pipe.cfg.validation.hurdle_t_stat)
    test_cfg = {
        "signal": sig.name,
        "source": sig.source,
        "sign": sig.sign,
        "industry_demean": sig.industry_demean,
        "custom_source_hash": _source_hash(sig),
        "horizons": horizons,
        "primary_horizon": primary,
        "step": step,
        "incremental_weight": w_inc,
        "baseline_config_hash": pipe.cfg.hash(),
        "use_ml": bool(use_ml),
    }
    tid = f"{utc_stamp()}_{sig.name}"
    logged = False
    if log_trial and ledger_path:
        from qts.validation import TrialLedger

        led = TrialLedger(ledger_path)
        monthly = book.gross.groupby(book.gross.index.to_period("M")).sum()
        monthly.index = monthly.index.to_timestamp()
        led.log(
            name=f"{family}:{sig.name}",
            family=family,
            config=test_cfg,
            metrics={
                "t_primary": t_primary,
                "mean_ic_primary": float(ic_h[str(primary)]["mean_ic"]),
                "sharpe_annual": float(book.stats()["sharpe_gross"]),
                "n_obs": len(book.gross),
                "incremental_t_primary": float(inc_h[str(primary)]["t_nw"]),
                "delta_sharpe_net": float(incremental["delta_sharpe_net"]),
            },
            data_hash=data_hash_of(md),
            market_hash=hash_market(md),
            code_version=code_version(),
            returns_monthly=monthly,
            replace_identical=True,
            notes=f"signal test {tid}",
            extra={"kind": "trial", "test": "signal", "signal_test_id": tid, "market_hash_full": mh_family},
        )
        logged = True
    tests = _family_tests(ledger_path, family)
    if not logged:
        tests.append({"signal": sig.name, "t": t_primary, "ts": None, "id": tid})
    p_all = [two_sided_p(float(r["t"])) for r in tests]
    adj = holm_adjust(p_all)
    me = next((i for i, r in enumerate(tests) if r.get("id") == tid), len(tests) - 1)
    trial = {
        "family": family,
        "logged": logged,
        "n_tests": len(tests),
        "t": t_primary,
        "p": two_sided_p(t_primary),
        "p_holm": float(adj[me]) if len(adj) else float("nan"),
        "hurdle": hurdle,
        "passes_hurdle": bool(math.isfinite(t_primary) and abs(t_primary) >= hurdle),
        "tests": [{**r, "p": p, "p_holm": float(a)} for r, p, a in zip(tests, p_all, adj, strict=True)],
    }
    timings["total"] = time.perf_counter() - t_all
    res = SignalTestResult(
        id=tid,
        signal=sig,
        horizons=list(horizons),
        primary_horizon=primary,
        step=step,
        window=window,
        baseline={
            "config_hash": pipe.cfg.hash(),
            "horizon_days": H,
            "signal_lag_days": L,
            "use_ml": bool(use_ml),
            "cache": cache_info,
            "sealed_end": str(md.dates[-1].date()) if len(md.dates) else None,
        },
        coverage_by_year=coverage,
        ic_by_horizon=ic_h,
        ic_by_year=ic_y,
        ic_by_fold=folds,
        decay=decay,
        paper=book.stats(),
        correlations=corr,
        incremental=incremental,
        trial=trial,
        timings=timings,
        notes=notes,
    )
    if runs_root:
        out = Path(runs_root) / SIGNALS_DIRNAME / f"{tid}.json"
        res.json_path = str(out)
        write_json(out, res.to_dict())
    return res


def _z(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score, missing -> 0 (a constant or empty row -> all 0)."""
    x = s.astype(float)
    sd = float(x.std(ddof=0))
    if not math.isfinite(sd) or sd <= 0:
        return pd.Series(0.0, index=x.index)
    return ((x - x.mean()) / sd).fillna(0.0)


def _family_tests(ledger_path: str | None, family: str) -> list[dict[str, Any]]:
    """The signal tests of ``family`` (de-duplicated trials), in order."""
    if not ledger_path or not Path(ledger_path).exists():
        return []
    from qts.validation import TrialLedger

    out = []
    for r in TrialLedger(ledger_path).records(family):
        m = r.get("metrics") or {}
        out.append(
            {
                "signal": (r.get("config") or {}).get("signal"),
                "t": float(m.get("t_primary", float("nan"))),
                "ts": r.get("ts"),
                "id": r.get("signal_test_id"),
            }
        )
    return out


__all__ = [
    "DEFAULT_HORIZONS",
    "INCREMENTAL_WEIGHT",
    "PaperBook",
    "SignalDef",
    "SignalTestResult",
    "get_signal",
    "holm_adjust",
    "ic_stats",
    "list_signals",
    "neutral_rank_weights",
    "paper_book",
    "process_signal",
    "register_signal",
    "run_signal_test",
]
