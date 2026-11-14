"""Fit the baseline once (stage cache) and re-run only its backtest under variants
that do not touch a fit: the NAV (``qts capacity``) and the risk model's alignment
variance scale (``qts risk calibrate``).

Why the fit can be shared: the stage-cache key of ``risk_fit`` / ``alpha_fit``
(``ResearchPipeline._cache_parts``) contains the data, the ``universe`` / ``risk`` /
``alpha`` config sections and the constructor knobs that change a fit -- not the
NAV (``nav0`` is a backtest input: the cost model's dollars and the ADV participation
cap scale with it) and not ``portfolio`` / ``cost`` / ``execution``.
``risk.alignment_nu_scale`` *is* in the risk section, but it is read only by
``RiskModel.as_of`` (the alignment variance added to the forecast), never by ``fit``:
the variant runs the fitted baseline risk model with ``as_of(..., nu_scale=nu)``
(:class:`NuScaledRisk`) and records the variant's own config (``risk.alignment_nu_scale
= nu``) on its run directory, so nothing is refitted and every number is labelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qts.research.common import BASELINE_CONFIG, pipeline_config, resolve_repo_path
from qts.research.data import Printer, StoreSpec, _quiet

if TYPE_CHECKING:
    from qts.core.config import SystemConfig
    from qts.core.types import MarketData
    from qts.pipeline import ResearchPipeline


@dataclass
class BaselineFit:
    """A fitted baseline pipeline on a (sealed) market."""

    md: MarketData
    cfg: SystemConfig
    pipe: ResearchPipeline
    market_hash_family: str  #: full-history market hash under the seal, else the load's
    load: Any = None

    @property
    def cache_info(self) -> dict[str, str]:
        return dict(self.pipe.cache_info_) or {"cache": "off"}


def fit_baseline(
    spec: StoreSpec | None = None,
    *,
    config: SystemConfig | None = None,
    baseline: str | Path = BASELINE_CONFIG,
    md: MarketData | None = None,
    use_ml: bool = True,
    runs_root: str | None = "runs",
    cache_root: str | None = None,
    log: Printer = _quiet,
) -> BaselineFit:
    """Load ``spec`` under the research seal (or use ``md``) and fit the baseline through
    the stage cache (``<runs_root>/_cache`` by default)."""
    from qts.core.config import SystemConfig
    from qts.core.manifest import hash_market
    from qts.pipeline import ResearchPipeline
    from qts.research.data import load_research_market

    base = config if config is not None else SystemConfig.from_yaml(resolve_repo_path(baseline))
    load = None
    mh = ""
    if md is None:
        if spec is None:
            raise ValueError("fit_baseline: give a store spec or a MarketData")
        load = load_research_market(
            spec, hash_cache_dir=f"{runs_root}/_cache/market_hash" if runs_root else None, log=log
        )
        md = load.md
        mh = load.market_hash_full
    cache = cache_root if cache_root is not None else (f"{runs_root}/_cache" if runs_root else None)
    cfg = pipeline_config(base, md, cache)
    pipe = ResearchPipeline(cfg, use_ml=use_ml, ledger_path=None, runs_root=None)
    pipe.fit(md)
    log(f"baseline fit: stage cache {dict(pipe.cache_info_) or 'off'}")
    return BaselineFit(md=md, cfg=cfg, pipe=pipe, market_hash_family=mh or hash_market(md), load=load)


class NuScaledRisk:
    """The fitted risk model with a fixed alignment variance scale: ``as_of(d, alpha)``
    is ``risk.as_of(d, alpha, nu_scale=nu)``; every other attribute is the model's."""

    def __init__(self, risk: Any, nu_scale: float):
        self._risk = risk
        self.nu_scale = float(nu_scale)

    def as_of(self, date: Any, alpha: Any = None, nu_scale: float | None = None) -> Any:
        return self._risk.as_of(date, alpha=alpha, nu_scale=self.nu_scale if nu_scale is None else nu_scale)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._risk, name)


def variant_run(
    bf: BaselineFit,
    *,
    nav: float,
    context: Any,
    ledger_path: str | None,
    runs_root: str | None,
    nu_scale: float | None = None,
    max_days: int | None = None,
    progress: bool = False,
) -> Any:
    """Backtest the fitted baseline once more at ``nav`` (and, with ``nu_scale``, the
    alignment scale) and evaluate / log / save it like any pipeline run.  The fitted
    models are shared, nothing is refitted (the pipeline's own refit rule only fires on
    a changed signal lag)."""
    import copy

    p = copy.copy(bf.pipe)  # shallow: the fitted models are shared, the knobs are this run's
    p.nav0 = float(nav)
    p.context = context
    p.ledger_path = ledger_path
    from qts.validation import TrialLedger

    p.ledger = TrialLedger(ledger_path) if ledger_path else None
    p.runs_root = runs_root
    if nu_scale is not None:
        p.cfg = bf.cfg.with_overrides({"risk.alignment_nu_scale": float(nu_scale)})
        p.risk = NuScaledRisk(bf.pipe.risk, nu_scale)  # type: ignore[assignment]
    res = p.run(bf.md, progress=progress, max_days=max_days)
    if p.alpha is not bf.pipe.alpha or (nu_scale is None and p.risk is not bf.pipe.risk):
        raise RuntimeError("variant_run: the pipeline refitted (it must not)")
    return res


__all__ = ["BaselineFit", "NuScaledRisk", "fit_baseline", "variant_run"]
