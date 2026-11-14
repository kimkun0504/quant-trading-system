"""Risk-forecast calibration report (``qts risk calibrate``): realised / ex-ante vol of
the baseline book by year for a small grid of ``risk.alignment_nu_scale``.

Report only -- it changes no default.  The baseline is fitted once (stage cache) and
its backtest re-run with the fitted risk model's alignment variance scaled to each
``nu`` (:class:`qts.research.baseline.NuScaledRisk`; ``alignment_nu_scale`` is read by
``as_of`` only, so nothing is refitted), under the research seal.  The alignment
factor (Saxena & Stubbs) adds variance along the part of the alpha the factor model
does not span, so a larger ``nu`` raises the ex-ante vol of alpha-heavy books and the
optimizer de-risks them; the right value makes the forecast unbiased for the book the
optimizer actually builds.

Per ``nu`` and calendar year, over the *invested* decision days
(``BacktestResult.invested_days``; forecast of decision day ``t`` against the gross
return of ``t + 1``): realised annual vol, mean ex-ante annual vol, their ratio and the
bias statistic ``sd(r_t / sigma_t)`` (1 = unbiased); and the whole-period numbers plus
net Sharpe.  **Recommendation rule (fixed before looking):** the ``nu`` whose
whole-period ratio is closest to 1 in log terms, ``|log(realised / ex-ante)|``; a tie
within 0.01 goes to the smaller ``nu`` (less change).  Adopting it is a configuration
change of the strategy: one logged trial in the research family, run once under the
seal (``qts run ... --set risk.alignment_nu_scale=<nu>``), not a free default change.

Each variant run is logged with ``kind: "calibration"`` in the family
``calibration:<market_hash[:12]>`` (not counted by any DSR).

Entry point: :func:`run_risk_calibration` -> :class:`RiskCalibrationResult`
(``to_dict()`` is ``runs/_risk/<id>.json``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from qts.research.common import BASELINE_CONFIG, fmt_float, json_safe, utc_stamp, write_json
from qts.research.data import Printer, StoreSpec, _quiet

if TYPE_CHECKING:
    from qts.backtest import BacktestResult
    from qts.core.config import SystemConfig
    from qts.core.types import MarketData

DEFAULT_NU_GRID: tuple[float, ...] = (0.0, 0.05, 0.25, 1.0)
FAMILY_PREFIX = "calibration:"
RISK_DIRNAME = "_risk"
TIE_TOL = 0.01


def vol_by_year(bt: BacktestResult, periods_per_year: int = 252) -> tuple[pd.DataFrame, dict[str, float]]:
    """Year x {realised_vol, exante_vol, ratio, bias_stat, n_days} over invested days, and
    the same for the whole period (``BacktestResult.realized_vs_exante``'s alignment)."""
    from qts.portfolio.diagnostics import realized_vs_exante_risk

    g = bt.gross_returns.dropna()
    inv = bt.invested_days().reindex(bt.exante_var.index).fillna(False).astype(bool)
    exv = bt.exante_var.where(inv).shift(1).reindex(g.index)
    rows = {}
    for y in sorted(set(g.index.year)):
        sel = g.index.year == y
        r = realized_vs_exante_risk(g[sel], exv[sel], periods_per_year, min_obs=20)
        n = int((exv[sel].notna() & (exv[sel] > 0)).sum())
        rows[int(y)] = {
            "realised_vol": r["realized_vol_annual"],
            "exante_vol": r["exante_vol_annual"],
            "ratio": r["ratio"],
            "bias_stat": r["bias_stat"],
            "n_days": n,
        }
    tot = realized_vs_exante_risk(g, exv, periods_per_year, min_obs=20)
    return pd.DataFrame(rows).T, {
        "realised_vol": tot["realized_vol_annual"],
        "exante_vol": tot["exante_vol_annual"],
        "ratio": tot["ratio"],
        "bias_stat": tot["bias_stat"],
    }


@dataclass
class NuResult:
    nu_scale: float
    overall: dict[str, float]
    by_year: dict[str, dict[str, float]]
    sharpe_net: float
    run_id: str | None = None


@dataclass
class RiskCalibrationResult:
    id: str
    config_hash: str
    current_nu_scale: float
    results: list[NuResult]
    recommended_nu_scale: float | None
    rule: str
    family: str
    sealed_end: str | None
    cache: dict[str, str]
    notes: list[str] = field(default_factory=list)
    json_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    def format(self) -> str:
        nus = [r.nu_scale for r in self.results]
        lines = [
            f"=== Risk forecast calibration: realised / ex-ante vol of the baseline book "
            f"(config {self.config_hash}, data through {self.sealed_end}, fit {self.cache}) ===",
            f"{'year':>6} " + " ".join(f"{'nu=' + format(n, 'g'):>16s}" for n in nus),
        ]
        years = sorted({y for r in self.results for y in r.by_year})
        for y in years:
            cells = []
            for r in self.results:
                c = r.by_year.get(y, {})
                cells.append(f"{fmt_float(c.get('ratio')):>7} ({fmt_float(c.get('bias_stat')):>5})")
            lines.append(f"{y:>6} " + " ".join(f"{c:>16s}" for c in cells))
        lines.append(
            f"{'all':>6} "
            + " ".join(
                f"{fmt_float(r.overall['ratio']):>7} ({fmt_float(r.overall['bias_stat']):>5})"
                for r in self.results
            )
        )
        lines.append("        cells: realised / ex-ante vol (bias statistic sd(r / sigma)); 1 = unbiased")
        lines.append(
            "ex-ante vol %  "
            + "  ".join(
                f"nu={r.nu_scale:g}: {fmt_float(100 * r.overall['exante_vol'], '.1f')}" for r in self.results
            )
        )
        lines.append(
            "realised vol % "
            + "  ".join(
                f"nu={r.nu_scale:g}: {fmt_float(100 * r.overall['realised_vol'], '.1f')}"
                for r in self.results
            )
        )
        lines.append(
            "net Sharpe     "
            + "  ".join(f"nu={r.nu_scale:g}: {fmt_float(r.sharpe_net)}" for r in self.results)
        )
        rec = self.recommended_nu_scale
        lines.append(
            f"recommended alignment_nu_scale: {'n/a' if rec is None else format(rec, 'g')} "
            f"(current {self.current_nu_scale:g}; rule: {self.rule})"
        )
        lines.append(
            "adopting it is ONE logged trial: qts run <the baseline flags> --set risk.alignment_nu_scale=<value> "
            "(under the seal); the default is not changed by this report"
        )
        if self.notes:
            lines.append("notes: " + " | ".join(self.notes))
        if self.json_path:
            lines.append(f"written {self.json_path}")
        return "\n".join(lines)


def recommend(results: Sequence[NuResult], tol: float = TIE_TOL) -> float | None:
    """The ``nu`` minimising ``|log(ratio)|`` over the whole period; ties within ``tol``
    go to the smaller ``nu``."""
    scored = [
        (abs(math.log(r.overall["ratio"])), r.nu_scale)
        for r in results
        if isinstance(r.overall.get("ratio"), float)
        and math.isfinite(r.overall["ratio"])
        and r.overall["ratio"] > 0
    ]
    if not scored:
        return None
    best = min(s for s, _ in scored)
    return min(n for s, n in scored if s <= best + tol)


def run_risk_calibration(
    spec: StoreSpec | None = None,
    *,
    nu_grid: Sequence[float] = DEFAULT_NU_GRID,
    baseline: str | Path = BASELINE_CONFIG,
    config: SystemConfig | None = None,
    use_ml: bool = True,
    nav: float = 5e6,
    ledger_path: str | None = "runs/trial_ledger.jsonl",
    runs_root: str | None = "runs",
    cache_root: str | None = None,
    max_days: int | None = None,
    md: MarketData | None = None,
    progress: bool = False,
    log: Printer = _quiet,
) -> RiskCalibrationResult:
    """The report (module docstring)."""
    import dataclasses

    from qts.research.baseline import fit_baseline, variant_run
    from qts.runs.context import RunContext

    grid = sorted({float(x) for x in nu_grid})
    if not grid or grid[0] < 0:
        raise ValueError("risk calibrate: nu values must be >= 0")
    bf = fit_baseline(
        spec,
        config=config,
        baseline=baseline,
        md=md,
        use_ml=use_ml,
        runs_root=runs_root,
        cache_root=cache_root,
        log=log,
    )
    if not bf.cfg.risk.alignment:
        raise ValueError("risk calibrate: the baseline has risk.alignment off; nu has no effect")
    family = f"{FAMILY_PREFIX}{bf.market_hash_family[:12]}"
    ctx_base = bf.load.context(2 if use_ml else 1) if bf.load is not None else RunContext()
    rid = f"{utc_stamp()}_{bf.cfg.hash()[:8]}"
    out: list[NuResult] = []
    for nu in grid:
        ctx = dataclasses.replace(
            ctx_base, family=family, record_kind="calibration", reference=False,
            record_extra={"calibration_id": rid, "alignment_nu_scale": nu},
        )  # fmt: skip
        res = variant_run(
            bf, nav=nav, context=ctx, ledger_path=ledger_path, runs_root=runs_root, nu_scale=nu,
            max_days=max_days, progress=progress,
        )  # fmt: skip
        by_year, overall = vol_by_year(res.backtest, bf.cfg.trading_days_per_year)
        out.append(
            NuResult(
                nu_scale=nu,
                overall={k: float(v) for k, v in overall.items()},
                by_year={str(int(y)): {k: float(v) for k, v in row.items()} for y, row in by_year.iterrows()},
                sharpe_net=float(res.summary["sharpe_net"]),
                run_id=None if res.manifest is None else res.manifest.run_id,
            )
        )
        log(
            f"nu {nu:g}: realised / ex-ante {fmt_float(overall['ratio'])}  net Sharpe {fmt_float(res.summary['sharpe_net'])}"
        )
    rec = recommend(out)
    result = RiskCalibrationResult(
        id=rid,
        config_hash=bf.cfg.hash(),
        current_nu_scale=float(bf.cfg.risk.alignment_nu_scale),
        results=out,
        recommended_nu_scale=rec,
        rule="minimise |log(realised / ex-ante vol)| over the invested days of the pre-seal period; ties (0.01) -> smaller nu",
        family=family,
        sealed_end=str(bf.md.dates[-1].date()) if len(bf.md.dates) else None,
        cache=bf.cache_info,
    )
    if runs_root:
        p = Path(runs_root) / RISK_DIRNAME / f"{rid}.json"
        result.json_path = str(p)
        write_json(p, result.to_dict())
    return result


__all__ = [
    "DEFAULT_NU_GRID",
    "NuResult",
    "RiskCalibrationResult",
    "recommend",
    "run_risk_calibration",
    "vol_by_year",
]
