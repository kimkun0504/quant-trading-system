"""Capacity curve of the baseline (``qts capacity --navs 5e6,5e7,2.5e8,1e9``).

The baseline is fitted once (stage cache; the NAV is not part of a fit's cache key)
and only the backtest is re-run at each NAV, under the research seal.  NAV enters the
backtest twice: the cost model prices dollars (impact ``b sigma sqrt(Q / ADV)`` grows
with the trade size ``Q = |dw| x NAV``) and the optimizer's per-name trade cap is
``max_adv_participation x ADV / NAV``.  Per NAV: gross / net Sharpe, net CAGR, cost
per side, turnover, the share of traded names at the ADV participation cap (mean over
rebalance solves of names whose cap binds / names traded) and the share of solves with
one, the breakeven cost per side; and the NAV at which the net Sharpe falls below half
of its value at the smallest NAV (linear interpolation in log NAV between the two
bracketing points; ``None`` when it never does in the range).

Capacity runs are not alpha trials: each is logged with ``kind: "capacity"`` in its
own family ``capacity:<market_hash[:12]>`` (full-history hash under the seal), which
no deflated Sharpe ratio counts.

Entry point: :func:`run_capacity` -> :class:`CapacityResult` (``to_dict()`` is
``runs/_capacity/<id>.json``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from qts.research.common import BASELINE_CONFIG, adv_cap_share, fmt_float, json_safe, utc_stamp, write_json
from qts.research.data import Printer, StoreSpec, _quiet

if TYPE_CHECKING:
    from qts.core.config import SystemConfig
    from qts.core.types import MarketData

DEFAULT_NAVS: tuple[float, ...] = (5e6, 5e7, 2.5e8, 1e9)
FAMILY_PREFIX = "capacity:"
CAPACITY_DIRNAME = "_capacity"
#: the threshold of the "capacity NAV": net Sharpe below this fraction of its value at the smallest NAV
HALF = 0.5


@dataclass
class CapacityPoint:
    nav: float
    sharpe_gross: float
    sharpe_net: float
    cagr_net: float
    cost_bps_per_side: float
    turnover_per_day: float
    traded_at_adv_cap: float
    solves_with_adv_cap: float
    breakeven_cost_bps: float
    #: the cost per side split by kind (impact grows with NAV; the per-order minimum
    #: commission makes fees per side *fall* with NAV, so the total can be U-shaped)
    impact_bps_per_side: float = float("nan")
    spread_bps_per_side: float = float("nan")
    fees_bps_per_side: float = float("nan")
    run_id: str | None = None
    seconds: float = float("nan")


@dataclass
class CapacityResult:
    id: str
    config_hash: str
    family: str
    points: list[CapacityPoint]
    half_sharpe_nav: float | None
    reference_sharpe_net: float
    cache: dict[str, str]
    sealed_end: str | None
    notes: list[str] = field(default_factory=list)
    json_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    def format(self) -> str:
        lines = [
            f"=== Capacity curve: config {self.config_hash}, data through {self.sealed_end} (fit: {self.cache}) ===",
            f"{'NAV':>12} {'SR gross':>9} {'SR net':>7} {'CAGR net':>9} {'cost bp':>8} {'TO %/d':>7} "
            f"{'at ADV cap':>10} {'solves cap':>10} {'breakeven':>9} {'impact/spread/fees bp':>22}",
        ]
        for p in self.points:
            lines.append(
                f"{_money(p.nav):>12} {fmt_float(p.sharpe_gross):>9} {fmt_float(p.sharpe_net):>7} "
                f"{fmt_float(100 * p.cagr_net, '.2f'):>8}% {fmt_float(p.cost_bps_per_side, '.1f'):>8} "
                f"{fmt_float(100 * p.turnover_per_day, '.2f'):>7} {fmt_float(100 * p.traded_at_adv_cap, '.1f'):>9}% "
                f"{fmt_float(100 * p.solves_with_adv_cap, '.0f'):>9}% {fmt_float(p.breakeven_cost_bps, '.1f'):>9} "
                f"{fmt_float(p.impact_bps_per_side, '.1f'):>8}/{fmt_float(p.spread_bps_per_side, '.1f')}/"
                f"{fmt_float(p.fees_bps_per_side, '.1f')}"
            )
        ref = self.points[0].nav if self.points else float("nan")
        if self.half_sharpe_nav is None:
            lines.append(
                f"net Sharpe stays above {HALF:g} x its {_money(ref)} value ({fmt_float(self.reference_sharpe_net)}) "
                f"over the whole range"
            )
        else:
            lines.append(
                f"net Sharpe falls below {HALF:g} x its {_money(ref)} value ({fmt_float(self.reference_sharpe_net)}) "
                f"at about {_money(self.half_sharpe_nav)} (log-NAV interpolation)"
            )
        if self.notes:
            lines.append("notes: " + " | ".join(self.notes))
        if self.json_path:
            lines.append(f"written {self.json_path}")
        return "\n".join(lines)


def _money(x: float) -> str:
    if not math.isfinite(x):
        return "n/a"
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= div:
            return f"${x / div:g}{unit}"
    return f"${x:g}"


def cost_bps_by_kind(bt: Any) -> dict[str, float]:
    """Spread / impact / fees in bp of traded notional (the per-side cost split)."""
    traded = float(bt.traded_usd.sum())
    out = {}
    for k in ("spread", "impact", "fees"):
        out[k] = float(bt.costs[k].sum()) / traded * 1e4 if traded > 0 else float("nan")
    return out


def half_sharpe_nav(navs: Sequence[float], sharpes: Sequence[float], frac: float = HALF) -> float | None:
    """First NAV at which ``sharpe < frac x sharpe[0]`` (NAVs ascending), linearly
    interpolated in ``log NAV`` between the bracketing points; ``None`` when it never
    happens (or the first Sharpe is not positive)."""
    n = np.asarray(navs, dtype=float)
    s = np.asarray(sharpes, dtype=float)
    if len(n) < 2 or not math.isfinite(s[0]) or s[0] <= 0:
        return None
    target = frac * s[0]
    for i in range(1, len(n)):
        if math.isfinite(s[i]) and s[i] < target:
            a, b = s[i - 1], s[i]
            w = (a - target) / (a - b) if a != b else 1.0
            return float(math.exp(math.log(n[i - 1]) + w * (math.log(n[i]) - math.log(n[i - 1]))))
    return None


def run_capacity(
    spec: StoreSpec | None = None,
    *,
    navs: Sequence[float] = DEFAULT_NAVS,
    baseline: str | Path = BASELINE_CONFIG,
    config: SystemConfig | None = None,
    use_ml: bool = True,
    ledger_path: str | None = "runs/trial_ledger.jsonl",
    runs_root: str | None = "runs",
    cache_root: str | None = None,
    max_days: int | None = None,
    md: MarketData | None = None,
    progress: bool = False,
    log: Printer = _quiet,
) -> CapacityResult:
    """The capacity curve (module docstring).  ``md``: an in-memory market (tests)."""
    import time

    from qts.research.baseline import fit_baseline, variant_run

    navs = sorted({float(x) for x in navs})
    if not navs or navs[0] <= 0:
        raise ValueError("capacity: NAVs must be positive")
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
    family = f"{FAMILY_PREFIX}{bf.market_hash_family[:12]}"
    ctx_base = bf.load.context(2 if use_ml else 1) if bf.load is not None else None
    import dataclasses

    from qts.runs.context import RunContext

    cid = f"{utc_stamp()}_{bf.cfg.hash()[:8]}"
    points: list[CapacityPoint] = []
    for nav in navs:
        t0 = time.perf_counter()
        ctx = dataclasses.replace(
            ctx_base if ctx_base is not None else RunContext(),
            family=family,
            record_kind="capacity",
            reference=False,
            record_extra={"capacity_id": cid, "nav": nav},
        )
        res = variant_run(
            bf,
            nav=nav,
            context=ctx,
            ledger_path=ledger_path,
            runs_root=runs_root,
            max_days=max_days,
            progress=progress,
        )
        s = res.summary
        cap = adv_cap_share(res.backtest)
        by_kind = cost_bps_by_kind(res.backtest)
        pt = CapacityPoint(
            nav=nav,
            sharpe_gross=float(s["sharpe_gross"]),
            sharpe_net=float(s["sharpe_net"]),
            cagr_net=float(s["cagr_net"]),
            cost_bps_per_side=float(s["avg_cost_bps_per_side"]),
            turnover_per_day=float(s["avg_daily_turnover"]),
            traded_at_adv_cap=float(cap["traded_at_cap"]),
            solves_with_adv_cap=float(cap["solves_with_cap"]),
            breakeven_cost_bps=float(res.breakeven_cost_bps),
            impact_bps_per_side=by_kind["impact"],
            spread_bps_per_side=by_kind["spread"],
            fees_bps_per_side=by_kind["fees"],
            run_id=None if res.manifest is None else res.manifest.run_id,
            seconds=time.perf_counter() - t0,
        )
        points.append(pt)
        log(
            f"NAV {_money(nav)}: net Sharpe {fmt_float(pt.sharpe_net)}  cost {fmt_float(pt.cost_bps_per_side, '.1f')} bp/side  "
            f"({pt.seconds:.0f}s)"
        )
    hs = half_sharpe_nav([p.nav for p in points], [p.sharpe_net for p in points])
    notes = []
    if navs[0] != 5e6:
        notes.append(f"the reference NAV is the smallest one given ({_money(navs[0])}), not $5M")
    out = CapacityResult(
        id=cid,
        config_hash=bf.cfg.hash(),
        family=family,
        points=points,
        half_sharpe_nav=hs,
        reference_sharpe_net=points[0].sharpe_net,
        cache=bf.cache_info,
        sealed_end=str(bf.md.dates[-1].date()) if len(bf.md.dates) else None,
        notes=notes,
    )
    if runs_root:
        p = Path(runs_root) / CAPACITY_DIRNAME / f"{cid}.json"
        out.json_path = str(p)
        write_json(p, out.to_dict())
    return out


__all__ = ["DEFAULT_NAVS", "CapacityPoint", "CapacityResult", "half_sharpe_nav", "run_capacity"]
