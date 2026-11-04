"""Statistical hurdles for admitting a candidate signal / model (BUILD_PLAN s.8).

Gates (all must pass unless marked optional):

* ``t_stat``            decile long-short t > ``hurdle_t_stat`` (3.0) — applied
                        **only to standalone signals** (``standalone=True``);
                        for incremental features inside an ensemble the
                        incremental tests below replace it.
* ``incremental_ic``    incremental rank-IC > 0.005 in >= 2/3 of folds.
* ``net_sharpe_delta``  net Sharpe improvement >= +0.05 in >= 2/3 of folds.
* ``breakeven_cost``    breakeven cost >= 2x modeled cost.
* ``dsr``               deflated Sharpe > 0.95.
* ``pbo``               probability of backtest overfitting < 0.5.
* ``phase_sharpe``      phase-1 net Sharpe >= 0.3, phase-2 (and later) >= 0.5.
* ``suspect``           any net backtest Sharpe > 2 is flagged as suspect
                        (fails overall; DSR cannot detect leakage).

Each gate has a status ``PASS`` / ``FAIL`` / ``skipped``; a gate whose input is
missing is *skipped* and only fails ``overall`` if it is in
``ValidationConfig.required_gates`` (default: phase_sharpe, suspect, dsr).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from qts.core.config import ValidationConfig

INCREMENTAL_IC_MIN = 0.005
NET_SHARPE_DELTA_MIN = 0.05
BREAKEVEN_MULTIPLE = 2.0
SUSPECT_SHARPE = 2.0


@dataclass
class GateResult:
    name: str
    passed: Optional[bool]          # None when skipped
    value: object
    threshold: object
    reason: str
    applied: bool = True            # False == skipped (input missing or gate not applicable)
    required: bool = False          # a skipped *required* gate fails ``overall``

    @property
    def status(self) -> str:
        if not self.applied:
            return "skipped"
        return "PASS" if self.passed else "FAIL"


@dataclass
class HurdleReport:
    """Gate evaluation with three statuses per gate: ``PASS`` / ``FAIL`` /
    ``skipped``.  A gate is *skipped* when its input is not supplied (or it does
    not apply, e.g. the standalone t-stat for an incremental feature).  Skipped
    gates do not fail ``overall`` unless they are in ``cfg.required_gates``
    (default ``{phase_sharpe, suspect, dsr}``) — those must always be
    evaluated, so a missing input there is a FAIL with reason "not supplied".

    Standalone-run conventions (``ResearchPipeline``): ``net_sharpe_delta_by_fold``
    is the fold net Sharpe minus a baseline of zero, so the gate reads
    "net Sharpe by fold > +0.05 in >= 2/3 of folds"; ``pbo`` is only available
    when a configuration grid was run (``SystemConfig.pipeline_pbo_configs``),
    otherwise the PBO gate is skipped.
    """
    cfg: ValidationConfig = field(default_factory=ValidationConfig)

    # ------------------------------------------------------------------ helpers
    def _fold_fraction(self, s: Optional[pd.Series], thresh: float) -> tuple:
        if s is None or len(s) == 0:
            return float("nan"), 0, 0
        s = pd.Series(s).dropna()
        ok = int((s >= thresh).sum())
        return (ok / len(s) if len(s) else float("nan")), ok, len(s)

    @staticmethod
    def _skip(name: str, threshold: object, reason: str, value: object = None) -> GateResult:
        return GateResult(name, None, value, threshold, reason, applied=False)

    # ------------------------------------------------------------------ main
    def evaluate_signal(self, ic_by_fold: Optional[pd.Series] = None, decile_tstat: Optional[float] = None,
                        incremental_ic_by_fold: Optional[pd.Series] = None,
                        net_sharpe_delta_by_fold: Optional[pd.Series] = None,
                        breakeven_cost_bps: Optional[float] = None, modeled_cost_bps: Optional[float] = None,
                        dsr: Optional[float] = None, pbo: Optional[float] = None, phase: int = 1,
                        net_sharpe: Optional[float] = None, standalone: bool = True,
                        required: Optional[Iterable[str]] = None) -> Dict[str, object]:
        cfg = self.cfg
        gates: List[GateResult] = []
        frac_req = cfg.min_fold_pass_fraction
        req = set(cfg.required_gates if required is None else required)

        def _finite(x) -> bool:
            return x is not None and np.isfinite(float(x))

        # 1. decile long-short t-stat (standalone only)
        if not standalone:
            gates.append(self._skip("t_stat", cfg.hurdle_t_stat, "not applied (incremental feature)", decile_tstat))
        elif not _finite(decile_tstat):
            gates.append(self._skip("t_stat", cfg.hurdle_t_stat, "decile t-stat not supplied"))
        else:
            ok = float(decile_tstat) > cfg.hurdle_t_stat
            gates.append(GateResult("t_stat", ok, float(decile_tstat), cfg.hurdle_t_stat,
                                    f"decile L/S t = {float(decile_tstat):.2f} {'>' if ok else '<='} {cfg.hurdle_t_stat}"))

        # 1b. standalone IC positivity (mean IC by fold > 0 in 2/3 folds)
        if ic_by_fold is not None and len(ic_by_fold):
            frac, ok_n, n = self._fold_fraction(ic_by_fold, 0.0)
            gates.append(GateResult("ic_sign", frac >= frac_req, frac, frac_req, f"IC > 0 in {ok_n}/{n} folds"))
        else:
            gates.append(self._skip("ic_sign", frac_req, "IC by fold not supplied"))

        # 2. incremental IC
        if incremental_ic_by_fold is not None and len(incremental_ic_by_fold):
            frac, ok_n, n = self._fold_fraction(incremental_ic_by_fold, INCREMENTAL_IC_MIN)
            gates.append(GateResult("incremental_ic", frac >= frac_req, frac, frac_req,
                                    f"incremental IC > {INCREMENTAL_IC_MIN} in {ok_n}/{n} folds"))
        else:
            gates.append(self._skip("incremental_ic", frac_req,
                                    "not applied (standalone signal)" if standalone
                                    else "incremental IC by fold not supplied"))

        # 3. net Sharpe delta (standalone: baseline = 0, i.e. "net Sharpe by fold > +0.05")
        if net_sharpe_delta_by_fold is not None and len(net_sharpe_delta_by_fold):
            frac, ok_n, n = self._fold_fraction(net_sharpe_delta_by_fold, NET_SHARPE_DELTA_MIN)
            label = "net Sharpe by fold (vs zero baseline)" if standalone else "net Sharpe delta"
            gates.append(GateResult("net_sharpe_delta", frac >= frac_req, frac, frac_req,
                                    f"{label} >= +{NET_SHARPE_DELTA_MIN} in {ok_n}/{n} folds"))
        else:
            gates.append(self._skip("net_sharpe_delta", frac_req, "net Sharpe delta by fold not supplied"))

        # 4. breakeven cost
        if _finite(breakeven_cost_bps) and _finite(modeled_cost_bps):
            need = BREAKEVEN_MULTIPLE * float(modeled_cost_bps)
            ok = float(breakeven_cost_bps) >= need
            gates.append(GateResult("breakeven_cost", ok, float(breakeven_cost_bps), need,
                                    f"breakeven {float(breakeven_cost_bps):.1f} bps vs 2x modeled {need:.1f} bps"))
        elif breakeven_cost_bps is not None and np.isinf(float(breakeven_cost_bps)) and _finite(modeled_cost_bps):
            gates.append(GateResult("breakeven_cost", True, float("inf"), 2 * float(modeled_cost_bps),
                                    "zero turnover: breakeven undefined (inf)"))
        else:
            gates.append(self._skip("breakeven_cost", None, "breakeven/modeled cost not supplied", breakeven_cost_bps))

        # 5. DSR
        if not _finite(dsr):
            gates.append(self._skip("dsr", cfg.dsr_threshold, "DSR not supplied", dsr))
        else:
            ok = float(dsr) > cfg.dsr_threshold
            gates.append(GateResult("dsr", ok, float(dsr), cfg.dsr_threshold,
                                    f"DSR = {float(dsr):.3f} {'>' if ok else '<='} {cfg.dsr_threshold}"))

        # 6. PBO
        if not _finite(pbo):
            gates.append(self._skip("pbo", cfg.pbo_threshold,
                                    "PBO not supplied (no configuration grid run; set pipeline_pbo_configs)", pbo))
        else:
            ok = float(pbo) < cfg.pbo_threshold
            gates.append(GateResult("pbo", ok, float(pbo), cfg.pbo_threshold,
                                    f"PBO = {float(pbo):.3f} {'<' if ok else '>='} {cfg.pbo_threshold}"))

        # 7. phase net Sharpe + suspect flag
        min_sr = cfg.phase1_min_net_sharpe if phase <= 1 else cfg.phase2_min_net_sharpe
        if not _finite(net_sharpe):
            gates.append(self._skip("phase_sharpe", min_sr, "net Sharpe not supplied", net_sharpe))
            gates.append(self._skip("suspect", SUSPECT_SHARPE, "net Sharpe not supplied", net_sharpe))
        else:
            ok = float(net_sharpe) >= min_sr
            gates.append(GateResult("phase_sharpe", ok, float(net_sharpe), min_sr,
                                    f"phase-{phase} net Sharpe {float(net_sharpe):.2f} {'>=' if ok else '<'} {min_sr}"))
            suspect = float(net_sharpe) > SUSPECT_SHARPE
            gates.append(GateResult("suspect", not suspect, float(net_sharpe), SUSPECT_SHARPE,
                                    ("SUSPECT: net Sharpe > 2 — check for leakage before trusting DSR"
                                     if suspect else "net Sharpe within plausible range")))

        for g in gates:
            g.required = g.name in req
            if not g.applied and g.required:          # a required gate cannot be skipped
                g.applied, g.passed = True, False
                g.reason = f"REQUIRED gate skipped: {g.reason}"
        overall = all(g.passed for g in gates if g.applied)
        reasons = [f"[{g.status}] {g.name}: {g.reason}" for g in gates]
        return {
            "overall": overall,
            "phase": phase,
            "gates": {g.name: g.passed for g in gates},            # True / False / None (skipped)
            "status": {g.name: g.status for g in gates},           # PASS / FAIL / skipped
            "details": {g.name: {"value": g.value, "threshold": g.threshold, "reason": g.reason,
                                 "applied": g.applied, "required": g.required, "status": g.status} for g in gates},
            "reasons": reasons,
            "failed": [g.name for g in gates if g.applied and not g.passed],
            "skipped": [g.name for g in gates if not g.applied],
            "required": sorted(req),
        }

    @staticmethod
    def format(result: Dict[str, object]) -> str:
        head = f"OVERALL: {'PASS' if result['overall'] else 'FAIL'} (phase {result['phase']})"
        skipped = result.get("skipped") or []
        tail = [f"skipped (not required): {', '.join(skipped)}"] if skipped else []
        return "\n".join([head] + list(result["reasons"]) + tail)
