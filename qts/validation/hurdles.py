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
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    passed: bool
    value: object
    threshold: object
    reason: str
    applied: bool = True


@dataclass
class HurdleReport:
    cfg: ValidationConfig = field(default_factory=ValidationConfig)

    # ------------------------------------------------------------------ helpers
    def _fold_fraction(self, s: Optional[pd.Series], thresh: float) -> tuple:
        if s is None or len(s) == 0:
            return float("nan"), 0, 0
        s = pd.Series(s).dropna()
        ok = int((s >= thresh).sum())
        return (ok / len(s) if len(s) else float("nan")), ok, len(s)

    # ------------------------------------------------------------------ main
    def evaluate_signal(self, ic_by_fold: Optional[pd.Series] = None, decile_tstat: Optional[float] = None,
                        incremental_ic_by_fold: Optional[pd.Series] = None,
                        net_sharpe_delta_by_fold: Optional[pd.Series] = None,
                        breakeven_cost_bps: Optional[float] = None, modeled_cost_bps: Optional[float] = None,
                        dsr: Optional[float] = None, pbo: Optional[float] = None, phase: int = 1,
                        net_sharpe: Optional[float] = None, standalone: bool = True) -> Dict[str, object]:
        cfg = self.cfg
        gates: List[GateResult] = []
        frac_req = cfg.min_fold_pass_fraction

        # 1. t-stat (standalone only)
        if standalone:
            if decile_tstat is None:
                gates.append(GateResult("t_stat", False, None, cfg.hurdle_t_stat, "decile t-stat not supplied"))
            else:
                ok = decile_tstat > cfg.hurdle_t_stat
                gates.append(GateResult("t_stat", ok, float(decile_tstat), cfg.hurdle_t_stat,
                                        f"decile L/S t = {decile_tstat:.2f} {'>' if ok else '<='} {cfg.hurdle_t_stat}"))
        else:
            gates.append(GateResult("t_stat", True, decile_tstat, cfg.hurdle_t_stat,
                                    "not applied (incremental feature)", applied=False))

        # 1b. standalone IC positivity (informational: mean IC by fold > 0 in 2/3 folds)
        if ic_by_fold is not None and len(ic_by_fold):
            frac, ok_n, n = self._fold_fraction(ic_by_fold, 0.0)
            gates.append(GateResult("ic_sign", frac >= frac_req, frac, frac_req,
                                    f"IC > 0 in {ok_n}/{n} folds"))

        # 2. incremental IC
        if incremental_ic_by_fold is not None and len(incremental_ic_by_fold):
            frac, ok_n, n = self._fold_fraction(incremental_ic_by_fold, INCREMENTAL_IC_MIN)
            gates.append(GateResult("incremental_ic", frac >= frac_req, frac, frac_req,
                                    f"incremental IC > {INCREMENTAL_IC_MIN} in {ok_n}/{n} folds"))
        elif not standalone:
            gates.append(GateResult("incremental_ic", False, None, frac_req, "incremental IC by fold not supplied"))

        # 3. net Sharpe delta
        if net_sharpe_delta_by_fold is not None and len(net_sharpe_delta_by_fold):
            frac, ok_n, n = self._fold_fraction(net_sharpe_delta_by_fold, NET_SHARPE_DELTA_MIN)
            gates.append(GateResult("net_sharpe_delta", frac >= frac_req, frac, frac_req,
                                    f"net Sharpe delta >= +{NET_SHARPE_DELTA_MIN} in {ok_n}/{n} folds"))
        elif not standalone:
            gates.append(GateResult("net_sharpe_delta", False, None, frac_req, "net Sharpe delta by fold not supplied"))

        # 4. breakeven cost
        if breakeven_cost_bps is not None and modeled_cost_bps is not None:
            need = BREAKEVEN_MULTIPLE * modeled_cost_bps
            ok = breakeven_cost_bps >= need
            gates.append(GateResult("breakeven_cost", ok, float(breakeven_cost_bps), need,
                                    f"breakeven {breakeven_cost_bps:.1f} bps vs 2x modeled {need:.1f} bps"))
        else:
            gates.append(GateResult("breakeven_cost", False, breakeven_cost_bps, None, "breakeven/modeled cost not supplied"))

        # 5. DSR
        if dsr is None or not np.isfinite(dsr):
            gates.append(GateResult("dsr", False, dsr, cfg.dsr_threshold, "DSR not supplied"))
        else:
            ok = dsr > cfg.dsr_threshold
            gates.append(GateResult("dsr", ok, float(dsr), cfg.dsr_threshold,
                                    f"DSR = {dsr:.3f} {'>' if ok else '<='} {cfg.dsr_threshold}"))

        # 6. PBO
        if pbo is None or not np.isfinite(pbo):
            gates.append(GateResult("pbo", False, pbo, cfg.pbo_threshold, "PBO not supplied"))
        else:
            ok = pbo < cfg.pbo_threshold
            gates.append(GateResult("pbo", ok, float(pbo), cfg.pbo_threshold,
                                    f"PBO = {pbo:.3f} {'<' if ok else '>='} {cfg.pbo_threshold}"))

        # 7. phase net Sharpe + suspect flag
        min_sr = cfg.phase1_min_net_sharpe if phase <= 1 else cfg.phase2_min_net_sharpe
        if net_sharpe is None:
            gates.append(GateResult("phase_sharpe", True, None, min_sr, "net Sharpe not supplied (gate skipped)", applied=False))
            gates.append(GateResult("suspect", True, None, SUSPECT_SHARPE, "net Sharpe not supplied", applied=False))
        else:
            ok = net_sharpe >= min_sr
            gates.append(GateResult("phase_sharpe", ok, float(net_sharpe), min_sr,
                                    f"phase-{phase} net Sharpe {net_sharpe:.2f} {'>=' if ok else '<'} {min_sr}"))
            suspect = net_sharpe > SUSPECT_SHARPE
            gates.append(GateResult("suspect", not suspect, float(net_sharpe), SUSPECT_SHARPE,
                                    ("SUSPECT: net Sharpe > 2 — check for leakage before trusting DSR"
                                     if suspect else "net Sharpe within plausible range")))

        overall = all(g.passed for g in gates if g.applied)
        reasons = [f"[{'PASS' if g.passed else 'FAIL'}] {g.name}: {g.reason}" for g in gates]
        return {
            "overall": overall,
            "phase": phase,
            "gates": {g.name: g.passed for g in gates},
            "details": {g.name: {"value": g.value, "threshold": g.threshold, "reason": g.reason, "applied": g.applied} for g in gates},
            "reasons": reasons,
            "failed": [g.name for g in gates if g.applied and not g.passed],
        }

    @staticmethod
    def format(result: Dict[str, object]) -> str:
        head = f"OVERALL: {'PASS' if result['overall'] else 'FAIL'} (phase {result['phase']})"
        return "\n".join([head] + list(result["reasons"]))
