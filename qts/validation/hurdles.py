"""Statistical hurdles for admitting a candidate signal / model (BUILD_PLAN §2.4
"Research protocol with hurdle rates"; §8 for the validation protocol).

Gates (comparison operators exactly as implemented):

* ``t_stat``            decile long-short t **>** ``hurdle_t_stat`` (3.0) — applied
                        **only to standalone signals** (``standalone=True``);
                        for incremental features inside an ensemble the
                        incremental tests below replace it.
* ``ic_sign``           mean IC by fold **>** 0 in **>=** ``min_fold_pass_fraction`` (2/3) of folds.
* ``incremental_ic``    incremental rank IC **>** 0.005 in **>=** 2/3 of folds.
* ``net_sharpe_delta``  net Sharpe improvement **>=** +0.05 in **>=** 2/3 of folds.
* ``breakeven_cost``    breakeven cost **>=** 2x modeled cost.
* ``dsr``               deflated Sharpe **>** 0.95.
* ``pbo``               probability of backtest overfitting **<** 0.5.
* ``phase_sharpe``      phase-1 net Sharpe **>=** 0.3, phase-2 (and later) **>=** 0.5.
* ``suspect``           any net backtest Sharpe **>** 2 is flagged as suspect
                        (fails overall; DSR cannot detect leakage).

Each gate has a status ``PASS`` / ``FAIL`` / ``skipped``; a gate whose input is
missing is *skipped* and only fails ``overall`` if it is in
``ValidationConfig.required_gates`` (default: phase_sharpe, suspect, dsr).

:meth:`HurdleReport.evaluate_signal` returns a typed :class:`HurdleResult`
(list of :class:`GateResult`).  Dict-style access (``res["status"]``,
``res["overall"]``...) is kept for backward compatibility during the transition.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

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
    passed: bool | None  # None when skipped
    value: object
    threshold: object
    reason: str
    applied: bool = True  # False == skipped (input missing or gate not applicable)
    required: bool = False  # a skipped *required* gate fails ``overall``

    @property
    def status(self) -> str:
        if not self.applied:
            return "skipped"
        return "PASS" if self.passed else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "passed": self.passed,
            "value": _json_scalar(self.value),
            "threshold": _json_scalar(self.threshold),
            "reason": self.reason,
            "applied": self.applied,
            "required": self.required,
        }


def _json_scalar(v: Any) -> Any:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return v if math.isfinite(v) else str(v)
    return v


@dataclass
class HurdleResult:
    """Typed outcome of :meth:`HurdleReport.evaluate_signal`.

    ``gates`` keeps the evaluation order.  Mapping views (``status``,
    ``passed``, ``details``, ``failed``, ``skipped``, ``reasons``) are derived.
    ``res["key"]`` reproduces the legacy dict layout: ``overall``, ``phase``,
    ``gates`` (name -> True/False/None), ``status``, ``details``, ``reasons``,
    ``failed``, ``skipped``, ``required``.
    """

    overall: bool
    phase: int
    gates: list[GateResult]
    required: list[str]

    # ---- views
    def gate(self, name: str) -> GateResult:
        for g in self.gates:
            if g.name == name:
                return g
        raise KeyError(name)

    @property
    def passed(self) -> dict[str, bool | None]:
        return {g.name: g.passed for g in self.gates}

    @property
    def status(self) -> dict[str, str]:
        return {g.name: g.status for g in self.gates}

    @property
    def details(self) -> dict[str, dict[str, Any]]:
        return {
            g.name: {
                "value": g.value,
                "threshold": g.threshold,
                "reason": g.reason,
                "applied": g.applied,
                "required": g.required,
                "status": g.status,
            }
            for g in self.gates
        }

    @property
    def reasons(self) -> list[str]:
        return [f"[{g.status}] {g.name}: {g.reason}" for g in self.gates]

    @property
    def failed(self) -> list[str]:
        return [g.name for g in self.gates if g.applied and not g.passed]

    @property
    def skipped(self) -> list[str]:
        return [g.name for g in self.gates if not g.applied]

    # ---- legacy dict access
    _LEGACY = ("overall", "phase", "gates", "status", "details", "reasons", "failed", "skipped", "required")

    def __getitem__(self, key: str) -> Any:
        if key == "gates":
            return self.passed
        if key in self._LEGACY:
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in self._LEGACY

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in self._LEGACY else default

    def keys(self) -> Iterator[str]:
        return iter(self._LEGACY)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (``hurdles.json`` in a run directory)."""
        return {
            "overall": bool(self.overall),
            "phase": int(self.phase),
            "required": list(self.required),
            "gates": [g.to_dict() for g in self.gates],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HurdleResult:
        gates = []
        for g in d["gates"]:
            val, thr = g.get("value"), g.get("threshold")
            val = float(val) if isinstance(val, str) and val in ("nan", "inf", "-inf") else val
            thr = float(thr) if isinstance(thr, str) and thr in ("nan", "inf", "-inf") else thr
            gates.append(
                GateResult(
                    g["name"],
                    g.get("passed"),
                    val,
                    thr,
                    g.get("reason", ""),
                    applied=bool(g.get("applied", True)),
                    required=bool(g.get("required", False)),
                )
            )
        return cls(
            overall=bool(d["overall"]),
            phase=int(d["phase"]),
            gates=gates,
            required=list(d.get("required", [])),
        )

    def format(self) -> str:
        return HurdleReport.format(self)


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
    def _fold_fraction(self, s: pd.Series | None, thresh: float, strict: bool = False) -> tuple:
        """(fraction of folds passing, n passing, n folds); a fold passes when
        its value is ``> thresh`` (``strict``) or ``>= thresh``."""
        if s is None or len(s) == 0:
            return float("nan"), 0, 0
        s = pd.Series(s).dropna()
        ok = int(((s > thresh) if strict else (s >= thresh)).sum())
        return (ok / len(s) if len(s) else float("nan")), ok, len(s)

    @staticmethod
    def _skip(name: str, threshold: object, reason: str, value: object = None) -> GateResult:
        return GateResult(name, None, value, threshold, reason, applied=False)

    # ------------------------------------------------------------------ main
    def evaluate_signal(
        self,
        ic_by_fold: pd.Series | None = None,
        decile_tstat: float | None = None,
        incremental_ic_by_fold: pd.Series | None = None,
        net_sharpe_delta_by_fold: pd.Series | None = None,
        breakeven_cost_bps: float | None = None,
        modeled_cost_bps: float | None = None,
        dsr: float | None = None,
        pbo: float | None = None,
        phase: int = 1,
        net_sharpe: float | None = None,
        standalone: bool = True,
        required: Iterable[str] | None = None,
    ) -> HurdleResult:
        cfg = self.cfg
        gates: list[GateResult] = []
        frac_req = cfg.min_fold_pass_fraction
        req = set(cfg.required_gates if required is None else required)

        def _finite(x) -> bool:
            return x is not None and np.isfinite(float(x))

        # 1. decile long-short t-stat (standalone only)
        if not standalone:
            gates.append(
                self._skip("t_stat", cfg.hurdle_t_stat, "not applied (incremental feature)", decile_tstat)
            )
        elif not _finite(decile_tstat):
            gates.append(self._skip("t_stat", cfg.hurdle_t_stat, "decile t-stat not supplied"))
        else:
            ok = float(decile_tstat) > cfg.hurdle_t_stat
            gates.append(
                GateResult(
                    "t_stat",
                    ok,
                    float(decile_tstat),
                    cfg.hurdle_t_stat,
                    f"decile L/S t = {float(decile_tstat):.2f} {'>' if ok else '<='} {cfg.hurdle_t_stat}",
                )
            )

        # 1b. standalone IC positivity (mean IC by fold > 0 in 2/3 folds)
        if ic_by_fold is not None and len(ic_by_fold):
            frac, ok_n, n = self._fold_fraction(ic_by_fold, 0.0, strict=True)
            gates.append(
                GateResult(
                    "ic_sign",
                    frac >= frac_req,
                    frac,
                    frac_req,
                    f"IC > 0 in {ok_n}/{n} folds (need >= {frac_req:.2f})",
                )
            )
        else:
            gates.append(self._skip("ic_sign", frac_req, "IC by fold not supplied"))

        # 2. incremental IC
        if incremental_ic_by_fold is not None and len(incremental_ic_by_fold):
            frac, ok_n, n = self._fold_fraction(incremental_ic_by_fold, INCREMENTAL_IC_MIN, strict=True)
            gates.append(
                GateResult(
                    "incremental_ic",
                    frac >= frac_req,
                    frac,
                    frac_req,
                    f"incremental IC > {INCREMENTAL_IC_MIN} in {ok_n}/{n} folds (need >= {frac_req:.2f})",
                )
            )
        else:
            gates.append(
                self._skip(
                    "incremental_ic",
                    frac_req,
                    "not applied (standalone signal)"
                    if standalone
                    else "incremental IC by fold not supplied",
                )
            )

        # 3. net Sharpe delta (standalone: baseline = 0, i.e. "net Sharpe by fold > +0.05")
        if net_sharpe_delta_by_fold is not None and len(net_sharpe_delta_by_fold):
            frac, ok_n, n = self._fold_fraction(net_sharpe_delta_by_fold, NET_SHARPE_DELTA_MIN)
            label = "net Sharpe by fold (vs zero baseline)" if standalone else "net Sharpe delta"
            gates.append(
                GateResult(
                    "net_sharpe_delta",
                    frac >= frac_req,
                    frac,
                    frac_req,
                    f"{label} >= +{NET_SHARPE_DELTA_MIN} in {ok_n}/{n} folds (need >= {frac_req:.2f})",
                )
            )
        else:
            gates.append(self._skip("net_sharpe_delta", frac_req, "net Sharpe delta by fold not supplied"))

        # 4. breakeven cost
        if _finite(breakeven_cost_bps) and _finite(modeled_cost_bps):
            need = BREAKEVEN_MULTIPLE * float(modeled_cost_bps)
            ok = float(breakeven_cost_bps) >= need
            gates.append(
                GateResult(
                    "breakeven_cost",
                    ok,
                    float(breakeven_cost_bps),
                    need,
                    f"breakeven {float(breakeven_cost_bps):.1f} bps {'>=' if ok else '<'} 2x modeled {need:.1f} bps",
                )
            )
        elif (
            breakeven_cost_bps is not None
            and np.isinf(float(breakeven_cost_bps))
            and _finite(modeled_cost_bps)
        ):
            gates.append(
                GateResult(
                    "breakeven_cost",
                    True,
                    float("inf"),
                    2 * float(modeled_cost_bps),
                    "zero turnover: breakeven undefined (inf)",
                )
            )
        else:
            gates.append(
                self._skip("breakeven_cost", None, "breakeven/modeled cost not supplied", breakeven_cost_bps)
            )

        # 5. DSR
        if not _finite(dsr):
            gates.append(self._skip("dsr", cfg.dsr_threshold, "DSR not supplied", dsr))
        else:
            ok = float(dsr) > cfg.dsr_threshold
            gates.append(
                GateResult(
                    "dsr",
                    ok,
                    float(dsr),
                    cfg.dsr_threshold,
                    f"DSR = {float(dsr):.3f} {'>' if ok else '<='} {cfg.dsr_threshold}",
                )
            )

        # 6. PBO
        if not _finite(pbo):
            gates.append(
                self._skip(
                    "pbo",
                    cfg.pbo_threshold,
                    "PBO not supplied (no configuration grid run; set pipeline_pbo_configs)",
                    pbo,
                )
            )
        else:
            ok = float(pbo) < cfg.pbo_threshold
            gates.append(
                GateResult(
                    "pbo",
                    ok,
                    float(pbo),
                    cfg.pbo_threshold,
                    f"PBO = {float(pbo):.3f} {'<' if ok else '>='} {cfg.pbo_threshold}",
                )
            )

        # 7. phase net Sharpe + suspect flag
        min_sr = cfg.phase1_min_net_sharpe if phase <= 1 else cfg.phase2_min_net_sharpe
        if not _finite(net_sharpe):
            gates.append(self._skip("phase_sharpe", min_sr, "net Sharpe not supplied", net_sharpe))
            gates.append(self._skip("suspect", SUSPECT_SHARPE, "net Sharpe not supplied", net_sharpe))
        else:
            ok = float(net_sharpe) >= min_sr
            gates.append(
                GateResult(
                    "phase_sharpe",
                    ok,
                    float(net_sharpe),
                    min_sr,
                    f"phase-{phase} net Sharpe {float(net_sharpe):.2f} {'>=' if ok else '<'} {min_sr}",
                )
            )
            suspect = float(net_sharpe) > SUSPECT_SHARPE
            gates.append(
                GateResult(
                    "suspect",
                    not suspect,
                    float(net_sharpe),
                    SUSPECT_SHARPE,
                    (
                        "SUSPECT: net Sharpe > 2 — check for leakage before trusting DSR"
                        if suspect
                        else "net Sharpe within plausible range"
                    ),
                )
            )

        for g in gates:
            g.required = g.name in req
            if not g.applied and g.required:  # a required gate cannot be skipped
                g.applied, g.passed = True, False
                g.reason = f"REQUIRED gate skipped: {g.reason}"
        overall = all(bool(g.passed) for g in gates if g.applied)
        return HurdleResult(overall=overall, phase=int(phase), gates=gates, required=sorted(req))

    @staticmethod
    def format(result: HurdleResult | dict[str, Any]) -> str:
        head = f"OVERALL: {'PASS' if result['overall'] else 'FAIL'} (phase {result['phase']})"
        skipped = result.get("skipped") or []
        tail = [f"skipped (not required): {', '.join(skipped)}"] if skipped else []
        return "\n".join([head, *list(result["reasons"]), *tail])
