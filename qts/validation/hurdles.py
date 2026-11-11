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
* ``failed_solves``     share of rebalance solves that failed (book held)
                        **<=** ``max_failed_solve_frac`` (2 %); skipped when no
                        solve count is supplied.
* ``risk_model_sanity`` share of rebalance solves on which the risk model's
                        ex-ante vol of the incoming (or reference) book exceeded
                        ``max_exante_vol`` (200 %) **<=**
                        ``max_exante_vol_alert_frac`` (0.5 %); skipped when no
                        alert count is supplied.  A covariance blown up by a
                        data error makes a target-vol book liquidate and stay
                        flat for many half-lives (docs/notes/risk_outlier_fix_2026-11-10.md).

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
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from qts.core.config import ValidationConfig
from qts.validation.leakage import SUSPECT_SHARPE

INCREMENTAL_IC_MIN = 0.005
NET_SHARPE_DELTA_MIN = 0.05
BREAKEVEN_MULTIPLE = 2.0


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
    def _apply_required(g: GateResult, required: bool) -> GateResult:
        """Mark ``g`` required; a required gate cannot be skipped (missing input = FAIL)."""
        if required and not g.applied:
            return replace(
                g, applied=True, passed=False, reason=f"REQUIRED gate skipped: {g.reason}", required=True
            )
        return replace(g, required=required)

    @staticmethod
    def _skip(name: str, threshold: object, reason: str, value: object = None) -> GateResult:
        return GateResult(name, None, value, threshold, reason, applied=False)

    @staticmethod
    def _finite(x: object) -> bool:
        return x is not None and bool(np.isfinite(float(x)))  # type: ignore[arg-type]

    # ------------------------------------------------------------------ gates
    def _t_stat_gate(self, decile_tstat: float | None, standalone: bool) -> GateResult:
        """1. decile long-short t-stat (standalone signals only)."""
        thr = self.cfg.hurdle_t_stat
        if not standalone:
            return self._skip("t_stat", thr, "not applied (incremental feature)", decile_tstat)
        if decile_tstat is None or not self._finite(decile_tstat):
            return self._skip("t_stat", thr, "decile t-stat not supplied")
        t = float(decile_tstat)
        ok = t > thr
        return GateResult("t_stat", ok, t, thr, f"decile L/S t = {t:.2f} {'>' if ok else '<='} {thr}")

    def _ic_sign_gate(self, ic_by_fold: pd.Series | None) -> GateResult:
        """1b. standalone IC positivity (mean IC by fold > 0 in 2/3 folds)."""
        frac_req = self.cfg.min_fold_pass_fraction
        if ic_by_fold is None or not len(ic_by_fold):
            return self._skip("ic_sign", frac_req, "IC by fold not supplied")
        frac, ok_n, n = self._fold_fraction(ic_by_fold, 0.0, strict=True)
        return GateResult(
            "ic_sign",
            frac >= frac_req,
            frac,
            frac_req,
            f"IC > 0 in {ok_n}/{n} folds (need >= {frac_req:.2f})",
        )

    def _incremental_ic_gate(self, incremental_ic_by_fold: pd.Series | None, standalone: bool) -> GateResult:
        """2. incremental IC."""
        frac_req = self.cfg.min_fold_pass_fraction
        if incremental_ic_by_fold is None or not len(incremental_ic_by_fold):
            reason = (
                "not applied (standalone signal)" if standalone else "incremental IC by fold not supplied"
            )
            return self._skip("incremental_ic", frac_req, reason)
        frac, ok_n, n = self._fold_fraction(incremental_ic_by_fold, INCREMENTAL_IC_MIN, strict=True)
        return GateResult(
            "incremental_ic",
            frac >= frac_req,
            frac,
            frac_req,
            f"incremental IC > {INCREMENTAL_IC_MIN} in {ok_n}/{n} folds (need >= {frac_req:.2f})",
        )

    def _net_sharpe_delta_gate(
        self, net_sharpe_delta_by_fold: pd.Series | None, standalone: bool
    ) -> GateResult:
        """3. net Sharpe delta (standalone: baseline = 0, i.e. "net Sharpe by fold > +0.05")."""
        frac_req = self.cfg.min_fold_pass_fraction
        if net_sharpe_delta_by_fold is None or not len(net_sharpe_delta_by_fold):
            return self._skip("net_sharpe_delta", frac_req, "net Sharpe delta by fold not supplied")
        frac, ok_n, n = self._fold_fraction(net_sharpe_delta_by_fold, NET_SHARPE_DELTA_MIN)
        label = "net Sharpe by fold (vs zero baseline)" if standalone else "net Sharpe delta"
        return GateResult(
            "net_sharpe_delta",
            frac >= frac_req,
            frac,
            frac_req,
            f"{label} >= +{NET_SHARPE_DELTA_MIN} in {ok_n}/{n} folds (need >= {frac_req:.2f})",
        )

    def _breakeven_gate(self, breakeven_cost_bps: float | None, modeled_cost_bps: float | None) -> GateResult:
        """4. breakeven cost >= 2x modeled cost (zero turnover: breakeven inf passes)."""
        if self._finite(breakeven_cost_bps) and self._finite(modeled_cost_bps):
            be, mc = float(breakeven_cost_bps), float(modeled_cost_bps)  # type: ignore[arg-type]
            need = BREAKEVEN_MULTIPLE * mc
            ok = be >= need
            return GateResult(
                "breakeven_cost",
                ok,
                be,
                need,
                (
                    f"breakeven {be:.1f} bps {'>=' if ok else '<'} "
                    f"{BREAKEVEN_MULTIPLE:g}x modeled {mc:.1f} bps "
                    f"(= {need:.1f} bps)"
                ),
            )
        if (
            breakeven_cost_bps is not None
            and np.isinf(float(breakeven_cost_bps))
            and self._finite(modeled_cost_bps)
        ):
            return GateResult(
                "breakeven_cost",
                True,
                float("inf"),
                BREAKEVEN_MULTIPLE * float(modeled_cost_bps),  # type: ignore[arg-type]
                "zero turnover: breakeven undefined (inf)",
            )
        return self._skip("breakeven_cost", None, "breakeven/modeled cost not supplied", breakeven_cost_bps)

    def _dsr_gate(self, dsr: float | None) -> GateResult:
        """5. deflated Sharpe."""
        thr = self.cfg.dsr_threshold
        if dsr is None or not self._finite(dsr):
            return self._skip("dsr", thr, "DSR not supplied", dsr)
        ok = float(dsr) > thr
        return GateResult("dsr", ok, float(dsr), thr, f"DSR = {float(dsr):.3f} {'>' if ok else '<='} {thr}")

    def _pbo_gate(self, pbo: float | None) -> GateResult:
        """6. probability of backtest overfitting."""
        thr = self.cfg.pbo_threshold
        if pbo is None or not self._finite(pbo):
            return self._skip(
                "pbo", thr, "PBO not supplied (no configuration grid run; set pipeline_pbo_configs)", pbo
            )
        ok = float(pbo) < thr
        return GateResult("pbo", ok, float(pbo), thr, f"PBO = {float(pbo):.3f} {'<' if ok else '>='} {thr}")

    def _sharpe_gates(self, net_sharpe: float | None, phase: int) -> list[GateResult]:
        """7. phase net Sharpe + suspect flag."""
        cfg = self.cfg
        min_sr = cfg.phase1_min_net_sharpe if phase <= 1 else cfg.phase2_min_net_sharpe
        if net_sharpe is None or not self._finite(net_sharpe):
            return [
                self._skip("phase_sharpe", min_sr, "net Sharpe not supplied", net_sharpe),
                self._skip("suspect", SUSPECT_SHARPE, "net Sharpe not supplied", net_sharpe),
            ]
        sr = float(net_sharpe)
        ok = sr >= min_sr
        suspect = sr > SUSPECT_SHARPE
        return [
            GateResult(
                "phase_sharpe",
                ok,
                sr,
                min_sr,
                f"phase-{phase} net Sharpe {sr:.2f} {'>=' if ok else '<'} {min_sr}",
            ),
            GateResult(
                "suspect",
                not suspect,
                sr,
                SUSPECT_SHARPE,
                (
                    "SUSPECT: net Sharpe > 2 — check for leakage before trusting DSR"
                    if suspect
                    else "net Sharpe within plausible range"
                ),
            ),
        ]

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
        failed_solve_frac: float | None = None,
        exante_vol_alert_frac: float | None = None,
    ) -> HurdleResult:
        """Every gate in the module docstring's order; a missing input skips its
        gate unless the gate is required (``required``, default
        ``cfg.required_gates``), in which case it FAILs."""
        req = set(self.cfg.required_gates if required is None else required)
        gates = [
            self._t_stat_gate(decile_tstat, standalone),
            self._ic_sign_gate(ic_by_fold),
            self._incremental_ic_gate(incremental_ic_by_fold, standalone),
            self._net_sharpe_delta_gate(net_sharpe_delta_by_fold, standalone),
            self._breakeven_gate(breakeven_cost_bps, modeled_cost_bps),
            self._dsr_gate(dsr),
            self._pbo_gate(pbo),
            *self._sharpe_gates(net_sharpe, phase),
            self._failed_solves_gate(failed_solve_frac),
            self._risk_model_sanity_gate(exante_vol_alert_frac),
        ]
        gates = [self._apply_required(g, g.name in req) for g in gates]
        overall = all(bool(g.passed) for g in gates if g.applied)
        return HurdleResult(overall=overall, phase=int(phase), gates=gates, required=sorted(req))

    def _failed_solves_gate(self, frac: float | None) -> GateResult:
        """``failed_solves``: PASS when the failed-solve share is at most
        ``cfg.max_failed_solve_frac``.  A failed solve holds the current book, so
        an optimizer that fails often produces a stale "hold everything" backtest
        whose other statistics look ordinary."""
        thr = float(self.cfg.max_failed_solve_frac)
        if frac is None or not np.isfinite(float(frac)):
            return self._skip("failed_solves", thr, "failed-solve fraction not supplied")
        ok = float(frac) <= thr
        return GateResult(
            "failed_solves",
            ok,
            float(frac),
            thr,
            f"failed solves {float(frac):.1%} of rebalances {'<=' if ok else '>'} {thr:.1%}",
        )

    def _risk_model_sanity_gate(self, frac: float | None) -> GateResult:
        """``risk_model_sanity``: PASS when the share of rebalance solves with an
        ex-ante vol alert (``BacktestResult.diagnostics['exante_vol_alerts']``) is
        at most ``cfg.max_exante_vol_alert_frac``."""
        thr = float(self.cfg.max_exante_vol_alert_frac)
        if frac is None or not np.isfinite(float(frac)):
            return self._skip("risk_model_sanity", thr, "ex-ante vol alert fraction not supplied")
        ok = float(frac) <= thr
        lim = float(self.cfg.max_exante_vol)
        return GateResult(
            "risk_model_sanity",
            ok,
            float(frac),
            thr,
            f"ex-ante vol > {lim:.0%} on {float(frac):.2%} of rebalances {'<=' if ok else '>'} {thr:.2%}",
        )

    @staticmethod
    def format(result: HurdleResult | dict[str, Any]) -> str:
        head = f"OVERALL: {'PASS' if result['overall'] else 'FAIL'} (phase {result['phase']})"
        skipped = result.get("skipped") or []
        tail = [f"skipped (not required): {', '.join(skipped)}"] if skipped else []
        return "\n".join([head, *list(result["reasons"]), *tail])
