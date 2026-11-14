"""Parameter sweeps with honest trial accounting and PBO (``qts sweep``).

``qts sweep --config configs/baseline.yaml --param portfolio.trade_aversion=0.5,1,2
[--param KEY=V1,V2,...] [--grid | --zip] [store flags] [--dry-run] [--yes]``

1. **Plan** (:func:`plan_sweep`): the parameters expand to ``k`` configurations (the
   cartesian product with ``--grid``, the default; element-wise with ``--zip``), each
   validated by ``SystemConfig.with_overrides``; more than ``max_runs`` (12) is refused.
   The plan reads the ledger family the members will be logged in (the sealed load's
   full-history family, as ``qts run``) and prints the trial count before and after
   (members whose configuration is already a trial of the family are reference
   re-runs, not new trials) and the haircut the extra trials cause to the current
   best trial's deflated Sharpe ratio: ``E[max SR]`` (Bailey & Lopez de Prado) at the
   family's effective ``N`` and at ``N + k_new`` with the family's current cross-trial
   variance ``V`` (assumed unchanged), and the best trial's DSR at both.
2. **Run** (:func:`run_sweep`, after a confirmation unless ``--yes``): the store is
   loaded once under the seal; members run sequentially, each a normal pipeline run
   (run directory, ledger record with ``sweep_id`` / ``sweep_member``), with the stage
   cache on, so a member whose ``universe`` / ``risk`` / ``alpha`` sections equal an
   earlier one reuses its fits (a portfolio-only sweep fits once).  The sweep
   manifest ``runs/_sweeps/<id>.json`` records the parameters, every member's
   overrides, config hash, status (``pending`` / ``running`` / ``done`` / ``failed``),
   run id and headline numbers; it is rewritten after every member.  **Resumable**: the
   id is a hash of the base config, the parameters, the mode, the load flags and the
   run settings, so re-running the same command continues the sweep (members ``done``
   with an existing run directory are skipped; ``--id`` names a sweep explicitly).
3. **PBO** (:func:`finalize_sweep`): Combinatorially Symmetric Cross-Validation
   (:func:`qts.validation.pbo.probability_of_backtest_overfitting`, Bailey, Borwein,
   Lopez de Prado & Zhu 2017) on the members' daily net returns (``returns.parquet``
   of each run, aligned on common dates), ``S = 10`` contiguous blocks (``C(10, 5) =
   252`` splits; 10 keeps each block about two years long on the 2003-2023 pre-seal
   history -- long enough to contain a regime, short enough to give 252 splits; fewer
   than ``2 S`` days or fewer than two distinct members: PBO is not computed and the
   reason is given), metric = Sharpe.  Plus the table metric-vs-parameter.

API: a sweep is a sequence of jobs.  ``plan = plan_sweep(...)`` then
``create_sweep(plan)`` writes the manifest; the API queue runs
``run_sweep_member(sweep_id, i, ...)`` for each pending member (one job each, any
order, resumable) and ``finalize_sweep(sweep_id)`` when all are done (also safe to
call earlier: PBO over the members done so far, flagged ``partial``).  Everything
returns dataclasses with ``to_dict()``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

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
    from qts.core.types import MarketData

#: Default cap on the number of members.
DEFAULT_MAX_RUNS = 12
#: CSCV blocks.
PBO_BLOCKS = 10
SWEEPS_DIRNAME = "_sweeps"
MEMBER_STATUSES = ("pending", "running", "done", "failed")


class SweepError(ValueError):
    """A sweep cannot be planned or run."""


# ----------------------------------------------------------------------------- parameters
def parse_param(text: str) -> tuple[str, list[Any]]:
    """``"portfolio.trade_aversion=0.5,1,2"`` -> ``("portfolio.trade_aversion", [0.5, 1, 2])``
    (values parsed as YAML scalars, like ``--set``)."""
    import yaml

    key, sep, raw = str(text).partition("=")
    key = key.strip()
    if not sep or not key:
        raise SweepError(f"--param expects KEY=V1,V2,..., got {text!r}")
    vals: list[Any] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            v = yaml.safe_load(item)
        except yaml.YAMLError as e:
            raise SweepError(f"--param {key}: cannot parse {item!r}: {e}") from e
        if isinstance(v, (dict, list)):
            raise SweepError(f"--param {key}: values must be scalars")
        vals.append(v)
    if not vals:
        raise SweepError(f"--param {key}: no values")
    if len(set(map(repr, vals))) != len(vals):
        raise SweepError(f"--param {key}: repeated value")
    return key, vals


def expand(params: Sequence[tuple[str, Sequence[Any]]], mode: str = "grid") -> list[dict[str, Any]]:
    """Member overrides: the cartesian product (``grid``) or element-wise (``zip``,
    equal lengths)."""
    if not params:
        raise SweepError("a sweep needs at least one --param")
    keys = [k for k, _ in params]
    if len(set(keys)) != len(keys):
        raise SweepError(f"a parameter is listed twice: {keys}")
    if mode == "grid":
        combos = itertools.product(*[list(v) for _, v in params])
    elif mode == "zip":
        lens = {len(v) for _, v in params}
        if len(lens) != 1:
            raise SweepError(f"--zip needs the same number of values for every --param, got {sorted(lens)}")
        combos = zip(*[list(v) for _, v in params], strict=True)
    else:
        raise SweepError(f"mode must be grid or zip, got {mode!r}")
    return [dict(zip(keys, c, strict=True)) for c in combos]


# ----------------------------------------------------------------------------- plan
@dataclass
class TrialImpact:
    """Effect of the sweep's new trials on the family's deflated Sharpe ratio."""

    family: str
    n_trials_raw: int
    n_eff_before: float
    n_new: int
    n_reference: int
    n_eff_after: float
    var_sr_annual: float
    exp_max_sr_before: float
    exp_max_sr_after: float
    best_sharpe: float | None
    best_n_obs: int | None
    dsr_best_before: float | None
    dsr_best_after: float | None

    def format(self) -> str:
        lines = [
            f"ledger family {self.family}: {self.n_trials_raw} trial(s), N_eff {self.n_eff_before:.1f} now; "
            f"this sweep adds {self.n_new} new trial(s) ({self.n_reference} member(s) re-run counted configurations) "
            f"-> N_eff about {self.n_eff_after:.1f}",
            f"expected max Sharpe of the search (annual, V = {self.var_sr_annual:.3f}): "
            f"{fmt_float(self.exp_max_sr_before)} -> {fmt_float(self.exp_max_sr_after)}",
        ]
        if self.best_sharpe is not None:
            lines.append(
                f"current best trial (net Sharpe {self.best_sharpe:.2f}): DSR {fmt_float(self.dsr_best_before, '.3f')} "
                f"-> {fmt_float(self.dsr_best_after, '.3f')} after the sweep"
            )
        return "\n".join(lines)


def trial_impact(ledger_path: str | None, family: str, n_new: int, n_reference: int = 0) -> TrialImpact:
    """:class:`TrialImpact` from the family's current ledger statistics.  ``N_eff`` after
    the sweep is estimated as ``N_eff + n_new`` (new members are treated as independent
    trials: conservative for the haircut; Nyholt will count correlated members as less)."""
    from qts.validation import TrialLedger
    from qts.validation.deflated_sharpe import deflated_sharpe_ratio_annual, expected_max_sharpe

    led = TrialLedger(ledger_path) if ledger_path and Path(ledger_path).exists() else None
    recs = led.records(family) if led is not None else []
    n_raw = len(recs)
    best = None
    for r in recs:
        m = r.get("metrics") or {}
        sr = m.get("sharpe_annual")
        if isinstance(sr, (int, float)) and math.isfinite(sr) and (best is None or sr > best[0]):
            best = (
                float(sr),
                int(m.get("n_obs") or 0),
                float(m.get("skew", 0.0) or 0.0),
                float(m.get("kurt", 3.0) or 3.0),
            )
    n_obs = best[1] if best else 5000
    if led is not None and n_raw:
        d = led.dsr_inputs(family, max(n_obs, 2))
        n_eff, var = float(d["n_trials"]), float(d["var_trials_sr_annual"])
    else:
        n_eff, var = 1.0, 0.0
    if var <= 0:  # one trial or none: use the sampling-noise floor of an annual SR
        var = 252.0 / max(n_obs, 2)
    after = n_eff + n_new if n_raw else max(float(n_new), 1.0)
    q = 252.0

    def emax(n: float) -> float:
        return expected_max_sharpe(n, var / q) * math.sqrt(q)

    dsr_b = dsr_a = None
    if best is not None and best[1] > 2:
        dsr_b = deflated_sharpe_ratio_annual(best[0], max(n_eff, 1.0), var, best[1], best[2], best[3])
        dsr_a = deflated_sharpe_ratio_annual(best[0], max(after, 1.0), var, best[1], best[2], best[3])
    return TrialImpact(
        family=family,
        n_trials_raw=n_raw,
        n_eff_before=n_eff if n_raw else 0.0,
        n_new=int(n_new),
        n_reference=int(n_reference),
        n_eff_after=after,
        var_sr_annual=var,
        exp_max_sr_before=emax(n_eff) if n_raw else 0.0,
        exp_max_sr_after=emax(after),
        best_sharpe=None if best is None else best[0],
        best_n_obs=None if best is None else best[1],
        dsr_best_before=dsr_b,
        dsr_best_after=dsr_a,
    )


@dataclass
class SweepMember:
    index: int
    overrides: dict[str, Any]
    config_hash: str
    status: str = "pending"
    run_id: str | None = None
    run_dir: str | None = None
    reference: bool = False
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None


@dataclass
class SweepPlan:
    """A planned (or loaded) sweep; ``to_dict()`` is the manifest."""

    id: str
    base_config: str
    base_config_hash: str
    params: list[tuple[str, list[Any]]]
    mode: str
    store: dict[str, Any] | None
    use_ml: bool
    nav: float
    max_days: int | None
    members: list[SweepMember]
    family: str
    impact: TrialImpact | None = None
    created_at: str = ""
    pbo: dict[str, Any] | None = None
    runs_root: str = "runs"

    @property
    def path(self) -> Path:
        return Path(self.runs_root) / SWEEPS_DIRNAME / f"{self.id}.json"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["params"] = [{"key": k, "values": list(v)} for k, v in self.params]
        d["n_members"] = len(self.members)
        d["status"] = self.status
        return json_safe(d)

    @property
    def status(self) -> str:
        st = {m.status for m in self.members}
        if st == {"done"}:
            return "done"
        if "running" in st:
            return "running"
        if "failed" in st:
            return "failed"
        return "pending" if st == {"pending"} else "partial"

    def save(self) -> Path:
        return write_json(self.path, self.to_dict())

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SweepPlan:
        imp = d.get("impact")
        return cls(
            id=str(d["id"]),
            base_config=str(d["base_config"]),
            base_config_hash=str(d["base_config_hash"]),
            params=[(p["key"], list(p["values"])) for p in d["params"]],
            mode=str(d["mode"]),
            store=d.get("store"),
            use_ml=bool(d["use_ml"]),
            nav=float(d["nav"]),
            max_days=d.get("max_days"),
            members=[SweepMember(**m) for m in d["members"]],
            family=str(d["family"]),
            impact=None if imp is None else TrialImpact(**imp),
            created_at=str(d.get("created_at") or ""),
            pbo=d.get("pbo"),
            runs_root=str(d.get("runs_root") or "runs"),
        )

    def format(self) -> str:
        keys = [k for k, _ in self.params]
        lines = [
            f"=== Sweep {self.id} ({self.mode}, {len(self.members)} runs) on {self.base_config} "
            f"(hash {self.base_config_hash}) ===",
            f"{'#':>3} {'status':8s} " + " ".join(f"{k:>28s}" for k in keys) + f" {'config':>16s}",
        ]
        for m in self.members:
            lines.append(
                f"{m.index:>3} {m.status:8s} "
                + " ".join(f"{m.overrides[k]!s:>28s}" for k in keys)
                + f" {m.config_hash:>16s}"
                + ("  (reference re-run)" if m.reference else "")
            )
        if self.impact is not None:
            lines += ["", self.impact.format()]
        return "\n".join(lines)


def sweep_id(
    base_hash: str,
    params: Sequence[tuple[str, Sequence[Any]]],
    mode: str,
    store: Mapping[str, Any] | None,
    use_ml: bool,
    nav: float,
    max_days: int | None,
) -> str:
    payload = json.dumps(
        [
            base_hash,
            [[k, list(v)] for k, v in params],
            mode,
            dict(store or {}),
            bool(use_ml),
            float(nav),
            max_days,
        ],
        sort_keys=True,
        default=str,
    )
    return "sw_" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def plan_sweep(
    params: Sequence[tuple[str, Sequence[Any]]],
    *,
    base_config: str | Path = BASELINE_CONFIG,
    config: SystemConfig | None = None,
    mode: str = "grid",
    spec: StoreSpec | None = None,
    family: str | None = None,
    use_ml: bool = True,
    nav: float = 5e6,
    max_runs: int = DEFAULT_MAX_RUNS,
    max_days: int | None = None,
    ledger_path: str | None = "runs/trial_ledger.jsonl",
    runs_root: str = "runs",
    sweep_name: str | None = None,
    log: Printer = _quiet,
) -> SweepPlan:
    """Expand and validate the members, resolve the family (``family``, else the
    sealed full-history family of ``spec`` -- computed from the close panel only, cached),
    mark members whose configuration is already a counted trial there, estimate the
    DSR haircut.  An existing manifest of the same id is loaded instead (resume)."""
    from qts.core.config import SystemConfig
    from qts.research.store_run import reference_trial

    base = config if config is not None else SystemConfig.from_yaml(resolve_repo_path(base_config))
    params = [(k, list(v)) for k, v in params]
    overrides = expand(params, mode)
    if len(overrides) > int(max_runs):
        raise SweepError(
            f"the sweep has {len(overrides)} runs > --max-runs {max_runs}: narrow it or raise --max-runs"
        )
    store = None if spec is None else spec.to_dict()
    sid = sweep_name or sweep_id(base.hash(), params, mode, store, use_ml, nav, max_days)
    existing = Path(runs_root) / SWEEPS_DIRNAME / f"{sid}.json"
    if existing.exists():
        plan = SweepPlan.from_dict(json.loads(existing.read_text()))
        log(
            f"sweep {sid}: resuming ({sum(m.status == 'done' for m in plan.members)} of {len(plan.members)} done)"
        )
        return plan
    phase = 2 if use_ml else 1
    if family is None:
        if spec is None:
            raise SweepError("plan_sweep: give a store spec or a family")
        family = _family_for(spec, phase, runs_root, log)
    members: list[SweepMember] = []
    seen: set[str] = set()
    for i, ov in enumerate(overrides):
        try:
            cfg = base.with_overrides(ov)
        except (KeyError, TypeError, ValueError) as e:
            raise SweepError(f"member {i} {ov}: {e.args[0] if e.args else e}") from e
        h = cfg.hash()
        if h in seen:
            raise SweepError(f"member {i} {ov} repeats another member's configuration")
        seen.add(h)
        ref = reference_trial(ledger_path, family, h, phase) is not None
        members.append(SweepMember(index=i, overrides=ov, config_hash=h, reference=ref))
    n_ref = sum(m.reference for m in members)
    impact = trial_impact(ledger_path, family, len(members) - n_ref, n_ref)
    return SweepPlan(
        id=sid,
        base_config=str(base_config) if config is None else "<config>",
        base_config_hash=base.hash(),
        params=params,
        mode=mode,
        store=store,
        use_ml=use_ml,
        nav=float(nav),
        max_days=max_days,
        members=members,
        family=family,
        impact=impact,
        created_at=utc_stamp(),
        runs_root=str(runs_root),
    )


def _family_for(spec: StoreSpec, phase: int, runs_root: str, log: Printer) -> str:
    """The family a ``qts run`` with ``spec``'s flags logs into: the full-history market
    hash's research family (under the seal and without it alike)."""
    from qts.research.data import full_history_market_hash
    from qts.validation.ledger import research_family

    return research_family(phase, full_history_market_hash(spec, f"{runs_root}/_cache/market_hash", log))


def create_sweep(plan: SweepPlan) -> Path:
    """Write the manifest (idempotent: an existing manifest is left as it is)."""
    if plan.path.exists():
        return plan.path
    return plan.save()


def load_sweep(sweep_id_: str, runs_root: str = "runs") -> SweepPlan:
    p = Path(runs_root) / SWEEPS_DIRNAME / f"{sweep_id_}.json"
    if not p.exists():
        raise SweepError(f"no sweep manifest {p}")
    return SweepPlan.from_dict(json.loads(p.read_text()))


# ----------------------------------------------------------------------------- run
@dataclass
class _Loaded:
    load: Any
    md: MarketData


def _load(plan: SweepPlan, log: Printer) -> _Loaded:
    from qts.research.data import load_research_market

    if plan.store is None:
        raise SweepError("this sweep has no store spec: pass md= to run_sweep_member")
    load = load_research_market(
        StoreSpec.from_dict(plan.store), hash_cache_dir=f"{plan.runs_root}/_cache/market_hash", log=log
    )
    return _Loaded(load=load, md=load.md)


def run_sweep_member(
    sweep: str | SweepPlan,
    index: int,
    *,
    runs_root: str = "runs",
    ledger_path: str | None = "runs/trial_ledger.jsonl",
    config: SystemConfig | None = None,
    md: MarketData | None = None,
    loaded: Any = None,
    force: bool = False,
    progress: bool = False,
    log: Printer = _quiet,
) -> SweepMember:
    """Run member ``index`` (skipped when already ``done`` with its run directory, unless
    ``force``); updates and saves the manifest.  ``md`` / ``loaded``: an already loaded
    market (tests; a :class:`qts.research.data.ResearchLoad`); ``config``: the base
    configuration when the manifest has none on disk (``base_config == "<config>"``)."""
    import time

    from qts.core.config import SystemConfig
    from qts.pipeline import ResearchPipeline
    from qts.research.common import pipeline_config
    from qts.research.data import ResearchLoad
    from qts.research.store_run import StoreRun

    plan = load_sweep(sweep, runs_root) if isinstance(sweep, str) else sweep
    m = plan.members[index]
    if m.status == "done" and m.run_dir and Path(m.run_dir).exists() and not force:
        return m
    base = config if config is not None else SystemConfig.from_yaml(resolve_repo_path(plan.base_config))
    if base.hash() != plan.base_config_hash:
        raise SweepError(
            f"base config hash {base.hash()} != the sweep's {plan.base_config_hash}: the base file changed"
        )
    if loaded is None:
        loaded = ResearchLoad(md=md) if md is not None else _load(plan, log).load
    cfg_m = base.with_overrides(m.overrides)
    cfg_run = pipeline_config(cfg_m, loaded.md, f"{runs_root}/_cache")
    phase = 2 if plan.use_ml else 1
    sr = StoreRun(load=loaded, phase=phase)
    ctx = sr.finalize(cfg_run, ledger_path, log=log)
    import dataclasses

    if md is not None and not ctx.family:
        ctx = dataclasses.replace(ctx, family=plan.family)  # library use on an in-memory market
    ctx = dataclasses.replace(
        ctx, record_extra={**dict(ctx.record_extra), "sweep_id": plan.id, "sweep_member": index}
    )
    m.status, m.started_at, m.error = "running", utc_stamp(), None
    m.reference = bool(ctx.reference)
    plan.save()
    t0 = time.perf_counter()
    try:
        pipe = ResearchPipeline(
            cfg_run,
            use_ml=plan.use_ml,
            ledger_path=ledger_path,
            nav0=plan.nav,
            runs_root=runs_root,
            context=ctx,
        )
        res = pipe.run(loaded.md, progress=progress, max_days=plan.max_days)
    except Exception as e:  # recorded; the sweep can be resumed after a fix
        m.status, m.error, m.finished_at = "failed", f"{type(e).__name__}: {e}", utc_stamp()
        plan.save()
        raise
    s = res.summary
    m.status, m.finished_at = "done", utc_stamp()
    m.run_id = None if res.manifest is None else res.manifest.run_id
    m.run_dir = res.run_dir
    m.metrics = json_safe(
        {
            "sharpe_net": s["sharpe_net"],
            "sharpe_gross": s["sharpe_gross"],
            "cagr_net": s["cagr_net"],
            "avg_daily_turnover": s["avg_daily_turnover"],
            "avg_cost_bps_per_side": s["avg_cost_bps_per_side"],
            "max_drawdown_net": s["max_drawdown_net"],
            "mean_ic": res.ic_report.mean_ic,
            "dsr": res.dsr,
            "n_trials": res.dsr_inputs.get("n_trials"),
            "cache": dict(pipe.cache_info_),
            "seconds": time.perf_counter() - t0,
        }
    )
    plan.save()
    log(
        f"sweep {plan.id} member {index} {m.overrides}: net Sharpe {fmt_float(s['sharpe_net'])}  "
        f"cache {pipe.cache_info_}  ({m.metrics['seconds']:.0f}s)"
    )
    return m


@dataclass
class SweepResult:
    plan: SweepPlan
    pbo: dict[str, Any]
    table: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return json_safe({"sweep": self.plan.to_dict(), "pbo": self.pbo, "table": self.table})

    def format(self) -> str:
        keys = [k for k, _ in self.plan.params]
        cols = [
            "sharpe_net",
            "sharpe_gross",
            "cagr_net",
            "avg_daily_turnover",
            "avg_cost_bps_per_side",
            "mean_ic",
            "dsr",
        ]
        lines = [
            "",
            f"=== Sweep {self.plan.id}: metric vs parameter ===",
            " ".join(f"{k.split('.')[-1]:>18s}" for k in keys)
            + " "
            + " ".join(f"{c[:12]:>12s}" for c in cols),
        ]
        for r in self.table:
            lines.append(
                " ".join(f"{r['params'][k]!s:>18s}" for k in keys)
                + " "
                + " ".join(f"{fmt_float(r.get(c), '.3f'):>12s}" for c in cols)
                + ("  ref" if r.get("reference") else "")
            )
        p = self.pbo
        if p.get("pbo") is None:
            lines.append(f"PBO not computed: {p.get('reason')}")
        else:
            lines.append(
                f"PBO (CSCV, S = {p['n_blocks']} blocks, {p['n_combinations']} splits, {p['n_days']} days, "
                f"{p['n_configs']} configs{', PARTIAL' if p.get('partial') else ''}): {p['pbo']:.3f}   "
                f"P(IS-best loses OOS) {p['prob_oos_loss']:.3f}   degradation slope {fmt_float(p['degradation_slope'])}"
            )
        lines.append(f"manifest {self.plan.path}")
        return "\n".join(lines)


def sweep_pbo(returns: pd.DataFrame, n_blocks: int = PBO_BLOCKS) -> dict[str, Any]:
    """CSCV PBO of the columns of ``returns`` (dates x members, aligned; NaN rows
    dropped) with metric Sharpe; ``{"pbo": None, "reason": ...}`` when undefined."""
    from qts.validation.pbo import probability_of_backtest_overfitting

    M = returns.dropna(how="any")
    base: dict[str, Any] = {"n_blocks": int(n_blocks), "n_days": len(M), "n_configs": int(M.shape[1])}
    if M.shape[1] < 2:
        return {**base, "pbo": None, "reason": "fewer than two members with returns"}
    if len(M) < 2 * n_blocks:
        return {**base, "pbo": None, "reason": f"{len(M)} common days < 2 x {n_blocks} blocks"}
    try:
        r = probability_of_backtest_overfitting(M, n_groups=int(n_blocks), metric="sharpe")
    except ValueError as e:
        return {**base, "pbo": None, "reason": str(e)}
    return {
        **base,
        "pbo": float(r["pbo"]),
        "n_combinations": int(r["n_combinations"]),
        "prob_oos_loss": float(r["prob_oos_loss"]),
        "degradation_slope": float(r["degradation_slope"]),
        "degradation_r2": float(r["degradation_r2"]),
        "mean_oos_rank_pct": float(r["mean_oos_rank_pct"]),
        "logits": [float(x) for x in np.asarray(r["logits"], dtype=float)],
    }


def finalize_sweep(
    sweep: str | SweepPlan, runs_root: str = "runs", n_blocks: int = PBO_BLOCKS
) -> SweepResult:
    """PBO over the members done so far (``partial`` when some are not) and the table
    metric-vs-parameter; stored in the manifest (``pbo``)."""
    plan = load_sweep(sweep, runs_root) if isinstance(sweep, str) else sweep
    cols: dict[str, pd.Series] = {}
    table = []
    for m in plan.members:
        row = {
            "index": m.index,
            "params": m.overrides,
            "status": m.status,
            "run_id": m.run_id,
            "reference": m.reference,
        }
        row.update({k: v for k, v in (m.metrics or {}).items() if k != "cache"})
        table.append(row)
        if m.status == "done" and m.run_dir and (Path(m.run_dir) / "returns.parquet").exists():
            r = pd.read_parquet(Path(m.run_dir) / "returns.parquet")
            if "net" in r:
                cols[f"m{m.index}"] = r["net"]
    pbo = sweep_pbo(pd.DataFrame(cols), n_blocks) if cols else {"pbo": None, "reason": "no member done"}
    pbo["partial"] = any(m.status != "done" for m in plan.members)
    plan.pbo = json_safe(pbo)
    plan.save()
    return SweepResult(plan=plan, pbo=pbo, table=table)


def run_sweep(
    plan: SweepPlan,
    *,
    ledger_path: str | None = "runs/trial_ledger.jsonl",
    config: SystemConfig | None = None,
    md: MarketData | None = None,
    progress: bool = False,
    log: Printer = _quiet,
) -> SweepResult:
    """Create the manifest, run every member not done (sequentially, one store load),
    then :func:`finalize_sweep`.  A failed member stops the sweep (re-run to resume)."""
    from qts.research.data import ResearchLoad

    create_sweep(plan)
    pending = [m for m in plan.members if not (m.status == "done" and m.run_dir and Path(m.run_dir).exists())]
    if pending:
        loaded = ResearchLoad(md=md) if md is not None else _load(plan, log).load
        for m in pending:
            run_sweep_member(
                plan, m.index, runs_root=plan.runs_root, ledger_path=ledger_path, config=config,
                md=md, loaded=loaded, progress=progress, log=log,
            )  # fmt: skip
    return finalize_sweep(plan, plan.runs_root)


__all__ = [
    "DEFAULT_MAX_RUNS",
    "PBO_BLOCKS",
    "SweepError",
    "SweepMember",
    "SweepPlan",
    "SweepResult",
    "TrialImpact",
    "create_sweep",
    "expand",
    "finalize_sweep",
    "load_sweep",
    "parse_param",
    "plan_sweep",
    "run_sweep",
    "run_sweep_member",
    "sweep_pbo",
    "trial_impact",
]
