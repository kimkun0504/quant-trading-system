"""Research and paper-trading tasks as background jobs (the worker side).

The run queue (:mod:`qts_api.services.jobs`) runs pipeline runs with ``run_job``; every
other long-running library call goes through the same queue as a *task*: one worker
process per job, the same persisted job file, log and progress, cancellation and
restart recovery.  ``run_task(job_path, payload)`` dispatches on ``payload["task"]``:

=====================  ==============================================================
``signal_test``        :func:`qts.research.signals.run_signal_test` (one trial in ``signal:<hash>``)
``sweep_member``       :func:`qts.research.sweep.run_sweep_member` (one trial each)
``sweep_finalize``     :func:`qts.research.sweep.finalize_sweep` (PBO + table)
``capacity``           :func:`qts.research.capacity.run_capacity` (``kind: capacity``, not trials)
``risk_calibration``   :func:`qts.research.risk_calibration.run_risk_calibration` (report only)
``midcap_check``       :func:`qts.research.midcap.run_midcap_check` (once per configuration)
``paper_run``          :func:`qts.paper.run_book` (``StaleDataError`` -> exit code 3)
``paper_reconcile``    :func:`qts.paper.reconcile`
``market_hash``        :func:`qts.research.data.full_history_market_hash` (resolves the
                       trial family of a store slice once, so previews need no store read)
=====================  ==============================================================

Each task is called exactly as the matching ``qts`` command calls it (same defaults, the
API's runs root and ledger), and its job records the equivalent command line
(:func:`task_command`).  Store loads go through the research seal inside the library
functions (``load_research_market``); the mid-cap check is exempt by design.

Progress is coarse and honest: a few fixed stages, plus "k of n" where the library logs
one line per unit of work (one NAV, one nu, one sweep member's backtest days).  The job's
``result`` is a flat summary (ids, the headline number); ``result_payload`` keeps the
library's whole dict for the results that are not written to disk by the library itself
(a mid-cap dry run, a reconciliation, a paper run's day list).
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shlex
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qts_api.services.jobs import _attach_log, _now, _Reporter, job_log

TASKS: tuple[str, ...] = (
    "signal_test",
    "sweep_member",
    "sweep_finalize",
    "capacity",
    "risk_calibration",
    "midcap_check",
    "paper_run",
    "paper_reconcile",
    "market_hash",
)
#: Exit codes of ``qts paper`` (qts/paper/cli.py), reported on failed paper jobs.
EXIT_STALE, EXIT_LOCKED, EXIT_ERROR = 3, 4, 1
DEFAULT_BASELINE = "configs/baseline.yaml"


# ----------------------------------------------------------------------------- command lines
def g(v: float) -> str:
    """Python's ``%g`` (what the CLI prints and what the UI's command builder mirrors)."""
    return f"{float(v):g}"


def store_flags(spec: dict[str, Any], store_path: str | None) -> list[str]:
    parts = ["--store", str(store_path or spec.get("store") or "<store>")]
    for flag, key in (("--start", "start"), ("--end", "end")):
        if spec.get(key):
            parts += [flag, str(spec[key])]
    if spec.get("max_names") is not None:
        parts += ["--max-names", str(int(spec["max_names"]))]
    if spec.get("min_mcap_ever") is not None:
        parts += ["--min-mcap-ever", g(spec["min_mcap_ever"])]
    if spec.get("max_rank") is not None:
        parts += ["--max-rank", str(int(spec["max_rank"]))]
    return parts


def _paper_root_flag(paper_root: str | None) -> list[str]:
    if not paper_root:
        return []
    return [] if Path(paper_root).resolve() == Path("paper").resolve() else ["--paper-root", str(paper_root)]


def task_command(task: str, params: dict[str, Any], store_path: str | None, paper_root: str | None = None) -> str:
    """The ``qts`` command line that does what the task does (recorded on the job; the
    UI's launch forms print the same text before launch)."""
    p = params
    spec = p.get("store") or {}
    parts: list[str]
    if task == "signal_test":
        parts = ["qts", "signal", "test", str(p["feature"]), *store_flags(spec, store_path)]
        parts += ["--horizons", ",".join(str(int(h)) for h in p["horizons"])]
        if int(p.get("step", 5)) != 5:
            parts += ["--step", str(int(p["step"]))]
        if p.get("cost_bps") is not None:
            parts += ["--cost-bps", g(p["cost_bps"])]
        if p.get("baseline", DEFAULT_BASELINE) != DEFAULT_BASELINE:
            parts += ["--baseline", str(p["baseline"])]
        if not p.get("use_ml", True):
            parts.append("--no-ml")
        if not p.get("cache", True):
            parts.append("--no-cache")
        if not p.get("log_trial", True):
            parts.append("--no-log")
    elif task in ("sweep_member", "sweep_finalize"):
        parts = ["qts", "sweep", "--config", str(p.get("base_config", DEFAULT_BASELINE))]
        for prm in p.get("params") or []:
            parts += ["--param", f"{prm['key']}={','.join(_yaml(v) for v in prm['values'])}"]
        if p.get("mode") == "zip":
            parts.append("--zip")
        parts += store_flags(spec, store_path)
        if float(p.get("nav", 5e6)) != 5e6:
            parts += ["--nav", g(p["nav"])]
        if p.get("max_days") is not None:
            parts += ["--max-days", str(int(p["max_days"]))]
        if not p.get("use_ml", True):
            parts.append("--no-ml")
        if p.get("sweep_id"):
            parts += ["--id", str(p["sweep_id"])]
    elif task == "capacity":
        parts = ["qts", "capacity", *store_flags(spec, store_path), "--navs", ",".join(g(x) for x in p["navs"])]
        if p.get("baseline", DEFAULT_BASELINE) != DEFAULT_BASELINE:
            parts += ["--config", str(p["baseline"])]
        if p.get("max_days") is not None:
            parts += ["--max-days", str(int(p["max_days"]))]
        if not p.get("use_ml", True):
            parts.append("--no-ml")
    elif task == "risk_calibration":
        parts = ["qts", "risk", "calibrate", *store_flags(spec, store_path), "--nu", ",".join(g(x) for x in p["nu"])]
        if float(p.get("nav", 5e6)) != 5e6:
            parts += ["--nav", g(p["nav"])]
        if p.get("baseline", DEFAULT_BASELINE) != DEFAULT_BASELINE:
            parts += ["--config", str(p["baseline"])]
        if p.get("max_days") is not None:
            parts += ["--max-days", str(int(p["max_days"]))]
        if not p.get("use_ml", True):
            parts.append("--no-ml")
    elif task == "midcap_check":
        parts = ["qts", "check", "midcap", *store_flags(spec, store_path)]
        if p.get("config", DEFAULT_BASELINE) != DEFAULT_BASELINE:
            parts += ["--config", str(p["config"])]
        if float(p.get("nav", 5e6)) != 5e6:
            parts += ["--nav", g(p["nav"])]
        if p.get("cache"):
            parts.append("--cache")
        if p.get("dry_run"):
            parts.append("--dry-run")
        if p.get("force"):
            parts.append("--force")
        if p.get("reference_run"):
            parts += ["--reference-run", str(p["reference_run"])]
    elif task == "paper_run":
        parts = ["qts", "paper", "run", "--name", str(p["book"]), *_paper_root_flag(paper_root)]
        for flag, key in (("--date", "date"), ("--until", "until")):
            if p.get(key):
                parts += [flag, str(p[key])]
        for flag, key in (("--no-sync", "no_sync"), ("--dry-run", "dry_run"), ("--redo-last", "redo_last")):
            if p.get(key):
                parts.append(flag)
        if p.get("signal_mode", "daily") != "daily":
            parts += ["--signal-mode", str(p["signal_mode"])]
    elif task == "paper_reconcile":
        parts = ["qts", "paper", "reconcile", "--name", str(p["book"]), *_paper_root_flag(paper_root)]
    elif task == "market_hash":
        # no command of its own: `qts sweep --dry-run` / a sealed `qts run` compute and cache it
        parts = ["qts", "sweep", "--param", "portfolio.trade_aversion=1.0", *store_flags(spec, store_path), "--dry-run"]
    else:
        raise ValueError(f"unknown task {task!r}")
    return " ".join(shlex.quote(x) for x in parts)


def _yaml(v: Any) -> str:
    if v is None:
        return "null"
    if v is True or v is False:
        return "true" if v else "false"
    if isinstance(v, float):
        return g(v) if not float(v).is_integer() else str(v)
    return str(v)


# ----------------------------------------------------------------------------- the worker
class _Progress:
    """Turns the library's log lines into job progress: the line itself becomes the
    job's message, and ``units`` regexes count finished units of work out of ``total``
    between ``lo`` and ``hi``."""

    def __init__(self, rep: _Reporter, lo: float = 0.1, hi: float = 0.95, total: int = 0, unit: str | None = None):
        self.rep, self.lo, self.hi, self.total = rep, lo, hi, max(int(total), 0)
        self.unit = re.compile(unit) if unit else None
        self.done = 0
        self.frac = lo

    def __call__(self, msg: str) -> None:
        for line in str(msg).splitlines() or [""]:
            if not line.strip():
                continue
            job_log.info("%s", line)
            if self.unit is not None and self.unit.search(line) and self.total:
                self.done = min(self.done + 1, self.total)
                self.frac = self.lo + (self.hi - self.lo) * self.done / self.total
            with contextlib.suppress(Exception):
                self.rep.update(force=False, progress=round(self.frac, 4), message=line[:300])


def _spec(params: dict[str, Any], store_path: str | None) -> Any:
    from qts.research.data import StoreSpec

    if not store_path:
        raise ValueError("research tasks read the data store: set QTS_STORE_PATH on the API server")
    d = {k: v for k, v in (params.get("store") or {}).items() if v is not None}
    d["store"] = str(store_path)
    return StoreSpec.from_dict(d)


def _install_stage_hooks(rep: _Reporter, prog: _Progress) -> Callable[[], None]:
    """Stage messages from ``qts.pipeline``'s factory functions (as ``run_job`` does);
    returns the undo."""
    import qts.pipeline as P

    orig = (P.build_universe_mask, P.build_risk_model, P.AlphaModel)

    def universe(*a: Any, **k: Any):
        prog("Building the point-in-time universe")
        return orig[0](*a, **k)

    def risk(*a: Any, **k: Any):
        prog("Fitting the risk model (or reading it from the stage cache)")
        return orig[1](*a, **k)

    def alpha(*a: Any, **k: Any):
        prog("Fitting the alpha model (or reading it from the stage cache)")
        return orig[2](*a, **k)

    P.build_universe_mask, P.build_risk_model, P.AlphaModel = universe, risk, alpha  # type: ignore[assignment]

    def undo() -> None:
        P.build_universe_mask, P.build_risk_model, P.AlphaModel = orig  # type: ignore[assignment]

    return undo


class _DayLines(logging.Handler):
    """The simulator's every-50-days ``progress`` line as the job's message."""

    def __init__(self, rep: _Reporter):
        super().__init__(logging.INFO)
        self.rep = rep

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if str(record.msg).startswith("%s nav=") and record.args:
                d = record.args[0] if isinstance(record.args, tuple) else None
                self.rep.update(force=False, message=f"Running the backtest: {d}")
        except Exception:
            pass


def _run(task: str, payload: dict[str, Any], rep: _Reporter) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run one task; ``(result, result_payload)``."""
    p = payload["params"]
    runs_root = str(payload["runs_root"])
    ledger = payload.get("ledger_path") or None
    store_path = payload.get("store_path")
    if task == "signal_test":
        from qts.research.signals import run_signal_test

        prog = _Progress(rep, 0.05, 0.95)
        prog("Loading the store under the seal and the baseline fit (stage cache)")
        r = run_signal_test(
            str(p["feature"]),
            _spec(p, store_path),
            horizons=[int(h) for h in p["horizons"]],
            baseline=str(p.get("baseline", DEFAULT_BASELINE)),
            step=int(p.get("step", 5)),
            cost_bps=p.get("cost_bps"),
            ledger_path=ledger,
            runs_root=runs_root,
            cache_root=None if p.get("cache", True) else "",
            use_ml=bool(p.get("use_ml", True)),
            log_trial=bool(p.get("log_trial", True)),
            log=prog,
        )
        t = r.trial
        return {
            "result_id": r.id,
            "path": r.json_path,
            "t": t.get("t"),
            "p_holm": t.get("p_holm"),
            "n_tests": t.get("n_tests"),
            "passes_hurdle": bool(t.get("passes_hurdle")),
        }, None
    if task == "sweep_member":
        from qts.research.sweep import load_sweep, run_sweep_member

        plan = load_sweep(str(p["sweep_id"]), runs_root)
        prog = _Progress(rep, 0.05, 0.95)
        prog(
            f"Sweep {plan.id}: member {int(p['index']) + 1} of {len(plan.members)} {plan.members[int(p['index'])].overrides}"
        )
        undo = _install_stage_hooks(rep, prog)
        sim = logging.getLogger("qts.backtest.simulator")
        h = _DayLines(rep)
        sim.addHandler(h)
        try:
            m = run_sweep_member(plan.id, int(p["index"]), runs_root=runs_root, ledger_path=ledger, log=prog)
        finally:
            sim.removeHandler(h)
            undo()
        return {
            "result_id": plan.id,
            "index": int(p["index"]),
            "run_id": m.run_id,
            "status": m.status,
            "reference": bool(m.reference),
            "sharpe_net": (m.metrics or {}).get("sharpe_net"),
        }, None
    if task == "sweep_finalize":
        from qts.research.sweep import finalize_sweep

        _Progress(rep, 0.1, 0.9)("Computing PBO (CSCV) over the members' daily net returns")
        res = finalize_sweep(str(p["sweep_id"]), runs_root)
        return {"result_id": res.plan.id, "pbo": res.pbo.get("pbo"), "partial": bool(res.pbo.get("partial"))}, None
    if task == "capacity":
        from qts.research.capacity import run_capacity

        navs = [float(x) for x in p["navs"]]
        prog = _Progress(rep, 0.25, 0.95, total=len(navs), unit=r"^NAV \S+: net Sharpe")
        prog(f"Fitting the baseline once (stage cache), then {len(navs)} backtests")
        r = run_capacity(
            _spec(p, store_path),
            navs=navs,
            baseline=str(p.get("baseline", DEFAULT_BASELINE)),
            use_ml=bool(p.get("use_ml", True)),
            ledger_path=ledger,
            runs_root=runs_root,
            max_days=p.get("max_days"),
            log=prog,
        )
        return {"result_id": r.id, "path": r.json_path, "half_sharpe_nav": r.half_sharpe_nav}, None
    if task == "risk_calibration":
        from qts.research.risk_calibration import run_risk_calibration

        nus = [float(x) for x in p["nu"]]
        prog = _Progress(rep, 0.25, 0.95, total=len(nus), unit=r"^nu \S+: realised")
        prog(f"Fitting the baseline once (stage cache), then {len(nus)} backtests")
        r = run_risk_calibration(
            _spec(p, store_path),
            nu_grid=nus,
            baseline=str(p.get("baseline", DEFAULT_BASELINE)),
            use_ml=bool(p.get("use_ml", True)),
            nav=float(p.get("nav", 5e6)),
            ledger_path=ledger,
            runs_root=runs_root,
            max_days=p.get("max_days"),
            log=prog,
        )
        return {"result_id": r.id, "path": r.json_path, "recommended_nu_scale": r.recommended_nu_scale}, None
    if task == "midcap_check":
        from qts.research.midcap import run_midcap_check

        prog = _Progress(rep, 0.05, 0.95)
        undo = _install_stage_hooks(rep, prog)
        sim = logging.getLogger("qts.backtest.simulator")
        h = _DayLines(rep)
        sim.addHandler(h)
        try:
            r = run_midcap_check(
                _spec(p, store_path),
                config_path=str(p.get("config", DEFAULT_BASELINE)),
                ledger_path=ledger,
                runs_root=runs_root,
                cache_root=f"{runs_root}/_cache" if p.get("cache") else None,
                nav=float(p.get("nav", 5e6)),
                force=bool(p.get("force")),
                dry_run=bool(p.get("dry_run")),
                reference_run=p.get("reference_run"),
                log=prog,
            )
        finally:
            sim.removeHandler(h)
            undo()
        v = r.verdict
        return {
            "result_id": None if v is None else v.run_id,
            "path": r.json_path,
            "dry_run": bool(r.dry_run),
            "outcome": None if v is None else v.outcome,
            "coverage_ok": None if r.coverage is None else bool(r.coverage.ok),
        }, r.to_dict()
    if task == "paper_run":
        from qts.paper import run_book

        prog = _Progress(rep, 0.05, 0.95)
        rep_ = run_book(
            str(p["book"]),
            date=p.get("date"),
            until=p.get("until"),
            no_sync=bool(p.get("no_sync")),
            dry_run=bool(p.get("dry_run")),
            redo_last=bool(p.get("redo_last")),
            paper_root=str(payload["paper_root"]),
            signal_mode=str(p.get("signal_mode", "daily")),
            progress=prog,
        )
        st = rep_.get("status") or {}
        return {
            "result_id": str(p["book"]),
            "message": rep_.get("message"),
            "target": rep_.get("target"),
            "n_days": len(rep_.get("days") or []),
            "level": st.get("level"),
            "dry_run": bool(rep_.get("dry_run", p.get("dry_run", False))),
        }, rep_
    if task == "paper_reconcile":
        from qts.paper import reconcile

        _Progress(rep, 0.1, 0.9)("Re-running the batch simulator over the book's dates (one fit)")
        out = reconcile(str(p["book"]), paper_root=str(payload["paper_root"]))
        return {
            "result_id": str(p["book"]),
            "max_abs_weight_diff": out.get("max_abs_weight_diff"),
            "max_rel_nav_diff": out.get("max_rel_nav_diff"),
            "daily_return_corr": out.get("daily_return_corr"),
        }, out
    if task == "market_hash":
        from qts.research.data import full_history_market_hash

        prog = _Progress(rep, 0.1, 0.95)
        prog("Hashing the full-history close panel of these load flags (once per store snapshot)")
        mh = full_history_market_hash(_spec(p, store_path), cache_dir=f"{runs_root}/_cache/market_hash", log=prog)
        return {"result_id": mh[:12], "market_hash_full": mh}, None
    raise ValueError(f"unknown task {task!r}")


def _error_kind(e: BaseException) -> tuple[str, int | None]:
    try:
        from qts.paper.book import BookLockedError, PaperError, StaleDataError
    except ImportError:  # pragma: no cover
        return "error", None
    from qts.research.seal import SealError

    if isinstance(e, StaleDataError):
        return "stale_data", EXIT_STALE
    if isinstance(e, BookLockedError):
        return "book_locked", EXIT_LOCKED
    if isinstance(e, PaperError):
        return "paper", EXIT_ERROR
    if isinstance(e, SealError):
        return "seal", 2
    name = type(e).__name__
    if name == "LoadCoverageError":
        return "coverage", 2
    if name in ("MidcapCheckError", "SweepError", "StoreLoadError"):
        return "refused", 2
    return "error", None


STALE_HINT = (
    "Data not yet published: the store does not hold the session the vendor should have published by now. "
    "Nothing was changed; run it again later (the next run catches up day by day)."
)
LOCKED_HINT = "The book is locked by another job (a run, an ack or a replay). Wait for it to finish, then try again."


def run_task(job_path: str, payload: dict[str, Any]) -> None:
    """Worker entry point of a task job (top level: importable under ``spawn``)."""
    import warnings

    rep = _Reporter(Path(job_path))
    rep.update(status="running", progress=0.01, message="Starting", stage="start", started_at=_now(), pid=os.getpid())
    handler: logging.Handler | None = None
    task = str(payload.get("task"))
    try:
        handler = _attach_log(Path(job_path))
        job_log.info("job %s: %s", payload["job_id"], payload.get("command") or task)
        rep.stage(0.02, f"Starting {task.replace('_', ' ')}", stage=task)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result, full = _run(task, payload, rep)
        job_log.info("finished: %s", {k: v for k, v in result.items() if v is not None})
        upd: dict[str, Any] = {
            "status": "done",
            "progress": 1.0,
            "stage": "",
            "message": _done_message(task, result),
            "result": result,
            "finished_at": _now(),
        }
        if full is not None:
            upd["result_payload"] = full
        rep.update(**upd)
    except BaseException as e:
        kind, code = _error_kind(e)
        msg = f"{type(e).__name__}: {e}"
        if kind == "stale_data":
            msg = f"{STALE_HINT} ({e})"
        elif kind == "book_locked":
            msg = f"{LOCKED_HINT} ({e})"
        with contextlib.suppress(Exception):
            job_log.error("failed: %s", msg)
        rep.update(
            status="failed",
            stage="",
            message=f"Failed: {msg}"[:600],
            error=traceback.format_exc()[-4000:],
            error_kind=kind,
            exit_code=code,
            finished_at=_now(),
        )
        if not isinstance(e, Exception):
            raise
    finally:
        if handler is not None:
            for name in ("qts", "qts_api.job"):
                logging.getLogger(name).removeHandler(handler)


def _done_message(task: str, r: dict[str, Any]) -> str:
    if task == "signal_test":
        t = r.get("t")
        return f"Finished: signal test {r.get('result_id')} (primary t {t:.2f})" if isinstance(t, float) else "Finished"
    if task == "sweep_member":
        return f"Finished: sweep member {int(r.get('index', 0)) + 1} ({r.get('run_id')})"
    if task == "sweep_finalize":
        pbo = r.get("pbo")
        return f"Finished: PBO {pbo:.2f}" if isinstance(pbo, float) else "Finished: PBO not computed"
    if task == "capacity":
        return f"Finished: capacity curve {r.get('result_id')}"
    if task == "risk_calibration":
        return f"Finished: risk calibration report {r.get('result_id')}"
    if task == "midcap_check":
        return "Finished: coverage checked (dry run)" if r.get("dry_run") else f"Finished: verdict {r.get('outcome')}"
    if task == "paper_run":
        if r.get("message"):
            return f"Finished: {r['message']}"
        return f"Finished: {r.get('n_days')} day(s) processed, status {r.get('level')}"
    if task == "paper_reconcile":
        return "Finished: reconciliation against the batch simulator"
    if task == "market_hash":
        return f"Finished: family hash {r.get('result_id')}"
    return "Finished"


__all__ = ["TASKS", "run_task", "store_flags", "task_command"]
