"""Background pipeline jobs.

A job runs the research pipeline in a **separate process** (``multiprocessing``,
``spawn`` by default): backtests are CPU- and memory-heavy and must not block the
event loop or share the API's heap.  At most ``Settings.max_workers`` (default 1)
run at once; the rest wait in a FIFO queue of at most ``Settings.max_queued`` (default
4) jobs -- :meth:`JobManager.submit` raises :class:`QueueFull` beyond that (the route
answers 429).  The dispatcher thread survives a failing ``Process.start()`` (EAGAIN,
a pickling or spawn-bootstrap error): that job is marked failed with the error and
the queue moves on.

State lives in ``<runs root>/_jobs/<job_id>.json`` (atomic replace).  The worker
writes its own progress there; the API reads the file on ``GET /api/jobs/{id}``, so a
restarted API still reports finished jobs (jobs that were queued or running when the
API stopped are marked failed: "interrupted").

Code path (same as ``python -m qts.cli demo`` / ``run``): market data
(``generate(SyntheticSpec(stocks, days, seed))`` or ``ParquetStore(...).as_market_data()``
+ ``survivorship_check``), ``cfg.seed = seed``, ``top_n_by_mcap`` clamped to the
number of names, ``ResearchPipeline(cfg, use_ml, phase, ledger_path, nav0=5e6)``,
``run(md, max_days)``, then ``qts.pipeline.save_run`` and
``qts.runs.artifacts.write_extra_artifacts``.  The one difference: the run id is
fixed when the job is *submitted* (``POST /api/runs`` returns it), so the worker
sets ``manifest.run_id`` before writing, into a staging directory that is renamed
into place once complete (readers never see a half-written run).

Progress is stage based: market data, universe, risk model, alpha model (hooks on
``qts.pipeline``'s factory functions inside the worker process), then the
backtest day counter from the simulator's ``progress`` log line (every 50 days),
then evaluation and writing.
"""

from __future__ import annotations

import contextlib
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qts_api.settings import Settings

log = logging.getLogger(__name__)

STAGING_DIR = "_staging"
#: Job ids are ``job_<12 hex>``; anything else is not a job (404, never a filesystem error).
JOB_ID_RE = re.compile(r"^job_[0-9a-f]{12}$")
TERMINAL = ("done", "failed")
NAV0 = 5_000_000.0  # qts.cli default (--nav)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def default_label(ml: bool, seed: int, signal_lag_days: int) -> str:
    label = f"{'Composite + ML' if ml else 'Composite'}, seed {seed}"
    return label + (", same-close execution (lag 0)" if signal_lag_days == 0 else "")


# ============================================================================ worker process
class _Reporter:
    """Read-modify-write of the job file from the worker (throttled)."""

    def __init__(self, path: Path):
        self.path = path
        self._last = 0.0

    def update(self, force: bool = True, **kw: Any) -> None:
        now = time.monotonic()
        if not force and now - self._last < 0.25:
            return
        self._last = now
        cur = _read_json(self.path) or {}
        cur.update(kw)
        cur["updated_at"] = _now()
        _write_json(self.path, cur)

    def stage(self, progress: float, message: str) -> None:
        self.update(progress=round(float(progress), 4), message=message)


class _BacktestDayHandler(logging.Handler):
    """Turns the simulator's every-50-days ``progress`` log line into a day counter."""

    def __init__(self, rep: _Reporter, positions: dict[Any, int], n: int, lo: float, hi: float):
        super().__init__(logging.INFO)
        self.rep, self.positions, self.n, self.lo, self.hi = rep, positions, max(n, 1), lo, hi

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if not str(record.msg).startswith("%s nav=") or not record.args:
                return
            d = record.args[0] if isinstance(record.args, tuple) else None
            i = self.positions.get(d)
            if i is None:
                return
            frac = (i + 1) / self.n
            self.rep.stage(self.lo + (self.hi - self.lo) * frac, f"Running the backtest: day {i + 1} of {self.n}")
        except Exception:  # never let progress reporting break the run
            pass


def _load_market(payload: dict[str, Any]):
    req = payload["request"]
    if req["kind"] == "synthetic":
        from qts.data.synthetic import SyntheticSpec, generate

        return generate(SyntheticSpec(n_stocks=req["stocks"], n_days=req["days"], seed=req["seed"]))
    from qts.data.store import ParquetStore
    from qts.data.universe import survivorship_check

    md = ParquetStore(payload["store_path"]).as_market_data()
    survivorship_check(md)
    return md


def run_job(job_path: str, payload: dict[str, Any]) -> None:
    """Worker entry point (top level: importable under ``spawn``)."""
    import warnings

    rep = _Reporter(Path(job_path))
    rep.update(status="running", progress=0.01, message="Starting", started_at=_now(), pid=os.getpid())
    staging: Path | None = None
    try:
        import pandas as pd
        import qts.pipeline as P
        from qts.core.config import SystemConfig
        from qts.runs.artifacts import write_extra_artifacts

        req = payload["request"]
        ml = bool(req["ml"])
        bt_lo, bt_hi = (0.5, 0.9) if ml else (0.3, 0.9)
        rep.stage(0.02, "Building the market data" if req["kind"] == "synthetic" else "Loading the data store")
        md = _load_market(payload)

        cfg = SystemConfig.from_dict(payload["config"])
        cfg.universe.top_n_by_mcap = min(cfg.universe.top_n_by_mcap, md.close.shape[1])

        # stage hooks (worker process only)
        orig_u, orig_r, orig_a = P.build_universe_mask, P.build_risk_model, P.AlphaModel

        def universe(*a: Any, **k: Any):
            rep.stage(0.05, "Building the point-in-time universe")
            return orig_u(*a, **k)

        def risk(*a: Any, **k: Any):
            rep.stage(0.08, "Fitting the risk model")
            return orig_r(*a, **k)

        def alpha(*a: Any, **k: Any):
            rep.stage(0.2, "Fitting the alpha model" + (" (walk-forward ML sleeve)" if ml else ""))
            return orig_a(*a, **k)

        P.build_universe_mask, P.build_risk_model, P.AlphaModel = universe, risk, alpha  # type: ignore[assignment]

        class _Pipeline(P.ResearchPipeline):
            def evaluate(self, *a: Any, **k: Any):
                rep.stage(0.91, "Evaluating gates and writing the trial ledger")
                return super().evaluate(*a, **k)

        pipe = _Pipeline(
            cfg,
            use_ml=ml,
            phase=int(req["phase"]),
            ledger_path=payload.get("ledger_path") or None,
            nav0=NAV0,
            venue=req.get("venue") or "routed",
            runs_root=None,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pipe.fit(md)
            dates = pipe.backtest_dates()
            max_days = req.get("max_days")
            if max_days is not None and len(dates) > int(max_days) + 1:
                dates = dates[-int(max_days) - 1 :]
            positions = {pd.Timestamp(d).date(): i for i, d in enumerate(dates)}
            sim_log = logging.getLogger("qts.backtest.simulator")
            handler = _BacktestDayHandler(rep, positions, len(dates) - 1, bt_lo, bt_hi)
            sim_log.addHandler(handler)
            sim_log.setLevel(logging.INFO)
            sim_log.propagate = False
            rep.stage(bt_lo, f"Running the backtest: day 0 of {len(dates) - 1}")
            try:
                res = pipe.run(md, progress=True, max_days=max_days)
            finally:
                sim_log.removeHandler(handler)

        rep.stage(0.96, "Writing the run directory")
        assert res.manifest is not None
        res.manifest.run_id = payload["run_id"]
        res.manifest.command = f"qts-api job {payload['job_id']}"
        root = Path(payload["runs_root"])
        staging_root = root / STAGING_DIR
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = staging_root / payload["run_id"]
        if staging.exists():
            shutil.rmtree(staging)
        run_dir = P.save_run(res, cfg, staging_root)
        write_extra_artifacts(run_dir, res, pipe, meta={"label": payload["label"], "kind": req["kind"]})
        os.replace(run_dir, root / payload["run_id"])
        staging = None
        s = res.summary
        rep.update(
            status="done",
            progress=1.0,
            message=f"Finished: gates evaluated (net Sharpe {float(s.get('sharpe_net', float('nan'))):.2f}, "
            f"overall {'PASS' if res.hurdles.overall else 'FAIL'})",
            finished_at=_now(),
        )
    except BaseException as e:  # report everything, including KeyboardInterrupt in the worker
        rep.update(
            status="failed",
            message=f"Failed: {type(e).__name__}: {e}"[:500],
            error=traceback.format_exc()[-4000:],
            finished_at=_now(),
        )
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        if not isinstance(e, Exception):
            raise


# ============================================================================ manager (API process)
class JobNotFound(LookupError):
    pass


class QueueFull(RuntimeError):
    """More than ``Settings.max_queued`` jobs are waiting for a worker."""


class JobManager:
    def __init__(self, settings: Settings, start: bool = True):
        self.settings = settings
        self.dir = settings.jobs_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ctx = mp.get_context(settings.job_start_method)
        self.max_workers = max(1, settings.max_workers)
        self.max_queued = max(0, int(getattr(settings, "max_queued", 4)))
        self._queue: deque[str] = deque()
        self._payloads: dict[str, dict[str, Any]] = {}
        self._procs: dict[str, Any] = {}
        self._cv = threading.Condition()
        self._stop = False
        self._recover()
        self._thread: threading.Thread | None = None
        if start:
            self.start()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="qts-api-jobs", daemon=True)
            self._thread.start()

    def _recover(self) -> None:
        """Jobs left queued/running by a previous API process cannot be resumed."""
        for f in self.dir.glob("*.json"):
            j = _read_json(f)
            if j and j.get("status") not in TERMINAL:
                j.update(
                    status="failed",
                    message="Interrupted: the API restarted before the job finished",
                    finished_at=_now(),
                )
                _write_json(f, j)
        shutil.rmtree(self.settings.runs_root / STAGING_DIR, ignore_errors=True)

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
            procs = dict(self._procs)
            queued = list(self._queue)
            self._queue.clear()
        for job_id, p in procs.items():
            with contextlib.suppress(Exception):
                if p.is_alive():
                    p.terminate()
                p.join(timeout)
            if (_read_json(self._path(job_id)) or {}).get("status") not in TERMINAL:
                self._mark(job_id, status="failed", message="Interrupted: the API shut down", finished_at=_now())
        for job_id in queued:
            self._mark(job_id, status="failed", message="Interrupted: the API shut down", finished_at=_now())
        if self._thread is not None:
            self._thread.join(timeout)
        shutil.rmtree(self.settings.runs_root / STAGING_DIR, ignore_errors=True)

    # ------------------------------------------------------------------ files
    def _path(self, job_id: str) -> Path:
        if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
            raise JobNotFound(job_id)
        return self.dir / f"{job_id}.json"

    def _mark(self, job_id: str, **kw: Any) -> dict[str, Any]:
        p = self._path(job_id)
        j = _read_json(p) or {"job_id": job_id}
        j.update(kw)
        j["updated_at"] = _now()
        _write_json(p, j)
        return j

    def get(self, job_id: str) -> dict[str, Any]:
        j = _read_json(self._path(job_id))
        if j is None:
            raise JobNotFound(job_id)
        if j.get("status") == "queued":
            with self._cv:
                q = list(self._queue)
            if job_id in q:
                ahead = q.index(job_id)
                j["message"] = "Queued" + (f" ({ahead} job{'s' if ahead != 1 else ''} ahead)" if ahead else "")
        return j

    def all(self) -> list[dict[str, Any]]:
        out = []
        for f in self.dir.glob("*.json"):
            j = _read_json(f)
            if j:
                out.append(j)
        return out

    def counts(self) -> dict[str, int]:
        with self._cv:
            return {"queued": len(self._queue), "running": len(self._procs)}

    # ------------------------------------------------------------------ submit / cancel
    def submit(
        self,
        *,
        request: dict[str, Any],
        config: dict[str, Any],
        config_hash: str,
        run_id: str,
        label: str,
        git_sha: str,
    ) -> dict[str, Any]:
        with self._cv:
            if len(self._queue) >= self.max_queued:
                raise QueueFull(
                    f"{len(self._queue)} jobs are already queued (limit {self.max_queued}, "
                    "QTS_MAX_QUEUED); wait for one to start or cancel one"
                )
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        rec = {
            "job_id": job_id,
            "run_id": run_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Queued",
            "created_at": _now(),
            "label": label,
            "kind": request["kind"],
            "phase": request["phase"],
            "use_ml": request["ml"],
            "config_hash": config_hash,
            "git_sha": git_sha,
            "request": request,
        }
        _write_json(self._path(job_id), rec)
        payload = {
            "job_id": job_id,
            "run_id": run_id,
            "runs_root": str(self.settings.runs_root),
            "ledger_path": str(self.settings.ledger),
            "store_path": str(self.settings.store_path) if self.settings.store_path else None,
            "request": request,
            "config": config,
            "label": label,
        }
        with self._cv:
            self._payloads[job_id] = payload
            self._queue.append(job_id)
            self._cv.notify_all()
        return rec

    def cancel(self, job_id: str) -> dict[str, Any]:
        j = self.get(job_id)
        if j.get("status") in TERMINAL:
            return j
        with self._cv:
            if job_id in self._queue:
                self._queue.remove(job_id)
                self._payloads.pop(job_id, None)
                return self._mark(job_id, status="failed", message="Cancelled before it started", finished_at=_now())
            proc = self._procs.pop(job_id, None)
            self._payloads.pop(job_id, None)
            self._cv.notify_all()
        if proc is not None:
            with contextlib.suppress(Exception):
                proc.terminate()
                proc.join(10)
        shutil.rmtree(self.settings.runs_root / STAGING_DIR / str(j.get("run_id")), ignore_errors=True)
        cur = _read_json(self._path(job_id)) or j
        if cur.get("status") == "done":  # finished just before the signal landed
            return cur
        return self._mark(job_id, status="failed", message="Cancelled", finished_at=_now())

    # ------------------------------------------------------------------ dispatcher
    def _reap(self) -> None:
        for job_id, p in list(self._procs.items()):
            if p.is_alive():
                continue
            p.join(0)
            self._procs.pop(job_id, None)
            self._payloads.pop(job_id, None)
            j = _read_json(self._path(job_id)) or {}
            if j.get("status") not in TERMINAL:
                self._mark(
                    job_id,
                    status="failed",
                    message=f"Failed: the worker process exited with code {p.exitcode}",
                    finished_at=_now(),
                )

    def _start_process(self, job_id: str, payload: dict[str, Any]) -> Any:
        p = self.ctx.Process(
            target=run_job,
            args=(str(self._path(job_id)), payload),
            name=f"qts-job-{job_id}",
            daemon=True,
        )
        p.start()
        return p

    def _dispatch(self) -> None:
        """Start queued jobs while workers are free (caller holds ``self._cv``).  A job
        whose process cannot start is marked failed; the others still run."""
        while self._queue and len(self._procs) < self.max_workers:
            job_id = self._queue.popleft()
            payload = self._payloads.get(job_id)
            if payload is None:
                continue
            try:
                self._procs[job_id] = self._start_process(job_id, payload)
            except Exception as e:  # EAGAIN, pickling, spawn bootstrap ...
                log.exception("could not start the worker process for %s", job_id)
                self._payloads.pop(job_id, None)
                with contextlib.suppress(Exception):
                    self._mark(
                        job_id,
                        status="failed",
                        message=f"Failed: the worker process could not start: {type(e).__name__}: {e}"[:500],
                        error=traceback.format_exc()[-4000:],
                        finished_at=_now(),
                    )

    def _loop(self) -> None:
        while True:
            with self._cv:
                if self._stop:
                    return
                try:
                    self._reap()
                    self._dispatch()
                except Exception:  # never let the dispatcher thread die
                    log.exception("job dispatcher error (continuing)")
                self._cv.wait(timeout=0.5)
