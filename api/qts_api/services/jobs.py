"""Background pipeline jobs.

A job runs the research pipeline in a **separate process** (``multiprocessing``,
``spawn`` by default): backtests are CPU- and memory-heavy and must not block the
event loop or share the API's heap.  At most ``Settings.max_workers`` (default 1)
run at once; the rest wait in a FIFO queue of at most ``Settings.max_queued`` (default
20) jobs -- :meth:`JobManager.submit` raises :class:`QueueFull` beyond that (the route
answers 429).  Statuses: ``queued`` -> ``running`` -> ``done`` | ``failed``, or
``cancelled`` (from either).  A queued job can be cancelled or moved in the queue.  The dispatcher thread survives a failing ``Process.start()`` (EAGAIN,
a pickling or spawn-bootstrap error): that job is marked failed with the error and
the queue moves on.

State lives in ``<runs root>/_jobs/``: ``<job_id>.json`` (atomic replace; state, request
and worker payload), ``<job_id>.log`` (the worker's log: stage lines plus the pipeline's
INFO log, served by ``GET /api/jobs/{id}/log``) and ``queue.json`` (the waiting ids in
order).  The worker writes its own progress to the job file; the API reads it on every
request.  **Restart:** a restarted API still reports finished jobs, puts ``queued`` jobs
back on the queue in their order, and marks a job that was ``running`` failed with a note
(its worker died with the API; nothing was written).

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
#: Stage lines of the worker (written to the job's log file next to the pipeline's own).
job_log = logging.getLogger("qts_api.job")

STAGING_DIR = "_staging"
#: Job ids are ``job_<12 hex>``; anything else is not a job (404, never a filesystem error).
JOB_ID_RE = re.compile(r"^job_[0-9a-f]{12}$")
TERMINAL = ("done", "failed", "cancelled")
QUEUE_FILE = "queue.json"
LAUNCH_FILE = "launch.json"
#: Finished jobs listed by ``GET /api/jobs`` after the running and queued ones.
RECENT_TERMINAL = 20
INTERRUPTED_RUNNING = (
    "Interrupted: the API restarted while this job was running; nothing was written. "
    "Queue it again from the run builder."
)
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

    def stage(self, progress: float, message: str, stage: str | None = None, **kw: Any) -> None:
        if stage is not None:
            kw["stage"] = stage
            job_log.info("%s", message)
        self.update(progress=round(float(progress), 4), message=message, **kw)


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
            self.rep.stage(
                self.lo + (self.hi - self.lo) * frac,
                f"Running the backtest: day {i + 1} of {self.n}",
                backtest_frac=round(frac, 4),
            )
        except Exception:  # never let progress reporting break the run
            pass


def _load_market(payload: dict[str, Any]):
    req = payload["request"]
    if req["kind"] == "synthetic":
        from qts.data.synthetic import SyntheticSpec, generate

        return generate(SyntheticSpec(n_stocks=req["stocks"], n_days=req["days"], seed=req["seed"]))
    from qts.data.store import ParquetStore
    from qts.data.universe import survivorship_check

    # same path as ``qts run --store``: optional prefilter by peak market cap (reads only
    # the long market_cap table), then the [start, end] slice
    store = ParquetStore(payload["store_path"])
    start, end = req.get("start"), req.get("end")
    tickers = None
    if req.get("min_mcap_ever") is not None or req.get("max_names") is not None:
        tickers = store.tickers_by_market_cap(
            start, end, min_mcap_ever=req.get("min_mcap_ever"), max_names=req.get("max_names")
        )
        job_log.info("prefilter: %d candidate ids by peak market cap", len(tickers))
        if not tickers:
            raise ValueError("the universe prefilter left no ids")
    md = store.as_market_data(universe_tickers=tickers, start=start, end=end)
    job_log.info("store: %d ids x %d days", md.close.shape[1], md.close.shape[0])
    survivorship_check(md)
    return md


def log_path(job_path: Path) -> Path:
    return job_path.with_suffix(".log")


def _attach_log(job_path: Path) -> logging.Handler:
    """Send the pipeline's INFO log (``qts.*``) and the worker's stage lines to
    ``<job_id>.log``; the API serves its tail."""
    h = logging.FileHandler(log_path(job_path), mode="a", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s", "%H:%M:%S"))
    h.setLevel(logging.INFO)
    for name in ("qts", "qts_api.job"):
        lg = logging.getLogger(name)
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return h


def cli_command(req: dict[str, Any], store_path: str | None, overrides: dict[str, Any] | None = None) -> str:
    """The ``qts`` command line equivalent to a launch request (recorded in
    ``launch.json``; the run builder shows the same text before launch)."""
    import shlex

    if req["kind"] == "synthetic":
        parts = ["qts", "demo", "--stocks", str(req["stocks"]), "--days", str(req["days"]), "--seed", str(req["seed"])]
    else:
        parts = ["qts", "run", "--store", str(store_path or "<store>")]
        for flag, key in (
            ("--start", "start"),
            ("--end", "end"),
            ("--max-names", "max_names"),
            ("--min-mcap-ever", "min_mcap_ever"),
        ):
            if req.get(key) is not None:
                v = req[key]
                parts += [flag, f"{v:g}" if isinstance(v, float) else str(v)]
    if req.get("base_run_id"):
        parts += ["--config", f"runs/{req['base_run_id']}/config.yaml"]
    if req.get("ml"):
        parts.append("--ml")
    if int(req.get("signal_lag_days", 1)) != 1:
        parts += ["--signal-lag", str(req["signal_lag_days"])]
    if req.get("max_days") is not None:
        parts += ["--max-days", str(req["max_days"])]
    if req.get("cache"):
        parts.append("--cache")
    ov = dict(overrides if overrides is not None else req.get("config_overrides") or {})
    if req["kind"] == "store" and (req.get("base_run_id") or int(req.get("seed", 0)) != 0):
        ov = {"seed": req["seed"], **ov}  # `qts run` has no --seed flag: the seed is a config key
    for k, v in ov.items():
        val = "null" if v is None else ("true" if v is True else "false" if v is False else str(v))
        parts += ["--set", f"{k}={val}"]
    return " ".join(shlex.quote(p) for p in parts)


def run_job(job_path: str, payload: dict[str, Any]) -> None:
    """Worker entry point (top level: importable under ``spawn``)."""
    import warnings

    rep = _Reporter(Path(job_path))
    rep.update(status="running", progress=0.01, message="Starting", stage="start", started_at=_now(), pid=os.getpid())
    staging: Path | None = None
    log_handler: logging.Handler | None = None
    try:
        log_handler = _attach_log(Path(job_path))
        job_log.info("job %s -> run %s", payload["job_id"], payload["run_id"])
        import pandas as pd
        import qts.pipeline as P
        from qts.core.config import SystemConfig
        from qts.runs.artifacts import write_extra_artifacts

        req = payload["request"]
        ml = bool(req["ml"])
        bt_lo, bt_hi = (0.5, 0.9) if ml else (0.3, 0.9)
        rep.stage(
            0.02, "Building the market data" if req["kind"] == "synthetic" else "Loading the data store", stage="data"
        )
        md = _load_market(payload)

        cfg = SystemConfig.from_dict(payload["config"])
        cfg = cfg.with_overrides({"universe.top_n_by_mcap": min(cfg.universe.top_n_by_mcap, md.close.shape[1])})
        if req.get("cache"):  # qts --cache: operational, outside the config hash
            cfg = cfg.with_overrides({"cache.enabled": True, "cache.root": str(Path(payload["runs_root"]) / "_cache")})

        # stage hooks (worker process only)
        orig_u, orig_r, orig_a = P.build_universe_mask, P.build_risk_model, P.AlphaModel

        def universe(*a: Any, **k: Any):
            rep.stage(0.05, "Building the point-in-time universe", stage="universe")
            return orig_u(*a, **k)

        def risk(*a: Any, **k: Any):
            rep.stage(0.08, "Fitting the risk model", stage="risk")
            return orig_r(*a, **k)

        def alpha(*a: Any, **k: Any):
            rep.stage(0.2, "Fitting the alpha model" + (" (walk-forward ML sleeve)" if ml else ""), stage="alpha")
            return orig_a(*a, **k)

        P.build_universe_mask, P.build_risk_model, P.AlphaModel = universe, risk, alpha  # type: ignore[assignment]

        class _Pipeline(P.ResearchPipeline):
            def evaluate(self, *a: Any, **k: Any):
                rep.stage(0.91, "Evaluating gates and writing the trial ledger", stage="evaluate")
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
            sim_log.addHandler(log_handler)  # propagation is off below: log the day lines too
            sim_log.setLevel(logging.INFO)
            sim_log.propagate = False
            rep.stage(
                bt_lo,
                f"Running the backtest: day 0 of {len(dates) - 1}",
                stage="backtest",
                backtest_started_at=_now(),
                backtest_frac=0.0,
            )
            try:
                res = pipe.run(md, progress=True, max_days=max_days)
            finally:
                sim_log.removeHandler(handler)
                sim_log.removeHandler(log_handler)

        rep.stage(0.96, "Writing the run directory", stage="write")
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
        # what the run builder needs to start "from" this run (data source, flags, base)
        _write_json(
            run_dir / LAUNCH_FILE,
            {
                "job_id": payload["job_id"],
                "request": req,
                "store_path": payload.get("store_path"),
                "command": cli_command(req, payload.get("store_path")),
            },
        )
        os.replace(run_dir, root / payload["run_id"])
        staging = None
        s = res.summary
        job_log.info("finished: net Sharpe %.2f", float(s.get("sharpe_net", float("nan"))))
        rep.update(
            status="done",
            progress=1.0,
            stage="",
            message=f"Finished: gates evaluated (net Sharpe {float(s.get('sharpe_net', float('nan'))):.2f}, "
            f"overall {'PASS' if res.hurdles.overall else 'FAIL'})",
            finished_at=_now(),
        )
    except BaseException as e:  # report everything, including KeyboardInterrupt in the worker
        with contextlib.suppress(Exception):
            job_log.error("failed: %s: %s", type(e).__name__, e)
        rep.update(
            status="failed",
            stage="",
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


class NotQueued(RuntimeError):
    """The job is not waiting in the queue (it started, finished or was cancelled)."""


def _parse_ts(v: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _duration(j: dict[str, Any]) -> float | None:
    a, b = _parse_ts(j.get("started_at")), _parse_ts(j.get("finished_at"))
    return b - a if a is not None and b is not None and b > a else None


def job_size(req: dict[str, Any]) -> tuple[float | None, tuple[Any, ...]]:
    """``(size, exact key)`` of a request for the ETA: names x days for a synthetic market
    (``max_days`` replaces the days when it is set, since the fit is cheap next to the
    backtest); a store slice has no size before it is loaded, only its exact key."""
    if req.get("kind") == "synthetic":
        days = req.get("max_days") or req.get("days") or 0
        return float(req.get("stocks") or 0) * float(days), ("synthetic", req.get("stocks"), req.get("days"))
    return None, ("store", req.get("start"), req.get("end"), req.get("max_names"), req.get("min_mcap_ever"))


def estimate_seconds(job: dict[str, Any], history: list[dict[str, Any]]) -> float | None:
    """Median duration of finished jobs of *similar size*: same data kind and ML flag,
    and either the same data slice or a synthetic market within a factor of two (scaled
    linearly by names x days)."""
    req = job.get("request") or {}
    size, key = job_size(req)
    scaled: list[float] = []
    for h in history:
        hreq = h.get("request") or {}
        if h.get("status") != "done" or h.get("job_id") == job.get("job_id"):
            continue
        if hreq.get("kind") != req.get("kind") or bool(hreq.get("ml")) != bool(req.get("ml")):
            continue
        if (hreq.get("max_days") is None) != (req.get("max_days") is None):
            continue
        d = _duration(h)
        if d is None:
            continue
        hsize, hkey = job_size(hreq)
        if hkey == key and hreq.get("max_days") == req.get("max_days"):
            scaled.append(d)
        elif size and hsize and 0.5 <= size / hsize <= 2.0:
            scaled.append(d * size / hsize)
    if not scaled:
        return None
    scaled.sort()
    mid = len(scaled) // 2
    return scaled[mid] if len(scaled) % 2 else 0.5 * (scaled[mid - 1] + scaled[mid])


def read_log_tail(path: Path, tail: int) -> tuple[list[str], int]:
    """``(last `tail` lines, total lines)`` of a job log (missing file: empty)."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return [], 0
    return lines[-tail:] if tail > 0 else [], len(lines)


def _last_line(path: Path, max_bytes: int = 4096) -> str | None:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            chunk = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = [ln for ln in chunk.splitlines() if ln.strip()]
    return lines[-1][:400] if lines else None


class JobManager:
    """The FIFO queue.  ``<runs root>/_jobs/`` holds one ``<job_id>.json`` per job (state,
    request and the worker payload), ``<job_id>.log`` (the worker's log) and
    ``queue.json`` (the ids waiting, in order), so the queue survives an API restart:
    queued jobs are queued again in their order; a job that was running is marked failed
    (its worker died with the API and wrote nothing)."""

    def __init__(self, settings: Settings, start: bool = True):
        self.settings = settings
        self.dir = settings.jobs_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ctx = mp.get_context(settings.job_start_method)
        self.max_workers = max(1, settings.max_workers)
        self.max_queued = max(0, int(getattr(settings, "max_queued", 20)))
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
        """After a restart: ``queued`` jobs go back on the queue in their persisted order
        (then by creation time for any the queue file does not name); a ``running`` job
        cannot be resumed and is marked failed with a note."""
        order = (_read_json(self.dir / QUEUE_FILE) or {}).get("order") or []
        queued: dict[str, dict[str, Any]] = {}
        for j in self.all():
            status = j.get("status")
            if status == "running":
                self._mark(j["job_id"], status="failed", stage="", message=INTERRUPTED_RUNNING, finished_at=_now())
            elif status == "queued":
                if isinstance(j.get("payload"), dict):
                    queued[j["job_id"]] = j
                else:  # written by an API version that did not persist the payload
                    self._mark(
                        j["job_id"],
                        status="failed",
                        message="Interrupted: the API restarted before the job started",
                        finished_at=_now(),
                    )
        rank = {jid: i for i, jid in enumerate(order)}
        for jid in sorted(queued, key=lambda k: (rank.get(k, len(rank)), str(queued[k].get("created_at", "")))):
            self._queue.append(jid)
            self._payloads[jid] = queued[jid]["payload"]
        self._save_queue()
        shutil.rmtree(self.settings.runs_root / STAGING_DIR, ignore_errors=True)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop the dispatcher and the running workers (marked failed).  Queued jobs stay
        queued on disk: the next API process runs them."""
        with self._cv:
            self._stop = True
            self._cv.notify_all()
            procs = dict(self._procs)
        for job_id, p in procs.items():
            with contextlib.suppress(Exception):
                if p.is_alive():
                    p.terminate()
                p.join(timeout)
            if (_read_json(self._path(job_id)) or {}).get("status") not in TERMINAL:
                self._mark(
                    job_id,
                    status="failed",
                    stage="",
                    message="Interrupted: the API shut down while this job was running",
                    finished_at=_now(),
                )
        if self._thread is not None:
            self._thread.join(timeout)
        shutil.rmtree(self.settings.runs_root / STAGING_DIR, ignore_errors=True)

    # ------------------------------------------------------------------ files
    def _path(self, job_id: str) -> Path:
        if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
            raise JobNotFound(job_id)
        return self.dir / f"{job_id}.json"

    def _save_queue(self) -> None:
        """Persist the waiting ids (caller holds ``self._cv`` or is single-threaded)."""
        with contextlib.suppress(OSError):
            _write_json(self.dir / QUEUE_FILE, {"order": list(self._queue)})

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
                j["position"] = ahead
                j["message"] = "Queued" + (f" ({ahead} job{'s' if ahead != 1 else ''} ahead)" if ahead else "")
        return j

    def all(self) -> list[dict[str, Any]]:
        out = []
        for f in self.dir.glob("job_*.json"):
            if not JOB_ID_RE.fullmatch(f.stem):
                continue
            j = _read_json(f)
            if j and j.get("job_id") == f.stem:
                out.append(j)
        return out

    def counts(self) -> dict[str, int]:
        with self._cv:
            return {"queued": len(self._queue), "running": len(self._procs)}

    # ------------------------------------------------------------------ views
    def view(self, j: dict[str, Any], history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """A job record as the API's ``Job``: adds the queue position, the last log line
        and the ETA (from finished jobs of similar size; while the backtest runs and
        there is no such history, from the backtest's own day counter)."""
        req = j.get("request") or {}
        status = j.get("status", "failed")
        hist = self.all() if history is None else history
        est = estimate_seconds(j, hist) if status in ("queued", "running") else None
        eta: float | None = None
        if status == "running":
            now = time.time()
            started = _parse_ts(j.get("started_at"))
            if est is not None and started is not None:
                eta = max(0.0, est - (now - started))
            else:
                bt0, frac = _parse_ts(j.get("backtest_started_at")), float(j.get("backtest_frac") or 0.0)
                if bt0 is not None and frac >= 0.05:
                    eta = max(0.0, (now - bt0) * (1.0 - frac) / frac)
        position = j.get("position")
        if status == "queued" and position is None:
            with self._cv:
                q = list(self._queue)
            position = q.index(j["job_id"]) if j["job_id"] in q else None
        return {
            "job_id": j["job_id"],
            "run_id": j.get("run_id", ""),
            "status": status if status in ("queued", "running", "done", "failed", "cancelled") else "failed",
            "progress": float(j.get("progress", 0.0) or 0.0),
            "message": str(j.get("message", "")),
            "label": str(j.get("label", "")),
            "kind": j.get("kind") if j.get("kind") in ("synthetic", "store") else "synthetic",
            "ml": bool(j.get("use_ml", False)),
            "phase": 2 if int(j.get("phase", 1) or 1) == 2 else 1,
            "config_hash": str(j.get("config_hash", "")),
            "base_run_id": req.get("base_run_id"),
            "n_overrides": len(req.get("config_overrides") or {}),
            "created_at": str(j.get("created_at", "")),
            "started_at": j.get("started_at"),
            "finished_at": j.get("finished_at"),
            "position": position if status == "queued" else None,
            "stage": str(j.get("stage", "") or "") if status == "running" else "",
            "last_log": _last_line(log_path(self._path(j["job_id"]))),
            "eta_seconds": eta,
            "estimated_seconds": est,
        }

    def listing(self, recent: int = RECENT_TERMINAL) -> dict[str, Any]:
        """Running first, then the queue in order, then the newest ``recent`` terminal jobs."""
        jobs = self.all()
        with self._cv:
            q = list(self._queue)
        rank = {jid: i for i, jid in enumerate(q)}
        running = sorted((j for j in jobs if j.get("status") == "running"), key=lambda j: str(j.get("started_at", "")))
        queued = sorted(
            (j for j in jobs if j.get("status") == "queued"), key=lambda j: rank.get(j["job_id"], len(rank))
        )
        for j in queued:
            j["position"] = rank.get(j["job_id"])
        terminal = sorted(
            (j for j in jobs if j.get("status") in TERMINAL),
            key=lambda j: str(j.get("finished_at") or j.get("updated_at") or ""),
            reverse=True,
        )[: max(0, recent)]
        counts: dict[str, int] = {}
        for j in jobs:
            counts[str(j.get("status"))] = counts.get(str(j.get("status")), 0) + 1
        return {
            "jobs": [self.view(j, jobs) for j in (*running, *queued, *terminal)],
            "max_workers": self.max_workers,
            "max_queued": self.max_queued,
            "counts": counts,
        }

    def log(self, job_id: str, tail: int = 200) -> dict[str, Any]:
        self.get(job_id)  # 404 for an unknown id
        lines, total = read_log_tail(log_path(self._path(job_id)), tail)
        return {"job_id": job_id, "lines": lines, "total_lines": total, "truncated": total > len(lines)}

    # ------------------------------------------------------------------ submit / cancel / reorder
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
        rec = {
            "job_id": job_id,
            "run_id": run_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Queued",
            "stage": "",
            "created_at": _now(),
            "label": label,
            "kind": request["kind"],
            "phase": request["phase"],
            "use_ml": request["ml"],
            "config_hash": config_hash,
            "git_sha": git_sha,
            "request": request,
            "payload": payload,  # what a restarted API needs to run a job that was still queued
        }
        _write_json(self._path(job_id), rec)
        with self._cv:
            self._payloads[job_id] = payload
            self._queue.append(job_id)
            self._save_queue()
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
                self._save_queue()
                return self._mark(job_id, status="cancelled", message="Cancelled before it started", finished_at=_now())
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
        return self._mark(job_id, status="cancelled", stage="", message="Cancelled", finished_at=_now())

    def move(self, job_id: str, position: int) -> dict[str, Any]:
        """Put a queued job at ``position`` (0 = next to run; clamped to the end)."""
        self.get(job_id)
        with self._cv:
            if job_id not in self._queue:
                raise NotQueued(job_id)
            self._queue.remove(job_id)
            self._queue.insert(max(0, min(int(position), len(self._queue))), job_id)
            self._save_queue()
        return self.get(job_id)

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
                    stage="",
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
            self._save_queue()
            payload = self._payloads.get(job_id)
            if payload is None:
                continue
            try:
                # "running" before the process exists: a restart in between must not find
                # the job neither queued (queue file) nor running
                self._mark(job_id, status="running", message="Starting", stage="start", started_at=_now())
                self._procs[job_id] = self._start_process(job_id, payload)
            except Exception as e:  # EAGAIN, pickling, spawn bootstrap ...
                log.exception("could not start the worker process for %s", job_id)
                self._payloads.pop(job_id, None)
                with contextlib.suppress(Exception):
                    self._mark(
                        job_id,
                        status="failed",
                        stage="",
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
