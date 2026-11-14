"""Library entry points of the paper book (the CLI and the future API call these).

Every function takes a book *name* (resolved under ``paper_root``, default
``paper/``) or a path to a book directory and returns JSON-serialisable dicts
(shapes in ``docs/handbook/14_paper_trading.md``, "Library entry points").

=====================  ==========================================================
:func:`init_book`      create the book: freeze config, expectations, stop rules
:func:`run_book`       the daily job: refresh, check freshness, catch up day by day
:func:`status`         latest status, every check, metrics and plot series
:func:`history`        one row per journal day
:func:`orders`         orders generated on a day (for the next close) and its fills
:func:`ack`            acknowledge a STOP with a note (trading resumes next job)
:func:`replay`         validate the journal and rebuild every derived file
:func:`schedule_text`  launchd plist + crontab line (printed, never installed)
:func:`reconcile`      paper vs the batch simulator over the book's dates
=====================  ==========================================================
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import shlex
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qts.core.config import SystemConfig
from qts.core.types import MarketData
from qts.paper import expectations as X
from qts.paper.book import (
    DEFAULT_PAPER_ROOT,
    BookHeader,
    BookLock,
    DataSpec,
    PaperError,
    PaperPaths,
    append_ack,
    append_record,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    book_root,
    effective_records,
    iso,
    json_safe,
    read_acks,
    read_header,
    read_journal,
    sha256_bytes,
    utc_now,
)
from qts.paper.data import (
    NY,
    Calendar,
    DataSource,
    MemorySource,
    StoreSource,
    check_freshness,
    expected_last_session,
    make_calendar,
    now_utc,
    sync_since,
    sync_store,
)
from qts.paper.engine import (
    ORDER_COLUMNS,
    BookState,
    PaperEngine,
    frame_from_columns,
    positions_frame,
)
from qts.paper.monitor import StopRules, evaluate
from qts.paper.serving import FrozenFoldStore, SignalProvider, effective_config

log = logging.getLogger(__name__)

#: MOC / LOC entry cutoff (NYSE 15:50 ET): orders generated before it on the fill
#: day could have been submitted (``mode: live``); later ones are ``backfill``
MOC_CUTOFF_HOUR, MOC_CUTOFF_MINUTE = 15, 50
FILL_COLUMNS = ["ticker", "side", "shares", "px", "usd", "spread", "impact", "fees", "kind", "date"]


def _paths(
    book: str | os.PathLike[str], paper_root: str | os.PathLike[str] = DEFAULT_PAPER_ROOT
) -> PaperPaths:
    return PaperPaths(book_root(str(book), paper_root))


def _calendar(header: BookHeader, md: MarketData | None = None) -> Calendar:
    return make_calendar(header.calendar, None if md is None else md.dates)


def _source(header: BookHeader, source: DataSource | None) -> DataSource:
    if source is not None:
        return source
    if header.data.kind == "store":
        return StoreSource(header.data)
    raise PaperError(f"book data kind {header.data.kind!r} needs an in-memory source (tests / benchmarks)")


def _load_config(paths: PaperPaths) -> SystemConfig:
    return SystemConfig.from_yaml(str(paths.config))


def _code_version() -> str:
    from qts.core.manifest import code_version

    return code_version()


def _stage_hashes() -> dict[str, str]:
    from qts.runs.cache import source_hash

    try:
        return {s: source_hash(s)[:16] for s in ("risk_fit", "alpha_fit")}
    except Exception:  # pragma: no cover - source unavailable
        return {}


# =========================================================================== init
def init_book(
    name: str,
    config_path: str,
    *,
    nav: float = 5e6,
    start: str,
    reference_run: str | None = None,
    paper_root: str | os.PathLike[str] = DEFAULT_PAPER_ROOT,
    data: DataSpec | None = None,
    sizing: str = "moc",
    use_ml: bool = True,
    rules: StopRules | None = None,
    calendar: str = "nyse",
    expectations: X.Expectations | None = None,
    source: DataSource | None = None,
) -> dict[str, Any]:
    """Create a paper book (``book.json``, frozen config copy, expectation tables).

    ``start`` is the first job day: its close produces the first orders, which fill
    at the next session's close (the first execution date and the anchor of the
    ``rebalance_every_days`` schedule).  ``reference_run``: the run directory the
    expectations come from (or pass ``expectations`` directly)."""
    root = book_root(name, paper_root)
    paths = PaperPaths(root)
    if paths.exists():
        raise PaperError(
            f"paper book {root} already exists (pick another --name; books are never overwritten)"
        )
    if sizing not in ("moc", "replica"):
        raise PaperError(f"--sizing must be moc or replica, got {sizing!r}")
    if not (math.isfinite(float(nav)) and float(nav) > 0):
        raise PaperError(f"--nav must be > 0, got {nav}")
    cfg_text = Path(config_path).read_text()
    cfg = SystemConfig.from_yaml(config_path).validate()
    if int(cfg.execution.signal_lag_days) != 1:
        raise PaperError("the paper book needs execution.signal_lag_days = 1 (MOC order for the next close)")
    data = data or DataSpec()
    start_ts = pd.Timestamp(start).normalize()
    md_dates = None
    if calendar == "index":
        if source is None:
            raise PaperError("calendar 'index' needs the in-memory source at init")
        md_dates = source.load().dates
    cal = make_calendar(calendar, md_dates)
    if not cal.is_session(start_ts):
        raise PaperError(f"--start {iso(start_ts)} is not a trading session")
    first_exec = cal.next_session(start_ts)
    if expectations is None:
        if not reference_run:
            raise PaperError("give --reference-run (a finished `qts run` directory): the stop rules need it")
        expectations = X.from_run_dir(reference_run)
    rules = rules or StopRules()
    if expectations.returns is not None and "calibration" not in expectations.summary:
        # boundaries of the drawdown / Sharpe rules: a 2.5 % (STOP) / 10 % (WATCH) chance
        # of a false alarm within the first year for a book that behaves like the backtest
        info, rows = X.calibrate(expectations.returns.to_numpy(dtype=float), rules)
        expectations = X.Expectations(
            summary={**expectations.summary, "calibration": info},
            table=pd.concat([expectations.table, rows], ignore_index=True),
            returns=expectations.returns,
        )
    notes: list[str] = []
    ref = dict(expectations.summary)
    if ref.get("run_config_hash") and ref.get("run_config_hash") != cfg.hash():
        notes.append(
            f"reference run config hash {ref.get('run_config_hash')} != book config hash {cfg.hash()} "
            "(expected when config fields were added since the run; check the config diff)"
        )
    if ref.get("use_ml") is not None and bool(ref.get("use_ml")) != bool(use_ml):
        notes.append(f"reference run use_ml={ref.get('use_ml')} but the book runs use_ml={use_ml}")
    root.mkdir(parents=True, exist_ok=False)
    for d in (paths.expectations, paths.models, paths.state_dir, paths.logs, root / "days"):
        d.mkdir(parents=True, exist_ok=True)
    atomic_write_text(paths.config, cfg_text)
    digest = X.save(expectations, paths.expectations)
    header = BookHeader(
        name=root.name,
        created_at=utc_now(),
        start=iso(start_ts),
        first_execution=iso(first_exec),
        nav0=float(nav),
        sizing=sizing,
        use_ml=bool(use_ml),
        config_source=str(config_path),
        config_file="config.yaml",
        config_text_sha256=sha256_bytes(cfg_text.encode()),
        config_hash=cfg.hash(),
        data=data,
        calendar=calendar,
        reference_run=None if reference_run is None else str(reference_run),
        reference={
            k: ref.get(k)
            for k in ("run_id", "run_config_hash", "run_data_hash", "start", "end", "n_days", "sharpe_ann")
        },
        stop_rules=rules.to_dict(),
        expectations_sha256=digest,
        code_version=_code_version(),
        source_hashes=_stage_hashes(),
        notes=notes,
    )
    atomic_write_json(paths.header, header.to_dict())  # written last: the book exists from here
    return json_safe({"root": str(root), "header": header.to_dict(), "expectations": expectations.summary})


# =========================================================================== run
def _record_mode(header: BookHeader, exec_date: str | None, generated_at: datetime) -> str:
    """``live`` when the orders were generated before the fill day's MOC cutoff,
    ``backfill`` otherwise (and always for the ``replica`` sizing)."""
    if header.sizing != "moc" or exec_date is None:
        return "backfill"
    d = pd.Timestamp(exec_date)
    cutoff = datetime(d.year, d.month, d.day, MOC_CUTOFF_HOUR, MOC_CUTOFF_MINUTE, tzinfo=NY)
    return "live" if generated_at < cutoff else "backfill"


def _days_to_process(
    cal: Calendar, header: BookHeader, last: str | None, end: pd.Timestamp
) -> list[pd.Timestamp]:
    first = pd.Timestamp(header.start) if last is None else cal.next_session(pd.Timestamp(last))
    if first > end:
        return []
    return [pd.Timestamp(d) for d in cal.sessions(first, end)]


def _write_views(
    paths: PaperPaths, date: str, state: BookState, fills: pd.DataFrame, orders: pd.DataFrame
) -> None:
    d = paths.day_dir(date)
    atomic_write_parquet(d / "positions.parquet", positions_frame(state))
    atomic_write_parquet(d / "fills.parquet", _typed(fills, FILL_COLUMNS))
    atomic_write_parquet(d / "orders.parquet", _typed(orders, ORDER_COLUMNS))


_STR_COLS = frozenset({"ticker", "side", "kind", "date", "exec_date", "px_date"})
_BOOL_COLS = frozenset({"close_out"})


def _typed(frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """``frame`` with exactly ``cols``, strings / booleans / floats typed (parquet-safe)."""
    out = frame.reindex(columns=list(cols))
    data: dict[str, Any] = {}
    for c in cols:
        v = out[c]
        if c in _STR_COLS:
            data[c] = v.astype(object).where(v.notna(), None)
        elif c in _BOOL_COLS:
            data[c] = v.fillna(False).astype(bool)
        else:
            data[c] = pd.to_numeric(v, errors="coerce").astype(float)
    return pd.DataFrame(data, columns=list(cols))


def _load_last_risk(paths: PaperPaths, state: BookState) -> Any:
    if state.last_risk_date is None or not paths.last_risk.exists():
        return None
    try:
        with open(paths.last_risk, "rb") as fh:
            date, risk = pickle.load(fh)  # trusted local file written by this module
    except Exception:  # a stale / unreadable cache only loses the hold-day ex-ante vol
        return None
    return risk if date == state.last_risk_date else None


def _ic_summary(fit: Any, first_execution: str) -> dict[str, Any] | None:
    try:
        ic = fit.realised_ic(pd.Timestamp(first_execution))
    except Exception:  # pragma: no cover - diagnostic only
        return None
    if ic is None or len(ic) == 0:
        return {"n": 0, "mean": None, "last_date": None}
    from qts.core import stats

    h = int(fit.cfg.alpha.horizon_days) + 1
    t = (
        float(stats.newey_west_tstat(ic.to_numpy(dtype=float), lags=max(1, min(h, len(ic) - 1))))
        if len(ic) > 2
        else None
    )
    return {"n": len(ic), "mean": float(ic.mean()), "nw_t": t, "last_date": iso(ic.index[-1])}


def run_book(
    book: str | os.PathLike[str],
    *,
    date: str | None = None,
    until: str | None = None,
    no_sync: bool = False,
    dry_run: bool = False,
    redo_last: bool = False,
    paper_root: str | os.PathLike[str] = DEFAULT_PAPER_ROOT,
    source: DataSource | None = None,
    sync_fn: Callable[..., Any] | None = None,
    now: datetime | None = None,
    inner_factory: Callable[[], object] | None = None,
    signal_mode: str = "daily",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """The daily job (module docstring of :mod:`qts.paper.engine` for the timeline).

    1. incremental Sharadar sync since the last data date minus a few days
       (``no_sync`` / ``dry_run`` skip it; store books only);
    2. freshness: without ``date`` / ``until`` the store must hold the session the
       vendor should have published by now (:func:`expected_last_session`), else
       :class:`~qts.paper.book.StaleDataError`;
    3. every session after the last journal day through the target day is processed
       in order (catch-up), each from data through that day only; a day already in
       the journal is never processed again (``redo_last`` re-processes the last one
       and supersedes its record).
    """
    say = progress or (lambda msg: log.info("%s", msg))
    t_run = time.perf_counter()
    now = now or now_utc()
    paths = _paths(book, paper_root)
    header = read_header(paths)
    lock = BookLock(
        paths.root, f"qts paper run {'--dry-run ' if dry_run else ''}{date or until or ''}".strip()
    )
    with lock:
        raw, torn = read_journal(paths)
        if torn:
            raise PaperError(
                f"{paths.journal} ends with a torn line (crash mid-write); run `qts paper replay` first"
            )
        records = effective_records(raw)
        if redo_last:
            if not records:
                raise PaperError("--redo-last: the journal is empty")
            redo = records.pop()
            say(f"re-processing {redo['date']} (supersedes journal record {redo.get('seq')})")
        else:
            redo = None
        last = records[-1]["date"] if records else None
        src = _source(header, source)
        report: dict[str, Any] = {
            "book": header.name,
            "root": str(paths.root),
            "dry_run": dry_run,
            "days": [],
        }
        # ---- 1. refresh
        if header.data.kind == "store" and not no_sync and not dry_run:
            since = sync_since(
                records[-1]["data_asof"] if records else header.start, header.data.sync_overlap_days
            )
            say(f"data refresh: Sharadar incremental sync since {since} ...")
            report["sync"] = sync_store(header.data, since, sync_fn)
            say(f"data refresh done in {report['sync']['seconds']:.0f}s")
        latest = src.latest_date()
        if latest is None:
            raise PaperError("the data source has no prices")
        t_load = time.perf_counter()
        md_full = src.load() if header.calendar == "index" else None
        cal = _calendar(header, md_full)
        # ---- 2. target day + freshness
        if date is not None:
            end = pd.Timestamp(date)
            if end > latest:
                raise PaperError(
                    f"--date {iso(end)} is after the latest price date {iso(latest)} in the data"
                )
        elif until is not None:
            end = min(pd.Timestamp(until), latest)
        else:
            expected = expected_last_session(now, cal, header.data_ready_hour_et)
            check_freshness(
                latest, expected, str(header.data.store if header.data.kind == "store" else "source")
            )
            end = latest
        report["target"] = iso(end)
        days = _days_to_process(cal, header, last, end)
        if not days:
            report["message"] = (
                f"nothing to do: {iso(end)} is already in the journal (last day {last})"
                if last is not None
                else f"nothing to do: the target {iso(end)} is before the book's start {header.start}"
            )
            report["status"] = records[-1].get("status") if records else None
            return json_safe(report)
        # ---- 3. models and state
        cfg_file = _load_config(paths)
        if md_full is not None and not (isinstance(src, MemorySource) and end in src.snapshots):
            md_end = md_full if end >= md_full.dates[-1] else md_full.slice_until(end)
        else:
            md_end = src.load(end)
        report["load_seconds"] = round(time.perf_counter() - t_load, 2)
        cfg = effective_config(cfg_file, md_end)
        missing = [iso(d) for d in days if d not in md_end.dates]
        if missing:
            raise PaperError(
                f"session(s) {missing[:5]} are missing from the data (vendor gap?); the book cannot skip a "
                "trading day -- fix the data or process up to the day before with --date"
            )

        def loader(d: pd.Timestamp) -> MarketData:
            d = pd.Timestamp(d)
            if isinstance(src, MemorySource) and d in src.snapshots:
                return src.snapshots[d]
            return md_end if d >= md_end.dates[-1] else md_end.slice_until(d)

        identity = {
            "config_text_sha256": header.config_text_sha256,
            "use_ml": header.use_ml,
            "inner": None
            if inner_factory is None
            else f"{getattr(inner_factory, '__module__', '')}.{getattr(inner_factory, '__qualname__', type(inner_factory).__name__)}",
        }
        store = FrozenFoldStore(paths.models, identity) if header.use_ml else None
        signals = SignalProvider(
            loader,
            cfg,
            use_ml=header.use_ml,
            store=store,
            inner_factory=inner_factory,
            nav0=header.nav0,
            mode=signal_mode,
            end=end,
        )
        engine = PaperEngine(
            cfg,
            sizing=header.sizing,
            first_execution=pd.Timestamp(header.first_execution),
            calendar=cal,
            venue=header.venue,
        )
        state = BookState.from_dict(records[-1]["state"]) if records else BookState.initial(header.nav0)
        last_risk = _load_last_risk(paths, state)
        exp = X.load(paths.expectations, header.expectations_sha256)
        rules = StopRules.from_dict(header.stop_rules)
        acks = read_acks(paths)
        code_ver = _code_version()
        cfg_hash = cfg_file.hash()
        hdr_warn: list[str] = []
        if code_ver != header.code_version:
            hdr_warn.append(f"code version {code_ver} differs from the version at init {header.code_version}")
        if cfg_hash != header.config_hash:
            hdr_warn.append(
                f"config hash {cfg_hash} differs from init {header.config_hash} (same frozen YAML; the code "
                "added or changed config fields)"
            )
        seq = len(raw)
        # ---- 4. day loop
        for t in days:
            t0 = time.perf_counter()
            n_fits = len(signals.fits)
            prev = records[-1] if records else None
            prev_status = (prev or {}).get("status") or {}
            acks_prev = [a for a in acks if prev is not None and str(a.get("date")) == str(prev["date"])]
            halted = prev_status.get("level") == "STOP" and not acks_prev
            suppressed_in = set(prev_status.get("suppressed", []) or [])
            for a in acks_prev:
                suppressed_in |= set(a.get("rules", []) or [])
            exec_date, _ = engine.decision_dates(t)
            if exec_date is not None and engine.is_rebalance(exec_date) and not halted:
                say(
                    f"{iso(t)}: rebalance decision for {iso(exec_date)}: fitting the frozen strategy on data "
                    "through the signal date (a few minutes on the real store; the book's first run trains every "
                    "walk-forward fold, ~20+ min, and a day that crosses a scheduled refit trains one new fold)"
                )
            md_t = loader(t)
            res = engine.run_day(state, t, md_t, signals, last_risk=last_risk, halted=halted)
            rec = res.record
            rec.update(
                {
                    "kind": "day",
                    "seq": seq,
                    "data_asof": iso(md_t.dates[-1]),
                    "generated_at": utc_now(),
                    "mode": _record_mode(header, rec.get("orders_for"), now),
                    "config_hash": cfg_hash,
                    "config_text_sha256": header.config_text_sha256,
                    "code_version": code_ver,
                    "halted": halted,
                    "warnings": list(rec.get("warnings", [])) + (hdr_warn if not records else []),
                }
            )
            if redo is not None and redo["date"] == rec["date"]:
                rec["supersedes"] = redo.get("seq")
            new_fits = signals.fits[n_fits:]
            latest_fit = signals.latest
            solve = rec.get("solve") or {}
            rec["models"] = {
                "fold": solve.get("fold"),
                "fit_asof": solve.get("fit_asof"),
                "fits": new_fits,
                "folds_trained": [f for x in new_fits for f in x.get("trained", [])],
                "stage_code": _stage_hashes() if new_fits else None,
            }
            if new_fits and latest_fit is not None:
                rec["signal"] = {"realised_ic": _ic_summary(latest_fit, header.first_execution)}
            for w in [w for x in new_fits for w in x.get("warnings", [])]:
                rec["warnings"].append(w)
            st = evaluate(
                [*records, rec],
                exp,
                rules,
                header.first_execution,
                float(cfg.portfolio.gross_leverage_max),
                prior_suppressed=sorted(suppressed_in),
            )
            if st["level"] == "STOP" and res.state.pending and not halted:
                rec["orders_cancelled_by_stop"] = len(res.state.pending)
                res.state.pending = []
                rec["orders"] = {c: [] for c in ORDER_COLUMNS}
                res = res.__class__(
                    res.state,
                    rec,
                    res.positions,
                    res.fills,
                    frame_from_columns(rec["orders"]),
                    res.last_risk,
                    res.solve,
                )
                rec["warnings"].append(
                    "STOP: today's orders were cancelled; trading resumes after `qts paper ack`"
                )
            rec["status"] = {k: st[k] for k in ("level", "reasons", "stop_rules", "suppressed")}
            rec["state"] = res.state.to_dict()
            rec["timings"] = {"day_seconds": round(time.perf_counter() - t0, 2)}
            if not dry_run:
                _write_views(paths, rec["date"], res.state, res.fills, res.orders)
                if res.solve is not None and res.last_risk is not None:
                    atomic_write_bytes(
                        paths.last_risk, pickle.dumps((res.state.last_risk_date, res.last_risk), protocol=5)
                    )
                append_record(paths, rec)  # commit point of the day
            records.append(rec)
            seq += 1
            state = res.state
            last_risk = res.last_risk
            report["days"].append(
                {
                    "date": rec["date"],
                    "nav": rec["nav"],
                    "ret": rec["ret"],
                    "n_fills": rec["n_fills"],
                    "n_orders": len(rec["orders"].get("ticker", [])),
                    "orders_for": rec.get("orders_for"),
                    "status": rec["status"]["level"],
                    "mode": rec["mode"],
                    "seconds": rec["timings"]["day_seconds"],
                    "warnings": rec["warnings"],
                    **({"orders": frame_from_columns(rec["orders"]).to_dict("records")} if dry_run else {}),
                }
            )
            say(
                f"{rec['date']}: NAV {rec['nav']:,.0f} ({100 * rec['ret']:+.2f}%), {rec['n_fills']} fills, "
                f"{report['days'][-1]['n_orders']} orders for {rec.get('orders_for')}, status {rec['status']['level']} "
                f"[{rec['mode']}, {rec['timings']['day_seconds']:.0f}s]"
            )
        report["status"] = records[-1]["status"]
        report["seconds"] = round(time.perf_counter() - t_run, 2)
        return json_safe(report)


# =========================================================================== read side
def _book(
    book: str | os.PathLike[str], paper_root: str | os.PathLike[str]
) -> tuple[PaperPaths, BookHeader, list[dict[str, Any]]]:
    paths = _paths(book, paper_root)
    header = read_header(paths)
    raw, _ = read_journal(paths)
    return paths, header, effective_records(raw)


def history(
    book: str | os.PathLike[str], paper_root: str | os.PathLike[str] = DEFAULT_PAPER_ROOT
) -> list[dict[str, Any]]:
    """One row per journal day: date, mode, nav, ret, gross, net, fills, orders,
    trading cost, borrow, status level."""
    _, _, recs = _book(book, paper_root)
    out = []
    for r in recs:
        pnl = r.get("pnl", {}) or {}
        out.append(
            {
                "date": r["date"],
                "mode": r.get("mode"),
                "nav": r.get("nav"),
                "ret": r.get("ret"),
                "gross": r.get("gross"),
                "net": r.get("net"),
                "n_fills": r.get("n_fills"),
                "n_orders": len((r.get("orders") or {}).get("ticker", [])),
                "orders_for": r.get("orders_for"),
                "pnl_positions": pnl.get("positions"),
                "trading_cost": pnl.get("trading_cost"),
                "borrow": pnl.get("borrow"),
                "turnover": r.get("turnover"),
                "exante_vol": r.get("exante_vol"),
                "status": (r.get("status") or {}).get("level"),
                "rebalance": r.get("rebalance_decision"),
                "n_warnings": len(r.get("warnings", []) or []),
            }
        )
    return json_safe(out)


def orders(
    book: str | os.PathLike[str],
    date: str | None = None,
    paper_root: str | os.PathLike[str] = DEFAULT_PAPER_ROOT,
) -> dict[str, Any]:
    """Orders generated by the job of ``date`` (default: the latest) for the next
    close, with estimated costs, and the fills applied that day."""
    _, _, recs = _book(book, paper_root)
    if not recs:
        raise PaperError("the journal is empty")
    rec = recs[-1] if date is None else next((r for r in recs if r["date"] == iso(date)), None)
    if rec is None:
        raise PaperError(f"no journal day {iso(date)}")
    o = frame_from_columns(rec.get("orders"))
    f = frame_from_columns(rec.get("fills"))
    return json_safe(
        {
            "date": rec["date"],
            "orders_for": rec.get("orders_for"),
            "mode": rec.get("mode"),
            "status": (rec.get("status") or {}).get("level"),
            "orders": o.to_dict("records"),
            "fills": f.to_dict("records"),
            "n_orders": len(o),
            "buy_usd": float(o.loc[o["usd"] > 0, "usd"].sum()) if len(o) else 0.0,
            "sell_usd": float(-o.loc[o["usd"] < 0, "usd"].sum()) if len(o) else 0.0,
            "est_cost_usd": float(o["est_cost_usd"].sum()) if len(o) else 0.0,
            "cancelled_by_stop": rec.get("orders_cancelled_by_stop", 0),
        }
    )


def status(
    book: str | os.PathLike[str], paper_root: str | os.PathLike[str] = DEFAULT_PAPER_ROOT
) -> dict[str, Any]:
    """Everything the status page needs (shape in handbook 14): header facts, the
    latest day, the status with every check (acknowledgements applied), metrics,
    and per-day series (NAV, return, drawdown, analytic and bootstrap cones,
    gross / net, turnover, ex-ante vol, rolling realised vol)."""
    paths, header, recs = _book(book, paper_root)
    exp = X.load(paths.expectations, header.expectations_sha256)
    rules = StopRules.from_dict(header.stop_rules)
    acks = read_acks(paths)
    cfg = _load_config(paths)
    out: dict[str, Any] = {
        "book": header.name,
        "root": str(paths.root),
        "header": header.to_dict(),
        "n_days": len(recs),
        "last_date": recs[-1]["date"] if recs else None,
        "acks": acks,
    }
    if not recs:
        out["status"] = {"level": "OK", "reasons": ["no day processed yet"], "checks": [], "metrics": {}}
        out["series"] = {}
        return json_safe(out)
    prev_sup = (recs[-1].get("status") or {}).get("suppressed", []) or []
    st = evaluate(
        recs,
        exp,
        rules,
        header.first_execution,
        float(cfg.portfolio.gross_leverage_max),
        acks=acks,
        prior_suppressed=prev_sup,
    )
    st["acknowledged"] = any(str(a.get("date")) == recs[-1]["date"] for a in acks)
    st["trading_halted"] = st["level"] == "STOP" and not st["acknowledged"]
    out["status"] = st
    last = recs[-1]
    out["latest"] = {
        k: last.get(k)
        for k in (
            "date",
            "data_asof",
            "mode",
            "nav",
            "nav_pre",
            "cash",
            "gross",
            "net",
            "n_long",
            "n_short",
            "ret",
            "orders_for",
            "exante_vol",
            "warnings",
            "data_quality",
            "models",
            "code_version",
            "config_hash",
        )
    }
    out["latest"]["n_orders"] = len((last.get("orders") or {}).get("ticker", []))
    out["series"] = _series(recs, header, exp)
    return json_safe(out)


def _series(recs: list[dict[str, Any]], header: BookHeader, exp: X.Expectations) -> dict[str, Any]:
    rows = [r for r in recs if r["date"] >= header.first_execution]
    dates = [r["date"] for r in rows]
    nav = np.array([float(r["nav"]) for r in rows])
    ret = np.array([float(r.get("ret", 0.0) or 0.0) for r in rows])
    nav0 = header.nav0
    peak = np.maximum.accumulate(np.concatenate([[nav0], nav]))[1:] if len(nav) else np.array([])
    cum_log = np.cumsum(np.log1p(np.clip(ret, -0.999999, None)))
    n = np.arange(1, len(rows) + 1)
    mu, sg = exp.summary.get("mean_log"), exp.summary.get("sd_log")
    s = pd.Series(ret)
    rv = (s.rolling(21, min_periods=21).std(ddof=1) * math.sqrt(252)).tolist()
    out = {
        "date": dates,
        "nav": nav.tolist(),
        "ret": ret.tolist(),
        "cum_log": cum_log.tolist(),
        "drawdown": (nav / peak - 1.0).tolist() if len(nav) else [],
        "gross": [r.get("gross") for r in rows],
        "net": [r.get("net") for r in rows],
        "turnover": [r.get("turnover") for r in rows],
        "exante_vol": [r.get("exante_vol") for r in rows],
        "realised_vol_21d": rv,
        "status": [(r.get("status") or {}).get("level") for r in rows],
    }
    if mu is not None and sg is not None:
        out["cone_analytic"] = {
            "expected": (n * mu).tolist(),
            "lower": (n * mu - 1.96 * sg * np.sqrt(n)).tolist(),
            "upper": (n * mu + 1.96 * sg * np.sqrt(n)).tolist(),
        }
    out["cone_bootstrap"] = {
        "lower": [exp.quantile_at("cum_log", 0.025, float(k)) for k in n],
        "median": [exp.quantile_at("cum_log", 0.5, float(k)) for k in n],
        "upper": [exp.quantile_at("cum_log", 0.975, float(k)) for k in n],
        "max_dd_q01": [exp.quantile_at("max_dd", 0.01, float(k)) for k in n],
        "max_dd_q05": [exp.quantile_at("max_dd", 0.05, float(k)) for k in n],
    }
    return out


def ack(
    book: str | os.PathLike[str], note: str, paper_root: str | os.PathLike[str] = DEFAULT_PAPER_ROOT
) -> dict[str, Any]:
    """Acknowledge the latest day's status (a STOP needs this to resume trading).
    The acknowledged STOP rules are reported as WATCH until they clear."""
    if not note or not note.strip():
        raise PaperError("--note is required: say why it is safe to continue")
    paths = _paths(book, paper_root)
    read_header(paths)
    with BookLock(paths.root, "qts paper ack"):
        raw, _ = read_journal(paths)
        recs = effective_records(raw)
        if not recs:
            raise PaperError("nothing to acknowledge: the journal is empty")
        st = recs[-1].get("status") or {}
        a = {
            "date": recs[-1]["date"],
            "ts": utc_now(),
            "level": st.get("level"),
            "rules": list(st.get("stop_rules", []) or []),
            "reasons": list(st.get("reasons", []) or []),
            "note": note.strip(),
            "user": os.environ.get("USER", ""),
        }
        append_ack(paths, a)
    return json_safe(a)


def replay(
    book: str | os.PathLike[str], paper_root: str | os.PathLike[str] = DEFAULT_PAPER_ROOT
) -> dict[str, Any]:
    """Validate the journal and rebuild every derived file from it: a torn last line
    (crash mid-append) is removed (the original is kept as ``journal.jsonl.torn-<ts>``),
    the day views are rewritten and the accounting identity of every record
    (``nav = cash + sum(shares x price)``) and the day-to-day chain are checked."""
    paths = _paths(book, paper_root)
    header = read_header(paths)
    with BookLock(paths.root, "qts paper replay"):
        raw, torn = read_journal(paths)
        report: dict[str, Any] = {"torn_line_removed": False, "n_records": len(raw), "problems": []}
        if torn:
            bak = paths.journal.with_name(f"journal.jsonl.torn-{utc_now().replace(':', '')}")
            bak.write_bytes(paths.journal.read_bytes())
            from qts.paper.book import dumps

            atomic_write_text(paths.journal, "".join(dumps(r) + "\n" for r in raw))
            report["torn_line_removed"] = True
            report["backup"] = str(bak)
        recs = effective_records(raw)
        prev_date = None
        for r in recs:
            st = BookState.from_dict(r["state"])
            if st.date != r["date"]:
                report["problems"].append(f"{r['date']}: state date {st.date}")
            if prev_date is not None and r["date"] <= prev_date:
                report["problems"].append(f"{r['date']}: not after {prev_date}")
            v = sum(float(st.shares[k]) * float(st.px[k]) for k in st.shares)
            if abs(st.cash + v - float(r["nav"])) > 1e-6 * max(1.0, abs(float(r["nav"]))):
                report["problems"].append(f"{r['date']}: nav {r['nav']} != cash + positions {st.cash + v}")
            _write_views(
                paths, r["date"], st, frame_from_columns(r.get("fills")), frame_from_columns(r.get("orders"))
            )
            prev_date = r["date"]
        if recs and header.start and recs[0]["date"] != header.start:
            report["problems"].append(f"first record {recs[0]['date']} is not the book start {header.start}")
        report["n_days"] = len(recs)
        report["last_date"] = recs[-1]["date"] if recs else None
        report["ok"] = not report["problems"]
        return json_safe(report)


# =========================================================================== schedule
def schedule_text(
    book: str = "baseline",
    *,
    repo: str | os.PathLike[str] | None = None,
    python: str | None = None,
    paper_root: str | os.PathLike[str] = DEFAULT_PAPER_ROOT,
    hour: int = 7,
    minute: int = 5,
) -> dict[str, str]:
    """A launchd plist and a crontab line that run ``qts paper run`` at 07:05 on
    weekdays (Mac local time: set the Mac to America/New_York, or shift the hour).
    Printed only; nothing is installed."""
    repo_p = Path(repo or Path.cwd()).resolve()
    py = python or sys.executable
    log_path = repo_p / paper_root / book / "logs" / "run.log"
    cmd = (
        f"cd {shlex.quote(str(repo_p))} && {shlex.quote(py)} -m qts.cli paper run --name {shlex.quote(book)}"
    )
    label = f"com.qts.paper.{book}"
    cmd_xml = cmd.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    days = "\n".join(
        f"""    <dict><key>Weekday</key><integer>{d}</integer><key>Hour</key><integer>{hour}</integer><key>Minute</key><integer>{minute}</integer></dict>"""
        for d in range(1, 6)
    )
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>{cmd_xml}</string>
  </array>
  <key>WorkingDirectory</key><string>{repo_p}</string>
  <key>StartCalendarInterval</key>
  <array>
{days}
  </array>
  <key>StandardOutPath</key><string>{log_path}</string>
  <key>StandardErrorPath</key><string>{log_path}</string>
  <key>RunAtLoad</key><false/>
</dict>
</plist>
"""
    cron = f"{minute} {hour} * * 1-5 {cmd} >> {shlex.quote(str(log_path))} 2>&1"
    install = (
        f"# save the plist as ~/Library/LaunchAgents/{label}.plist, then:\n"
        f"#   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/{label}.plist\n"
        f"#   launchctl kickstart -k gui/$(id -u)/{label}      # run once now (test)\n"
        f"#   launchctl bootout gui/$(id -u)/{label}            # remove\n"
        "# or the crontab line with `crontab -e` (cron uses the Mac's local time zone).\n"
        "# The Mac must be awake at 07:05 (System Settings > Battery > Options, or\n"
        "#   sudo pmset repeat wakeorpoweron MTWRF 07:00:00 ); launchd runs a job missed\n"
        "# while asleep at the next wake, cron does not."
    )
    return {"label": label, "plist": plist, "crontab": cron, "log": str(log_path), "install": install}


# =========================================================================== reconcile
def reconcile(
    book: str | os.PathLike[str],
    *,
    paper_root: str | os.PathLike[str] = DEFAULT_PAPER_ROOT,
    source: DataSource | None = None,
    inner_factory: Callable[[], object] | None = None,
) -> dict[str, Any]:
    """Paper book vs ``DailyBarSimulator`` over the book's execution dates (flat start,
    same NAV, same rebalance schedule): the backfill pipeline check.  The batch side
    fits once on the data through the book's last day (the fold models come from the
    book's frozen store, so no refit) and runs the simulator exactly as
    ``ResearchPipeline.run`` does.  Returns per-day NAV / return / weight differences."""
    from qts.backtest import BacktestSpec, DailyBarSimulator

    paths, header, recs = _book(book, paper_root)
    rows = [r for r in recs if r["date"] >= header.first_execution]
    if len(rows) < 2:
        raise PaperError("need at least two execution days in the journal to reconcile")
    src = _source(header, source)
    end = pd.Timestamp(rows[-1]["date"])
    md = src.load(end)
    cfg = effective_config(_load_config(paths), md)
    identity = {
        "config_text_sha256": header.config_text_sha256,
        "use_ml": header.use_ml,
        "inner": None
        if inner_factory is None
        else f"{getattr(inner_factory, '__module__', '')}.{getattr(inner_factory, '__qualname__', type(inner_factory).__name__)}",
    }
    store = FrozenFoldStore(paths.models, identity) if header.use_ml else None
    from qts.paper.serving import fit_signals

    t0 = time.perf_counter()
    fit = fit_signals(
        md, cfg, use_ml=header.use_ml, store=store, inner_factory=inner_factory, nav0=header.nav0
    )
    pipe = fit.pipe
    dates = pd.DatetimeIndex([pd.Timestamp(r["date"]) for r in rows])
    elig = pipe.md_u.eligibility()
    spec = BacktestSpec.create(
        dates,
        nav0=header.nav0,
        rebalance_every=int(cfg.portfolio.rebalance_every_days),
        venue=header.venue,
        neutral_styles=pipe.neutral_styles(),
        eligible_fn=lambda d: elig.columns[elig.loc[d].to_numpy()],
        signal_lag_days=1,
    )
    sim = DailyBarSimulator(cfg, pipe.cost)
    bt = sim.run_spec(
        pipe.md_u, spec, pipe._eligible_alpha_fn(), lambda d, a: fit.risk_as_of(d, a), pipe.optimizer
    )
    batch_nav = bt.nav  # nav at each date before that date's costs: == paper nav_pre
    paper_nav_pre = pd.Series([float(r["nav_pre"]) for r in rows], index=dates)
    # the book's first execution day has nav_pre = nav0 (no position yet)
    paper_ret = pd.Series([float(r.get("ret_sim", 0.0) or 0.0) for r in rows], index=dates).iloc[1:]
    batch_ret = bt.returns["net"].reindex(paper_ret.index)
    w_diff = []
    for i, r in enumerate(rows[:-1]):
        st = BookState.from_dict(r["state"])
        nav_pre = float(r["nav_pre"])
        wp = pd.Series({k: st.shares[k] * st.px[k] / nav_pre for k in st.shares}, dtype=float)
        wb = bt.weights.iloc[i]
        u = wb.index.union(wp.index)
        w_diff.append(float((wp.reindex(u).fillna(0.0) - wb.reindex(u).fillna(0.0)).abs().max()))
    nav_rel = (paper_nav_pre - batch_nav.reindex(dates)).abs() / batch_nav.reindex(dates)
    d = (paper_ret - batch_ret).dropna()
    out = {
        "sizing": header.sizing,
        "dates": [iso(dates[0]), iso(dates[-1]), len(dates)],
        "seconds": round(time.perf_counter() - t0, 1),
        "max_abs_weight_diff": float(np.max(w_diff)) if w_diff else float("nan"),
        "max_rel_nav_diff": float(nav_rel.max()),
        "final_nav_paper_pre": float(paper_nav_pre.iloc[-1]),
        "final_nav_batch": float(batch_nav.reindex(dates).iloc[-1]),
        "cum_return_paper": float(np.prod(1.0 + paper_ret.to_numpy(dtype=float)) - 1.0),
        "cum_return_batch": float(np.prod(1.0 + batch_ret.fillna(0.0).to_numpy(dtype=float)) - 1.0),
        "daily_return_corr": float(paper_ret.corr(batch_ret)) if len(d) > 2 else float("nan"),
        "tracking_error_ann": float(d.std(ddof=1) * math.sqrt(252)) if len(d) > 2 else float("nan"),
        "max_abs_daily_return_diff": float(d.abs().max()) if len(d) else float("nan"),
        "batch_summary": {
            k: v
            for k, v in bt.summary().items()
            if k in ("sharpe_net", "avg_daily_turnover", "avg_cost_bps_per_side", "total_costs_usd")
        },
        "per_day": [
            {
                "date": iso(dt),
                "paper_nav_pre": float(paper_nav_pre[dt]),
                "batch_nav": float(batch_nav[dt]),
                "weight_diff": (w_diff[i] if i < len(w_diff) else None),
            }
            for i, dt in enumerate(dates)
        ],
        "note": "pipeline check on history, not out-of-sample evidence"
        + (
            ""
            if header.sizing == "replica"
            else "; moc sizing differs from the batch by one day of sizing drift and whole-share rounding"
        ),
    }
    return json_safe(out)
