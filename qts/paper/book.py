"""Durable storage of a paper book: header, append-only journal, acks, lock, views.

Layout of a paper root (default ``paper/<book_name>/``)::

    book.json              BookHeader, frozen at init (config + hash, start, NAV, data
                           spec, reference run, stop rules, expectation-table hash)
    config.yaml            frozen copy of the strategy config (the book reads this file,
                           never the source path again)
    expectations/          frozen expectation tables from the reference backtest
        summary.json       scalar expectations (daily mean / sd, turnover, cost, IC, ...)
        bootstrap.parquet  horizon x quantile table (cumulative return, max drawdown, Sharpe)
    journal.jsonl          append-only, one JSON record per processed trading day; the
                           book's single source of truth (positions, fills, orders, P&L
                           split, engine state, status, model versions)
    acks.jsonl             append-only acknowledgements of STOP statuses (``qts paper ack``)
    days/<YYYY-MM-DD>/     derived views of one day, rebuilt by ``qts paper replay``:
        positions.parquet  post-fill positions at the close (shares, price, value, weight)
        fills.parquet      fills applied at the close
        orders.parquet     orders generated for the next close
    models/                frozen walk-forward fold models (``qts.paper.serving``)
    state/last_risk.pkl    the last solve's risk-model output (ex-ante vol on hold days)
    logs/                  where the scheduled job writes its log
    .lock                  advisory lock held by every writer (``BookLock``)

Write discipline: every file except the journal and the acks is written to a
temporary file in the same directory, fsync'ed and moved into place
(``os.replace``), so a reader never sees half a file.  The journal and acks are
appended one line per ``write`` on an ``O_APPEND`` descriptor followed by
``fsync``; the journal line is the **commit point** of a day (views are written
before it, so a crash between the two leaves views that the next run or
``replay`` overwrites).  A torn last line (crash mid-write) is ignored by readers
and removed by ``replay``.

A day can be re-processed only explicitly (``qts paper run --redo-last``): the new
record carries ``supersedes`` (the sequence number it replaces) and readers keep the
last record per date (:func:`effective_records`).  Nothing is ever deleted from the
journal.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Bump when the book layout changes incompatibly.
BOOK_SCHEMA_VERSION = 1
DEFAULT_PAPER_ROOT = "paper"


class PaperError(RuntimeError):
    """A paper-book operation could not be carried out (message says why and what to do)."""


class BookLockedError(PaperError):
    """Another process holds the book's lock."""


class StaleDataError(PaperError):
    """The store does not have the trading day the job expects yet."""


# --------------------------------------------------------------------------- JSON helpers
def json_safe(v: Any) -> Any:
    """Recursively convert to JSON-serialisable values (non-finite floats -> None,
    timestamps -> ISO date / datetime strings, numpy scalars -> Python)."""
    if isinstance(v, Mapping):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return [json_safe(x) for x in v.tolist()]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, pd.Timestamp):
        if pd.isna(v):
            return None
        return str(v.date()) if v == v.normalize() else v.isoformat()
    if v is pd.NaT:
        return None
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return json_safe(dataclasses.asdict(v))
    return v


def dumps(obj: Any, indent: int | None = None) -> str:
    return json.dumps(json_safe(obj), indent=indent, sort_keys=indent is not None, allow_nan=False)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def iso(d: Any) -> str:
    """``YYYY-MM-DD`` of a date-like."""
    return str(pd.Timestamp(d).date())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- atomic IO
def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file in the same dir, fsync, replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | os.PathLike[str], obj: Any) -> None:
    atomic_write_text(path, dumps(obj, indent=1) + "\n")


def atomic_write_parquet(path: str | os.PathLike[str], frame: pd.DataFrame) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    os.close(fd)
    try:
        frame.to_parquet(tmp)
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def append_line(path: str | os.PathLike[str], line: str) -> None:
    """Append one line (``\\n`` added) with a single ``write`` on an ``O_APPEND``
    descriptor, then fsync."""
    data = (line.rstrip("\n") + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        n = os.write(fd, data)
        if n != len(data):  # pragma: no cover - regular files write fully
            raise OSError(f"short write to {path}: {n} of {len(data)} bytes")
        os.fsync(fd)
    finally:
        os.close(fd)


def read_jsonl(path: str | os.PathLike[str]) -> tuple[list[dict[str, Any]], bool]:
    """``(records, torn)``: every parseable line of ``path``; ``torn`` is True when the
    last line is incomplete or unparseable (a crash mid-append).  An unparseable line
    *before* the last one is an error (the file was edited or corrupted)."""
    p = Path(path)
    if not p.exists():
        return [], False
    raw = p.read_bytes().decode("utf-8")
    lines = raw.split("\n")
    out: list[dict[str, Any]] = []
    torn = False
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError as e:
            last = all(not x.strip() for x in lines[i + 1 :])
            if last:
                torn = True
                break
            raise PaperError(f"{p}: line {i + 1} is not valid JSON ({e}); run `qts paper replay`") from e
    if raw and not raw.endswith("\n") and not torn:
        torn = False  # a complete last JSON object without a newline is accepted
    return out, torn


# --------------------------------------------------------------------------- lock
class BookLock:
    """Exclusive, non-blocking advisory lock on ``<root>/.lock`` (``fcntl.flock``).

    The holder's pid, host, command and start time are written into the lock file
    so a second process can say who holds it.  The OS releases the lock when the
    process dies, so a crashed job never leaves a stale lock behind."""

    def __init__(self, root: str | os.PathLike[str], what: str = ""):
        self.path = Path(root) / ".lock"
        self.what = what
        self._fd: int | None = None

    def acquire(self) -> BookLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                holder = Path(self.path).read_text().strip() or "unknown process"
            except OSError:
                holder = "unknown process"
            os.close(fd)
            raise BookLockedError(
                f"paper book {self.path.parent} is locked by another job ({holder}); "
                "wait for it to finish (the lock is released when that process exits)"
            ) from None
        os.ftruncate(fd, 0)
        os.write(fd, f"pid {os.getpid()} on {os.uname().nodename}: {self.what} since {utc_now()}\n".encode())
        self._fd = fd
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        with contextlib.suppress(OSError):
            os.ftruncate(self._fd, 0)
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None

    def __enter__(self) -> BookLock:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()


# --------------------------------------------------------------------------- header
@dataclass(frozen=True)
class DataSpec:
    """How the book loads market data (the ``qts run --store`` loader flags).

    ``kind``: ``"store"`` (a :class:`~qts.data.store.ParquetStore` at ``store``) or
    ``"memory"`` (tests / benchmarks: the caller passes the data).  ``start`` is the
    first *data* date (the walk-forward refit schedule and the half-life blocks are
    anchored on it, so it must never change for a book); ``max_rank`` is the
    screened-rank prefilter of ``qts run --max-rank`` (default 1,500: covers the
    baseline's top-1,000 universe by construction); ``max_names`` / ``min_mcap_ever``
    the peak-market-cap prefilter, which can miss universe names
    (docs/notes/prefilter_lookahead_2026-11-14.md; refused by the load-coverage guard
    unless it covers).  A ``book.json`` written before ``max_rank`` existed keeps its
    peak-cap prefilter (``max_rank`` = ``None``)."""

    kind: str = "store"
    store: str = "data/store"
    start: str | None = "2000-01-01"
    max_names: int | None = None
    min_mcap_ever: float | None = None
    max_rank: int | None = 1500
    raw_dir: str = "data/raw/sharadar"
    sync_tables: tuple[str, ...] = ("SEP", "TICKERS", "DAILY", "SF1", "ACTIONS", "SP500")
    #: the incremental sync downloads rows updated since (last data date - this many days)
    sync_overlap_days: int = 5


@dataclass(frozen=True)
class BookHeader:
    """``book.json``: everything about a paper book that is fixed at init."""

    name: str
    created_at: str
    start: str  #: first job day: its close produces the first orders
    first_execution: str  #: next session after ``start``: first fills; rebalance anchor
    nav0: float
    sizing: str  #: "moc" (whole-share orders sized at the job's close) | "replica"
    use_ml: bool
    config_source: str  #: path the config was copied from at init
    config_file: str  #: frozen copy, relative to the book root
    config_text_sha256: str  #: SHA-256 of the frozen YAML text
    config_hash: str  #: SystemConfig.hash() at init (code of that day)
    data: DataSpec
    venue: str = "routed"
    calendar: str = "nyse"
    reference_run: str | None = None  #: run directory the expectations come from
    reference: dict[str, Any] = field(default_factory=dict)  #: ids/hashes of the reference run
    stop_rules: dict[str, Any] = field(default_factory=dict)
    expectations_sha256: str = ""  #: SHA-256 of expectations/summary.json + bootstrap.parquet
    code_version: str = "unknown"  #: git sha at init (``+dirty`` for an uncommitted tree)
    source_hashes: dict[str, str] = field(default_factory=dict)  #: stage code hashes at init
    #: hour (America/New_York) after which the vendor is expected to have day t's
    #: close; the job is meant to run at 07:05 ET the next morning
    data_ready_hour_et: int = 6
    schema_version: int = BOOK_SCHEMA_VERSION
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> BookHeader:
        d = dict(d)
        data = dict(d.pop("data", {}) or {})
        if "sync_tables" in data and data["sync_tables"] is not None:
            data["sync_tables"] = tuple(data["sync_tables"])
        known = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        dknown = {f.name for f in dataclasses.fields(DataSpec)}
        dd = {k: v for k, v in data.items() if k in dknown}
        if "max_rank" not in dd:  # written before max_rank existed: its own (peak-cap) prefilter
            dd["max_rank"] = None
            dd.setdefault("max_names", 3000)
            dd.setdefault("min_mcap_ever", 3e8)
        return cls(data=DataSpec(**dd), **kw)


# --------------------------------------------------------------------------- paths
class PaperPaths:
    """File locations of one paper book."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)

    @property
    def header(self) -> Path:
        return self.root / "book.json"

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def journal(self) -> Path:
        return self.root / "journal.jsonl"

    @property
    def acks(self) -> Path:
        return self.root / "acks.jsonl"

    @property
    def expectations(self) -> Path:
        return self.root / "expectations"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def state_dir(self) -> Path:
        return self.root / "state"

    @property
    def last_risk(self) -> Path:
        return self.state_dir / "last_risk.pkl"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def day_dir(self, date: Any) -> Path:
        return self.root / "days" / iso(date)

    def exists(self) -> bool:
        return self.header.exists()


def book_root(name: str, paper_root: str | os.PathLike[str] = DEFAULT_PAPER_ROOT) -> Path:
    """``<paper_root>/<name>``; ``name`` may also be a path to an existing book."""
    p = Path(name)
    if (p / "book.json").exists():
        return p
    if "/" in name or name in ("", ".", ".."):
        raise PaperError(f"invalid book name {name!r} (use a plain name such as 'baseline')")
    return Path(paper_root) / name


# --------------------------------------------------------------------------- journal
def read_header(paths: PaperPaths) -> BookHeader:
    if not paths.header.exists():
        raise PaperError(f"no paper book at {paths.root} (create it with `qts paper init`)")
    return BookHeader.from_dict(json.loads(paths.header.read_text()))


def read_journal(paths: PaperPaths) -> tuple[list[dict[str, Any]], bool]:
    return read_jsonl(paths.journal)


def effective_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Day records with superseded ones removed (the last record per date wins),
    in date order."""
    by_date: dict[str, dict[str, Any]] = {}
    for r in records:
        if r.get("kind", "day") != "day":
            continue
        by_date[str(r["date"])] = r
    return [by_date[d] for d in sorted(by_date)]


def append_record(paths: PaperPaths, record: Mapping[str, Any]) -> None:
    append_line(paths.journal, dumps(record))


def read_acks(paths: PaperPaths) -> list[dict[str, Any]]:
    recs, _ = read_jsonl(paths.acks)
    return recs


def append_ack(paths: PaperPaths, ack: Mapping[str, Any]) -> None:
    append_line(paths.acks, dumps(ack))


def wait_for(predicate: Any, timeout: float = 5.0, step: float = 0.05) -> bool:  # pragma: no cover
    """Poll ``predicate()`` until true or ``timeout`` seconds (test helper)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if predicate():
            return True
        time.sleep(step)
    return False
