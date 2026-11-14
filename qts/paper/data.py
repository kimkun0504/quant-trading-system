"""Market data for the paper book: sources, the daily Sharadar refresh, freshness
checks and the trading calendar.

* :class:`StoreSource` loads a :class:`~qts.data.store.ParquetStore` exactly as
  ``qts run --store`` does (same ``start`` / ``max_names`` / ``min_mcap_ever``
  prefilter, so the book trades the universe the reference backtest was run on);
  :class:`MemorySource` wraps an in-memory :class:`~qts.core.types.MarketData`
  (tests, benchmarks).  Both return data through a requested date only.
* :func:`sync_store` runs the existing incremental Sharadar sync
  (:func:`qts.data.sharadar.sync` with ``since``); the API key is read by that
  module from ``$NASDAQ_DATA_LINK_API_KEY`` or ``.env`` and is never seen here.
* :func:`expected_last_session` / :func:`check_freshness`: the vendor publishes a
  day's prices after the close, so the job (07:05 ET) expects the previous session
  to be in the store and fails with a clear message when it is not.
* Calendars: :class:`NYSECalendar` (:mod:`qts.data.calendar`) for real data and
  :class:`IndexCalendar` for a synthetic market's business-day dates.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import pandas as pd

from qts.core.types import MarketData
from qts.paper.book import DataSpec, PaperError, StaleDataError, iso

log = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- calendars
class Calendar(Protocol):
    def next_session(self, d: Any) -> pd.Timestamp: ...

    def sessions(self, start: Any, end: Any) -> pd.DatetimeIndex: ...

    def is_session(self, d: Any) -> bool: ...


class NYSECalendar:
    """NYSE full-day sessions (:mod:`qts.data.calendar`, 2000-2035)."""

    name = "nyse"

    def next_session(self, d: Any) -> pd.Timestamp:
        from qts.data.calendar import add_sessions

        out: Any = add_sessions(pd.Timestamp(d), 1)
        return pd.Timestamp(out)

    def sessions(self, start: Any, end: Any) -> pd.DatetimeIndex:
        from qts.data.calendar import nyse_sessions

        return nyse_sessions(start, end)

    def is_session(self, d: Any) -> bool:
        from qts.data.calendar import is_session

        return bool(is_session(d))


class IndexCalendar:
    """Sessions = the given dates; after the last one, business days (a synthetic
    market's ``pd.bdate_range`` calendar)."""

    name = "index"

    def __init__(self, dates: pd.DatetimeIndex):
        self.dates = pd.DatetimeIndex(dates)

    def _extended(self, end: Any) -> pd.DatetimeIndex:
        end = pd.Timestamp(end)
        if len(self.dates) and end <= self.dates[-1]:
            return self.dates
        start = self.dates[-1] + pd.offsets.BDay(1) if len(self.dates) else end
        return pd.DatetimeIndex(self.dates.append(pd.bdate_range(start, max(end, start))))

    def next_session(self, d: Any) -> pd.Timestamp:
        d = pd.Timestamp(d)
        ext = self._extended(d + pd.Timedelta(days=10))
        pos = int(ext.searchsorted(d, side="right"))
        return pd.Timestamp(ext[pos])

    def sessions(self, start: Any, end: Any) -> pd.DatetimeIndex:
        ext = self._extended(end)
        return ext[(ext >= pd.Timestamp(start)) & (ext <= pd.Timestamp(end))]

    def is_session(self, d: Any) -> bool:
        return pd.Timestamp(d) in self._extended(d)


def make_calendar(name: str, dates: pd.DatetimeIndex | None = None) -> Calendar:
    if name == "nyse":
        return NYSECalendar()
    if name == "index":
        if dates is None:
            raise PaperError("calendar 'index' needs the market's dates")
        return IndexCalendar(dates)
    raise PaperError(f"unknown calendar {name!r} (nyse | index)")


# --------------------------------------------------------------------------- sources
class DataSource(Protocol):
    def latest_date(self) -> pd.Timestamp | None: ...

    def load(self, end: Any = None) -> MarketData: ...

    def describe(self) -> dict[str, Any]: ...


class StoreSource:
    """A ParquetStore loaded like ``qts run --store --start --max-rank`` (or the peak-cap
    ``--max-names --min-mcap-ever``, checked by the load-coverage guard)."""

    def __init__(self, spec: DataSpec):
        self.spec = spec

    def _store(self):
        from qts.data.store import ParquetStore

        return ParquetStore(self.spec.store)

    def latest_date(self) -> pd.Timestamp | None:
        st = self._store()
        if not st.exists("prices_close"):
            return None
        out = st.duckdb_query('SELECT max(date) AS d FROM "prices_close"')
        d = out["d"].iloc[0] if len(out) else None
        return None if d is None or pd.isna(d) else pd.Timestamp(d).normalize()

    def tickers(self) -> list[str] | None:
        s = self.spec
        max_rank = getattr(s, "max_rank", None)
        if max_rank is not None:
            # every id that ranked <= max_rank by screened market cap on some date (qts run --max-rank)
            return self._store().tickers_by_screened_rank(int(max_rank), s.start, None)
        if s.min_mcap_ever is None and s.max_names is None:
            return None
        return self._store().tickers_by_market_cap(
            s.start, None, min_mcap_ever=s.min_mcap_ever, max_names=s.max_names
        )

    def check_coverage(self) -> Any:
        """The load-coverage guard (:mod:`qts.research.coverage`) for this spec and the
        default universe; raises ``PaperError`` when the prefilter misses universe names."""
        from qts.research.coverage import LoadCoverageError, require_load_coverage
        from qts.research.data import StoreSpec

        s = self.spec
        spec = StoreSpec(
            store=s.store,
            start=s.start,
            max_names=s.max_names if getattr(s, "max_rank", None) is None else None,
            min_mcap_ever=s.min_mcap_ever if getattr(s, "max_rank", None) is None else None,
            max_rank=getattr(s, "max_rank", None),
        )
        try:
            return require_load_coverage(spec)
        except LoadCoverageError as e:
            raise PaperError(str(e)) from e

    def load(self, end: Any = None) -> MarketData:
        if getattr(self.spec, "max_rank", None) is None and (
            self.spec.max_names is not None or self.spec.min_mcap_ever is not None
        ):
            self.check_coverage()  # a peak-cap prefilter must cover the universe (--max-rank does by construction)
        tick = self.tickers()
        if tick is not None and not tick:
            raise PaperError("the store prefilter (max_rank / max_names / min_mcap_ever) left no ids")
        md = self._store().as_market_data(universe_tickers=tick, start=self.spec.start, end=None)
        if end is not None and len(md.dates) and pd.Timestamp(end) < md.dates[-1]:
            md = md.slice_until(pd.Timestamp(end))
        return md

    def describe(self) -> dict[str, Any]:
        s = self.spec
        return {
            "kind": "store",
            "store": s.store,
            "start": s.start,
            "max_names": s.max_names,
            "min_mcap_ever": s.min_mcap_ever,
            "max_rank": getattr(s, "max_rank", None),
        }


class MemorySource:
    """An in-memory market (tests and benchmarks): ``load(end)`` = ``md.slice_until(end)``.

    ``snapshots`` (optional) maps a date to the market *as the vendor delivered it
    that day* (e.g. with the adjusted history restated after a dividend): the job
    for day ``t`` then reads ``snapshots[t]`` instead of the slice."""

    def __init__(self, md: MarketData, snapshots: dict[pd.Timestamp, MarketData] | None = None):
        self.md = md
        self.snapshots = {pd.Timestamp(k): v for k, v in (snapshots or {}).items()}
        self.available_until: pd.Timestamp | None = None  # simulate a vendor lag in tests

    def latest_date(self) -> pd.Timestamp | None:
        last = self.md.dates[-1] if len(self.md.dates) else None
        if self.available_until is not None and last is not None:
            ok = self.md.dates[self.md.dates <= self.available_until]
            return pd.Timestamp(ok[-1]) if len(ok) else None
        return None if last is None else pd.Timestamp(last)

    def load(self, end: Any = None) -> MarketData:
        if end is not None and pd.Timestamp(end) in self.snapshots:
            return self.snapshots[pd.Timestamp(end)]
        last = self.latest_date()
        cut = last if end is None else min(pd.Timestamp(end), last) if last is not None else None
        if cut is None or cut >= self.md.dates[-1]:
            return self.md
        return self.md.slice_until(cut)

    def describe(self) -> dict[str, Any]:
        return {"kind": "memory", "n_names": len(self.md.tickers), "n_days": len(self.md.dates)}


# --------------------------------------------------------------------------- sync
SyncFn = Callable[..., Any]


def sync_store(spec: DataSpec, since: str | None, sync_fn: SyncFn | None = None) -> dict[str, Any]:
    """Incremental Sharadar refresh of ``spec.store`` (rows with ``lastupdated >=
    since``); ``sync_fn`` defaults to :func:`qts.data.sharadar.sync` (injectable for
    tests).  Returns a JSON-safe summary; a vendor / network error is re-raised as
    :class:`PaperError` (message already redacted by the sync module)."""
    from qts.data import sharadar

    fn = sync_fn or sharadar.sync
    try:
        rep = fn(store_root=spec.store, raw_dir=spec.raw_dir, tables=list(spec.sync_tables), since=since)
    except sharadar.SharadarError as e:
        raise PaperError(f"data refresh failed: {e}") from e
    tables: dict[str, Any] = {}
    for name, st in (getattr(rep, "tables", {}) or {}).items():
        tables[str(name)] = {
            "rows_kept": int(getattr(st, "rows_kept", 0) or 0),
            "date_max": getattr(st, "date_max", None),
            "skipped": bool(getattr(st, "skipped", False)),
        }
    return {
        "mode": getattr(rep, "mode", "incremental"),
        "since": since,
        "seconds": float(getattr(rep, "seconds", 0.0) or 0.0),
        "warnings": list(getattr(rep, "warnings", []) or []),
        "tables": tables,
    }


def sync_since(last_data_date: Any, overlap_days: int) -> str:
    """The ``--since`` date of the incremental refresh: the last data date the book
    used minus ``overlap_days`` calendar days (re-reads late vendor corrections)."""
    return iso(pd.Timestamp(last_data_date) - pd.Timedelta(days=int(overlap_days)))


# --------------------------------------------------------------------------- freshness
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def expected_last_session(now: datetime, cal: Calendar, ready_hour_et: int = 6) -> pd.Timestamp:
    """The latest session whose prices the vendor should have published by ``now``:
    session ``d`` counts once ``now`` is past ``ready_hour_et``:00 New York time on
    the calendar day after ``d`` (Sharadar posts the day's SEP rows in the evening;
    the job runs at 07:05 ET the next morning)."""
    now_ny = now.astimezone(NY)
    probe = pd.Timestamp(now_ny.date()) - pd.Timedelta(days=1)
    for _ in range(15):
        if cal.is_session(probe):
            ready = datetime(probe.year, probe.month, probe.day, ready_hour_et, tzinfo=NY) + timedelta(days=1)
            if now_ny >= ready:
                return probe
        probe -= pd.Timedelta(days=1)
    raise PaperError(f"no session found in the two weeks before {now_ny.date()}")  # pragma: no cover


def check_freshness(latest: pd.Timestamp | None, expected: pd.Timestamp, source: str) -> None:
    """Raise :class:`StaleDataError` when the store's latest price date is older
    than the session the vendor should have published by now."""
    if latest is None:
        raise StaleDataError(f"{source} has no prices (prices_close is empty)")
    if latest < expected:
        raise StaleDataError(
            f"{source}: latest price date is {iso(latest)} but {iso(expected)} should be published by now. "
            "Sharadar updates SEP after the close; if this persists the vendor is late -- re-run later "
            "(the recommended schedule is 07:05 ET on weekdays), or process the available day explicitly "
            f"with `qts paper run --date {iso(latest)}`."
        )
