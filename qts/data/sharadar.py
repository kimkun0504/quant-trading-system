"""Sharadar Core US Equities Bundle (Nasdaq Data Link) -> :class:`~qts.data.store.ParquetStore`.

The bundle's tables (``SHARADAR/<TABLE>``) and what this module makes of them:

==========  ==============================================================================
``TICKERS`` security master -> ``security_master`` (long), ``industry`` (ticker -> FF48),
            ``delisting_dates`` (ticker -> last price date, NaT = alive); defines the
            kept universe and the panel ids (see *Identifiers*)
``SEP``     daily prices -> ``prices_close`` (``closeadj``), ``prices_close_raw``
            (``closeunadj``), ``prices_volume`` (actual shares traded, derived)
``DAILY``   daily valuation -> ``sharadar_marketcap`` (USD), ``sharadar_ev`` (USD),
            ``sharadar_pb``, ``sharadar_pe``, ``sharadar_ps``; then the derived
            ``market_cap`` and ``shares_outstanding`` (SF1 ``sharesbas`` fallback)
``SF1``     as-reported fundamentals (``ART`` + ``ARQ`` rows kept) -> ``sharadar_sf1``
            (long, keyed on ``datekey``); then the derived PIT panels ``book_to_price``,
            ``earnings_yield``, ``gross_profitability``, ``asset_growth`` through
            :func:`qts.data.loaders.sharadar_sf1_to_pit_fundamentals`
``ACTIONS`` corporate actions -> ``corporate_actions`` (long, reference; the
            ``delisted`` rows cross-check ``delisting_dates`` in :func:`validate`);
            then the derived ``delisting_reasons`` (one row per delisted id:
            acquisition / performance / voluntary / unknown, see *Delisting reasons*)
``SP500``   index membership events -> ``sp500_membership`` (long); the helper
            :func:`sp500_membership_panel` turns it into a date x id boolean panel.
            Stored and exposed only: it is **not** part of the universe rules.
``METRICS``, ``EVENTS``, ``SFP``, ``INDICATORS``
            optional (``--tables``); stored verbatim as ``sharadar_<table>`` long tables
==========  ==============================================================================

Identifiers
-----------
Sharadar restates a company's whole price history under its **latest** ticker, and a
ticker can later be reused by a different company (a different ``permaticker`` with a
disjoint ``[firstpricedate, lastpricedate]``).  Every panel is therefore keyed by a
stable *id*: the Sharadar ticker when exactly one kept ``permaticker`` ever used it, and
``f"{ticker}.{permaticker}"`` for **every** holder when several did (``DDD.100004``,
``DDD.100005``).  SEP/DAILY rows are resolved by ``(ticker, date)`` and SF1 rows by
``(ticker, datekey)`` against each holder's price-date range (the alive holder's range is
open-ended); rows that match no kept holder are dropped and counted in the
:class:`SyncReport`.  Sharadar tickers may themselves contain a dot (``BRK.A``); an id's
suffix is a permaticker only when it is all digits.

Delisting reasons
-----------------
:func:`derive_delisting_reasons` classifies every delisted id from the ACTIONS rows
of its ticker dated within :data:`DELISTING_REASON_WINDOW_DAYS` business days
(weekdays; +-5) of its delisting date (TICKERS ``lastpricedate``), through
:data:`DELISTING_REASON_ACTIONS`:

* ``acquisitionby`` / ``mergerto`` -> **acquisition**.  Reading of the ACTIONS
  columns used here: a row describes the company in ``ticker``; ``contraticker``
  is the counterparty.  ``acquisitionby`` = "``ticker`` was acquired by
  ``contraticker``" and ``mergerto`` = "``ticker`` merged into ``contraticker``"
  (the row's company ceases to trade); the mirror rows ``acquisitionof`` /
  ``mergerfrom`` sit on the surviving company and are not mapped.  This reading
  follows Sharadar's action names; it has not been checked against the vendor's
  field documentation from here, so the derive report prints every action code
  seen next to a delisting: an acquirer does not delist the week it buys
  something, so if ``acquisitionof`` -- not ``acquisitionby`` -- dominates
  there, the reading is inverted and the constant must be corrected.
* ``bankruptcyliquidation`` / ``regulatorydelisting`` -> **performance**.
* ``voluntarydelisting`` -> **voluntary**.
* none of these (a plain ``delisted`` row, or no row at all) -> **unknown**.

Several mapped actions on one delisting: performance > acquisition > voluntary
(:data:`qts.core.delisting.REASON_PRECEDENCE`).  The table is rebuilt from the
stored ``corporate_actions`` + ``security_master`` (no download):
``qts data sharadar derive --store <store>``; ``sync`` also writes it.
:meth:`ParquetStore.as_market_data` loads it into ``MarketData.delisting_reasons``.

Adjustments and units
---------------------
* SEP ``open/high/low/close/volume`` are **split-adjusted** (not dividend-adjusted);
  ``closeadj`` is split- and dividend-adjusted; ``closeunadj`` is the as-traded close.
  Actual shares traded are recovered as ``volume * close / closeunadj``: before an
  ``m``-for-1 split ``close = closeunadj / m`` and the adjusted volume is ``m x`` the
  shares that actually traded, so the ratio undoes the adjustment exactly.  Where
  ``closeunadj`` is missing or zero the row keeps the vendor volume (counted).
* DAILY ``marketcap`` and ``ev`` are in **USD millions** (:data:`DAILY_USD_SCALE`);
  ``market_cap = marketcap * 1e6`` and ``shares_outstanding = market_cap / closeunadj``
  (same day).  Days without a DAILY row use SF1 ``sharesbas`` carried forward from the
  filing date and ``market_cap = closeunadj * shares``.  :func:`validate` checks the
  unit (largest caps ~1e11-1e13 USD; implied shares vs ``sharesbas``).
* SF1 values are in plain USD / shares; every panel built from them is visible from
  ``datekey + 1`` calendar day (snapped to the next trading day).
* Industry labels: ``famaindustry`` (Sharadar's Fama-French 48 names) mapped to the
  short names of :data:`qts.data.loaders.FF48_NAMES`, else ``sic_to_ff48(siccode)``,
  else ``"Other"``.  TICKERS carries only the *current* classification, so no
  ``industry_history`` is produced (the static map applies; see
  :func:`build_industry_history`).

Not covered by the bundle: spreads (``spread_bps``), borrow fees, short interest, dated
industry history.

Download
--------
Bulk export of the Tables API: ``GET {API_BASE}/<TABLE>.json?qopts.export=true&api_key=KEY``
(plus filters such as ``lastupdated.gte=YYYY-MM-DD``), polled until
``datatable_bulk_download.file.status == "fresh"``, then the ``file.link`` zip (one CSV)
is downloaded and cached under ``<raw_dir>/<TABLE>/<YYYYMMDD>_<full|since-DATE>.zip``.
The key comes from ``api_key=``, the environment variable
:data:`API_KEY_ENV` or a ``.env`` file in the working directory (``KEY=VALUE`` lines,
parsed here without ``python-dotenv``).  The key is never logged and is redacted from
every error message.  ``from_files=DIR`` reads already-downloaded zips/CSVs (the
website's *Data Export* button produces the same files) and never touches the network.

Memory: SEP is ~20M rows.  Nothing is ever pivoted whole; CSVs are streamed in chunks,
rows are resolved to ids and spilled to per-year staging parquet files, and each year is
written once with :meth:`ParquetStore.write_long`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import shutil
import time
import uuid
import zipfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from qts.core.delisting import DELISTING_REASONS, REASON_PRECEDENCE
from qts.core.delisting import UNKNOWN as UNKNOWN_REASON
from qts.data.loaders import (
    FF48_LONG_NAMES,
    FF48_NAMES,
    ff48_from_sic_series,
    pit_forward_fill,
    sharadar_sf1_to_pit_fundamentals,
)
from qts.data.store import ParquetStore

try:  # optional dependency ("data" extra); the transform layer never needs it
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================
API_BASE = "https://data.nasdaq.com/api/v3/datatables/SHARADAR"
API_KEY_ENV = "NASDAQ_DATA_LINK_API_KEY"
DEFAULT_RAW_DIR = "data/raw/sharadar"
DEFAULT_STORE = "data/store"

#: tables loaded by default (order = processing order)
DEFAULT_TABLES: tuple[str, ...] = ("TICKERS", "SEP", "DAILY", "SF1", "ACTIONS", "SP500")
#: tables allowed with ``--tables`` but skipped by default (stored verbatim)
OPTIONAL_TABLES: tuple[str, ...] = ("METRICS", "EVENTS", "SFP", "INDICATORS")
KNOWN_TABLES: tuple[str, ...] = DEFAULT_TABLES + OPTIONAL_TABLES
#: tables that support ``lastupdated.gte`` incremental exports; the others are always
#: downloaded whole (small, or without a ``lastupdated`` column)
INCREMENTAL_TABLES: tuple[str, ...] = ("SEP", "DAILY", "SF1", "SFP", "METRICS")

#: DAILY ``marketcap`` / ``ev`` are reported in USD millions
DAILY_USD_SCALE = 1e6

#: TICKERS ``category`` values kept by default.  ADRs (foreign issuers; different
#: filings, FX and settlement), Canadian stocks, preferred stock, ETFs/ETNs/CEFs/units
#: and warrants are not the US common-equity cross-section the models are built for;
#: secondary share classes would double-count one company (one capitalisation, two
#: price series), so only the primary class is kept.
DEFAULT_CATEGORIES: frozenset[str] = frozenset(
    {"Domestic Common Stock", "Domestic Common Stock Primary Class"}
)
#: TICKERS ``exchange`` values kept by default (OTC excluded: no consolidated
#: auction, unreliable prints; ``NYSEARCA`` is ETF territory; ``INDEX`` is not a stock)
DEFAULT_EXCHANGES: frozenset[str] = frozenset({"NYSE", "NASDAQ", "NYSEMKT", "BATS"})

#: SF1 columns kept in ``sharadar_sf1`` (besides the keys)
SF1_FIELDS: tuple[str, ...] = (
    "bvps",
    "eps",
    "epsusd",
    "gp",
    "assets",
    "equity",
    "revenue",
    "netinc",
    "sharesbas",
    "shareswa",
    "marketcap",
    "pe",
    "pb",
    "ev",
    "ebitda",
    "fcf",
    "debt",
    "liabilities",
)
SF1_KEYS: tuple[str, ...] = ("ticker", "dimension", "calendardate", "datekey", "reportperiod", "lastupdated")
SF1_DIMENSIONS: tuple[str, ...] = ("ART", "ARQ")
#: fields handed to ``sharadar_sf1_to_pit_fundamentals`` (keeps its memory small)
FUNDAMENTAL_FIELDS: tuple[str, ...] = ("bvps", "eps", "gp", "assets")
FUNDAMENTAL_PANELS: tuple[str, ...] = (
    "book_to_price",
    "earnings_yield",
    "gross_profitability",
    "asset_growth",
)

SEP_COLUMNS = (
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "closeadj",
    "closeunadj",
    "lastupdated",
)
DAILY_COLUMNS = ("ticker", "date", "lastupdated", "ev", "evebit", "evebitda", "marketcap", "pb", "pe", "ps")
DAILY_PANELS: dict[str, str] = {
    "ev": "sharadar_ev",
    "pb": "sharadar_pb",
    "pe": "sharadar_pe",
    "ps": "sharadar_ps",
}

#: ACTIONS ``action`` code -> delisting reason (module docstring "Delisting reasons").
#: A module constant so it can be corrected after looking at real counts
#: (``qts data sharadar derive`` prints them); codes not listed do not set a reason.
DELISTING_REASON_ACTIONS: dict[str, str] = {
    "acquisitionby": "acquisition",
    "mergerto": "acquisition",
    "bankruptcyliquidation": "performance",
    "regulatorydelisting": "performance",
    "voluntarydelisting": "voluntary",
}
#: an action counts for a delisting when it is dated within this many business
#: days (weekdays, holidays not excluded) of the id's delisting date, either side
DELISTING_REASON_WINDOW_DAYS = 5
#: wider window used only for the report's hint on unknowns ("is the window too tight?")
DELISTING_REASON_HINT_WINDOW_DAYS = 63
DELISTING_REASON_COLUMNS: tuple[str, ...] = ("id", "delisting_date", "reason", "source_action")

_OPEN_START = pd.Timestamp("1900-01-01")
_OPEN_END = pd.Timestamp("2200-01-01")
_STATE_DIR = "_sharadar"


def _ns(s: pd.Series) -> pd.Series:
    """Datetime Series at nanosecond resolution (parquet round-trips may come back as
    ``datetime64[us]``; ``merge_asof`` and comparisons want one resolution)."""
    return pd.to_datetime(s).astype("datetime64[ns]")


class SharadarError(RuntimeError):
    """Base class for loader errors (messages never contain the API key)."""


class SharadarDownloadError(SharadarError):
    """HTTP / export errors from Nasdaq Data Link."""


# =============================================================================
# API key handling
# =============================================================================
def parse_dotenv(text: str) -> dict[str, str]:
    """Parse the simple ``KEY=VALUE`` format of a ``.env`` file.

    Supports ``export KEY=VALUE``, ``#`` comments, blank lines and single/double
    quoted values.  No variable expansion.  Kept in-house so the ``data`` extra
    does not need ``python-dotenv``."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        elif " #" in v:
            v = v.split(" #", 1)[0].rstrip()
        if k:
            out[k] = v
    return out


def load_api_key(explicit: str | None = None, env_file: str | os.PathLike[str] | None = ".env") -> str | None:
    """API key from ``explicit``, else ``$NASDAQ_DATA_LINK_API_KEY``, else ``env_file``
    (default ``.env`` in the working directory).  Returns ``None`` when none is set."""
    if explicit:
        return explicit.strip()
    v = os.environ.get(API_KEY_ENV, "").strip()
    if v:
        return v
    if env_file is not None and os.path.exists(env_file):
        with open(env_file, encoding="utf-8") as f:
            v = parse_dotenv(f.read()).get(API_KEY_ENV, "").strip()
        if v:
            return v
    return None


def redact(text: object, key: str | None) -> str:
    """Replace every occurrence of ``key`` (and any ``api_key=...`` query value) in ``text``."""
    s = str(text)
    if key:
        s = s.replace(key, "<redacted>")
    return re.sub(r"(api_key=)[^&\s'\"]+", r"\1<redacted>", s)


# =============================================================================
# Download layer
# =============================================================================
class NasdaqDataLinkClient:
    """Minimal Tables-API client for bulk exports (``qopts.export=true``).

    ``session`` is any object with ``get(url, params=, timeout=, stream=)`` returning a
    response with ``status_code``, ``json()``, ``iter_content()``, ``raise_for_status()``
    (a ``requests.Session`` by default; tests inject a fake)."""

    def __init__(
        self,
        api_key: str,
        session: Any | None = None,
        base_url: str = API_BASE,
        poll_interval: float = 5.0,
        max_wait: float = 3600.0,
        timeout: float = 120.0,
    ):
        if not api_key:
            raise SharadarDownloadError(
                f"no API key: set {API_KEY_ENV} (environment or .env) or pass api_key="
            )
        self._key = api_key
        if session is None:
            if requests is None:  # pragma: no cover
                raise ImportError("pip install requests  (optional dependency 'data')")
            session = requests.Session()
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.poll_interval = float(poll_interval)
        self.max_wait = float(max_wait)
        self.timeout = float(timeout)

    # -- helpers ----------------------------------------------------------------
    def _get(self, url: str, **kw: Any) -> Any:
        try:
            r = self.session.get(url, timeout=self.timeout, **kw)
        except Exception as e:  # requests exceptions embed the full URL (with the key)
            raise SharadarDownloadError(f"request failed: {redact(e, self._key)}") from None
        if getattr(r, "status_code", 200) >= 400:
            body = redact(str(getattr(r, "text", ""))[:300], self._key)
            raise SharadarDownloadError(
                f"HTTP {r.status_code} for {redact(url, self._key)}: {body}"
            ) from None
        return r

    def export_status(self, table: str, filters: Mapping[str, str] | None = None) -> dict[str, Any]:
        """One export request; returns ``{"status", "link", "data_snapshot_time"}``."""
        url = f"{self.base_url}/{table}.json"
        params: dict[str, str] = {"qopts.export": "true"}
        params.update({k: str(v) for k, v in (filters or {}).items()})
        params["api_key"] = self._key
        r = self._get(url, params=params)
        try:
            info = r.json()["datatable_bulk_download"]["file"]
        except (KeyError, TypeError, ValueError) as e:
            raise SharadarDownloadError(
                f"unexpected export response for {table}: {redact(e, self._key)}"
            ) from None
        return {
            "status": str(info.get("status", "")),
            "link": str(info.get("link", "")),
            "data_snapshot_time": info.get("data_snapshot_time"),
        }

    def export_table(
        self, table: str, dest: str | os.PathLike[str], filters: Mapping[str, str] | None = None
    ) -> Path:
        """Request the export, poll until ``status == "fresh"``, download the zip to
        ``dest`` (atomic rename) and return its path."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.max_wait
        n = 0
        while True:
            st = self.export_status(table, filters)
            n += 1
            if st["status"] == "fresh" and st["link"]:
                break
            if time.monotonic() > deadline:
                raise SharadarDownloadError(
                    f"export of {table} not ready after {self.max_wait:.0f}s (last status {st['status']!r})"
                )
            log.info(
                "SHARADAR/%s export status %r (poll %d); waiting %.0fs",
                table,
                st["status"],
                n,
                self.poll_interval,
            )
            time.sleep(self.poll_interval)
        link = st["link"]
        tmp = dest.with_name(f".tmp-{uuid.uuid4().hex}.partial")
        try:
            r = self._get(link, stream=True)
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                tmp.unlink()
        log.info("SHARADAR/%s downloaded -> %s (%.1f MB)", table, dest, dest.stat().st_size / 1e6)
        return dest


def raw_cache_path(
    raw_dir: str | os.PathLike[str], table: str, since: str | None, day: _date | None = None
) -> Path:
    """``<raw_dir>/<TABLE>/<YYYYMMDD>_<full|since-YYYY-MM-DD>.zip``."""
    day = day or _date.today()
    tag = f"since-{pd.Timestamp(since).date().isoformat()}" if since else "full"
    return Path(raw_dir) / table.upper() / f"{day.strftime('%Y%m%d')}_{tag}.zip"


_TOKEN_RE = re.compile(r"[^A-Za-z0-9]+")


_DATA_FILE_RE = re.compile(r"\.(zip|csv|csv\.gz)$", re.I)
#: tables a sync cannot do without when they are requested from files
REQUIRED_TABLES: tuple[str, ...] = ("TICKERS", "SEP", "DAILY", "SF1")


def _data_files(directory: str | os.PathLike[str]) -> list[Path]:
    root = Path(directory)
    return sorted(p for p in root.rglob("*") if p.is_file() and _DATA_FILE_RE.search(p.name))


def find_table_files(directory: str | os.PathLike[str], table: str) -> list[Path]:
    """Files under ``directory`` (searched recursively) holding ``table``: ``.zip`` (one
    CSV inside), ``.csv`` or ``.csv.gz`` whose file name contains the table code as a
    token, case-insensitively (``SHARADAR_SEP_2f3a.zip``, ``sep.csv``,
    ``sharadar-tickers.zip``), or that sit in a sub-directory named after the table
    (the raw cache layout ``<raw_dir>/SEP/20261109_full.zip``).

    Selection when several files match: in the cache layout the newest ``*_full.zip``
    plus every later ``*_since-*.zip`` (oldest first, so increments apply after the
    full file); otherwise **the newest file by modification time** only, so a folder
    with two exports of one table uses the latest one.  The choice is logged."""
    t = table.upper()
    out: list[Path] = []
    for p in _data_files(directory):
        tokens = {x.upper() for x in _TOKEN_RE.split(p.name) if x}
        if t in tokens or p.parent.name.upper() == t:
            out.append(p)
    if not out:
        return []
    dated = [p for p in out if re.fullmatch(r"\d{8}_(full|since-\d{4}-\d{2}-\d{2})\.zip", p.name)]
    if dated and len(dated) == len(out):
        fulls = sorted(p for p in dated if p.name.endswith("_full.zip"))
        if fulls:
            base = fulls[-1]
            return [base, *sorted(p for p in dated if p.name > base.name and "_since-" in p.name)]
    if len(out) > 1:
        newest = max(out, key=lambda p: p.stat().st_mtime)
        log.warning(
            "SHARADAR/%s: %d files match under %s; using the newest by mtime: %s (ignored: %s)",
            table,
            len(out),
            directory,
            newest,
            ", ".join(p.name for p in out if p != newest),
        )
        return [newest]
    return out


def resolve_sources(directory: str | os.PathLike[str], tables: Sequence[str]) -> dict[str, list[Path]]:
    """Table -> files for a ``--from-files`` folder (:func:`find_table_files` per table).

    Raises :class:`SharadarError` listing every data file found when a requested table
    of :data:`REQUIRED_TABLES` has no match; a missing optional table is returned with an
    empty list (the sync skips it with a warning)."""
    if not Path(directory).is_dir():
        raise SharadarError(f"--from-files: {directory!s} is not a directory")
    found = {t.upper(): find_table_files(directory, t) for t in tables}
    missing = [t for t, f in found.items() if not f and t in REQUIRED_TABLES]
    if missing:
        files = _data_files(directory)
        listing = (
            "\n  ".join(str(p.relative_to(directory)) for p in files[:40])
            or "(no .zip / .csv / .csv.gz files)"
        )
        raise SharadarError(
            f"no file for required table(s) {missing} under {directory!s}. A file is matched when its name "
            f"contains the table code as a word (SHARADAR_SEP_*.zip, sep.csv, sharadar-tickers.zip) or it sits "
            f"in a sub-folder named after the table. Files found:\n  {listing}"
        )
    return found


def format_source_map(found: Mapping[str, Sequence[Path]]) -> str:
    lines = ["table -> file"]
    for t, files in found.items():
        if not files:
            lines.append(f"  {t:9s} (none; optional table skipped)")
        for p in files:
            mb = p.stat().st_size / 1e6 if p.exists() else float("nan")
            lines.append(f"  {t:9s} {p}  ({mb:.1f} MB)")
    return "\n".join(lines)


# =============================================================================
# Chunked CSV reading (zip or plain)
# =============================================================================
def _csv_member(z: zipfile.ZipFile) -> str:
    names = [n for n in z.namelist() if n.lower().endswith(".csv")] or z.namelist()
    if not names:
        raise SharadarError("zip archive is empty")
    return max(names, key=lambda n: z.getinfo(n).file_size)


def iter_csv_chunks(
    path: str | os.PathLike[str],
    chunksize: int = 500_000,
    usecols: Iterable[str] | None = None,
    dtype: Mapping[str, Any] | None = None,
) -> Iterator[pd.DataFrame]:
    """Stream a Sharadar CSV (plain, gzip or inside a zip) in chunks with lower-cased
    column names.  ``usecols`` names are matched case-insensitively and missing ones
    are ignored (so a column the vendor adds or drops does not break the load)."""
    want = {c.lower() for c in usecols} if usecols is not None else None
    kw: dict[str, Any] = {"chunksize": int(chunksize), "low_memory": False}
    if want is not None:
        kw["usecols"] = lambda c: str(c).strip().lower() in want
    if dtype:
        kw["dtype"] = dict(dtype)
    p = Path(path)
    if zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as z, z.open(_csv_member(z)) as fh:
            for chunk in pd.read_csv(fh, **kw):
                chunk.columns = [str(c).strip().lower() for c in chunk.columns]
                yield chunk
    else:
        for chunk in pd.read_csv(p, **kw):
            chunk.columns = [str(c).strip().lower() for c in chunk.columns]
            yield chunk


def read_csv_table(
    path: str | os.PathLike[str], usecols: Iterable[str] | None = None, dtype: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    """Whole small table (TICKERS, ACTIONS, SP500) as one frame."""
    parts = list(iter_csv_chunks(path, chunksize=2_000_000, usecols=usecols, dtype=dtype))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# =============================================================================
# Security master and identifiers
# =============================================================================
@dataclass(frozen=True)
class MasterFilters:
    """Which TICKERS rows form the kept universe (see :data:`DEFAULT_CATEGORIES`)."""

    categories: frozenset[str] = DEFAULT_CATEGORIES
    exchanges: frozenset[str] = DEFAULT_EXCHANGES
    table: str = "SEP"


def _norm_label(s: object) -> str:
    return (
        re.sub(r"[^a-z0-9]", "", str(s).lower())
        if s is not None and not (isinstance(s, float) and np.isnan(s))
        else ""
    )


#: Sharadar ``famaindustry`` spelling -> FF48 short name, where the normalised long name
#: of ``FF48_DEFINITIONS`` does not match (Sharadar's own spellings).
FAMAINDUSTRY_ALIASES: dict[str, str] = {
    "restarauntshotelsmotels": "Meals",  # Sharadar's spelling
    "restaurantshotelsmotels": "Meals",
    "shipbuildingrailroadequipment": "Ships",
    "steelworksetc": "Steel",
    "nonmetallicandindustrialmetalmining": "Mines",
    "almostnothing": "Other",
    "other": "Other",
}
#: normalised FF48 long name -> short name (the table the mapping is checked against)
FAMAINDUSTRY_TABLE: dict[str, str] = {_norm_label(long): short for short, long in FF48_LONG_NAMES.items()}
FAMAINDUSTRY_TABLE.update(FAMAINDUSTRY_ALIASES)


def map_famaindustry(
    famaindustry: pd.Series, siccode: pd.Series | None = None
) -> tuple[pd.Series, pd.Series]:
    """FF48 short names from Sharadar's ``famaindustry``; ``siccode`` fallback; else ``"Other"``.

    Returns ``(labels, source)`` with ``source`` in ``{"famaindustry", "sic", "other"}``."""
    keys = famaindustry.map(_norm_label)
    lab = keys.map(FAMAINDUSTRY_TABLE)
    src = pd.Series(np.where(lab.notna(), "famaindustry", ""), index=famaindustry.index, dtype=object)
    if siccode is not None:
        sic = ff48_from_sic_series(siccode)
        sic_ok = lab.isna() & pd.to_numeric(siccode, errors="coerce").notna() & (sic != "Other")
        lab = lab.where(~sic_ok, sic)
        src = src.where(~sic_ok, "sic")
    lab = lab.fillna("Other").astype(str)
    src = src.where(src != "", "other")
    assert set(lab.unique()) <= set(FF48_NAMES)
    return lab, src


@dataclass
class SecurityMaster:
    """Kept TICKERS rows with the panel ``id`` column, plus the resolution ranges."""

    frame: pd.DataFrame
    ranges: pd.DataFrame  # ticker, permaticker, id, start, end
    filters: MasterFilters = field(default_factory=MasterFilters)
    n_input_rows: int = 0

    @property
    def ids(self) -> list[str]:
        return sorted(self.frame["id"].astype(str))

    def industry_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"ticker": self.frame["id"].astype(str), "industry": self.frame["industry_ff48"].astype(str)}
        )

    def delisting_table(self) -> pd.DataFrame:
        dd = self.frame["lastpricedate"].where(self.frame["isdelisted"], pd.NaT)
        return pd.DataFrame(
            {"ticker": self.frame["id"].astype(str), "delisting_date": pd.to_datetime(dd).to_numpy()}
        )

    def summary(self) -> dict[str, Any]:
        f = self.frame
        holders = self.ranges.groupby("ticker")["permaticker"].size()
        reused = int((holders > 1).sum())
        return {
            "n_tickers_rows": int(self.n_input_rows),
            "n_kept": len(f),
            "n_ids": int(f["id"].nunique()),
            "n_reused_tickers": reused,
            "n_delisted": int(f["isdelisted"].sum()),
            "industry_source": {k: int(v) for k, v in f["industry_source"].value_counts().items()},
            "n_other_industry": int((f["industry_ff48"] == "Other").sum()),
            "exchanges": {k: int(v) for k, v in f["exchange"].value_counts().items()},
            "categories": {k: int(v) for k, v in f["category"].value_counts().items()},
        }


def build_security_master(tickers: pd.DataFrame, filters: MasterFilters | None = None) -> SecurityMaster:
    """TICKERS -> :class:`SecurityMaster` (pure).

    Keeps ``table == filters.table`` rows (TICKERS has one row per table a security
    appears in), ``category`` in ``filters.categories`` and ``exchange`` in
    ``filters.exchanges``, one row per ``permaticker`` (latest ``lastupdated``); adds
    ``industry_ff48``, ``industry_source``, ``isdelisted`` (bool), parsed
    ``firstpricedate`` / ``lastpricedate`` and the panel ``id``."""
    filters = filters or MasterFilters()
    df = tickers.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    need = {"permaticker", "ticker", "category", "exchange", "isdelisted"}
    missing = need - set(df.columns)
    if missing:
        raise SharadarError(f"TICKERS is missing columns {sorted(missing)}; got {list(df.columns)[:12]}...")
    n_in = len(df)
    for c in (
        "name",
        "cusips",
        "location",
        "famaindustry",
        "sicsector",
        "sicindustry",
        "sector",
        "industry",
        "scalemarketcap",
        "relatedtickers",
        "currency",
        "table",
        "siccode",
        "lastupdated",
        "firstpricedate",
        "lastpricedate",
        "firstquarter",
        "lastquarter",
    ):
        if c not in df.columns:
            df[c] = np.nan
    if df["table"].notna().any():
        df = df[df["table"].astype(str).str.upper() == filters.table.upper()]
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["category"] = df["category"].astype(str).str.strip()
    df["exchange"] = df["exchange"].astype(str).str.strip().str.upper()
    df = df[
        df["category"].isin(set(filters.categories))
        & df["exchange"].isin({e.upper() for e in filters.exchanges})
    ]
    df["permaticker"] = pd.to_numeric(df["permaticker"], errors="coerce")
    df = df[df["permaticker"].notna()].copy()
    df["permaticker"] = df["permaticker"].astype("int64")
    df["lastupdated"] = pd.to_datetime(df["lastupdated"], errors="coerce")
    df = df.sort_values(["permaticker", "lastupdated"]).drop_duplicates("permaticker", keep="last")
    df["isdelisted"] = df["isdelisted"].astype(str).str.strip().str.upper().eq("Y")
    for c in ("firstpricedate", "lastpricedate"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["siccode"] = pd.to_numeric(df["siccode"], errors="coerce")
    lab, src = map_famaindustry(df["famaindustry"], df["siccode"])
    df["industry_ff48"], df["industry_source"] = lab, src
    n_holders = df.groupby("ticker")["permaticker"].transform("nunique")
    df["id"] = np.where(n_holders > 1, df["ticker"] + "." + df["permaticker"].astype(str), df["ticker"])
    df = df.sort_values(["ticker", "firstpricedate", "permaticker"]).reset_index(drop=True)
    # resolution ranges: unique holders are unrestricted; reused tickers by price-date range
    start = df["firstpricedate"].fillna(_OPEN_START)
    end = df["lastpricedate"].where(df["isdelisted"], pd.NaT).fillna(_OPEN_END)
    uniq = df.groupby("ticker")["permaticker"].transform("nunique") <= 1
    ranges = pd.DataFrame(
        {
            "ticker": df["ticker"],
            "permaticker": df["permaticker"],
            "id": df["id"],
            "start": start.where(~uniq, _OPEN_START),
            "end": end.where(~uniq, _OPEN_END),
        }
    ).reset_index(drop=True)
    cols = [
        "id",
        "permaticker",
        "ticker",
        "name",
        "exchange",
        "category",
        "famaindustry",
        "industry_ff48",
        "industry_source",
        "siccode",
        "sicsector",
        "sicindustry",
        "firstpricedate",
        "lastpricedate",
        "isdelisted",
        "cusips",
        "location",
        "scalemarketcap",
        "relatedtickers",
        "currency",
        "firstquarter",
        "lastquarter",
        "lastupdated",
    ]
    return SecurityMaster(frame=df[cols].copy(), ranges=ranges, filters=filters, n_input_rows=n_in)


def master_from_store(store: ParquetStore) -> SecurityMaster:
    """Rebuild a :class:`SecurityMaster` from the stored ``security_master`` table."""
    if not store.exists("security_master"):
        raise SharadarError("store has no security_master: sync TICKERS first")
    f = store.read_long("security_master")
    f["isdelisted"] = f["isdelisted"].astype(bool)
    for c in ("firstpricedate", "lastpricedate"):
        f[c] = pd.to_datetime(f[c])
    uniq = f.groupby("ticker")["permaticker"].transform("nunique") <= 1
    start = f["firstpricedate"].fillna(_OPEN_START).where(~uniq, _OPEN_START)
    end = f["lastpricedate"].where(f["isdelisted"], pd.NaT).fillna(_OPEN_END).where(~uniq, _OPEN_END)
    ranges = pd.DataFrame(
        {"ticker": f["ticker"], "permaticker": f["permaticker"], "id": f["id"], "start": start, "end": end}
    ).reset_index(drop=True)
    return SecurityMaster(frame=f, ranges=ranges)


@dataclass
class ResolveCounts:
    n_rows: int = 0
    n_kept: int = 0
    n_not_in_master: int = 0
    n_out_of_range: int = 0
    n_overlap_resolved: int = 0
    n_gap_assigned: int = 0

    def add(self, other: ResolveCounts) -> None:
        for fl in dataclasses.fields(self):
            setattr(self, fl.name, getattr(self, fl.name) + getattr(other, fl.name))


def resolve_ids(
    df: pd.DataFrame,
    ranges: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    gap_policy: str = "drop",
) -> tuple[pd.DataFrame, ResolveCounts]:
    """Attach the panel ``id`` to every row of ``df`` by ``(ticker, date_col)`` (pure).

    Rows whose ticker is not in the kept master are dropped and counted.  For a
    reused ticker a row is matched to the holder whose ``[firstpricedate,
    lastpricedate]`` contains the date; if two holders overlap the later-starting one
    wins (``n_overlap_resolved``).  A date in no holder's range (a gap between two
    holders, or before the first) is dropped with ``gap_policy="drop"`` (prices: a
    print while nobody held the ticker is a data error) or assigned to the **next**
    holder with ``gap_policy="next"`` (filings: a company files before its first
    price date; counted in ``n_gap_assigned``)."""
    if gap_policy not in ("drop", "next"):
        raise ValueError("gap_policy must be 'drop' or 'next'")
    c = ResolveCounts(n_rows=len(df))
    if df.empty:
        return df.assign(id=pd.Series(dtype=str)), c
    work = df.copy()
    work["_row"] = np.arange(len(work))
    work[ticker_col] = work[ticker_col].astype(str).str.strip().str.upper()
    r = ranges[["ticker", "id", "start", "end"]].rename(columns={"ticker": ticker_col})
    m = work.merge(r, on=ticker_col, how="left")
    in_master = m["id"].notna()
    d = _ns(m[date_col])
    ok = in_master & (d >= m["start"]) & (d <= m["end"])
    kept = m[ok]
    if kept.duplicated("_row").any():
        before = len(kept)
        kept = kept.sort_values(["_row", "start"]).drop_duplicates("_row", keep="last")
        c.n_overlap_resolved = int(before - len(kept))
    if gap_policy == "next":
        unmatched = in_master & ~m["_row"].isin(kept["_row"])
        cand = (
            m[unmatched & (m["start"] > d)]
            .sort_values(["_row", "start"])
            .drop_duplicates("_row", keep="first")
        )
        if len(cand):
            kept = pd.concat([kept, cand], ignore_index=True)
            c.n_gap_assigned = len(cand)
    n_in_master = int(np.unique(m.loc[in_master, "_row"].to_numpy()).size)
    c.n_kept = len(kept)
    c.n_not_in_master = c.n_rows - n_in_master
    c.n_out_of_range = n_in_master - c.n_kept
    out = kept.sort_values("_row").drop(columns=["start", "end", "_row"]).reset_index(drop=True)
    out["id"] = out["id"].astype(str)
    return out, c


# =============================================================================
# Transforms (pure, chunk-friendly)
# =============================================================================
@dataclass
class SepCounts(ResolveCounts):
    n_missing_closeadj: int = 0
    n_missing_closeunadj: int = 0  # rows keeping the vendor (split-adjusted) volume


def transform_sep_chunk(chunk: pd.DataFrame, ranges: pd.DataFrame) -> tuple[pd.DataFrame, SepCounts]:
    """SEP rows -> ``date, ticker(id), close, close_raw, volume`` (see module doc).

    ``close = closeadj``; ``close_raw = closeunadj`` (NaN when missing or <= 0);
    ``volume = volume * close_sa / closeunadj`` where ``close_sa`` is SEP's
    split-adjusted ``close``; rows without ``closeadj`` are dropped."""
    df = chunk.copy()
    df.columns = [str(c).lower() for c in df.columns]
    for c in ("close", "volume", "closeadj", "closeunadj"):
        if c not in df.columns:
            raise SharadarError(f"SEP chunk lacks column {c!r}; got {list(df.columns)}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]
    res, rc = resolve_ids(df, ranges, "date")
    counts = SepCounts(**dataclasses.asdict(rc))
    miss_adj = res["closeadj"].isna()
    counts.n_missing_closeadj = int(miss_adj.sum())
    res = res[~miss_adj]
    unadj = res["closeunadj"].where(res["closeunadj"] > 0)
    counts.n_missing_closeunadj = int(unadj.isna().sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = res["close"] / unadj
    vol = res["volume"] * ratio
    vol = vol.where(ratio.notna(), res["volume"])
    out = pd.DataFrame(
        {
            "date": res["date"].to_numpy(),
            "ticker": res["id"].to_numpy(),
            "close": res["closeadj"].to_numpy(dtype=float),
            "close_raw": unadj.to_numpy(dtype=float),
            "volume": vol.to_numpy(dtype=float),
        }
    )
    counts.n_kept = len(out)
    return out, counts


def transform_daily_chunk(chunk: pd.DataFrame, ranges: pd.DataFrame) -> tuple[pd.DataFrame, ResolveCounts]:
    """DAILY rows -> ``date, ticker(id), marketcap, ev`` (USD) ``, pb, pe, ps``."""
    df = chunk.copy()
    df.columns = [str(c).lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]
    res, c = resolve_ids(df, ranges, "date")
    out = pd.DataFrame({"date": res["date"].to_numpy(), "ticker": res["id"].to_numpy()})
    for col in ("marketcap", "ev", "pb", "pe", "ps"):
        v = (
            pd.to_numeric(res[col], errors="coerce")
            if col in res.columns
            else pd.Series(np.nan, index=res.index)
        )
        if col in ("marketcap", "ev"):
            v = v * DAILY_USD_SCALE
        out[col] = v.to_numpy(dtype=float)
    out["marketcap"] = out["marketcap"].where(out["marketcap"] > 0)
    return out, c


@dataclass
class Sf1Counts(ResolveCounts):
    n_other_dimension: int = 0  # rows of dimensions not kept (MR*, ARY by default)


def transform_sf1(
    df: pd.DataFrame,
    ranges: pd.DataFrame,
    dimensions: Sequence[str] = SF1_DIMENSIONS,
    fields: Sequence[str] = SF1_FIELDS,
) -> tuple[pd.DataFrame, Sf1Counts]:
    """SF1 rows -> long table ``date(=datekey), ticker(id), dimension, calendardate,
    reportperiod, <fields>`` for the kept dimensions, resolved by ``datekey``."""
    d = df.copy()
    d.columns = [str(c).lower() for c in d.columns]
    need = {"ticker", "dimension", "calendardate", "datekey"}
    if not need <= set(d.columns):
        raise SharadarError(f"SF1 needs columns {sorted(need)}")
    n_in = len(d)
    d["dimension"] = d["dimension"].astype(str).str.upper()
    d = d[d["dimension"].isin({x.upper() for x in dimensions})]
    n_dim = n_in - len(d)
    for c in ("datekey", "calendardate", "reportperiod"):
        d[c] = pd.to_datetime(d[c], errors="coerce") if c in d.columns else pd.NaT
    d = d[d["datekey"].notna() & d["calendardate"].notna()]
    res, rc = resolve_ids(d, ranges, "datekey", gap_policy="next")
    c = Sf1Counts(**dataclasses.asdict(rc))
    c.n_rows, c.n_other_dimension = n_in, n_dim
    out = pd.DataFrame(
        {
            "date": res["datekey"].to_numpy(),
            "ticker": res["id"].to_numpy(),
            "dimension": res["dimension"].to_numpy(),
            "calendardate": res["calendardate"].to_numpy(),
            "reportperiod": res["reportperiod"].to_numpy(),
        }
    )
    for f in fields:
        if f in res.columns:
            out[f] = pd.to_numeric(res[f], errors="coerce").to_numpy(dtype=float)
    return out, c


def transform_events(
    df: pd.DataFrame, ranges: pd.DataFrame, value_cols: Sequence[str]
) -> tuple[pd.DataFrame, ResolveCounts]:
    """ACTIONS / SP500 rows -> ``date, action, ticker, id, <value_cols>``; rows whose
    ticker is outside the kept master keep ``id = NaN`` (reference tables keep every
    row; the resolve counts report how many are unresolved)."""
    d = df.copy()
    d.columns = [str(c).lower() for c in d.columns]
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d[d["date"].notna()].reset_index(drop=True)
    d["ticker"] = d["ticker"].astype(str).str.strip().str.upper()
    d["_k"] = np.arange(len(d))
    res, c = resolve_ids(d, ranges, "date")
    ids = pd.Series(res["id"].to_numpy(), index=res["_k"].to_numpy()).reindex(d["_k"].to_numpy())
    out = pd.DataFrame(
        {
            "date": d["date"].to_numpy(),
            "action": d["action"].astype(str).str.strip().str.lower().to_numpy(),
            "ticker": d["ticker"].to_numpy(),
            "id": ids.to_numpy(),
        }
    )
    for col in value_cols:
        out[col] = d[col].to_numpy() if col in d.columns else np.nan
    return out, c


@dataclass
class DelistingReasonReport:
    """Counts of :func:`derive_delisting_reasons` (printed by ``qts data sharadar derive``)."""

    n_delisted: int = 0
    window_days: int = DELISTING_REASON_WINDOW_DAYS
    by_reason: dict[str, int] = field(default_factory=dict)
    by_year: dict[int, dict[str, int]] = field(default_factory=dict)
    #: action codes seen in the window of *any* delisted id (ids counted once per code)
    actions_near_delistings: dict[str, int] = field(default_factory=dict)
    #: for ids left unknown: the action codes seen in the window ("(no action row)" = none)
    unknown_actions: dict[str, int] = field(default_factory=dict)
    #: for ids left unknown: mapped codes found in the wider hint window
    unknown_mapped_in_hint_window: dict[str, int] = field(default_factory=dict)
    mapping: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def format(self, top: int = 15) -> str:
        n = max(self.n_delisted, 1)
        lines = [
            f"Delisting reasons: {self.n_delisted} delisted ids; ACTIONS rows within "
            f"+-{self.window_days} business days of the delisting date",
            "  mapping: " + ", ".join(f"{k}->{v}" for k, v in self.mapping.items()),
            "",
            "  by reason",
        ]
        for r in DELISTING_REASONS:
            c = self.by_reason.get(r, 0)
            lines.append(f"    {r:12s} {c:>7,d}  {100 * c / n:5.1f}%")
        lines += ["", "  by year      " + "".join(f"{r:>13s}" for r in DELISTING_REASONS) + f"{'total':>9s}"]
        for y in sorted(self.by_year):
            row = self.by_year[y]
            lines.append(
                f"    {y:<11d}"
                + "".join(f"{row.get(r, 0):>13,d}" for r in DELISTING_REASONS)
                + f"{sum(row.values()):>9,d}"
            )
        lines += ["", "  action codes within the window of a delisting (all delisted ids; ids per code)"]
        for k, v in list(self.actions_near_delistings.items())[:top]:
            mapped = self.mapping.get(k)
            lines.append(f"    {k:24s} {v:>7,d}" + (f"  -> {mapped}" if mapped else ""))
        lines += ["", "  unknown ids: action codes within the window (top)"]
        for k, v in list(self.unknown_actions.items())[:top]:
            lines.append(f"    {k:24s} {v:>7,d}")
        if self.unknown_mapped_in_hint_window:
            lines += [
                "",
                f"  unknown ids with a mapped action within +-{DELISTING_REASON_HINT_WINDOW_DAYS} business days "
                "(outside the window; a large count means the window is too tight)",
            ]
            for k, v in list(self.unknown_mapped_in_hint_window.items())[:top]:
                lines.append(f"    {k:24s} {v:>7,d}")
        return "\n".join(lines)


def _busdays_between(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Signed weekday count from ``a`` to ``b`` (``b`` after ``a`` -> positive)."""
    a_d = pd.to_datetime(a).to_numpy().astype("datetime64[D]")
    b_d = pd.to_datetime(b).to_numpy().astype("datetime64[D]")
    return np.busday_count(a_d, b_d)


def _sorted_counts(s: pd.Series) -> dict[str, int]:
    vc = s.value_counts()
    order = sorted(vc.index, key=lambda k: (-int(vc[k]), str(k)))
    return {str(k): int(vc[k]) for k in order}


def derive_delisting_reasons(
    actions: pd.DataFrame,
    master: pd.DataFrame,
    window_days: int = DELISTING_REASON_WINDOW_DAYS,
    mapping: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, DelistingReasonReport]:
    """``corporate_actions`` + ``security_master`` -> ``delisting_reasons`` (pure).

    ``actions``: ``date, action, ticker`` (+ optional ``id``: when resolved, the
    row only counts for that id -- reused tickers).  ``master``: ``id, ticker,
    lastpricedate, isdelisted``.  Returns the long table
    (:data:`DELISTING_REASON_COLUMNS`, one row per delisted id, sorted by id) and
    the report.  Rule, mapping and precedence: module docstring "Delisting reasons".
    """
    mp = dict(DELISTING_REASON_ACTIONS if mapping is None else mapping)
    bad = sorted(set(mp.values()) - set(DELISTING_REASONS))
    if bad:
        raise ValueError(f"mapping targets {bad} are not delisting reasons {DELISTING_REASONS}")
    m = master.loc[master["isdelisted"].astype(bool), ["id", "ticker", "lastpricedate"]].copy()
    m["id"] = m["id"].astype(str)
    m["ticker"] = m["ticker"].astype(str).str.strip().str.upper()
    m["delisting_date"] = pd.to_datetime(m["lastpricedate"])
    m = m[m["delisting_date"].notna()].drop(columns="lastpricedate").drop_duplicates("id")
    rep = DelistingReasonReport(n_delisted=len(m), window_days=int(window_days), mapping=mp)
    a = pd.DataFrame(
        {
            "date": pd.to_datetime(actions["date"], errors="coerce").to_numpy(),
            "action": actions["action"].astype(str).str.strip().str.lower().to_numpy(),
            "ticker": actions["ticker"].astype(str).str.strip().str.upper().to_numpy(),
            "action_id": (
                actions["id"].to_numpy() if "id" in actions.columns else np.full(len(actions), None, dtype=object)
            ),
        }
    )
    a = a[a["date"].notna()]
    j = m.merge(a, on="ticker", how="inner")
    # a row resolved to another holder of a reused ticker does not belong to this id
    other = j["action_id"].notna() & (j["action_id"].astype(str) != j["id"])
    j = j[~other]
    j["gap"] = _busdays_between(j["delisting_date"], j["date"]) if len(j) else np.zeros(0, dtype=int)
    near = j[j["gap"].abs() <= int(window_days)].copy()
    near["reason"] = near["action"].map(mp)
    rank = {r: k for k, r in enumerate(REASON_PRECEDENCE)}
    hit = near[near["reason"].notna()].copy()
    hit["rank"] = hit["reason"].map(rank)
    # strongest reason per id; ties: the action closest to the delisting date, then by code
    hit["absgap"] = hit["gap"].abs()
    hit = hit.sort_values(["id", "rank", "absgap", "action"], kind="mergesort").drop_duplicates("id")
    out = m[["id", "delisting_date"]].merge(
        hit[["id", "reason", "action"]].rename(columns={"action": "source_action"}), on="id", how="left"
    )
    plain = set(near.loc[near["action"] == "delisted", "id"])
    unknown = out["reason"].isna()
    out.loc[unknown, "source_action"] = np.where(out.loc[unknown, "id"].isin(plain), "delisted", "")
    out["reason"] = out["reason"].fillna(UNKNOWN_REASON).astype(str)
    out["source_action"] = out["source_action"].astype(str)
    out = out.sort_values("id", kind="mergesort").reset_index(drop=True)[list(DELISTING_REASON_COLUMNS)]

    rep.by_reason = {r: int((out["reason"] == r).sum()) for r in DELISTING_REASONS}
    yr = out["delisting_date"].dt.year
    for y, g in out.groupby(yr):
        rep.by_year[int(cast(Any, y))] = {r: int((g["reason"] == r).sum()) for r in DELISTING_REASONS}
    per_id = near.drop_duplicates(["id", "action"])
    rep.actions_near_delistings = _sorted_counts(per_id["action"])
    unk_ids = set(out.loc[out["reason"] == UNKNOWN_REASON, "id"])
    seen = per_id[per_id["id"].isin(unk_ids)]
    cnt = _sorted_counts(seen["action"])
    n_none = len(unk_ids - set(seen["id"]))
    if n_none:
        cnt["(no action row)"] = n_none
    rep.unknown_actions = dict(sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])))
    wide = j[
        j["id"].isin(unk_ids)
        & (j["gap"].abs() <= DELISTING_REASON_HINT_WINDOW_DAYS)
        & j["action"].isin(list(mp))
    ].drop_duplicates(["id", "action"])
    rep.unknown_mapped_in_hint_window = _sorted_counts(wide["action"])
    return out, rep


def derive_from_store(store_root: str, window_days: int = DELISTING_REASON_WINDOW_DAYS) -> DelistingReasonReport:
    """Rebuild the ``delisting_reasons`` dataset of an existing store from its
    ``corporate_actions`` and ``security_master`` (no download; idempotent)."""
    store = ParquetStore(store_root)
    missing = [n for n in ("corporate_actions", "security_master") if not store.exists(n)]
    if missing:
        raise SharadarError(
            f"store {store_root} has no {' / '.join(missing)}: run `qts data sharadar sync` "
            "with TICKERS and ACTIONS first"
        )
    master = store.read_long("security_master")
    out, rep = derive_delisting_reasons(store.read_long("corporate_actions"), master, window_days)
    store.write_long("delisting_reasons", out, mode="replace", key=["id"])
    return rep


def derive_caps(
    close_raw: pd.DataFrame, daily: pd.DataFrame | None, sharesbas: pd.DataFrame | None
) -> pd.DataFrame:
    """``market_cap`` / ``shares_outstanding`` rows for one slice of dates (pure).

    ``close_raw``: ``date, ticker, close_raw``; ``daily``: ``date, ticker, marketcap``
    (USD); ``sharesbas``: ``datekey, ticker, sharesbas`` (SF1, any dimension).  Where
    DAILY has a cap: ``shares = cap / close_raw``; else ``shares`` = the latest
    ``sharesbas`` filed on or before the date and ``cap = close_raw * shares``.
    Returns ``date, ticker, market_cap, shares_outstanding, cap_source``."""
    base = close_raw[["date", "ticker", "close_raw"]].copy()
    base["date"] = _ns(base["date"])
    base["ticker"] = base["ticker"].astype(str)
    if daily is not None and not daily.empty:
        d = daily[["date", "ticker", "marketcap"]].drop_duplicates(["date", "ticker"], keep="last")
        d = d.assign(date=_ns(d["date"]), ticker=d["ticker"].astype(str))
        base = base.merge(d, on=["date", "ticker"], how="left")
    else:
        base["marketcap"] = np.nan
    base["shares_sf1"] = np.nan
    sb = None
    if sharesbas is not None and not sharesbas.empty:
        sb = sharesbas[["datekey", "ticker", "sharesbas"]].dropna()
        sb = sb[sb["sharesbas"] > 0].rename(columns={"datekey": "date", "sharesbas": "shares_sf1"})
        sb = sb.assign(date=_ns(sb["date"]), ticker=sb["ticker"].astype(str)).sort_values(["date", "ticker"])
        sb = sb.drop_duplicates(["date", "ticker"], keep="last")
    if sb is not None and not sb.empty:
        base = base.drop(columns=["shares_sf1"]).sort_values(["date", "ticker"])
        base = pd.merge_asof(base, sb, on="date", by="ticker", direction="backward", allow_exact_matches=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = base["close_raw"].where(base["close_raw"] > 0)
        has_daily = base["marketcap"].notna()
        shares_daily = base["marketcap"] / raw  # NaN where the as-traded close is missing
        shares = shares_daily.where(shares_daily.notna(), base["shares_sf1"])
        cap = base["marketcap"].where(has_daily, raw * base["shares_sf1"])
    src = np.where(has_daily, "daily", np.where(cap.notna(), "sf1", "none"))
    return pd.DataFrame(
        {
            "date": base["date"].to_numpy(),
            "ticker": base["ticker"].to_numpy(),
            "market_cap": cap.to_numpy(dtype=float),
            "shares_outstanding": shares.to_numpy(dtype=float),
            "cap_source": src,
        }
    )


def sp500_membership_panel(
    source: pd.DataFrame | ParquetStore,
    dates: pd.DatetimeIndex | None = None,
    ids: Iterable[str] | None = None,
    id_col: str = "id",
) -> pd.DataFrame:
    """Date x id boolean S&P 500 membership from the ``sp500_membership`` events.

    ``added`` makes an id a member from that date on, ``removed`` ends membership
    from that date on (member on ``[added, removed)``); ``current`` / ``historical``
    rows are listings, not events, and are ignored here (:func:`validate` compares
    the ``current`` list with the panel's last row).  ``source`` is the events frame
    or a store (reads ``sp500_membership``; ``dates`` default to the ``prices_close``
    dates).  Events before the first date are carried in; rows without an id (ticker
    outside the kept master) fall back to the raw ticker.  Not a universe rule."""
    if isinstance(source, ParquetStore):
        ev = source.read_long("sp500_membership")
        if dates is None:
            dates = pd.DatetimeIndex(
                source.duckdb_query("SELECT DISTINCT date FROM prices_close ORDER BY date")["date"]
            )
    else:
        ev = source.copy()
    if dates is None:
        raise ValueError("dates are required when source is a DataFrame")
    dates = pd.DatetimeIndex(dates)
    ev["date"] = pd.to_datetime(ev["date"])
    key = ev[id_col] if id_col in ev.columns else ev["ticker"]
    ev = ev.assign(ticker=key.where(key.notna(), ev["ticker"]).astype(str))
    ev = ev[ev["action"].isin(["added", "removed"])].copy()
    ev["value"] = (ev["action"] == "added").astype(float)
    ev = ev.sort_values(["ticker", "date", "value"])  # a same-day remove+add resolves to "added"
    if ev.empty:
        return pd.DataFrame(False, index=dates, columns=pd.Index(list(ids or []), name="ticker"))
    # events before the panel start: carry the latest state into the first date
    ev["available_date"] = ev["date"].where(ev["date"] >= dates[0], dates[0])
    panel = pit_forward_fill(ev, dates).fillna(0.0) > 0.5
    if ids is not None:
        panel = panel.reindex(columns=list(ids), fill_value=False)
    panel.columns.name = "ticker"
    return panel


def build_industry_history(
    master: SecurityMaster, dates: pd.DatetimeIndex
) -> pd.DataFrame:  # pragma: no cover
    """TODO hook: dated industry labels (store dataset ``industry_history``).

    TICKERS carries only the *current* ``famaindustry`` / ``siccode``, so the bundle
    cannot date reclassifications and this loader writes no ``industry_history`` (the
    static ``industry`` map applies everywhere; FF48 reclassifications are rare, a mild
    look-ahead).  A source of dated SIC codes (EDGAR filing headers carry the SIC at
    each filing date) would plug in here: return a date x id panel of FF48 labels
    with one row per change and write it with ``store.write_wide("industry_history",
    panel.astype(object))``."""
    raise NotImplementedError("no dated industry source in the Sharadar bundle; see docstring")


# =============================================================================
# Staging: per-year spill files so nothing is pivoted whole
# =============================================================================
class _YearSpill:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._seq = 0

    def add(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df = df.assign(date=_ns(df["date"]))
        years = df["date"].dt.year
        for y, part in df.groupby(years):
            d = self.root / f"year={int(y)}"
            d.mkdir(exist_ok=True)
            self._seq += 1
            pq.write_table(
                pa.Table.from_pandas(part.reset_index(drop=True), preserve_index=False),
                d / f"part-{self._seq:06d}.parquet",
                compression="zstd",
            )

    def years(self) -> list[int]:
        return sorted(int(p.name.split("=", 1)[1]) for p in self.root.glob("year=*") if p.is_dir())

    def read_year(self, year: int) -> pd.DataFrame:
        files = sorted((self.root / f"year={year}").glob("part-*.parquet"))
        if not files:
            return pd.DataFrame()
        df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
        # later files win (chunk order = file order = vendor delivery order)
        return df.drop_duplicates(["date", "ticker"], keep="last").reset_index(drop=True)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def _write_panel(store: ParquetStore, name: str, df: pd.DataFrame, col: str, mode: str) -> int:
    part = df[["date", "ticker", col]].rename(columns={col: "value"})
    part = part[part["value"].notna()]
    if part.empty:
        return 0
    store.write_long(name, part, mode=mode, kind="wide")
    return len(part)


# =============================================================================
# Sync orchestration
# =============================================================================
@dataclass
class TableStats:
    table: str
    sources: list[str] = field(default_factory=list)
    rows_read: int = 0
    rows_kept: int = 0
    dropped_not_in_master: int = 0
    dropped_out_of_range: int = 0
    dropped_other: dict[str, int] = field(default_factory=dict)
    date_min: str | None = None
    date_max: str | None = None
    datasets: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    skipped: bool = False
    note: str = ""

    def observe_dates(self, s: pd.Series) -> None:
        if s.empty:
            return
        lo, hi = pd.Timestamp(s.min()), pd.Timestamp(s.max())
        self.date_min = str(min(pd.Timestamp(self.date_min or hi), lo).date())
        self.date_max = str(max(pd.Timestamp(self.date_max or lo), hi).date())


@dataclass
class SyncReport:
    store_root: str
    raw_dir: str
    mode: str  # "full" | "incremental"
    since: str | None
    tables: dict[str, TableStats] = field(default_factory=dict)
    master: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    dry_run: bool = False
    seconds: float = 0.0
    delisting_reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

    def format(self) -> str:
        lines = [
            f"Sharadar sync -> {self.store_root}  ({self.mode}{', since ' + self.since if self.since else ''}"
            f"{', DRY RUN' if self.dry_run else ''}; {self.seconds:.0f}s)"
        ]
        if self.master:
            m = self.master
            lines.append(
                f"  master: {m.get('n_kept')} kept of {m.get('n_tickers_rows')} TICKERS rows; "
                f"{m.get('n_ids')} ids, {m.get('n_reused_tickers')} reused tickers, "
                f"{m.get('n_delisted')} delisted; industry source {m.get('industry_source')}"
            )
        for t in self.tables.values():
            if t.skipped:
                lines.append(f"  {t.table:8s} skipped ({t.note})")
                continue
            if self.dry_run:
                lines.append(f"  {t.table:8s} {t.note}")
                for src in t.sources:
                    lines.append(
                        f"           source: {src}{'' if os.path.exists(src) else '  (to download)'}"
                    )
                continue
            drop = f"not-in-master {t.dropped_not_in_master}, out-of-range {t.dropped_out_of_range}"
            if t.dropped_other:
                drop += ", " + ", ".join(f"{k} {v}" for k, v in t.dropped_other.items())
            lines.append(
                f"  {t.table:8s} read {t.rows_read:>11,d}  kept {t.rows_kept:>11,d}  [{drop}]  "
                f"{t.date_min or '-'}..{t.date_max or '-'}  {t.seconds:.0f}s"
            )
            for src in t.sources:
                lines.append(f"           source: {src}")
            if t.note:
                lines.append(f"           note: {t.note}")
            for n, r in t.datasets.items():
                lines.append(f"           -> {n} ({r:,d} rows)")
        if self.delisting_reasons:
            lines.append(
                "  delisting_reasons: "
                + ", ".join(f"{k} {v}" for k, v in self.delisting_reasons.items())
                + "  (full table: qts data sharadar derive --store <store>)"
            )
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def _state_path(store_root: str) -> Path:
    return Path(store_root) / _STATE_DIR / "state.json"


def _load_state(store_root: str) -> dict[str, Any]:
    p = _state_path(store_root)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except ValueError:
            return {}
    return {}


def _save_state(store_root: str, state: dict[str, Any]) -> None:
    p = _state_path(store_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True, default=str))
    os.replace(tmp, p)


def _fingerprint(paths: Sequence[Path]) -> list[list[Any]]:
    return [[str(p), int(p.stat().st_size), int(p.stat().st_mtime)] for p in paths]


class _Sync:
    """One ``sync`` invocation (state kept on the instance; :func:`sync` is the API)."""

    def __init__(
        self,
        store_root: str,
        raw_dir: str,
        tables: Sequence[str],
        since: str | None,
        from_files: str | None,
        api_key: str | None,
        master_filters: MasterFilters,
        chunksize: int,
        dry_run: bool,
        force: bool,
        client: NasdaqDataLinkClient | None,
        fundamentals_batch: int,
        keep_staging: bool,
    ):
        self.store_root, self.raw_dir = str(store_root), str(raw_dir)
        self.tables = [t.upper() for t in tables]
        unknown = [t for t in self.tables if t not in KNOWN_TABLES]
        if unknown:
            raise SharadarError(f"unknown tables {unknown}; known: {KNOWN_TABLES}")
        self.since = str(pd.Timestamp(since).date()) if since else None
        self.mode = "incremental" if self.since else "full"
        self.from_files = from_files
        self.api_key = api_key
        self.filters = master_filters
        self.chunksize = int(chunksize)
        self.dry_run, self.force = dry_run, force
        self.fund_batch = int(fundamentals_batch)
        self.keep_staging = keep_staging
        self.store = ParquetStore(self.store_root)
        self.report = SyncReport(self.store_root, self.raw_dir, self.mode, self.since, dry_run=dry_run)
        self.state: dict[str, Any] = {} if force else _load_state(self.store_root)
        self.master: SecurityMaster | None = None
        self.touched_years: set[int] = set()
        self.changed = False  # any raw table (re)written this run -> rebuild the derived datasets
        self.source_map: dict[str, list[Path]] | None = None  # --from-files: table -> files
        self.staging_root = Path(self.store_root) / _STATE_DIR / "staging"
        self._client_obj: NasdaqDataLinkClient | None = client

    # -- sources ----------------------------------------------------------------
    def _client(self) -> NasdaqDataLinkClient:
        if self._client_obj is None:
            key = load_api_key(self.api_key)
            if not key:
                raise SharadarDownloadError(
                    f"no API key: set {API_KEY_ENV} in the environment or a .env file, pass --api-key, "
                    "or use --from-files DIR with manually exported files"
                )
            self._client_obj = NasdaqDataLinkClient(key)
        return self._client_obj

    def sources(self, table: str) -> list[Path]:
        """Raw files for ``table``: from ``from_files`` or the raw cache (downloading
        when today's file is absent)."""
        if self.from_files:
            if self.source_map is None:
                self.source_map = resolve_sources(self.from_files, self.tables)
                log.info("%s", format_source_map(self.source_map))
            return list(self.source_map.get(table, []))
        since = self.since if table in INCREMENTAL_TABLES else None
        dest = raw_cache_path(self.raw_dir, table, since)
        if dest.exists():
            log.info("SHARADAR/%s: using cached %s", table, dest)
            return [dest]
        if self.dry_run:
            return [dest]
        filters = {"lastupdated.gte": since} if since else {}
        self._client().export_table(table, dest, filters)
        return [dest]

    def _skip(self, step: str, inputs: Sequence[Path]) -> bool:
        if self.force or self.dry_run:
            return False
        st = self.state.get(step)
        return bool(
            st
            and st.get("done")
            and st.get("inputs") == _fingerprint(inputs)
            and st.get("mode") == self.mode
            and st.get("since") == self.since
        )

    def _done(self, step: str, inputs: Sequence[Path]) -> None:
        self.changed = True
        self.state[step] = {
            "done": True,
            "inputs": _fingerprint(inputs),
            "mode": self.mode,
            "since": self.since,
            "finished_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        }
        _save_state(self.store_root, self.state)

    def _stats(self, table: str) -> TableStats:
        return self.report.tables.setdefault(table, TableStats(table))

    @property
    def panel_mode(self) -> str:
        return "upsert" if self.mode == "incremental" else "replace_partition"

    def _reset(self, *names: str) -> None:
        """Full load: a dataset is rebuilt from scratch (stale years must not survive)."""
        if self.mode == "full":
            for n in names:
                if self.store.exists(n):
                    self.store.drop(n)

    # -- steps ------------------------------------------------------------------
    def run(self) -> SyncReport:
        t0 = time.perf_counter()
        order = [t for t in KNOWN_TABLES if t in self.tables]
        if "TICKERS" in order:
            self.step_tickers()
        else:
            try:
                self.master = master_from_store(self.store)
                self.report.master = {
                    "note": "master read from store (TICKERS not in --tables)",
                    **self.master.summary(),
                }
            except SharadarError:
                if not self.dry_run:
                    raise
                self.report.warnings.append("no security_master in the store yet: add TICKERS to --tables")
        for t in order:
            if t == "TICKERS":
                continue
            if self.from_files and not self.sources(t):
                self._stats(t).skipped, self._stats(t).note = True, f"no file under {self.from_files}"
                self.report.warnings.append(f"{t}: no matching file under {self.from_files}; table skipped")
                continue
            fn = getattr(self, f"step_{t.lower()}", None)
            if fn is None:
                self.step_generic(t)
            else:
                fn()
        if not self.dry_run:
            derived_missing = not all(
                self.store.exists(n) for n in ("market_cap", "shares_outstanding", *FUNDAMENTAL_PANELS)
            )
            if ({"SEP", "DAILY", "SF1"} & set(order)) and (self.changed or self.force or derived_missing):
                self.step_caps()
                self.step_fundamentals()
            elif {"SEP", "DAILY", "SF1"} & set(order):
                for n in ("derived:caps", "derived:fundamentals"):
                    self._stats(n).skipped, self._stats(n).note = True, "no raw table changed"
            if self.store.exists("corporate_actions") and self.store.exists("security_master"):
                # cheap, deterministic, always rebuilt (also when ACTIONS was skipped as unchanged)
                out, dr = derive_delisting_reasons(
                    self.store.read_long("corporate_actions"), self.store.read_long("security_master")
                )
                self.store.write_long("delisting_reasons", out, mode="replace", key=["id"])
                self.report.delisting_reasons = dict(dr.by_reason)
            if not self.keep_staging:
                shutil.rmtree(self.staging_root, ignore_errors=True)
        self.report.seconds = time.perf_counter() - t0
        if not self.dry_run:
            p = Path(self.store_root) / _STATE_DIR / "last_sync.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self.report.to_dict(), indent=1, default=str))
        return self.report

    def step_tickers(self) -> None:
        st, t0 = self._stats("TICKERS"), time.perf_counter()
        files = self.sources("TICKERS")
        st.sources = [str(p) for p in files]
        if self.dry_run:
            st.note = "would build security_master, industry, delisting_dates"
            return
        raw = pd.concat(
            [read_csv_table(p, dtype={"ticker": str, "cusips": str}) for p in files], ignore_index=True
        )
        st.rows_read = len(raw)
        self.master = build_security_master(raw, self.filters)
        m = self.master
        st.rows_kept = len(m.frame)
        self.report.master = m.summary()
        st.observe_dates(m.frame["firstpricedate"].dropna())
        st.observe_dates(m.frame["lastpricedate"].dropna())
        if not self._skip("TICKERS", files):
            sm = m.frame.copy()
            sm["isdelisted"] = sm["isdelisted"].astype(bool)
            self.store.write_long("security_master", sm, mode="replace", key=["id"])
            self.store.write_long("industry", m.industry_table(), mode="replace")
            self.store.write_long("delisting_dates", m.delisting_table(), mode="replace")
            st.datasets = {"security_master": len(sm), "industry": len(sm), "delisting_dates": len(sm)}
            self._done("TICKERS", files)
        else:
            st.note = "unchanged (state)"
        if m.summary()["n_other_industry"] > 0.1 * max(1, len(m.frame)):
            self.report.warnings.append(
                "more than 10% of names map to industry 'Other'; check famaindustry spellings"
            )
        st.seconds = time.perf_counter() - t0

    def _stream(
        self,
        table: str,
        files: Sequence[Path],
        usecols: Sequence[str],
        dtype: Mapping[str, Any],
        transform: Any,
        spill: _YearSpill,
        st: TableStats,
    ) -> ResolveCounts:
        assert self.master is not None
        total = ResolveCounts()
        extra: dict[str, int] = {}
        for p in files:
            for chunk in iter_csv_chunks(p, self.chunksize, usecols=usecols, dtype=dtype):
                out, c = transform(chunk, self.master.ranges)
                total.add(c)
                for fl in dataclasses.fields(c):
                    if fl.name not in ResolveCounts.__dataclass_fields__:
                        extra[fl.name] = extra.get(fl.name, 0) + int(getattr(c, fl.name))
                if not out.empty:
                    st.observe_dates(out["date"])
                    self.touched_years.update(int(y) for y in pd.to_datetime(out["date"]).dt.year.unique())
                    spill.add(out)
                log.info(
                    "SHARADAR/%s: %s rows read, %s kept", table, f"{total.n_rows:,}", f"{total.n_kept:,}"
                )
        st.rows_read, st.rows_kept = total.n_rows, total.n_kept
        st.dropped_not_in_master, st.dropped_out_of_range = total.n_not_in_master, total.n_out_of_range
        st.dropped_other = {k: v for k, v in extra.items() if v}
        if total.n_overlap_resolved:
            st.dropped_other["overlap_resolved"] = total.n_overlap_resolved
        if total.n_gap_assigned:
            st.dropped_other["gap_assigned_to_next_holder"] = total.n_gap_assigned
        return total

    def step_sep(self) -> None:
        st, t0 = self._stats("SEP"), time.perf_counter()
        files = self.sources("SEP")
        st.sources = [str(p) for p in files]
        if self.dry_run:
            st.note = "would write prices_close, prices_close_raw, prices_volume"
            return
        if self._skip("SEP", files):
            st.skipped, st.note = True, "unchanged (state)"
            return
        spill = _YearSpill(self.staging_root / "sep")
        dtype = {
            "ticker": str,
            "close": "float64",
            "volume": "float64",
            "closeadj": "float64",
            "closeunadj": "float64",
        }
        self._stream(
            "SEP",
            files,
            ("ticker", "date", "close", "volume", "closeadj", "closeunadj"),
            dtype,
            transform_sep_chunk,
            spill,
            st,
        )
        self._reset("prices_close", "prices_close_raw", "prices_volume")
        names = {"prices_close": "close", "prices_close_raw": "close_raw", "prices_volume": "volume"}
        for y in spill.years():
            part = spill.read_year(y)
            for n, col in names.items():
                st.datasets[n] = st.datasets.get(n, 0) + _write_panel(
                    self.store, n, part, col, self.panel_mode
                )
            log.info("SEP year %d written (%s rows)", y, f"{len(part):,}")
        if not self.keep_staging:
            spill.cleanup()
        miss = st.dropped_other.get("n_missing_closeunadj", 0)
        if st.rows_kept and miss / st.rows_kept > 0.01:
            self.report.warnings.append(
                f"{miss / st.rows_kept:.1%} of SEP rows lack closeunadj (volume left split-adjusted there)"
            )
        self._done("SEP", files)
        st.seconds = time.perf_counter() - t0

    def step_daily(self) -> None:
        st, t0 = self._stats("DAILY"), time.perf_counter()
        files = self.sources("DAILY")
        st.sources = [str(p) for p in files]
        if self.dry_run:
            st.note = (
                "would write sharadar_marketcap, sharadar_ev/pb/pe/ps (then market_cap, shares_outstanding)"
            )
            return
        if self._skip("DAILY", files):
            st.skipped, st.note = True, "unchanged (state)"
            return
        spill = _YearSpill(self.staging_root / "daily")
        dtype = {
            "ticker": str,
            "marketcap": "float64",
            "ev": "float64",
            "pb": "float64",
            "pe": "float64",
            "ps": "float64",
        }
        self._stream(
            "DAILY",
            files,
            ("ticker", "date", "marketcap", "ev", "pb", "pe", "ps"),
            dtype,
            transform_daily_chunk,
            spill,
            st,
        )
        names = {"sharadar_marketcap": "marketcap", **{v: k for k, v in DAILY_PANELS.items()}}
        self._reset(*names)
        for y in spill.years():
            part = spill.read_year(y)
            for n, col in names.items():
                st.datasets[n] = st.datasets.get(n, 0) + _write_panel(
                    self.store, n, part, col, self.panel_mode
                )
        if not self.keep_staging:
            spill.cleanup()
        self._done("DAILY", files)
        st.seconds = time.perf_counter() - t0

    def step_sf1(self) -> None:
        st, t0 = self._stats("SF1"), time.perf_counter()
        files = self.sources("SF1")
        st.sources = [str(p) for p in files]
        if self.dry_run:
            st.note = "would write sharadar_sf1 (then book_to_price, earnings_yield, gross_profitability, asset_growth)"
            return
        if self._skip("SF1", files):
            st.skipped, st.note = True, "unchanged (state)"
            return
        assert self.master is not None
        parts: list[pd.DataFrame] = []
        total = Sf1Counts()
        usecols = SF1_KEYS + SF1_FIELDS
        for p in files:
            for chunk in iter_csv_chunks(
                p, self.chunksize, usecols=usecols, dtype={"ticker": str, "dimension": str}
            ):
                out, c = transform_sf1(chunk, self.master.ranges)
                total.add(c)
                if not out.empty:
                    parts.append(out)
        sf1 = (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame(columns=["date", "ticker", "dimension", "calendardate"])
        )
        sf1 = sf1.drop_duplicates(["date", "ticker", "dimension", "calendardate"], keep="last")
        st.rows_read, st.rows_kept = total.n_rows, len(sf1)
        st.dropped_not_in_master, st.dropped_out_of_range = total.n_not_in_master, total.n_out_of_range
        st.dropped_other = {"other_dimension": total.n_other_dimension}
        if total.n_gap_assigned:
            st.dropped_other["gap_assigned_to_next_holder"] = total.n_gap_assigned
        if not sf1.empty:
            st.observe_dates(sf1["date"])
            self.touched_years.update(int(y) for y in pd.to_datetime(sf1["date"]).dt.year.unique())
        self._reset("sharadar_sf1")
        if not sf1.empty:
            self.store.write_long(
                "sharadar_sf1",
                sf1,
                mode=self.panel_mode,
                kind="long",
                key=["date", "ticker", "dimension", "calendardate"],
            )
        st.datasets["sharadar_sf1"] = len(sf1)
        self._done("SF1", files)
        st.seconds = time.perf_counter() - t0

    def step_actions(self) -> None:
        st, t0 = self._stats("ACTIONS"), time.perf_counter()
        files = self.sources("ACTIONS")
        st.sources = [str(p) for p in files]
        if self.dry_run:
            st.note = "would write corporate_actions"
            return
        if self._skip("ACTIONS", files):
            st.skipped, st.note = True, "unchanged (state)"
            return
        assert self.master is not None
        raw = pd.concat(
            [read_csv_table(p, dtype={"ticker": str, "contraticker": str}) for p in files], ignore_index=True
        )
        out, c = transform_events(raw, self.master.ranges, ("name", "value", "contraticker", "contraname"))
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        st.rows_read, st.rows_kept = c.n_rows, len(out)
        st.dropped_not_in_master, st.dropped_out_of_range = c.n_not_in_master, c.n_out_of_range
        st.note = "unresolved rows are kept with id = NaN"
        st.observe_dates(out["date"])
        self.store.write_long(
            "corporate_actions", out, mode="replace", kind="long", key=["date", "ticker", "action"]
        )
        st.datasets["corporate_actions"] = len(out)
        self._done("ACTIONS", files)
        st.seconds = time.perf_counter() - t0

    def step_sp500(self) -> None:
        st, t0 = self._stats("SP500"), time.perf_counter()
        files = self.sources("SP500")
        st.sources = [str(p) for p in files]
        if self.dry_run:
            st.note = "would write sp500_membership"
            return
        if self._skip("SP500", files):
            st.skipped, st.note = True, "unchanged (state)"
            return
        assert self.master is not None
        raw = pd.concat(
            [read_csv_table(p, dtype={"ticker": str, "contraticker": str}) for p in files], ignore_index=True
        )
        out, c = transform_events(raw, self.master.ranges, ("name", "contraticker", "contraname", "note"))
        st.rows_read, st.rows_kept = c.n_rows, len(out)
        st.dropped_not_in_master, st.dropped_out_of_range = c.n_not_in_master, c.n_out_of_range
        st.note = "unresolved rows are kept with id = NaN; membership is stored, not a universe rule"
        st.observe_dates(out["date"])
        self.store.write_long(
            "sp500_membership", out, mode="replace", kind="long", key=["date", "ticker", "action"]
        )
        st.datasets["sp500_membership"] = len(out)
        self._done("SP500", files)
        st.seconds = time.perf_counter() - t0

    def step_generic(self, table: str) -> None:
        """Optional tables stored verbatim as ``sharadar_<table>`` (ids attached when
        the table has ``ticker`` and ``date`` columns)."""
        st, t0 = self._stats(table), time.perf_counter()
        files = self.sources(table)
        st.sources = [str(p) for p in files]
        name = f"sharadar_{table.lower()}"
        if self.dry_run:
            st.note = f"would write {name} (verbatim)"
            return
        if self._skip(table, files):
            st.skipped, st.note = True, "unchanged (state)"
            return
        assert self.master is not None
        self._reset(name)
        first = True
        for p in files:
            for chunk in iter_csv_chunks(p, self.chunksize, dtype={"ticker": str}):
                st.rows_read += len(chunk)
                if {"ticker", "date"} <= set(chunk.columns):
                    chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
                    chunk = chunk[chunk["date"].notna()]
                    res, c = resolve_ids(chunk, self.master.ranges, "date")
                    st.dropped_not_in_master += c.n_not_in_master
                    st.dropped_out_of_range += c.n_out_of_range
                    chunk = res.rename(columns={"ticker": "sharadar_ticker", "id": "ticker"})
                    st.observe_dates(chunk["date"])
                if chunk.empty:
                    continue
                mode = "upsert" if (self.mode == "incremental" or not first) else "replace_partition"
                key = [c for c in ("date", "ticker") if c in chunk.columns] or None
                self.store.write_long(name, chunk, mode=mode, kind="long", key=key)
                first = False
                st.rows_kept += len(chunk)
        st.datasets[name] = st.rows_kept
        st.note = "verbatim"
        self._done(table, files)
        st.seconds = time.perf_counter() - t0

    # -- derived datasets -------------------------------------------------------
    def _years_to_rebuild(self) -> list[int]:
        if not self.store.exists("prices_close_raw"):
            return []
        all_years = sorted(
            {
                int(os.path.basename(os.path.dirname(f)).split("=")[1])
                for f in self.store._files("prices_close_raw")
                if "year=" in f
            }
        )
        if self.mode == "full" or not self.touched_years:
            return all_years
        y0 = min(self.touched_years)
        return [y for y in all_years if y >= y0]

    def _sharesbas(self) -> pd.DataFrame | None:
        if not self.store.exists("sharadar_sf1"):
            return None
        sb = self.store.read_long("sharadar_sf1", columns=["date", "ticker", "dimension", "sharesbas"])
        if sb.empty or "sharesbas" not in sb.columns:
            return None
        return sb.rename(columns={"date": "datekey"})[["datekey", "ticker", "sharesbas"]]

    def step_caps(self) -> None:
        """``market_cap`` / ``shares_outstanding`` per year from the stored
        ``prices_close_raw`` + ``sharadar_marketcap`` (+ SF1 ``sharesbas``)."""
        st, t0 = self._stats("derived:caps"), time.perf_counter()
        years = self._years_to_rebuild()
        if not years:
            st.skipped, st.note = True, "no prices_close_raw in store"
            return
        sb = self._sharesbas()
        n_src: dict[str, int] = {}
        self._reset("market_cap", "shares_outstanding")
        for y in years:
            lo, hi = f"{y}-01-01", f"{y}-12-31"
            raw = self.store.read_long("prices_close_raw", lo, hi).rename(columns={"value": "close_raw"})
            if raw.empty:
                continue
            daily = None
            if self.store.exists("sharadar_marketcap"):
                daily = self.store.read_long("sharadar_marketcap", lo, hi).rename(
                    columns={"value": "marketcap"}
                )
            caps = derive_caps(raw, daily, sb)
            for k, v in caps["cap_source"].value_counts().items():
                n_src[str(k)] = n_src.get(str(k), 0) + int(v)
            st.datasets["market_cap"] = st.datasets.get("market_cap", 0) + _write_panel(
                self.store, "market_cap", caps, "market_cap", "replace_partition"
            )
            st.datasets["shares_outstanding"] = st.datasets.get("shares_outstanding", 0) + _write_panel(
                self.store, "shares_outstanding", caps, "shares_outstanding", "replace_partition"
            )
            st.observe_dates(caps["date"])
            st.rows_read += len(caps)
        st.rows_kept = st.datasets.get("market_cap", 0)
        st.dropped_other = {f"cap_source_{k}": v for k, v in n_src.items()}
        st.note = f"years {years[0]}..{years[-1]}"
        st.seconds = time.perf_counter() - t0

    def step_fundamentals(self) -> None:
        """PIT fundamentals panels rebuilt from ``sharadar_sf1`` (ART) + the stored
        ``market_cap`` / ``shares_outstanding``, in id batches, spilled per year."""
        st, t0 = self._stats("derived:fundamentals"), time.perf_counter()
        if not (
            self.store.exists("sharadar_sf1")
            and self.store.exists("market_cap")
            and self.store.exists("prices_close")
        ):
            st.skipped, st.note = True, "needs sharadar_sf1, market_cap and prices_close"
            return
        sf1 = self.store.read_long("sharadar_sf1")
        sf1 = sf1[sf1["dimension"] == "ART"].rename(columns={"date": "datekey"})
        if sf1.empty:
            st.skipped, st.note = True, "no ART rows"
            return
        dates = pd.DatetimeIndex(
            self.store.duckdb_query("SELECT DISTINCT date FROM prices_close ORDER BY date")["date"]
        )
        ids = sorted(sf1["ticker"].unique())
        spill = _YearSpill(self.staging_root / "fundamentals")
        fields = [f for f in FUNDAMENTAL_FIELDS if f in sf1.columns]
        for i in range(0, len(ids), self.fund_batch):
            batch = ids[i : i + self.fund_batch]
            mc = self.store.read_wide("market_cap", tickers=batch).reindex(index=dates)
            so = (
                self.store.read_wide("shares_outstanding", tickers=batch).reindex(index=dates)
                if self.store.exists("shares_outstanding")
                else None
            )
            sub = sf1[sf1["ticker"].isin(batch)]
            panels = sharadar_sf1_to_pit_fundamentals(
                sub,
                dimension="ART",
                lag_days=1,
                dates=dates,
                fields=fields,
                shares_outstanding=so,
                market_cap=mc,
            )
            frames = []
            for name in FUNDAMENTAL_PANELS:
                if name in panels:
                    s = cast(pd.Series, panels[name].stack(future_stack=True))
                    s.index.names = ["date", "ticker"]
                    frames.append(s.rename(name))
            if frames:
                long = pd.concat(frames, axis=1).reset_index()
                long = long.dropna(how="all", subset=[c for c in FUNDAMENTAL_PANELS if c in long.columns])
                spill.add(long)
            log.info("fundamentals: %d/%d ids", min(i + self.fund_batch, len(ids)), len(ids))
            st.rows_read += len(sub)
        years = self._years_to_rebuild()
        self._reset(*FUNDAMENTAL_PANELS)
        for y in spill.years():
            if years and y not in years:
                continue
            part = spill.read_year(y)
            for name in FUNDAMENTAL_PANELS:
                if name in part.columns:
                    st.datasets[name] = st.datasets.get(name, 0) + _write_panel(
                        self.store, name, part, name, "replace_partition"
                    )
            st.observe_dates(part["date"])
        if not self.keep_staging:
            spill.cleanup()
        st.rows_kept = st.datasets.get("book_to_price", 0)
        st.note = f"{len(ids)} ids, ART, lag 1 day"
        st.seconds = time.perf_counter() - t0


def sync(
    store_root: str = DEFAULT_STORE,
    raw_dir: str = DEFAULT_RAW_DIR,
    tables: Sequence[str] = DEFAULT_TABLES,
    since: str | None = None,
    from_files: str | None = None,
    api_key: str | None = None,
    master_filters: MasterFilters | None = None,
    chunksize: int = 500_000,
    dry_run: bool = False,
    force: bool = False,
    client: NasdaqDataLinkClient | None = None,
    fundamentals_batch: int = 500,
    keep_staging: bool = False,
) -> SyncReport:
    """Download (or read ``from_files``) -> transform -> write the store; returns a
    :class:`SyncReport` (also saved to ``<store>/_sharadar/last_sync.json``).

    * ``since=None`` is a **full load**: every dataset a table feeds is dropped and
      rebuilt (``mode="replace_partition"``); ``since="YYYY-MM-DD"`` downloads only rows
      with ``lastupdated >= since`` for SEP/DAILY/SF1 and upserts them (TICKERS,
      ACTIONS, SP500 are always whole).  Sharadar restates ``closeadj`` for the whole
      history at every dividend, so do a full load periodically.
    * Idempotent and resumable per table: a table whose raw file (path, size, mtime)
      was already processed in the same mode is skipped (state in
      ``<store>/_sharadar/state.json``; ``force=True`` ignores it).
    * ``market_cap``/``shares_outstanding`` and the four fundamentals panels are
      derived from the store after the raw tables, for every year a raw table touched
      (all years on a full load).
    """
    s = _Sync(
        store_root,
        raw_dir,
        tables,
        since,
        from_files,
        api_key,
        master_filters or MasterFilters(),
        chunksize,
        dry_run,
        force,
        client,
        fundamentals_batch,
        keep_staging,
    )
    return s.run()


# =============================================================================
# Validation
# =============================================================================
@dataclass
class CheckResult:
    name: str
    status: str  # PASS | WARN | SKIP
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    store_root: str
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def n_warn(self) -> int:
        return sum(c.status == "WARN" for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def format(self) -> str:
        lines = [
            f"Sharadar store validation: {self.store_root}",
            "",
            f"{'dataset':24s} {'kind':5s} {'rows':>12s} {'names':>7s}  range",
        ]
        for n, d in self.datasets.items():
            rng = f"{d.get('start') or '-'}..{d.get('end') or '-'}" if d.get("start") else "(no dates)"
            lines.append(
                f"{n:24s} {d.get('kind', ''):5s} {d.get('rows', 0):>12,d} {d.get('n_tickers', 0):>7,d}  {rng}"
            )
        lines.append("")
        for c in self.checks:
            lines.append(f"[{c.status:4s}] {c.name}: {c.message}")
            for k, v in c.detail.items():
                if isinstance(v, (list, dict)) and len(str(v)) > 200:
                    v = str(v)[:200] + "..."
                lines.append(f"         {k}: {v}")
        lines.append("")
        lines.append(f"{len(self.checks)} checks, {self.n_warn} WARN")
        return "\n".join(lines)


def _q(store: ParquetStore, sql: str) -> pd.DataFrame:
    return store.duckdb_query(sql)


def _sample_ids(store: ParquetStore, n: int, seed: int = 0) -> list[str]:
    ids = sorted(_q(store, "SELECT DISTINCT ticker FROM prices_close")["ticker"].astype(str))
    if len(ids) <= n:
        return ids
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(np.array(ids, dtype=object), size=n, replace=False).tolist())


def _sql_list(ids: Sequence[str]) -> str:
    return ",".join("'" + i.replace("'", "''") + "'" for i in ids)


def validate(store_root: str, sample_size: int = 50, seed: int = 0) -> ValidationReport:
    """Structural and economic sanity checks of a Sharadar-filled store (PASS/WARN/SKIP
    per check; nothing raises on a bad store).  See the check names in the report."""
    store = ParquetStore(store_root)
    rep = ValidationReport(store_root=str(store_root))
    names = store.datasets()
    for n in names:
        meta = store.meta(n)
        files = store._files(n)
        if not files:
            continue
        try:
            cols = pq.read_schema(files[0]).names
            nt = "count(DISTINCT ticker)" if "ticker" in cols else "0"
            if "date" in cols:
                d = _q(
                    store, f'SELECT count(*) AS rows, min(date) AS lo, max(date) AS hi, {nt} AS nt FROM "{n}"'
                ).iloc[0]
                rep.datasets[n] = {
                    "kind": meta.get("kind", ""),
                    "rows": int(d["rows"]),
                    "n_tickers": int(d["nt"]),
                    "start": str(pd.Timestamp(d["lo"]).date()),
                    "end": str(pd.Timestamp(d["hi"]).date()),
                }
            else:
                d = _q(store, f'SELECT count(*) AS rows, {nt} AS nt FROM "{n}"').iloc[0]
                rep.datasets[n] = {
                    "kind": meta.get("kind", ""),
                    "rows": int(d["rows"]),
                    "n_tickers": int(d["nt"]),
                }
        except Exception as e:  # pragma: no cover - defensive
            rep.datasets[n] = {"kind": meta.get("kind", ""), "rows": -1, "n_tickers": 0, "error": str(e)}
    missing = [n for n in ("prices_close", "prices_volume", "market_cap", "industry") if n not in names]
    rep.checks.append(
        CheckResult(
            "required_datasets",
            "WARN" if missing else "PASS",
            f"missing {missing}" if missing else "prices_close, prices_volume, market_cap, industry present",
        )
    )
    if "prices_close" not in names:
        return rep
    ids = _sample_ids(store, sample_size, seed)
    idl = _sql_list(ids)

    # coverage per year: names alive, names delisting that year
    cov = _q(
        store,
        "SELECT year(date) AS y, count(DISTINCT ticker) AS names FROM prices_close GROUP BY y ORDER BY y",
    )
    dl = (
        store.read_long("delisting_dates")
        if store.exists("delisting_dates")
        else pd.DataFrame(columns=["ticker", "delisting_date"])
    )
    dl["delisting_date"] = (
        pd.to_datetime(dl["delisting_date"]) if len(dl) else pd.Series(dtype="datetime64[ns]")
    )
    per_year = dl["delisting_date"].dt.year.value_counts().sort_index() if len(dl) else pd.Series(dtype=int)
    table = {
        int(r.y): {"names": int(r.names), "delisted": int(per_year.get(int(r.y), 0))}
        for r in cov.itertuples()
    }
    n_ids = int(_q(store, "SELECT count(DISTINCT ticker) AS n FROM prices_close").iloc[0]["n"])
    n_dl = int(dl["delisting_date"].notna().sum())
    share = n_dl / max(1, n_ids)
    rep.checks.append(
        CheckResult(
            "coverage_by_year",
            "PASS" if n_dl > 0 else "WARN",
            "delisted names present per year" if n_dl > 0 else "no delisted names at all: survivorship bias",
            {"by_year": table},
        )
    )
    rep.checks.append(
        CheckResult(
            "delisting_share",
            "PASS" if share >= 0.2 else "WARN",
            f"{n_dl} of {n_ids} ids ({share:.1%}) have a delisting date; "
            + (
                "plausible for a delisting-complete history"
                if share >= 0.2
                else "low for a multi-decade US history"
            ),
            {"covered": len(dl), "n_ids": n_ids},
        )
    )

    # market cap identity and units
    if store.exists("market_cap") and store.exists("prices_close_raw") and store.exists("shares_outstanding"):
        mc = _q(
            store,
            f"SELECT m.date, m.ticker, m.value AS mcap, r.value AS raw, s.value AS sh FROM market_cap m "
            f"JOIN prices_close_raw r ON m.date = r.date AND m.ticker = r.ticker "
            f"JOIN shares_outstanding s ON m.date = s.date AND m.ticker = s.ticker WHERE m.ticker IN ({idl})",
        )
        ratio = (mc["mcap"] / (mc["raw"] * mc["sh"])).replace([np.inf, -np.inf], np.nan).dropna()
        med = float(ratio.median()) if len(ratio) else float("nan")
        mx = float(_q(store, "SELECT max(value) AS v FROM market_cap").iloc[0]["v"])
        q50 = float(_q(store, "SELECT median(value) AS v FROM market_cap").iloc[0]["v"])
        unit_ok = 1e11 <= mx <= 2e13
        det: dict[str, Any] = {
            "median_ratio": med,
            "max_market_cap": mx,
            "median_market_cap": q50,
            "n": len(ratio),
        }
        status = "PASS" if (len(ratio) and abs(med - 1) < 1e-6 and unit_ok) else "WARN"
        msg = f"median mcap/(close_raw x shares) = {med:.4f}; largest cap {mx:.3g} USD ({'plausible' if unit_ok else 'check DAILY_USD_SCALE'})"
        if store.exists("sharadar_sf1"):
            sb = _q(
                store,
                f"SELECT f.ticker, f.date, f.sharesbas, s.value AS sh FROM sharadar_sf1 f "
                f"JOIN shares_outstanding s ON s.ticker = f.ticker AND s.date = f.date "
                f"WHERE f.dimension = 'ARQ' AND f.sharesbas > 0 AND f.ticker IN ({idl})",
            )
            if len(sb):
                r2 = float((sb["sh"] / sb["sharesbas"]).median())
                det["median_shares_vs_sharesbas"] = r2
                if not 0.5 <= r2 <= 2.0:
                    status = "WARN"
                    msg += f"; implied shares / SF1 sharesbas median {r2:.3g} (expected ~1)"
                else:
                    msg += f"; implied shares / SF1 sharesbas median {r2:.3f}"
        rep.checks.append(CheckResult("market_cap_identity", status, msg, det))
    else:
        rep.checks.append(
            CheckResult(
                "market_cap_identity", "SKIP", "needs market_cap, prices_close_raw, shares_outstanding"
            )
        )

    # splits: raw jumps with continuous adjusted close
    if store.exists("prices_close_raw"):
        px = _q(
            store,
            f"SELECT c.date, c.ticker, c.value AS adj, r.value AS raw FROM prices_close c "
            f"JOIN prices_close_raw r ON c.date = r.date AND c.ticker = r.ticker WHERE c.ticker IN ({idl}) "
            f"ORDER BY c.ticker, c.date",
        )
        prev_raw = px.groupby("ticker")["raw"].shift(1)
        prev_adj = px.groupby("ticker")["adj"].shift(1)
        px["raw_ret"] = px["raw"] / prev_raw - 1.0  # no gap filling: a NaN neighbour is NaN
        px["adj_ret"] = px["adj"] / prev_adj - 1.0
        jump = px[(px["raw_ret"].abs() > 0.4) & (px["adj_ret"].abs() < 0.2)]
        n_jump = len(jump)
        matched = None
        if n_jump and store.exists("corporate_actions"):
            ca = store.read_long("corporate_actions")
            ca = ca[ca["action"].astype(str).str.contains("split")]
            ca_id = ca["id"].where(ca["id"].notna(), ca["ticker"]).astype(str)
            keyed = {(i, d) for i, d in zip(ca_id, pd.to_datetime(ca["date"]), strict=True)}
            hits = 0
            for r in jump.itertuples():
                d0 = pd.Timestamp(r.date)
                if any((str(r.ticker), d0 + pd.Timedelta(days=k)) in keyed for k in range(-3, 4)):
                    hits += 1
            matched = hits / n_jump
        status = "PASS" if (n_jump == 0 or matched is None or matched >= 0.8) else "WARN"
        sample = [
            (str(r.ticker), str(pd.Timestamp(r.date).date()), round(float(r.raw_ret), 3))
            for r in jump.head(10).itertuples()
        ]
        rep.checks.append(
            CheckResult(
                "splits",
                status,
                f"{n_jump} raw-close jumps > 40% with a continuous adjusted close in the sample"
                + (f"; {matched:.0%} within 3 days of a split action" if matched is not None else ""),
                {"sample": sample},
            )
        )
    else:
        rep.checks.append(CheckResult("splits", "SKIP", "no prices_close_raw"))

    # volume sanity (dollar volume)
    if store.exists("prices_volume") and store.exists("prices_close_raw"):
        dv = _q(
            store,
            f"SELECT median(v.value * r.value) AS med, quantile_cont(v.value * r.value, 0.01) AS p1, "
            f"quantile_cont(v.value * r.value, 0.99) AS p99 FROM prices_volume v JOIN prices_close_raw r "
            f"ON v.date = r.date AND v.ticker = r.ticker WHERE v.ticker IN ({idl})",
        ).iloc[0]
        med = float(dv["med"]) if pd.notna(dv["med"]) else float("nan")
        ok = 1e4 <= med <= 1e9
        rep.checks.append(
            CheckResult(
                "volume_sanity",
                "PASS" if ok else "WARN",
                f"median daily dollar volume {med:,.0f} USD (p1 {float(dv['p1']):,.0f}, p99 {float(dv['p99']):,.0f})",
                {"median": med},
            )
        )
    else:
        rep.checks.append(CheckResult("volume_sanity", "SKIP", "needs prices_volume and prices_close_raw"))

    # PIT: no fundamentals value visible before datekey + 1
    if store.exists("sharadar_sf1") and store.exists("gross_profitability"):
        sf1 = _q(
            store,
            f"SELECT date AS datekey, ticker, gp, assets FROM sharadar_sf1 WHERE dimension = 'ART' "
            f"AND ticker IN ({idl}) AND gp IS NOT NULL AND assets IS NOT NULL AND assets <> 0 ORDER BY ticker, datekey",
        )
        gp = store.read_wide("gross_profitability", tickers=ids)
        viol: list[tuple[str, str, str]] = []
        checked = 0
        prev: dict[str, tuple[pd.Timestamp, float]] = {}
        for r in sf1.itertuples():
            t = str(r.ticker)
            dk = pd.Timestamp(r.datekey)
            v = float(r.gp) / float(r.assets)
            p_dk, p_v = prev.get(t, (pd.Timestamp.min, np.nan))
            prev[t] = (dk, v)
            if t not in gp.columns or (np.isfinite(p_v) and np.isclose(v, p_v, rtol=1e-9, atol=1e-12)):
                continue  # an unchanged ratio cannot be dated
            s = gp[t].dropna()
            s = s[s.index > p_dk]  # the previous filing may legitimately show an equal value earlier
            hit = s[np.isclose(s.to_numpy(), v, rtol=1e-9, atol=1e-12)]
            if hit.empty:
                continue
            checked += 1
            first = pd.Timestamp(hit.index[0])
            if first < dk + pd.Timedelta(days=1):
                viol.append((t, str(dk.date()), str(first.date())))
        rep.checks.append(
            CheckResult(
                "pit_fundamentals",
                "PASS" if not viol else "WARN",
                f"{len(viol)} of {checked} filings visible before datekey + 1",
                {"violations": viol[:10]},
            )
        )
    else:
        rep.checks.append(
            CheckResult("pit_fundamentals", "SKIP", "needs sharadar_sf1 and gross_profitability")
        )

    # missing closeunadj
    if store.exists("prices_close_raw"):
        m = _q(
            store,
            "SELECT (SELECT count(*) FROM prices_close) AS n, "
            "(SELECT count(*) FROM prices_close c LEFT JOIN prices_close_raw r ON c.date = r.date AND c.ticker = r.ticker "
            "WHERE r.value IS NULL) AS miss",
        ).iloc[0]
        frac = float(m["miss"]) / max(1.0, float(m["n"]))
        rep.checks.append(
            CheckResult(
                "missing_close_raw",
                "PASS" if frac <= 0.01 else "WARN",
                f"{frac:.3%} of prices_close rows have no as-traded close",
                {"missing": int(m["miss"]), "rows": int(m["n"])},
            )
        )
    else:
        rep.checks.append(
            CheckResult(
                "missing_close_raw",
                "WARN",
                "no prices_close_raw dataset: price screens fall back to adjusted prices",
            )
        )

    # duplicates
    dups: dict[str, int] = {}
    for n in ("prices_close", "prices_close_raw", "prices_volume", "market_cap", "shares_outstanding"):
        if store.exists(n):
            d = _q(
                store,
                f'SELECT count(*) AS n FROM (SELECT date, ticker, count(*) c FROM "{n}" GROUP BY date, ticker HAVING c > 1)',
            ).iloc[0]
            dups[n] = int(d["n"])
    rep.checks.append(
        CheckResult(
            "duplicates",
            "PASS" if not any(dups.values()) else "WARN",
            "no duplicate (id, date) rows" if not any(dups.values()) else f"duplicates: {dups}",
            dups,
        )
    )

    # delisting dates vs ACTIONS
    if store.exists("corporate_actions") and len(dl):
        ca = store.read_long("corporate_actions")
        ca = ca[(ca["action"] == "delisted") & ca["id"].notna()]
        if len(ca):
            j = ca[["id", "date"]].merge(dl.rename(columns={"ticker": "id"}), on="id", how="inner")
            j = j[j["delisting_date"].notna()]
            diff = (pd.to_datetime(j["date"]) - j["delisting_date"]).dt.days.abs()
            frac = float((diff <= 5).mean()) if len(j) else float("nan")
            rep.checks.append(
                CheckResult(
                    "delisting_vs_actions",
                    "PASS" if (len(j) == 0 or frac >= 0.9) else "WARN",
                    f"{len(j)} delisted actions matched to TICKERS lastpricedate; {frac:.0%} within 5 days"
                    if len(j)
                    else "no overlap between delisted actions and the delisting table",
                    {"n_actions": len(ca), "n_matched": len(j)},
                )
            )
        else:
            rep.checks.append(
                CheckResult("delisting_vs_actions", "SKIP", "no delisted actions in corporate_actions")
            )
    else:
        rep.checks.append(
            CheckResult("delisting_vs_actions", "SKIP", "needs corporate_actions and delisting_dates")
        )

    # S&P 500: current list vs the membership panel's last row
    if store.exists("sp500_membership"):
        ev = store.read_long("sp500_membership")
        cur = ev[ev["action"] == "current"]
        if len(cur):
            dates = pd.DatetimeIndex(
                _q(store, "SELECT DISTINCT date FROM prices_close ORDER BY date")["date"]
            )
            panel = sp500_membership_panel(ev, dates)
            last = set(panel.columns[panel.iloc[-1].to_numpy()])
            cur_ids = set(cur["id"].where(cur["id"].notna(), cur["ticker"]).astype(str))
            jac = len(last & cur_ids) / max(1, len(last | cur_ids))
            rep.checks.append(
                CheckResult(
                    "sp500_current_vs_panel",
                    "PASS" if jac >= 0.95 else "WARN",
                    f"panel has {len(last)} members on {dates[-1].date()}, vendor lists {len(cur_ids)} current; Jaccard {jac:.3f}",
                    {"only_panel": sorted(last - cur_ids)[:10], "only_current": sorted(cur_ids - last)[:10]},
                )
            )
        else:
            rep.checks.append(
                CheckResult("sp500_current_vs_panel", "SKIP", "no 'current' rows in sp500_membership")
            )
    else:
        rep.checks.append(CheckResult("sp500_current_vs_panel", "SKIP", "no sp500_membership dataset"))
    return rep


__all__ = [
    "API_BASE",
    "API_KEY_ENV",
    "DAILY_USD_SCALE",
    "DEFAULT_CATEGORIES",
    "DEFAULT_EXCHANGES",
    "DEFAULT_TABLES",
    "FAMAINDUSTRY_TABLE",
    "FUNDAMENTAL_PANELS",
    "KNOWN_TABLES",
    "OPTIONAL_TABLES",
    "REQUIRED_TABLES",
    "CheckResult",
    "MasterFilters",
    "NasdaqDataLinkClient",
    "ResolveCounts",
    "SecurityMaster",
    "SharadarDownloadError",
    "SharadarError",
    "SyncReport",
    "TableStats",
    "ValidationReport",
    "build_industry_history",
    "build_security_master",
    "derive_caps",
    "find_table_files",
    "format_source_map",
    "iter_csv_chunks",
    "load_api_key",
    "map_famaindustry",
    "master_from_store",
    "parse_dotenv",
    "raw_cache_path",
    "read_csv_table",
    "redact",
    "resolve_ids",
    "resolve_sources",
    "sp500_membership_panel",
    "sync",
    "transform_daily_chunk",
    "transform_events",
    "transform_sep_chunk",
    "transform_sf1",
    "validate",
]
