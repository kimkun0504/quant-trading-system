"""Local point-in-time feature store: partitioned Parquet + DuckDB.

Layout (``root`` is any local directory)::

    root/
      <dataset>/_meta.json            {"kind": "wide"|"long", "value_dtype": ...}
      <dataset>/year=YYYY/data.parquet
      <dataset>/all.parquet           (long tables without a date column)

Wide panels (date x ticker) are stored **long** (``date, ticker, value``) so
that the ticker set may change between writes, delistings are naturally
represented as missing rows, and DuckDB/Polars can query every dataset with
one schema.  ``read_wide`` pivots back to the wide convention of
``qts.core.types``.

Dataset names understood by :meth:`ParquetStore.as_market_data`:

==========================  ===========================================
``prices_close``            adjusted close (required)
``prices_volume``           actual shares traded (required)
``prices_close_raw``        as-traded (unadjusted) close -> ``close_raw``
``shares_outstanding``      actual shares outstanding -> ``shares_outstanding``
``industry_history``        dated industry labels (wide, string values)
``market_cap``              USD market cap (required)
``industry``                long table ``ticker, industry`` (required)
``spread_bps``              half-spread proxy in bps
``short_interest_ratio``    short interest / shares outstanding
``borrow_fee``              annualised borrow fee (fraction)
``book_to_price``           PIT book-to-price
``earnings_yield``          PIT earnings yield
``gross_profitability``     PIT gross profits / assets
``asset_growth``            PIT YoY asset growth
``delisting_dates``         long table ``ticker, delisting_date`` (last
                            trading date; NaT = alive) -> ``delisting_dates``
``delisting_reasons``       long table ``id, delisting_date, reason,
                            source_action`` (one row per delisted id; reason in
                            acquisition / performance / voluntary / unknown)
                            -> ``delisting_reasons`` (absent -> all unknown)
``earnings_dates``          long table ``date (= announce_date), ticker (id),
                            source, datekey, calendardate, reportperiod``: one
                            row per earnings announcement -> ``earnings_dates``
``earnings_fundamentals``   long table ``date (= datekey), ticker (id),
                            dimension (ARQ / ART), calendardate, eps, split_cum,
                            netinc, ncfo, assets`` -> ``earnings_fundamentals``
==========================  ===========================================

The two earnings tables (:mod:`qts.data.earnings`) are the inputs of
``AlphaConfig.feature_set = "v2"``; ``qts data sharadar derive-earnings`` builds
them from the stored ``sharadar_sf1`` / ``sharadar_events`` / ``corporate_actions``.
Rows up to ``end`` are loaded (no ``start`` cut: the seasonal earnings surprise
needs three years of quarters before the first panel date).

Dated industries: dates before a ticker's first dated classification read
``"UNKNOWN"`` (never the static map, which in a store is the *current*
classification -- a look-ahead; audit 2 #13).  Supply ``industry_history``
from the panel start for real data.

Any additional wide dataset listed in ``extra_names`` is put in
``MarketData.extra``.  :mod:`qts.data.sharadar` fills every dataset above and adds
``security_master``, ``sharadar_sf1``, ``sharadar_marketcap`` / ``sharadar_ev`` /
``sharadar_pb`` / ``sharadar_pe`` / ``sharadar_ps``, ``corporate_actions`` and
``sp500_membership`` (reference tables, not read by ``as_market_data``).  For a store
of many thousand names use :meth:`ParquetStore.tickers_by_market_cap` to choose the
ids to load (``read_long`` / ``read_wide`` push a ``tickers`` filter into the parquet
reader) instead of pivoting everything.

Write modes (:meth:`ParquetStore.write_long` / :meth:`ParquetStore.write_wide`)
-------------------------------------------------------------------------------
* ``"upsert"`` (default) -- merge into the existing year partitions: rows of
  the stored partition whose key (``date, ticker`` when present) appears in
  the new data are replaced, every other stored row is kept.  Appending one
  day to a year keeps the rest of the year.  For wide panels a NaN cell means
  "no information" and keeps the stored value (``nan_policy="keep_existing"``,
  default); ``nan_policy="delete"`` makes a NaN cell delete the stored value.
* ``"replace_partition"`` -- every year partition touched by the new data is
  rewritten with only the new rows (intentional full rewrite of those years).
* ``"replace"`` -- the whole dataset is dropped and rewritten.

Every parquet file is written to a temporary file in the same directory and
moved into place with ``os.replace`` (atomic on POSIX), so a crash mid-write
never leaves a truncated partition.

Concurrency (audit 2 #9): every write of a dataset (read-merge-replace of its
partitions and ``_meta.json``) holds an exclusive ``fcntl.flock`` on
``<root>/.locks/<dataset>.lock``; every read holds the shared lock, so a reader
sees a consistent snapshot (never a half-rewritten dataset) and concurrent
upserts from several processes never lose rows.  Locks are advisory and local
(POSIX ``flock``; not reliable on NFS).
"""
from __future__ import annotations

import fcntl
import glob
import hashlib
import json
import os
import shutil
import uuid
import warnings
from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from qts.core.types import MarketData, Wide

REQUIRED = ("prices_close", "prices_volume", "market_cap", "industry")
OPTIONAL_PANELS = ("spread_bps", "short_interest_ratio", "borrow_fee", "book_to_price",
                   "earnings_yield", "gross_profitability", "asset_growth")
#: dataset name -> MarketData field for the optional price-level panels
RAW_PANELS: Dict[str, str] = {"prices_close_raw": "close_raw", "shares_outstanding": "shares_outstanding"}
WRITE_MODES = ("upsert", "replace_partition", "replace")
NAN_POLICIES = ("keep_existing", "delete")


def _atomic_write_table(table: pa.Table, path: str) -> None:
    """Write ``table`` to ``path`` via a temp file in the same directory + ``os.replace``."""
    d = os.path.dirname(path)
    tmp = os.path.join(d, f".tmp-{uuid.uuid4().hex}.partial")
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _normalise_keys(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str)
    return df


def _upsert(old: pd.DataFrame, new: pd.DataFrame, key: Sequence[str]) -> pd.DataFrame:
    """Stored rows whose ``key`` appears in ``new`` are dropped; ``new`` is appended."""
    old = _normalise_keys(old)
    if not key or old.empty:
        return new
    if "date" in old.columns and "date" in new.columns and old["date"].dtype != new["date"].dtype:
        old["date"] = old["date"].astype(new["date"].dtype)
    k_old = pd.MultiIndex.from_frame(old[list(key)])
    k_new = pd.MultiIndex.from_frame(new[list(key)])
    kept = old[~k_old.isin(k_new)]
    if kept.empty:
        return new
    return pd.concat([kept, new], ignore_index=True)


class ParquetStore:
    def __init__(self, root: str):
        self.root = str(root)
        os.makedirs(self.root, exist_ok=True)

    # ------------------------------------------------------------------ locking
    @contextmanager
    def _lock(self, name: str, exclusive: bool) -> Iterator[None]:
        """Process-level advisory lock of one dataset (exclusive for writers,
        shared for readers); the lock file lives outside the dataset directory so
        ``mode="replace"`` / :meth:`drop` can remove the directory under it."""
        d = os.path.join(self.root, ".locks")
        os.makedirs(d, exist_ok=True)
        fd = os.open(os.path.join(d, f"{name}.lock"), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    # ------------------------------------------------------------------ paths
    def _dir(self, name: str) -> str:
        return os.path.join(self.root, name)

    def _meta_path(self, name: str) -> str:
        return os.path.join(self._dir(name), "_meta.json")

    def _files(self, name: str) -> List[str]:
        d = self._dir(name)
        return sorted(glob.glob(os.path.join(d, "year=*", "*.parquet")) + glob.glob(os.path.join(d, "all.parquet")))

    def datasets(self) -> List[str]:
        return sorted(d for d in os.listdir(self.root) if os.path.exists(self._meta_path(d)))

    def exists(self, name: str) -> bool:
        return os.path.exists(self._meta_path(name))

    def meta(self, name: str) -> dict:
        with self._lock(name, exclusive=False):
            return self._meta_unlocked(name)

    def _meta_unlocked(self, name: str) -> dict:
        with open(self._meta_path(name)) as f:
            return json.load(f)

    def _write_meta(self, name: str, meta: dict) -> None:
        os.makedirs(self._dir(name), exist_ok=True)
        tmp = self._meta_path(name) + f".tmp-{uuid.uuid4().hex}"
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=1, sort_keys=True)
        os.replace(tmp, self._meta_path(name))

    def drop(self, name: str) -> None:
        with self._lock(name, exclusive=True):
            self._drop_unlocked(name)

    def _drop_unlocked(self, name: str) -> None:
        if os.path.isdir(self._dir(name)):
            shutil.rmtree(self._dir(name))

    # ------------------------------------------------------------------ long
    def write_long(self, name: str, df: pd.DataFrame, mode: str = "upsert", kind: str = "long",
                   key: Optional[Sequence[str]] = None) -> None:
        """Write a long table (see the module docstring for ``mode``).

        If a ``date`` column exists the table is partitioned by year.  ``key``
        (default: the columns among ``("date", "ticker")`` present) identifies a
        row for ``mode="upsert"``; tables without a ``date`` column are upserted
        on ``ticker`` if present, otherwise replaced.
        """
        with self._lock(name, exclusive=True):
            self._write_long(name, df, mode=mode, kind=kind, key=key, drop_null=None)

    def _write_long(self, name: str, df: pd.DataFrame, mode: str, kind: str,
                    key: Optional[Sequence[str]], drop_null: Optional[str]) -> None:
        if mode == "overwrite_partitions":
            warnings.warn("mode='overwrite_partitions' is deprecated (it silently dropped the rest of each "
                          "year); use 'upsert' (merge) or 'replace_partition' (explicit rewrite)",
                          DeprecationWarning, stacklevel=3)
            mode = "replace_partition"
        if mode not in WRITE_MODES:
            raise ValueError(f"mode must be one of {WRITE_MODES}, got {mode!r}")
        df = df.reset_index() if isinstance(df.index, pd.MultiIndex) or df.index.name in ("date", "ticker") else df.copy()
        if "index" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"index": "date"})
        df = _normalise_keys(df)
        if key is None:
            key = [c for c in ("date", "ticker") if c in df.columns]
        key = list(key)
        missing = set(key) - set(df.columns)
        if missing:
            raise ValueError(f"key columns {sorted(missing)} not in frame")
        prev_cols: List[str] = []
        if mode == "replace":
            self._drop_unlocked(name)
        elif self.exists(name):
            prev_cols = list(self._meta_unlocked(name).get("columns", []))
        os.makedirs(self._dir(name), exist_ok=True)

        def finish(part: pd.DataFrame, path: str) -> None:
            if mode == "upsert" and os.path.exists(path):
                part = _upsert(pq.read_table(path).to_pandas(), part, key)
            if drop_null is not None:
                part = part[part[drop_null].notna()]
            sort_cols = [c for c in ("date", "ticker") if c in part.columns]
            if sort_cols:
                part = part.sort_values(sort_cols, kind="mergesort")
            if part.empty:
                if os.path.exists(path):
                    os.remove(path)
                return
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _atomic_write_table(pa.Table.from_pandas(part.reset_index(drop=True), preserve_index=False), path)

        if "date" in df.columns:
            for year, part in df.groupby(df["date"].dt.year):
                finish(part, os.path.join(self._dir(name), f"year={int(year)}", "data.parquet"))
        else:
            finish(df, os.path.join(self._dir(name), "all.parquet"))
        cols = list(dict.fromkeys(prev_cols + list(map(str, df.columns))))
        self._write_meta(name, {"kind": kind, "columns": cols, "partitioned": "date" in df.columns})

    def read_long(self, name: str, start=None, end=None, columns: Optional[List[str]] = None,
                  tickers: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """Rows of a dataset.  ``start``/``end`` open only the year partitions in
        range; ``tickers`` is pushed down into the parquet reader (rows of other
        tickers are never materialised), which is what makes a pre-filtered
        ``as_market_data`` cheap on a store of ten thousand names."""
        with self._lock(name, exclusive=False):
            return self._read_long_unlocked(name, start, end, columns, tickers)

    def _read_long_unlocked(self, name: str, start=None, end=None,
                            columns: Optional[List[str]] = None,
                            tickers: Optional[Iterable[str]] = None) -> pd.DataFrame:
        files = self._files(name)
        if not files:
            raise FileNotFoundError(f"dataset {name!r} not in store {self.root}")
        if start is not None or end is not None:
            y0 = pd.Timestamp(start).year if start is not None else -1
            y1 = pd.Timestamp(end).year if end is not None else 10 ** 6
            files = [f for f in files if _year_of(f) is None or y0 <= _year_of(f) <= y1]
        filters = None
        if tickers is not None:
            filters = [("ticker", "in", [str(t) for t in tickers])]
        parts = [pq.read_table(f, columns=columns, filters=filters).to_pandas() for f in files]
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            if start is not None:
                df = df[df["date"] >= pd.Timestamp(start)]
            if end is not None:
                df = df[df["date"] <= pd.Timestamp(end)]
            df = df.sort_values(["date"] + (["ticker"] if "ticker" in df.columns else [])).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ wide
    def write_wide(self, name: str, df: Wide, mode: str = "upsert", nan_policy: str = "keep_existing") -> None:
        """Store a wide (date x ticker) panel as ``date, ticker, value`` rows.

        NaN cells are never stored (sparse storage).  Under ``mode="upsert"``:
        ``nan_policy="keep_existing"`` (default) -- a NaN cell carries no
        information and leaves any stored value for that (date, ticker) in place
        (a partial re-delivery of a sparse panel cannot delete data);
        ``nan_policy="delete"`` -- a NaN cell deletes the stored value."""
        if nan_policy not in NAN_POLICIES:
            raise ValueError(f"nan_policy must be one of {NAN_POLICIES}, got {nan_policy!r}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("wide panel must have a DatetimeIndex")
        long = df.stack(future_stack=True)
        long.index.names = ["date", "ticker"]
        long = long.rename("value").reset_index()
        long["ticker"] = long["ticker"].astype(str)
        if nan_policy == "keep_existing":
            long = long[long["value"].notna()]
        with self._lock(name, exclusive=True):
            self._write_long(name, long, mode=mode, kind="wide", key=("date", "ticker"), drop_null="value")

    def read_wide(self, name: str, start=None, end=None, tickers: Optional[Iterable[str]] = None) -> Wide:
        tick = None if tickers is None else [str(t) for t in tickers]
        long = self.read_long(name, start, end, tickers=tick)
        if long.empty:
            return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
        wide = long.pivot(index="date", columns="ticker", values="value").sort_index()
        wide.index.name = None
        wide.columns.name = "ticker"
        return wide

    def tickers_by_market_cap(self, start=None, end=None, min_mcap_ever: Optional[float] = None,
                              max_names: Optional[int] = None, dataset: str = "market_cap") -> List[str]:
        """Candidate ids for a pre-filtered :meth:`as_market_data` (the ``qts run
        --min-mcap-ever / --max-names`` path).

        Reads only the long ``market_cap`` dataset (through DuckDB, never pivoted)
        and keeps the ids whose **maximum** market cap over ``[start, end]`` is at
        least ``min_mcap_ever`` USD, then the ``max_names`` largest by that maximum.
        A name's *peak* size is used so that a company that was large at any time in
        the window is loaded for its whole history, delisted names included; this is
        a loading convenience, not a universe rule -- ``build_universe_mask`` still
        applies the point-in-time price, ADV and size screens on top."""
        where = []
        if start is not None:
            where.append(f"date >= DATE '{pd.Timestamp(start).date()}'")
        if end is not None:
            where.append(f"date <= DATE '{pd.Timestamp(end).date()}'")
        sql = f'SELECT ticker, max(value) AS mx FROM "{dataset}"'
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " GROUP BY ticker"
        if min_mcap_ever is not None:
            sql += f" HAVING max(value) >= {float(min_mcap_ever)!r}"
        sql += " ORDER BY mx DESC, ticker"
        if max_names is not None:
            sql += f" LIMIT {int(max_names)}"
        return [str(t) for t in self.duckdb_query(sql)["ticker"]]

    # ------------------------------------------------------------------ duckdb
    def duckdb_connection(self):
        import duckdb
        con = duckdb.connect()
        for name in self.datasets():
            files = self._files(name)
            if not files:
                continue
            pattern = os.path.join(self._dir(name), "year=*", "*.parquet")
            if not glob.glob(pattern):
                pattern = os.path.join(self._dir(name), "all.parquet")
            con.execute(f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM read_parquet(\'{pattern}\')')
        return con

    def duckdb_query(self, sql: str) -> pd.DataFrame:
        """Run SQL with every dataset exposed as a view named after the dataset
        (wide panels have columns ``date, ticker, value``)."""
        con = self.duckdb_connection()
        try:
            return con.execute(sql).df()
        finally:
            con.close()

    # ------------------------------------------------------------------ hashing
    def snapshot_hash(self, name: str) -> str:
        """SHA-256 over the dataset's parquet files (for ``TrialLedger.data_hash``)."""
        h = hashlib.sha256()
        with self._lock(name, exclusive=False):
            for f in self._files(name):
                h.update(os.path.relpath(f, self.root).encode())
                with open(f, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
        return h.hexdigest()

    def snapshot_hash_all(self, names: Optional[Iterable[str]] = None) -> str:
        h = hashlib.sha256()
        for n in sorted(names or self.datasets()):
            h.update(f"{n}:{self.snapshot_hash(n)}".encode())
        return h.hexdigest()

    # ------------------------------------------------------------------ assembly
    def as_market_data(self, universe_tickers: Optional[Iterable[str]] = None, start=None, end=None,
                       extra_names: Iterable[str] = ()) -> MarketData:
        """Assemble a :class:`MarketData` from the stored datasets (see module doc)."""
        for n in REQUIRED:
            if not self.exists(n):
                raise FileNotFoundError(f"required dataset {n!r} missing from store")
        tick = list(universe_tickers) if universe_tickers is not None else None
        close = self.read_wide("prices_close", start, end, tick)
        idx, cols = close.index, close.columns

        def panel(n):
            if not self.exists(n):
                return None
            return self.read_wide(n, start, end, tick).reindex(index=idx, columns=cols)

        volume = panel("prices_volume")
        mcap = panel("market_cap")
        ind_long = self.read_long("industry")
        industry = ind_long.set_index("ticker")["industry"].astype(str)
        industry = industry.reindex(cols).fillna("UNKNOWN")
        industry.index.name = "ticker"
        md = MarketData(
            close=close, volume=volume, market_cap=mcap, industry=industry,
            spread_bps=panel("spread_bps"), short_interest_ratio=panel("short_interest_ratio"),
            borrow_fee=panel("borrow_fee"), book_to_price=panel("book_to_price"),
            earnings_yield=panel("earnings_yield"), gross_profitability=panel("gross_profitability"),
            asset_growth=panel("asset_growth"),
            extra={n: panel(n) for n in extra_names if self.exists(n)},
            close_raw=panel("prices_close_raw"), shares_outstanding=panel("shares_outstanding"),
            industry_history=self._industry_history(end, cols, idx[0] if len(idx) else None),
            delisting_dates=self._delisting_dates(cols),
            delisting_reasons=self._delisting_reasons(cols),
            earnings_dates=self._dated_table("earnings_dates", "announce_date", cols, end),
            earnings_fundamentals=self._dated_table("earnings_fundamentals", "datekey", cols, end),
        )
        md.validate()
        return md

    def _dated_table(self, name: str, date_col: str, cols: pd.Index, end) -> Optional[pd.DataFrame]:
        """A long earnings table (stored with ``date`` = ``date_col``) for the loaded
        ids, rows dated ``<= end``; ``None`` when the dataset is absent.  Sorted by
        ``ticker, date_col`` with a fresh index (deterministic for the data hash)."""
        if not self.exists(name):
            return None
        df = self.read_long(name, None, end, tickers=[str(c) for c in cols])
        df = df.rename(columns={"date": date_col})
        for c in ("announce_date", "datekey", "calendardate", "reportperiod"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
        df["ticker"] = df["ticker"].astype(str)
        by = [c for c in ("ticker", "dimension", date_col, "calendardate") if c in df.columns]
        first = ["ticker", *[c for c in ("dimension", date_col) if c in df.columns]]
        df = df[first + [c for c in df.columns if c not in first]]
        return df.sort_values(by, kind="mergesort").reset_index(drop=True)

    def _industry_history(self, end, cols: pd.Index, first_date=None) -> Optional[Wide]:
        """Dated labels up to ``end`` (no ``start`` cut: the label in force at the
        first panel date may have been set earlier).

        A row of ``"UNKNOWN"`` is prepended at the earliest of the history's and
        the panel's first date for every ticker not yet classified then, so
        ``MarketData.industry_as_of`` never back-fills a date before a ticker's
        first dated classification with the static (= current) map."""
        if not self.exists("industry_history"):
            return None
        ih = self.read_wide("industry_history", None, end).reindex(columns=cols).astype(object)
        starts = [d for d in (ih.index.min() if len(ih) else None, first_date) if d is not None]
        if not starts:
            return ih
        d0 = pd.Timestamp(min(starts))
        if d0 not in ih.index:
            ih = pd.concat([pd.DataFrame(np.nan, index=[d0], columns=ih.columns, dtype=object), ih]).sort_index()
        ih.loc[d0] = ih.loc[d0].where(ih.loc[d0].notna(), "UNKNOWN")
        return ih

    def _delisting_dates(self, cols: pd.Index) -> Optional[pd.Series]:
        """``delisting_dates`` long table -> Series ticker -> last trading date
        (only tickers listed in the table are covered; see ``MarketData``)."""
        if not self.exists("delisting_dates"):
            return None
        df = self.read_long("delisting_dates")
        if df.empty:
            return None
        s = pd.Series(pd.to_datetime(df["delisting_date"]).to_numpy(), index=df["ticker"].astype(str))
        s = s[~s.index.duplicated(keep="last")]
        s = s[s.index.isin(cols)]
        s.index.name = "ticker"
        s.name = "delisting_date"
        return s

    def _delisting_reasons(self, cols: pd.Index) -> Optional[pd.Series]:
        """``delisting_reasons`` long table -> Series id -> reason for the loaded
        ids (``None`` when the dataset is absent: every delisting is "unknown").
        Written by ``qts data sharadar derive`` / ``sync``."""
        if not self.exists("delisting_reasons"):
            return None
        df = self.read_long("delisting_reasons")
        if df.empty:
            return None
        s = pd.Series(df["reason"].astype(str).to_numpy(), index=df["id"].astype(str), dtype=object)
        s = s[~s.index.duplicated(keep="last")]
        s = s[s.index.isin(cols)].sort_index()
        s.index.name = "ticker"
        s.name = "reason"
        return s

    def write_market_data(self, md: MarketData, include_extra: bool = False) -> None:
        """Persist every panel of a MarketData under the documented dataset names."""
        self.write_wide("prices_close", md.close)
        self.write_wide("prices_volume", md.volume)
        self.write_wide("market_cap", md.market_cap)
        self.write_long("industry", md.industry.rename("industry").rename_axis("ticker").reset_index(), mode="replace")
        for n in OPTIONAL_PANELS:
            f = getattr(md, n)
            if f is not None:
                self.write_wide(n, f)
        for n, attr in RAW_PANELS.items():
            f = getattr(md, attr)
            if f is not None:
                self.write_wide(n, f)
        if md.industry_history is not None:
            self.write_wide("industry_history", md.industry_history.astype(object))
        if md.delisting_dates is not None:
            dd = pd.DataFrame({"ticker": md.delisting_dates.index.astype(str),
                               "delisting_date": pd.to_datetime(md.delisting_dates).to_numpy()})
            self.write_long("delisting_dates", dd, mode="replace")
        if md.delisting_reasons is not None:
            dd_src = md.delisting_dates
            ids = md.delisting_reasons.index.astype(str)
            when = (pd.to_datetime(dd_src.reindex(ids)).to_numpy() if dd_src is not None
                    else np.full(len(ids), np.datetime64("NaT"), dtype="datetime64[ns]"))
            dr = pd.DataFrame({"id": ids, "delisting_date": when,
                               "reason": md.delisting_reasons.astype(str).to_numpy(),
                               "source_action": ""})
            self.write_long("delisting_reasons", dr, mode="replace")
        if md.earnings_dates is not None:
            self.write_long("earnings_dates", md.earnings_dates.rename(columns={"announce_date": "date"}),
                            mode="replace", key=["date", "ticker"])
        if md.earnings_fundamentals is not None:
            self.write_long("earnings_fundamentals", md.earnings_fundamentals.rename(columns={"datekey": "date"}),
                            mode="replace", key=["date", "ticker", "dimension", "calendardate"])
        if include_extra:
            for k, v in md.extra.items():
                self.write_wide(k, v)


def _year_of(path: str) -> Optional[int]:
    part = os.path.basename(os.path.dirname(path))
    if part.startswith("year="):
        return int(part.split("=", 1)[1])
    return None
