"""Earnings announcement dates and the quarterly earnings record (feature set 2).

Two long tables feed the ``v2`` alpha features (:mod:`qts.alpha.features_v2`); both
are plain functions of vendor tables already in a Sharadar-filled store, so they are
*derived* (``qts data sharadar derive-earnings --store ...``), never downloaded.

``earnings_dates`` -- one row per earnings announcement
-------------------------------------------------------
Columns :data:`EARNINGS_DATES_COLUMNS`: ``ticker`` (the store id), ``announce_date``,
``source`` (``"events"`` | ``"datekey"``), and -- informational only, attached with
hindsight and **never read by a feature** -- the fiscal filing the announcement
belongs to (``datekey``, ``calendardate``, ``reportperiod``).

The announcement series of a name is the union of

1. every Sharadar ``EVENTS`` row that carries an earnings event code
   (:data:`EARNINGS_EVENT_CODES`: SEC form 8-K item 2.02, "Results of Operations and
   Financial Condition"); ``source = "events"``; and
2. every SF1 ``ARQ`` filing date (``datekey``) that has **no** such event in the
   ``ANNOUNCE_WINDOW_DAYS`` (45) calendar days up to and including the filing date
   (and after the previous filing date); ``source = "datekey"``.

Each row is public on its own ``announce_date``: an 8-K is public the day it is
filed, and on a filing date it is known whether an 8-K preceded it.  No row is
selected or dropped using anything dated later, so the series as of ``t`` is the
series restricted to ``announce_date <= t`` (what ``MarketData.slice_until`` does).
A pre-announcement filed under item 2.02 is an announcement of its own.

``earnings_fundamentals`` -- the as-reported quarterly record
-------------------------------------------------------------
Columns :data:`EARNINGS_FUNDAMENTALS_COLUMNS`: one row per SF1 filing row of the
``ARQ`` (single quarter: ``eps``) and ``ART`` (trailing twelve months: ``netinc``,
``ncfo``; ``assets`` is the balance-sheet level) dimensions, keyed on ``datekey``
(the filing date: when the numbers became public), plus ``split_cum``.

``split_cum`` is the product of the split ratios of every split of the name dated
on or before the row's ``datekey`` (1.0 without splits; a 2-for-1 split has ratio
2).  Sharadar's ``eps`` is *as reported*: per share of the share count at the time
of the filing.  ``eps * split_cum`` is therefore earnings per share of the
**original** (pre-split) share count, continuous across splits, and depends on
splits dated ``<= datekey`` only -- the adjustment never looks forward.

Split ratios come from the ``corporate_actions`` rows with ``action == "split"``.
Whether the vendor's ``value`` is the ratio or its inverse is checked against the
prices: on the split date ``(raw_{t-1} / raw_t) * (adj_t / adj_{t-1})`` is the
ratio implied by the as-traded and the adjusted close (:func:`confirm_split_ratios`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

#: Sharadar ``EVENTS.eventcodes`` values that mark an earnings announcement.  The
#: codes are the 8-K item numbers without the dot: ``22`` = item 2.02 "Results of
#: Operations and Financial Condition".  NOT verified against the vendor's
#: documentation offline: ``qts data sharadar derive-earnings`` prints the frequency
#: table of every code with two tell-tales (events per name-year, share filed in the
#: 45 days before a quarterly filing) so the code can be confirmed on the data.
EARNINGS_EVENT_CODES: tuple[str, ...] = ("22",)
#: An event counts for a filing when it is dated in the (calendar) days
#: ``(datekey - window, datekey]``.
ANNOUNCE_WINDOW_DAYS = 45
SOURCE_EVENTS = "events"
SOURCE_DATEKEY = "datekey"

EARNINGS_DATES_COLUMNS: tuple[str, ...] = (
    "ticker",
    "announce_date",
    "source",
    "datekey",
    "calendardate",
    "reportperiod",
)
EARNINGS_FUNDAMENTALS_COLUMNS: tuple[str, ...] = (
    "ticker",
    "dimension",
    "datekey",
    "calendardate",
    "eps",
    "split_cum",
    "netinc",
    "ncfo",
    "assets",
)
#: numeric SF1 fields copied into ``earnings_fundamentals`` (NaN column when the
#: stored ``sharadar_sf1`` lacks one)
EARNINGS_SF1_FIELDS: tuple[str, ...] = ("eps", "netinc", "ncfo", "assets")
EARNINGS_DIMENSIONS: tuple[str, ...] = ("ARQ", "ART")

_NAT = np.datetime64("NaT", "ns")
_CODE_RE = re.compile(r"[^0-9A-Za-z]+")
#: a price-implied split ratio confirms the vendor value within this factor
SPLIT_CONFIRM_TOL = 1.10


def parse_event_codes(value: object) -> tuple[str, ...]:
    """``"22|91"`` -> ``("22", "91")`` (any non-alphanumeric separator; ``22.0`` from a
    numeric column -> ``"22"``)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    s = str(value).strip()
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return tuple(x for x in _CODE_RE.split(s) if x)


def _dt(s: Any) -> pd.Series:
    return pd.to_datetime(pd.Series(s), errors="coerce").astype("datetime64[ns]").reset_index(drop=True)


def _events_long(events: pd.DataFrame | None) -> pd.DataFrame:
    """``ticker, date, code`` (one row per event code of a row)."""
    if events is None or len(events) == 0:
        return pd.DataFrame({"ticker": pd.Series(dtype=str), "date": _dt([]), "code": pd.Series(dtype=str)})
    need = {"ticker", "date", "eventcodes"}
    if not need <= set(events.columns):
        raise ValueError(f"events needs columns {sorted(need)}, got {sorted(events.columns)}")
    e = pd.DataFrame(
        {
            "ticker": events["ticker"].astype(str).to_numpy(),
            "date": _dt(events["date"].to_numpy()).to_numpy(),
            "code": [parse_event_codes(v) for v in events["eventcodes"]],
        }
    )
    e = e[e["date"].notna()].explode("code")
    e = e[e["code"].notna()]
    e["code"] = e["code"].astype(str)
    return e.drop_duplicates().reset_index(drop=True)


@dataclass
class EarningsDatesReport:
    """Counts of :func:`derive_earnings_dates` (printed by ``qts data sharadar
    derive-earnings``)."""

    has_events: bool = False
    codes: tuple[str, ...] = EARNINGS_EVENT_CODES
    window_days: int = ANNOUNCE_WINDOW_DAYS
    n_filings: int = 0
    n_filings_with_event: int = 0
    n_rows: int = 0
    n_event_rows: int = 0
    n_datekey_rows: int = 0
    n_event_rows_without_filing: int = 0
    #: calendar days from the announcement to the filing, matched events
    lead_days_quantiles: dict[str, float] = field(default_factory=dict)
    #: year -> {"filings", "with_event", "events", "datekey"}
    by_year: dict[int, dict[str, int]] = field(default_factory=dict)
    #: code -> {"rows", "tickers", "per_name_year", "share_before_filing"}
    code_table: dict[str, dict[str, float]] = field(default_factory=dict)
    # ---- earnings_fundamentals / splits (filled by the store-level derive)
    n_fundamental_rows: dict[str, int] = field(default_factory=dict)
    fields_missing: list[str] = field(default_factory=list)
    splits: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self, top: int = 25) -> str:
        nf = max(self.n_filings, 1)
        lines = [
            f"Earnings dates: {self.n_rows:,d} announcements "
            f"({self.n_event_rows:,d} from EVENTS code {'/'.join(self.codes)}, "
            f"{self.n_datekey_rows:,d} filing-date fallbacks)",
            f"  quarterly filings (SF1 ARQ datekeys): {self.n_filings:,d}; with an earnings event in the "
            f"{self.window_days} days before: {self.n_filings_with_event:,d} "
            f"({100 * self.n_filings_with_event / nf:.1f}%)",
        ]
        if not self.has_events:
            lines.append(
                "  EVENTS is NOT in this store: every announcement date is the SF1 filing date (datekey)."
            )
            lines.append(
                "  To add it from the bulk folder:  qts data sharadar sync --store <store> "
                "--tables EVENTS --from-files <bulk dir>   (then run derive-earnings again)"
            )
        else:
            lines.append(
                f"  events not followed by a quarterly filing within {self.window_days} days: "
                f"{self.n_event_rows_without_filing:,d} (kept: each is an announcement of its own)"
            )
            if self.lead_days_quantiles:
                q = self.lead_days_quantiles
                lines.append(
                    "  days from announcement to filing (matched): "
                    + ", ".join(f"{k} {v:.0f}" for k, v in q.items())
                )
        if self.by_year:
            lines += ["", "  by year      filings  with event   share   events  datekey-fallback"]
            for y in sorted(self.by_year):
                r = self.by_year[y]
                f_ = r.get("filings", 0)
                share = 100 * r.get("with_event", 0) / f_ if f_ else float("nan")
                lines.append(
                    f"    {y:<8d} {f_:>9,d} {r.get('with_event', 0):>11,d} {share:>6.1f}% "
                    f"{r.get('events', 0):>8,d} {r.get('datekey', 0):>17,d}"
                )
        if self.code_table:
            lines += [
                "",
                "  EVENTS code frequency (an earnings code shows ~4 per name-year and a high share "
                f"dated in the {self.window_days} days before a quarterly filing)",
                "    code        rows   tickers  per name-year  before a filing  used",
            ]
            items = sorted(self.code_table.items(), key=lambda kv: (-kv[1]["rows"], kv[0]))
            for code, r in items[:top]:
                lines.append(
                    f"    {code:<6s} {int(r['rows']):>9,d} {int(r['tickers']):>9,d} "
                    f"{r['per_name_year']:>14.2f} {100 * r['share_before_filing']:>15.1f}%"
                    + ("  <- earnings" if code in self.codes else "")
                )
        if self.n_fundamental_rows:
            lines += [
                "",
                "  earnings_fundamentals: "
                + ", ".join(f"{k} {v:,d} rows" for k, v in self.n_fundamental_rows.items()),
            ]
        if self.splits:
            s = self.splits
            lines.append(
                f"  splits: {s.get('n', 0):,d} rows; price-confirmed as given {s.get('as_given', 0):,d}, "
                f"as the inverse {s.get('inverse', 0):,d}, neither {s.get('unconfirmed', 0):,d}, "
                f"no price {s.get('no_price', 0):,d}  (orientation used where unconfirmed: "
                f"{s.get('orientation', 'given')})"
            )
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def _filings(arq: pd.DataFrame) -> pd.DataFrame:
    """Distinct quarterly filings ``ticker, datekey, calendardate, reportperiod``
    (two periods filed on one day: the later period), sorted, with ``prev_datekey``."""
    f = pd.DataFrame(
        {
            "ticker": arq["ticker"].astype(str).to_numpy(),
            "datekey": _dt(arq["datekey"].to_numpy()).to_numpy(),
            "calendardate": _dt(arq["calendardate"].to_numpy()).to_numpy(),
            "reportperiod": (
                _dt(arq["reportperiod"].to_numpy()).to_numpy()
                if "reportperiod" in arq.columns
                else np.full(len(arq), np.datetime64("NaT", "ns"))
            ),
        }
    )
    f = f[f["datekey"].notna()]
    f = f.sort_values(["ticker", "datekey", "calendardate"], kind="mergesort")
    f = f.drop_duplicates(["ticker", "datekey"], keep="last").reset_index(drop=True)
    f["prev_datekey"] = f.groupby("ticker")["datekey"].shift(1)
    return f


def derive_earnings_dates(
    arq: pd.DataFrame,
    events: pd.DataFrame | None = None,
    codes: Sequence[str] = EARNINGS_EVENT_CODES,
    window_days: int = ANNOUNCE_WINDOW_DAYS,
) -> tuple[pd.DataFrame, EarningsDatesReport]:
    """Announcement table (module docstring) from the quarterly filings ``arq``
    (``ticker, datekey, calendardate[, reportperiod]``) and the vendor ``events``
    (``ticker, date, eventcodes``; ``None`` / empty -> every announcement is a filing
    date).  Pure; returns ``(table with EARNINGS_DATES_COLUMNS, report)``, sorted by
    ``ticker, announce_date``."""
    codes = tuple(str(c) for c in codes)
    win = pd.Timedelta(days=int(window_days))
    rep = EarningsDatesReport(codes=codes, window_days=int(window_days))
    f = _filings(arq)
    rep.n_filings = len(f)
    el = _events_long(events)
    rep.has_events = len(el) > 0
    ev = el[el["code"].isin(codes)][["ticker", "date"]].drop_duplicates()
    ev = ev.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)

    # -- filings: the last earnings event on or before the filing date, if in the window
    fs = f.sort_values(["datekey", "ticker"], kind="mergesort").reset_index(drop=True)
    if len(ev) and len(fs):
        m = pd.merge_asof(
            fs,
            ev.rename(columns={"date": "event_date"}),
            left_on="datekey",
            right_on="event_date",
            by="ticker",
            direction="backward",
            allow_exact_matches=True,
        )
        lower = m["datekey"] - win
        lower = lower.where(m["prev_datekey"].isna() | (m["prev_datekey"] < lower), m["prev_datekey"])
        m["has_event"] = m["event_date"].notna() & (m["event_date"] > lower)
    else:
        m = fs.assign(event_date=_NAT, has_event=False)
    rep.n_filings_with_event = int(m["has_event"].sum())
    fb = m.loc[~m["has_event"]]
    fallback = pd.DataFrame(
        {
            "ticker": fb["ticker"].to_numpy(),
            "announce_date": fb["datekey"].to_numpy(),
            "source": SOURCE_DATEKEY,
            "datekey": fb["datekey"].to_numpy(),
            "calendardate": fb["calendardate"].to_numpy(),
            "reportperiod": fb["reportperiod"].to_numpy(),
        }
    )

    # -- events: every one is an announcement; the filing it precedes is informational
    if len(ev):
        if len(fs):
            me = pd.merge_asof(
                ev,
                fs.rename(columns={"datekey": "_dk"}),
                left_on="date",
                right_on="_dk",
                by="ticker",
                direction="forward",
                allow_exact_matches=True,
            )
            ok = (
                me["_dk"].notna()
                & (me["date"] > me["_dk"] - win)
                & (me["prev_datekey"].isna() | (me["date"] > me["prev_datekey"]))
            )
            for c in ("_dk", "calendardate", "reportperiod"):
                me[c] = me[c].where(ok)
        else:
            me = ev.assign(_dk=_NAT, calendardate=_NAT, reportperiod=_NAT)
        evrows = pd.DataFrame(
            {
                "ticker": me["ticker"].to_numpy(),
                "announce_date": me["date"].to_numpy(),
                "source": SOURCE_EVENTS,
                "datekey": _dt(me["_dk"].to_numpy()).to_numpy(),
                "calendardate": _dt(me["calendardate"].to_numpy()).to_numpy(),
                "reportperiod": _dt(me["reportperiod"].to_numpy()).to_numpy(),
            }
        )
        rep.n_event_rows_without_filing = int(evrows["datekey"].isna().sum())
        lead = (evrows["datekey"] - evrows["announce_date"]).dt.days.dropna()
        if len(lead):
            rep.lead_days_quantiles = {
                "p10": float(lead.quantile(0.10)),
                "median": float(lead.median()),
                "p90": float(lead.quantile(0.90)),
            }
    else:
        evrows = fallback.iloc[:0]
    out = pd.concat([evrows, fallback], ignore_index=True)
    for c in ("announce_date", "datekey", "calendardate", "reportperiod"):
        out[c] = _dt(out[c].to_numpy()).to_numpy()
    out["ticker"] = out["ticker"].astype(str)
    out["source"] = out["source"].astype(str)
    # an event and a fallback can never share (ticker, date): a filing with an event
    # on its own date has an event in the window
    out = out.sort_values(["ticker", "announce_date"], kind="mergesort").reset_index(drop=True)
    out = out[list(EARNINGS_DATES_COLUMNS)]

    rep.n_rows = len(out)
    rep.n_event_rows = int((out["source"] == SOURCE_EVENTS).sum())
    rep.n_datekey_rows = int((out["source"] == SOURCE_DATEKEY).sum())
    fy = m["datekey"].dt.year
    for y, g in m.groupby(fy):
        rep.by_year[int(y)] = {"filings": len(g), "with_event": int(g["has_event"].sum())}  # type: ignore[call-overload]
    for (y, src), n in out.groupby([out["announce_date"].dt.year, "source"]).size().items():  # type: ignore[misc]
        rep.by_year.setdefault(int(y), {"filings": 0, "with_event": 0})[str(src)] = int(n)
    rep.code_table = event_code_table(el, f, window_days)
    if rep.has_events and rep.n_event_rows == 0:
        rep.warnings.append(
            f"EVENTS has rows but none with code {'/'.join(codes)}: check the code table above and "
            "qts.data.earnings.EARNINGS_EVENT_CODES"
        )
    return out, rep


def event_code_table(
    events_long: pd.DataFrame, filings: pd.DataFrame, window_days: int = ANNOUNCE_WINDOW_DAYS
) -> dict[str, dict[str, float]]:
    """Per event code: rows, tickers, events per name-year and the share of events
    dated in the ``window_days`` up to a quarterly filing of the same name."""
    if len(events_long) == 0:
        return {}
    e = events_long.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    if len(filings):
        fs = filings[["ticker", "datekey"]].sort_values(["datekey", "ticker"], kind="mergesort")
        me = pd.merge_asof(
            e,
            fs,
            left_on="date",
            right_on="datekey",
            by="ticker",
            direction="forward",
            allow_exact_matches=True,
        )
        near = (me["datekey"] - me["date"]) < pd.Timedelta(days=int(window_days))
        e["near"] = near.fillna(False).to_numpy()
    else:
        e["near"] = False
    e["year"] = e["date"].dt.year
    out: dict[str, dict[str, float]] = {}
    for code, g in e.groupby("code"):
        name_years = g.drop_duplicates(["ticker", "year"]).shape[0]
        out[str(code)] = {
            "rows": float(len(g)),
            "tickers": float(g["ticker"].nunique()),
            "per_name_year": float(len(g) / max(name_years, 1)),
            "share_before_filing": float(g["near"].mean()),
        }
    return out


# ----------------------------------------------------------------------------- splits
def confirm_split_ratios(splits: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Orientation of the vendor's split ``value`` checked against the prices.

    ``splits``: ``ticker, date, value`` and, where the store has a close on the split
    date and the trading day before, ``raw, raw_prev, adj, adj_prev`` (as-traded and
    adjusted close).  The price-implied ratio is ``(raw_prev / raw) * (adj / adj_prev)``
    (a 2-for-1 split halves the as-traded price while the adjusted price is
    continuous -> 2).  Per row: the vendor value when the implied ratio confirms it
    within :data:`SPLIT_CONFIRM_TOL`; its inverse when the implied ratio confirms
    that instead; otherwise (no price, or neither) the orientation the confirmed
    majority uses.  Returns ``(ticker, date, ratio)`` and the counts."""
    s = splits.copy()
    v = pd.to_numeric(s["value"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(v) & (v > 0)
    s, v = s.loc[ok].reset_index(drop=True), v[ok]
    cols = {
        c: pd.to_numeric(s[c], errors="coerce").to_numpy(dtype=float) if c in s.columns else None
        for c in ("raw", "raw_prev", "adj", "adj_prev")
    }
    if all(x is not None for x in cols.values()):
        with np.errstate(divide="ignore", invalid="ignore"):
            implied = (cols["raw_prev"] / cols["raw"]) * (cols["adj"] / cols["adj_prev"])  # type: ignore[operator]
    else:
        implied = np.full(len(s), np.nan)
    priced = np.isfinite(implied) & (implied > 0)
    tol = np.log(SPLIT_CONFIRM_TOL)
    with np.errstate(divide="ignore", invalid="ignore"):
        given = priced & (np.abs(np.log(implied / v)) <= tol)
        inverse = priced & ~given & (np.abs(np.log(implied * v)) <= tol)
    orientation = "inverse" if int(inverse.sum()) > int(given.sum()) else "given"
    ratio = np.where(inverse, 1.0 / v, v)
    rest = ~(given | inverse)
    if orientation == "inverse":
        ratio = np.where(rest, 1.0 / v, ratio)
    out = pd.DataFrame(
        {
            "ticker": s["ticker"].astype(str).to_numpy(),
            "date": _dt(s["date"].to_numpy()).to_numpy(),
            "ratio": ratio,
        }
    )
    out = out[out["date"].notna()].reset_index(drop=True)
    counts = {
        "n": len(s),
        "as_given": int(given.sum()),
        "inverse": int(inverse.sum()),
        "unconfirmed": int((priced & rest).sum()),
        "no_price": int((~priced).sum()),
        "orientation": orientation,
    }
    return out, counts


def cumulative_split_factor(tickers: Any, dates: Any, splits: pd.DataFrame | None) -> np.ndarray:
    """Per ``(ticker, date)`` pair: product of the ``ratio`` of every split of the
    ticker dated ``<= date`` (1.0 when there is none).  ``splits``: ``ticker, date,
    ratio``."""
    tk = pd.Series(tickers).astype(str).reset_index(drop=True)
    dt = _dt(np.asarray(dates))
    out = np.ones(len(tk))
    if splits is None or len(splits) == 0 or len(tk) == 0:
        return out
    sp = pd.DataFrame(
        {
            "ticker": splits["ticker"].astype(str).to_numpy(),
            "date": _dt(splits["date"].to_numpy()).to_numpy(),
            "ratio": pd.to_numeric(splits["ratio"], errors="coerce").to_numpy(dtype=float),
        }
    )
    sp = sp[sp["date"].notna() & np.isfinite(sp["ratio"]) & (sp["ratio"] > 0)]
    if sp.empty:
        return out
    sp = sp.sort_values(["ticker", "date"], kind="mergesort")
    sp["cum"] = sp.groupby("ticker")["ratio"].cumprod()
    sp = sp.drop_duplicates(["ticker", "date"], keep="last").sort_values(["date", "ticker"], kind="mergesort")
    q = pd.DataFrame({"ticker": tk, "date": dt, "_i": np.arange(len(tk))})
    ok = q["date"].notna()
    m = pd.merge_asof(
        q[ok].sort_values(["date", "ticker"], kind="mergesort"),
        sp[["ticker", "date", "cum"]].rename(columns={"date": "_sd"}),
        left_on="date",
        right_on="_sd",
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
    )
    out[m["_i"].to_numpy()] = m["cum"].fillna(1.0).to_numpy()
    return out


def build_earnings_fundamentals(sf1: pd.DataFrame, splits: pd.DataFrame | None = None) -> pd.DataFrame:
    """SF1 rows (``ticker, dimension, datekey, calendardate`` + any of
    :data:`EARNINGS_SF1_FIELDS`) of the ARQ / ART dimensions -> the
    ``earnings_fundamentals`` table (:data:`EARNINGS_FUNDAMENTALS_COLUMNS`) with
    ``split_cum`` from ``splits`` (``ticker, date, ratio``).  A missing field becomes
    a NaN column.  Sorted by ``ticker, dimension, datekey, calendardate``."""
    d = sf1.copy()
    d.columns = [str(c).lower() for c in d.columns]
    need = {"ticker", "dimension", "datekey", "calendardate"}
    if not need <= set(d.columns):
        raise ValueError(f"SF1 frame needs columns {sorted(need)}")
    d["dimension"] = d["dimension"].astype(str).str.upper()
    d = d[d["dimension"].isin(EARNINGS_DIMENSIONS)]
    out = pd.DataFrame(
        {
            "ticker": d["ticker"].astype(str).to_numpy(),
            "dimension": d["dimension"].to_numpy(),
            "datekey": _dt(d["datekey"].to_numpy()).to_numpy(),
            "calendardate": _dt(d["calendardate"].to_numpy()).to_numpy(),
        }
    )
    for c in EARNINGS_SF1_FIELDS:
        out[c] = pd.to_numeric(d[c], errors="coerce").to_numpy(dtype=float) if c in d.columns else np.nan
    out = out[out["datekey"].notna() & out["calendardate"].notna()].reset_index(drop=True)
    out["split_cum"] = cumulative_split_factor(out["ticker"], out["datekey"], splits)
    out = out.sort_values(["ticker", "dimension", "datekey", "calendardate"], kind="mergesort")
    return out.reset_index(drop=True)[list(EARNINGS_FUNDAMENTALS_COLUMNS)]


__all__ = [
    "ANNOUNCE_WINDOW_DAYS",
    "EARNINGS_DATES_COLUMNS",
    "EARNINGS_DIMENSIONS",
    "EARNINGS_EVENT_CODES",
    "EARNINGS_FUNDAMENTALS_COLUMNS",
    "EARNINGS_SF1_FIELDS",
    "SOURCE_DATEKEY",
    "SOURCE_EVENTS",
    "EarningsDatesReport",
    "build_earnings_fundamentals",
    "confirm_split_ratios",
    "cumulative_split_factor",
    "derive_earnings_dates",
    "event_code_table",
    "parse_event_codes",
]
