"""NYSE trading calendar (2000-2035), dependency-free.

Used wherever a lag is stated in *exchange sessions* (e.g. FINRA short-interest
publication: settlement date + 8 sessions).  ``pandas.offsets.BDay`` ignores
exchange holidays and is early by one session per holiday it spans (audit 2
#12: settlement 2024-12-31 -> 2025-01-10 with ``BDay(8)``, 2025-01-14 on the
NYSE calendar, which was closed on 1 and 9 January 2025).

Full-day closures (NYSE Rule 7.2 and its predecessors), with observed-day
rules:

* New Year's Day, 1 Jan -- on a Sunday observed Monday 2 Jan; on a Saturday
  *not* observed (the NYSE does not close on Friday 31 Dec, e.g. 2022-01-01).
* Martin Luther King Jr. Day -- third Monday of January (from 1998).
* Washington's Birthday / Presidents' Day -- third Monday of February.
* Good Friday -- the Friday before Easter Sunday (Gregorian computus).
* Memorial Day -- last Monday of May.
* Juneteenth, 19 Jun -- from 2022; Saturday -> Friday, Sunday -> Monday.
* Independence Day, 4 Jul -- Saturday -> Friday, Sunday -> Monday.
* Labor Day -- first Monday of September.
* Thanksgiving -- fourth Thursday of November.
* Christmas, 25 Dec -- Saturday -> Friday, Sunday -> Monday.

Plus the unscheduled closures in the range (``SPECIAL_CLOSURES``): 11-14 Sep
2001, 11 Jun 2004 (Reagan), 2 Jan 2007 (Ford), 29-30 Oct 2012 (Sandy), 5 Dec
2018 (G.H.W. Bush), 9 Jan 2025 (Carter).  Early closes (13:00) are sessions and
are not modelled.  Dates outside 2000-2035 raise ``ValueError``.
"""

from __future__ import annotations

import datetime as _dt
from functools import cache, lru_cache

import numpy as np
import pandas as pd

FIRST_YEAR = 2000
LAST_YEAR = 2035

SPECIAL_CLOSURES: tuple[str, ...] = (
    "2001-09-11",
    "2001-09-12",
    "2001-09-13",
    "2001-09-14",
    "2004-06-11",
    "2007-01-02",
    "2012-10-29",
    "2012-10-30",
    "2018-12-05",
    "2025-01-09",
)

__all__ = [
    "FIRST_YEAR",
    "LAST_YEAR",
    "SPECIAL_CLOSURES",
    "add_sessions",
    "easter_sunday",
    "is_session",
    "nyse_holidays",
    "nyse_sessions",
]


def easter_sunday(year: int) -> _dt.date:
    """Gregorian Easter Sunday (anonymous / Meeus-Jones-Butcher algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month = (h + ll - 7 * m + 114) // 31
    day = (h + ll - 7 * m + 114) % 31 + 1
    return _dt.date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> _dt.date:
    """``n``-th (1-based) ``weekday`` (Mon=0) of the month."""
    d = _dt.date(year, month, 1)
    d += _dt.timedelta(days=(weekday - d.weekday()) % 7)
    return d + _dt.timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> _dt.date:
    nxt = _dt.date(year + (month == 12), month % 12 + 1, 1)
    d = nxt - _dt.timedelta(days=1)
    return d - _dt.timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: _dt.date) -> _dt.date:
    """Saturday -> Friday, Sunday -> Monday."""
    if d.weekday() == 5:
        return d - _dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + _dt.timedelta(days=1)
    return d


def _check_year(year: int) -> None:
    if not FIRST_YEAR <= year <= LAST_YEAR:
        raise ValueError(f"NYSE calendar covers {FIRST_YEAR}-{LAST_YEAR}, got {year}")


@cache
def _holidays_of_year(year: int) -> tuple[_dt.date, ...]:
    _check_year(year)
    out: list[_dt.date] = []
    ny = _dt.date(year, 1, 1)
    if ny.weekday() == 6:
        out.append(ny + _dt.timedelta(days=1))
    elif ny.weekday() != 5:
        out.append(ny)
    out.append(_nth_weekday(year, 1, 0, 3))  # MLK (observed by the NYSE since 1998)
    out.append(_nth_weekday(year, 2, 0, 3))  # Washington's Birthday
    out.append(easter_sunday(year) - _dt.timedelta(days=2))  # Good Friday
    out.append(_last_weekday(year, 5, 0))  # Memorial Day
    if year >= 2022:
        out.append(_observed(_dt.date(year, 6, 19)))  # Juneteenth
    out.append(_observed(_dt.date(year, 7, 4)))
    out.append(_nth_weekday(year, 9, 0, 1))  # Labor Day
    out.append(_nth_weekday(year, 11, 3, 4))  # Thanksgiving
    out.append(_observed(_dt.date(year, 12, 25)))
    out += [_dt.date.fromisoformat(s) for s in SPECIAL_CLOSURES if int(s[:4]) == year]
    return tuple(sorted(set(out)))


def nyse_holidays(start: int = FIRST_YEAR, end: int = LAST_YEAR) -> pd.DatetimeIndex:
    """Weekday full-day closures in years ``start..end`` (inclusive)."""
    days = [d for y in range(int(start), int(end) + 1) for d in _holidays_of_year(y)]
    return pd.DatetimeIndex(sorted(days))


@lru_cache(maxsize=1)
def _all_sessions() -> np.ndarray:
    bd = pd.bdate_range(f"{FIRST_YEAR}-01-01", f"{LAST_YEAR}-12-31")
    return bd[~bd.isin(nyse_holidays())].values.astype("datetime64[D]")


def nyse_sessions(start, end) -> pd.DatetimeIndex:
    """NYSE sessions in ``[start, end]``."""
    s = _all_sessions()
    a, b = np.datetime64(pd.Timestamp(start).date(), "D"), np.datetime64(pd.Timestamp(end).date(), "D")
    return pd.DatetimeIndex(s[(s >= a) & (s <= b)])


def is_session(date) -> bool:
    d = np.datetime64(pd.Timestamp(date).date(), "D")
    _check_year(pd.Timestamp(date).year)
    s = _all_sessions()
    i = int(np.searchsorted(s, d))
    return i < len(s) and s[i] == d


def add_sessions(dates, n: int):
    """The ``n``-th NYSE session strictly after each date (``n >= 1``; a
    non-session date counts from the next session, e.g. settlement on a holiday).
    ``n = 0`` returns the dates unchanged (normalised).  Accepts a scalar or
    array-like; returns a ``Timestamp`` or ``DatetimeIndex`` (time of day
    dropped).  NaT stays NaT."""
    if int(n) < 0:
        raise ValueError("n must be >= 0")
    scalar = np.ndim(dates) == 0
    idx = pd.DatetimeIndex([dates] if scalar else pd.to_datetime(dates))
    if int(n) == 0:
        res0 = idx.normalize()
        return res0[0] if scalar else res0
    out = np.full(len(idx), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = ~idx.isna()
    if ok.any():
        years = idx[ok].year
        if years.min() < FIRST_YEAR or years.max() > LAST_YEAR:
            raise ValueError(f"NYSE calendar covers {FIRST_YEAR}-{LAST_YEAR}")
        s = _all_sessions()
        d = idx[ok].normalize().values.astype("datetime64[D]")
        pos = np.searchsorted(s, d, side="right") + int(n) - 1
        if (pos >= len(s)).any():
            raise ValueError(f"result beyond the calendar's last year {LAST_YEAR}")
        out[np.flatnonzero(ok)] = s[pos].astype("datetime64[ns]")
    res = pd.DatetimeIndex(out)
    return res[0] if scalar else res
