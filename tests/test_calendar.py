"""NYSE calendar (audit 2 #12): holiday rules against the published 2024/2025
schedules and the FINRA publication lag in exchange sessions."""

from __future__ import annotations

import pandas as pd
import pytest

from qts.data.calendar import add_sessions, easter_sunday, is_session, nyse_holidays, nyse_sessions
from qts.data.loaders import finra_publication_date

NYSE_2024 = [
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
]  # fmt: skip
NYSE_2025 = [
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
]  # fmt: skip


def test_known_holiday_lists():
    assert list(nyse_holidays(2024, 2024).strftime("%Y-%m-%d")) == NYSE_2024
    assert list(nyse_holidays(2025, 2025).strftime("%Y-%m-%d")) == NYSE_2025
    # 252 sessions in 2024, 250 in 2025 (incl. the 9 Jan national day of mourning)
    assert len(nyse_sessions("2024-01-01", "2024-12-31")) == 252
    assert len(nyse_sessions("2025-01-01", "2025-12-31")) == 250


def test_observed_day_rules():
    h = set(nyse_holidays(2000, 2035).strftime("%Y-%m-%d"))
    assert "2021-12-31" not in h and "2022-01-03" not in h  # NY Day on Sat: not observed
    assert "2017-01-02" in h  # NY Day on Sunday -> Monday
    assert "2021-12-24" in h and "2022-12-26" in h  # Christmas Sat -> Fri, Sun -> Mon
    assert "2021-07-05" in h and "2020-07-03" in h  # Independence Sun -> Mon, Sat -> Fri
    assert "2022-06-20" in h and "2027-06-18" in h and "2021-06-18" not in h  # Juneteenth from 2022
    assert "2001-09-11" in h and "2012-10-29" in h and "2018-12-05" in h
    assert easter_sunday(2024) == pd.Timestamp("2024-03-31").date()
    assert easter_sunday(2000) == pd.Timestamp("2000-04-23").date()
    assert not is_session("2025-01-09") and is_session("2025-01-10")


def test_add_sessions_and_finra_lag():
    # audit repro: 2024-12-31 + 8 sessions skips 1 Jan and 9 Jan 2025
    assert finra_publication_date("2024-12-31", 8) == pd.Timestamp("2025-01-14")
    assert pd.Timestamp("2024-12-31") + pd.offsets.BDay(8) == pd.Timestamp("2025-01-10")  # the old answer
    idx = add_sessions(pd.DatetimeIndex(["2024-07-03", "2024-07-04", "NaT"]), 1)
    assert idx[0] == pd.Timestamp("2024-07-05") and idx[1] == pd.Timestamp("2024-07-05") and pd.isna(idx[2])
    assert add_sessions("2024-03-28", 1) == pd.Timestamp("2024-04-01")  # Good Friday skipped
    assert add_sessions("2024-03-28", 0) == pd.Timestamp("2024-03-28")
    with pytest.raises(ValueError):
        add_sessions("1999-06-01", 1)
