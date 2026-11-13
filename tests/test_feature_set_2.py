"""Feature set 2 (``AlphaConfig.feature_set = "v2"``): the earnings tables
(``qts.data.earnings``), the eight features (``qts.alpha.features_v2``) against
hand-computed values, their point-in-time rules, the wiring (config, roles, cache
key, store, CLI, diagnostics) and "v1 is bitwise what it was".
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from qts.alpha import FEATURE_ORDER, AlphaModel, FeatureBuilder, feature_order
from qts.alpha import features_v2 as v2
from qts.alpha.features import CONDITIONING_FEATURES, V2_INDUSTRY_DEMEAN
from qts.alpha.models import RidgeModel
from qts.core import roles
from qts.core.config import AlphaConfig, SystemConfig, UniverseConfig
from qts.core.manifest import hash_market_data
from qts.core.types import MarketData
from qts.data import earnings as E
from qts.data import sharadar as sh
from qts.data.store import ParquetStore
from qts.data.synthetic import SyntheticSpec, generate
from qts.data.universe import apply_mask, build_universe_mask
from qts.pipeline import ResearchPipeline

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "sharadar"
V2 = list(roles.FEATURE_SET_2_FEATURES)
V1_ORDER = [
    "value_bp", "value_ep", "mom_12_1", "mom_6_1", "mom_change", "industry_momentum", "quality_gpa",
    "investment_ag", "beta_252", "idio_vol_252", "max_ret_21", "turnover_63", "amihud_63",
    "log_dollar_vol_63", "log_mcap", "short_interest", "borrow_fee", "strev_21",
]  # fmt: skip


# ============================================================================= helpers
def tiny_md(n_days: int = 420, n: int = 6, seed: int = 0) -> MarketData:
    """Hand-checkable market: random-walk closes, constant shares, noisy volume."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    tickers = pd.Index([f"T{i}" for i in range(n)], name="ticker")
    close = pd.DataFrame(
        50 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_days, n)), axis=0)), index=dates, columns=tickers
    )
    volume = pd.DataFrame(
        rng.integers(10_000, 50_000, (n_days, n)).astype(float), index=dates, columns=tickers
    )
    shares = pd.DataFrame(1e6, index=dates, columns=tickers)
    return MarketData(
        close=close,
        volume=volume,
        market_cap=close * shares,
        industry=pd.Series("IND", index=tickers),
        close_raw=close.copy(),
        shares_outstanding=shares,
    )


def ann_table(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "announce_date": pd.to_datetime([r[1] for r in rows]),
            "source": "events",
            "datekey": pd.NaT,
            "calendardate": pd.NaT,
        }
    )


def fund_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in E.EARNINGS_FUNDAMENTALS_COLUMNS:
        if c not in df.columns:
            df[c] = 1.0 if c == "split_cum" else np.nan
    for c in ("datekey", "calendardate"):
        df[c] = pd.to_datetime(df[c])
    return df[list(E.EARNINGS_FUNDAMENTALS_COLUMNS)]


def quarters(first: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(first, periods=n, freq="QE")


def market_of(md: MarketData) -> pd.Series:
    return md.returns.where(md.eligibility()).mean(axis=1)


def reaction(md: MarketData, ed: pd.DataFrame) -> dict[str, pd.DataFrame]:
    m = dataclasses.replace(md, earnings_dates=ed)
    return v2.earnings_reaction_panels(m, market_of(m), v2.base_dollar_volume(m))


# ============================================================================= earnings dates
def test_parse_event_codes():
    assert E.parse_event_codes("22|91") == ("22", "91")
    assert E.parse_event_codes("22") == ("22",) and E.parse_event_codes(22.0) == ("22",)
    assert E.parse_event_codes(" 13, 22 ; 91 ") == ("13", "22", "91")
    assert E.parse_event_codes(None) == () and E.parse_event_codes(float("nan")) == ()


def _arq(rows):
    return pd.DataFrame(rows, columns=["ticker", "datekey", "calendardate"])


def test_announcement_is_the_event_and_the_filing_date_is_the_fallback():
    arq = _arq(
        [
            ("A", "2020-02-10", "2019-12-31"),  # 8-K five days earlier
            ("A", "2020-05-08", "2020-03-31"),  # no 8-K at all -> filing date
            (
                "A",
                "2020-08-07",
                "2020-06-30",
            ),  # 8-K 60 days earlier: outside the 45-day window -> filing date
            ("B", "2020-02-20", "2019-12-31"),  # 8-K the same day
            ("B", "2020-03-01", "2020-03-31"),  # the only 8-K in the window predates the previous filing
            ("C", "2020-02-12", "2019-12-31"),  # 8-K with another code only
        ]
    )
    ev = pd.DataFrame(
        [
            ("A", "2020-02-05", "22|91"),
            ("A", "2020-06-08", "22"),  # 60 days before the August filing
            ("A", "2020-12-01", "22"),  # no filing follows: still an announcement
            ("B", "2020-02-20", "91|22"),
            ("C", "2020-02-11", "52|91"),
        ],
        columns=["ticker", "date", "eventcodes"],
    )
    out, rep = E.derive_earnings_dates(arq, ev)
    got = {(r.ticker, r.announce_date.date().isoformat(), r.source) for r in out.itertuples()}
    assert got == {
        ("A", "2020-02-05", "events"),
        ("A", "2020-05-08", "datekey"),
        ("A", "2020-06-08", "events"),
        ("A", "2020-08-07", "datekey"),
        ("A", "2020-12-01", "events"),
        ("B", "2020-02-20", "events"),
        ("B", "2020-03-01", "datekey"),
        ("C", "2020-02-12", "datekey"),
    }
    a = out[(out["ticker"] == "A")].set_index("announce_date")
    assert a.loc["2020-02-05", "datekey"] == pd.Timestamp("2020-02-10")  # informational link
    assert pd.isna(a.loc["2020-06-08", "datekey"]) and pd.isna(a.loc["2020-12-01", "datekey"])
    assert rep.has_events and rep.n_filings == 6 and rep.n_filings_with_event == 2
    assert rep.n_event_rows == 4 and rep.n_datekey_rows == 4 and rep.n_event_rows_without_filing == 2
    assert rep.code_table["22"]["rows"] == 4 and rep.code_table["91"]["rows"] == 3
    assert "52" in rep.code_table and "<- earnings" in rep.format()
    assert list(out.columns) == list(E.EARNINGS_DATES_COLUMNS)


def test_without_events_every_announcement_is_the_filing_date_and_the_report_says_how_to_add_them():
    arq = _arq([("A", "2020-02-10", "2019-12-31"), ("A", "2020-05-08", "2020-03-31")])
    for ev in (None, pd.DataFrame(columns=["ticker", "date", "eventcodes"])):
        out, rep = E.derive_earnings_dates(arq, ev)
        assert (out["source"] == "datekey").all() and (out["announce_date"] == out["datekey"]).all()
        assert not rep.has_events
        txt = rep.format()
        assert "EVENTS is NOT in this store" in txt and "--tables EVENTS --from-files" in txt


def test_the_series_as_of_t_is_the_series_restricted_to_t():
    """No row is selected with later information: deriving from inputs cut at t gives
    the rows dated <= t of the full derivation (announcement date and source)."""
    md = generate(SyntheticSpec(n_stocks=20, n_days=500, seed=2, with_earnings=True))
    ed, ef = md.earnings_dates, md.earnings_fundamentals
    arq = ef[ef["dimension"] == "ARQ"]
    events = ed.loc[ed["source"] == "events", ["ticker", "announce_date"]].rename(
        columns={"announce_date": "date"}
    )
    events = events.assign(eventcodes="22")
    full, _ = E.derive_earnings_dates(arq, events)
    key = ["ticker", "announce_date", "source"]
    pd.testing.assert_frame_equal(full[key], ed[key].reset_index(drop=True))
    for t in pd.to_datetime(["2018-06-15", "2019-03-29", "2019-11-01"]):
        cut, _ = E.derive_earnings_dates(arq[arq["datekey"] <= t], events[events["date"] <= t])
        ref = full.loc[full["announce_date"] <= t, key].reset_index(drop=True)
        pd.testing.assert_frame_equal(cut[key], ref)


def test_cumulative_split_factor_and_price_confirmation():
    splits = pd.DataFrame(
        {"ticker": ["A", "A", "B"], "date": pd.to_datetime(["2020-03-02", "2021-03-01", "2020-06-01"]),
         "ratio": [2.0, 3.0, 0.1]}
    )  # fmt: skip
    tk = ["A", "A", "A", "A", "B", "B", "C"]
    dt = ["2020-03-01", "2020-03-02", "2021-02-28", "2021-06-01", "2020-05-31", "2020-06-01", "2020-06-01"]
    got = E.cumulative_split_factor(tk, dt, splits)
    np.testing.assert_allclose(got, [1.0, 2.0, 2.0, 6.0, 1.0, 0.1, 1.0])
    assert E.cumulative_split_factor(tk, dt, None).tolist() == [1.0] * 7
    # vendor value given as the ratio (2-for-1: raw halves, adjusted continuous) ...
    px = pd.DataFrame(
        {"ticker": ["A", "B", "C", "D"], "date": pd.to_datetime(["2020-03-02"] * 4), "value": [2.0, 0.5, 4.0, 3.0],
         "raw_prev": [100.0, 100.0, 100.0, np.nan], "raw": [51.0, 49.0, 80.0, np.nan],
         "adj_prev": [50.0, 50.0, 50.0, np.nan], "adj": [51.0, 49.0, 40.0, np.nan]}
    )  # fmt: skip
    out, c = E.confirm_split_ratios(px)
    # A confirmed as given; B is the inverse convention (0.5 for a 2-for-1); C neither; D no price
    np.testing.assert_allclose(out["ratio"], [2.0, 2.0, 4.0, 3.0])
    assert (c["as_given"], c["inverse"], c["unconfirmed"], c["no_price"]) == (1, 1, 1, 1)
    # ... a vendor whose values are mostly the inverse: unconfirmed rows follow the majority
    inv = px.assign(value=[0.5, 0.5, 4.0, 0.25])
    out, c = E.confirm_split_ratios(inv)
    assert c["orientation"] == "inverse" and c["inverse"] == 2
    np.testing.assert_allclose(out["ratio"], [2.0, 2.0, 0.25, 4.0])


# ============================================================================= SUE
def _arq_rows(ticker: str, eps, first="2016-03-31", lag_days=35, split_cum=None) -> pd.DataFrame:
    q = quarters(first, len(eps))
    return pd.DataFrame(
        {
            "ticker": ticker,
            "dimension": "ARQ",
            "datekey": q + pd.Timedelta(days=lag_days),
            "calendardate": q,
            "eps": np.asarray(eps, dtype=float),
            "split_cum": 1.0 if split_cum is None else np.asarray(split_cum, dtype=float),
        }
    )


def test_sue_matches_the_hand_computation_and_needs_four_differences():
    eps = np.array([1.0, 1.2, 0.9, 1.5, 1.1, 1.25, 1.0, 1.7, 1.3, 1.2, 1.4, 1.6, 1.0, 1.9])
    got = v2.sue_by_filing(_arq_rows("A", eps))["sue"].to_numpy()
    u = eps[4:] - eps[:-4]  # u[k] belongs to quarter k + 4
    assert np.isnan(got[:7]).all()  # quarters 0-3: no seasonal lag; 4-6: fewer than four differences
    for k in range(7, len(eps)):
        window = u[max(0, k - 4 - 7) : k - 4 + 1]  # the trailing eight differences, quarter k included
        assert got[k] == pytest.approx(u[k - 4] / np.std(window, ddof=1), rel=1e-12)
    # by hand for quarter 7: differences 0.10, 0.05, 0.10, 0.20 -> sd 0.062915...
    assert got[7] == pytest.approx(0.2 / np.std([0.1, 0.05, 0.1, 0.2], ddof=1))
    assert got[7] == pytest.approx(3.178878, abs=1e-6)  # 0.2 / sqrt(0.011875 / 3)


def test_sue_is_unchanged_by_a_split_inside_the_window():
    """Sharadar EPS is as reported: after a 2-for-1 split it halves.  ``split_cum``
    (splits dated <= the filing) puts every quarter on the original share basis."""
    e_true = np.array([1.0, 1.2, 0.9, 1.5, 1.1, 1.25, 1.0, 1.7, 1.3, 1.2, 1.4, 1.6, 1.0, 1.9])
    base = _arq_rows("A", e_true)
    ref = v2.sue_by_filing(base)["sue"].to_numpy()
    # 2-for-1 after the 9th filing, then 3-for-1 after the 12th
    splits = pd.DataFrame(
        {"ticker": "A", "date": [base["datekey"][8] + pd.Timedelta(days=10), base["datekey"][11] + pd.Timedelta(days=5)],
         "ratio": [2.0, 3.0]}
    )  # fmt: skip
    cum = E.cumulative_split_factor(base["ticker"], base["datekey"], splits)
    assert cum.tolist() == [1.0] * 9 + [2.0] * 3 + [6.0] * 2
    reported = base.assign(eps=e_true / cum, split_cum=cum)
    got = v2.sue_by_filing(reported)["sue"].to_numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-12, equal_nan=True)
    # without the adjustment the split quarters are wrong (the test would be vacuous otherwise)
    naive = v2.sue_by_filing(reported.assign(split_cum=1.0))["sue"].to_numpy()
    assert np.nanmax(np.abs(naive[9:] - ref[9:])) > 0.5
    # a split *after* the last filing cannot change anything: split_cum never looks forward
    later = pd.concat(
        [splits, pd.DataFrame({"ticker": "A", "date": [pd.Timestamp("2030-01-01")], "ratio": [5.0]})]
    )
    assert E.cumulative_split_factor(base["ticker"], base["datekey"], later).tolist() == cum.tolist()


def test_sue_ignores_a_quarter_filed_after_the_current_one_and_keeps_the_first_filing():
    eps = np.array([1.0, 1.2, 0.9, 1.5, 1.1, 1.25, 1.0, 1.7, 1.3, 1.2])
    rows = _arq_rows("A", eps)
    ref = v2.sue_by_filing(rows)
    # the quarter four back from quarter 9 is filed late, after quarter 9 itself
    late = rows.copy()
    late.loc[5, "datekey"] = rows.loc[9, "datekey"] + pd.Timedelta(days=3)
    got = v2.sue_by_filing(late).set_index("calendardate")["sue"]
    assert np.isnan(got.iloc[9])  # its seasonal lag was not known at the filing
    # quarter 8: the difference of quarter 5 (not filed yet) leaves the window -> 0.1, 0.1, 0.2, 0.2
    assert got.iloc[8] == pytest.approx(0.2 / np.std([0.1, 0.1, 0.2, 0.2], ddof=1))
    assert got.iloc[8] != pytest.approx(ref["sue"].iloc[8])
    assert np.isnan(got.iloc[7]) and np.isfinite(ref["sue"].iloc[7])  # only three known differences
    # a restatement of a period (second row, later datekey) does not replace the first filing
    dup = pd.concat(
        [rows, rows.iloc[[6]].assign(eps=99.0, datekey=rows.loc[9, "datekey"])], ignore_index=True
    )
    np.testing.assert_allclose(v2.sue_by_filing(dup)["sue"], ref["sue"], equal_nan=True)


def test_sue_panel_is_visible_the_day_after_the_filing_and_expires():
    md = tiny_md(n_days=520)
    eps = np.array([1.0, 1.2, 0.9, 1.5, 1.1, 1.25, 1.0, 1.7, 1.3, 1.2, 1.4, 1.6])
    rows = _arq_rows("T0", eps, first="2017-06-30", lag_days=40)  # last filing 2020-05-10 (a Sunday)
    by = v2.sue_by_filing(rows)
    panel = v2.filing_panel(md, by, "sue", max_age=v2.EARNINGS_MAX_AGE_DAYS)["T0"]
    dk10 = rows["datekey"][10]
    assert dk10 == pd.Timestamp("2020-02-09") and dk10.weekday() == 6  # a Sunday
    s9, s10 = by["sue"][9], by["sue"][10]
    assert np.isfinite(s9) and np.isfinite(s10) and s9 != s10
    # weekend filing: first usable on the Monday; the Friday before still shows the previous quarter
    assert panel.loc["2020-02-07"] == s9 and panel.loc["2020-02-10"] == s10
    # a weekday filing dated d is visible from d + 1
    wk = rows.assign(datekey=rows["datekey"].where(rows.index != 10, pd.Timestamp("2020-02-12")))
    p2 = v2.filing_panel(md, v2.sue_by_filing(wk), "sue", max_age=None)["T0"]
    assert p2.loc["2020-02-12"] == s9 and p2.loc["2020-02-13"] == s10
    # expiry: 126 trading days after the row it became visible on
    last = rows["datekey"][11]
    pos = md.dates.searchsorted(last + pd.Timedelta(days=1))
    assert np.isfinite(panel.iloc[pos + 126]) and np.isnan(panel.iloc[pos + 127])
    assert np.isnan(v2.filing_panel(md, by, "sue", max_age=None)["T1"]).all()


# ============================================================================= earnings reaction
def test_ear_3d_hand_computed_and_visible_only_from_a_plus_two():
    md = tiny_md()
    dates = md.dates
    p1, p2 = 150, 213
    out = reaction(md, ann_table([("T0", str(dates[p1].date())), ("T0", str(dates[p2].date()))]))
    close, mkt = md.close["T0"].to_numpy(), market_of(md).to_numpy()

    def ear(p):
        return close[p + 1] / close[p - 2] - 1 - (np.prod(1 + mkt[p - 1 : p + 2]) - 1)

    e = out["ear_3d"]["T0"].to_numpy()
    assert np.isnan(e[: p1 + 2]).all()
    assert e[p1 + 2] == pytest.approx(ear(p1), rel=1e-12) and e[p2 - 1] == e[p1 + 2]
    # on a and a+1 the previous announcement's value is still shown (after-close announcements)
    assert e[p2] == e[p1 + 2] and e[p2 + 1] == e[p1 + 2]
    assert e[p2 + 2] == pytest.approx(ear(p2), rel=1e-12) and e[p2 + 2] != e[p1 + 2]
    d = out["days_since_earnings"]["T0"].to_numpy()
    assert np.isnan(d[: p1 + 2]).all() and d[p1 + 2] == 2 and d[p1 + 10] == 10
    assert d[p2 + 1] == p2 + 1 - p1 and d[p2 + 2] == 2
    # expiry / cap at 126 trading days after the announcement
    assert np.isfinite(e[p2 + 126]) and np.isnan(e[p2 + 127]) and d[p2 + 126] == 126 and d[p2 + 200] == 126
    # names without announcements
    assert out["ear_3d"]["T1"].isna().all() and out["days_since_earnings"]["T1"].isna().all()


def test_announcement_on_a_non_trading_day_and_at_the_panel_edges():
    md = tiny_md()
    dates = md.dates
    sat = dates[200] + pd.Timedelta(days=(5 - dates[200].weekday()) % 7 or 7)
    assert sat.weekday() == 5
    p = int(dates.searchsorted(sat))  # the Monday
    edge = [
        ("T1", str(dates[1].date())),
        ("T2", str(dates[-1].date())),
        ("T3", "2010-01-04"),
        ("T3", "2031-01-06"),
    ]
    out = reaction(md, ann_table([("T0", str(sat.date())), *edge]))
    e = out["ear_3d"]["T0"]
    assert e.iloc[: p + 2].isna().all() and np.isfinite(e.iloc[p + 2])
    close, mkt = md.close["T0"].to_numpy(), market_of(md).to_numpy()
    assert e.iloc[p + 2] == pytest.approx(
        close[p + 1] / close[p - 2] - 1 - (np.prod(1 + mkt[p - 1 : p + 2]) - 1)
    )
    # window not inside the panel -> no reaction, but the announcement still counts for the clock
    assert out["ear_3d"]["T1"].isna().all() and out["days_since_earnings"]["T1"].iloc[3] == 2
    assert out["days_since_earnings"]["T2"].isna().all()  # visible only after the panel ends
    assert out["days_since_earnings"]["T3"].isna().all()  # outside the panel on both sides


def test_abnormal_volume_hand_computed():
    md = tiny_md()
    dates = md.dates
    dv = (md.close * md.volume)["T0"].to_numpy()
    base = v2.base_dollar_volume(md)
    a5 = v2.abnormal_volume_5d(md, base)["T0"].to_numpy()
    t = 300
    assert a5[t] == pytest.approx(np.log(dv[t - 4 : t + 1].mean() / dv[t - 256 : t - 4].mean()), rel=1e-12)
    # needs 126 base observations before the five-day window
    assert np.isnan(a5[: 125 + 5]).all() and np.isfinite(a5[125 + 5])
    assert a5[130] == pytest.approx(np.log(dv[126:131].mean() / dv[:126].mean()), rel=1e-12)
    p = 280
    out = reaction(md, ann_table([("T0", str(dates[p].date()))]))["abn_volume_earn"]["T0"].to_numpy()
    want = np.log(dv[p - 1 : p + 2].mean() / dv[p - 253 : p - 1].mean())
    assert np.isnan(out[: p + 2]).all() and out[p + 2] == pytest.approx(want, rel=1e-12)
    assert out[p + 50] == out[p + 2]  # carried forward like ear_3d


# ============================================================================= accruals / issuance / high
def test_accruals_hand_computed_and_pit():
    q = quarters("2018-03-31", 6)
    rows = pd.DataFrame(
        {
            "ticker": "T0",
            "dimension": "ART",
            "datekey": q + pd.Timedelta(days=40),
            "calendardate": q,
            "netinc": [100.0, 110.0, 120.0, 130.0, 150.0, 90.0],
            "ncfo": [130.0, 120.0, 100.0, 160.0, 120.0, 140.0],
            "assets": [1000.0, 1040.0, 1080.0, 1120.0, 1200.0, 1300.0],
        }
    )
    by = v2.accruals_by_filing(rows)["accruals"].to_numpy()
    assert np.isnan(by[:4]).all()  # no year-earlier assets
    assert by[4] == pytest.approx((150.0 - 120.0) / ((1200.0 + 1000.0) / 2))
    assert by[5] == pytest.approx((90.0 - 140.0) / ((1300.0 + 1040.0) / 2))
    md = tiny_md()
    panel = v2.filing_panel(md, v2.accruals_by_filing(rows), "accruals", max_age=None)["T0"]
    dk4, dk5 = rows["datekey"][4], rows["datekey"][5]  # 2019-05-10 (Fri), 2019-08-09 (Fri)
    assert panel.loc[:dk4].isna().all()  # on the filing date itself: not yet visible
    assert panel.loc[dk4 + pd.Timedelta(days=3)] == pytest.approx(by[4])  # the Monday
    assert panel.loc[dk5] == pytest.approx(by[4]) and panel.loc[dk5 + pd.Timedelta(days=3)] == pytest.approx(
        by[5]
    )
    assert panel.iloc[-1] == pytest.approx(by[5])  # carried like every fundamental (no expiry)


def test_net_issuance_sign_and_split_invariance():
    md = tiny_md(n_days=480)
    # T0 issues 25 % new shares on day 200; T1 splits 2-for-1 on day 220 (shares double, raw price
    # halves, market cap and adjusted close unchanged)
    shares = md.shares_outstanding.copy()
    shares.iloc[200:, 0] *= 1.25
    shares.iloc[220:, 1] *= 2.0
    raw = md.close_raw.copy()
    raw.iloc[220:, 1] /= 2.0
    m = dataclasses.replace(md, shares_outstanding=shares, close_raw=raw, market_cap=raw * shares)
    ni = v2.net_issuance(m)
    assert ni.iloc[:252].isna().all().all()
    assert ni["T0"].iloc[260] == pytest.approx(-np.log(1.25))  # issuance is bad: negative
    assert ni["T0"].iloc[200 + 252] == 0.0  # the issue has left the window
    assert (ni["T1"].iloc[252:] == 0.0).all() and (
        ni["T2"].iloc[252:] == 0.0
    ).all()  # a split is not issuance
    # a buy-back is good
    shares2 = md.shares_outstanding.copy()
    shares2.iloc[300:, 2] *= 0.9
    m2 = dataclasses.replace(md, shares_outstanding=shares2, market_cap=md.close * shares2)
    assert v2.net_issuance(m2)["T2"].iloc[310] == pytest.approx(-np.log(0.9)) and -np.log(0.9) > 0


def test_high_52w_hand_computed():
    md = tiny_md()
    h = v2.high_52w(md)["T3"].to_numpy()
    c = md.close["T3"].to_numpy()
    assert np.isnan(h[:125]).all() and h[125] == pytest.approx(c[125] / c[:126].max())
    assert h[400] == pytest.approx(c[400] / c[149:401].max())
    allh = v2.high_52w(md).to_numpy()
    assert np.nanmax(allh) == 1.0 and (allh[np.isfinite(allh)] > 0).all()  # at the high: exactly 1


# ============================================================================= wiring
@pytest.fixture(scope="module")
def raw_md() -> MarketData:
    return generate(
        SyntheticSpec(n_stocks=60, n_days=420, seed=5, ic_true=0.08, n_splits=6, with_earnings=True)
    )


@pytest.fixture(scope="module")
def md(raw_md) -> MarketData:
    return apply_mask(raw_md, build_universe_mask(raw_md, UniverseConfig(top_n_by_mcap=45)))


@pytest.fixture(scope="module")
def full_features(md):
    fb = FeatureBuilder(AlphaConfig(feature_set="v2"))
    raw = fb.build(md)
    return raw, fb.process(raw, md, valid=md.eligibility())


def test_feature_sets_and_roles():
    assert FEATURE_ORDER == V1_ORDER == feature_order("v1")  # v1 untouched, same object contents
    assert feature_order("v2") == V1_ORDER + V2 and len(V2) == 8
    assert V2 == ["ear_3d", "sue", "days_since_earnings", "abn_volume_5d", "abn_volume_earn", "accruals",
                  "net_issuance", "high_52w"]  # fmt: skip
    with pytest.raises(ValueError, match="unknown feature_set"):
        feature_order("v3")
    with pytest.raises(ValueError, match="feature_set"):
        AlphaConfig(feature_set="v3")
    assert AlphaConfig().feature_set == "v1" and SystemConfig().alpha.feature_set == "v1"
    # every new feature is an alpha feature, none proxies a risk-only style
    assert set(V2) <= roles.ALPHA_THEME_FEATURES and not set(V2) & set(roles.RISK_ONLY_FEATURES)
    assert V2_INDUSTRY_DEMEAN == set(V2) - {"days_since_earnings"}
    assert CONDITIONING_FEATURES == {"days_since_earnings": (126.0, 1.0)}
    # the composite is the same under both sets: no theme reads a v2 feature
    from qts.alpha.composite import THEMES

    assert not {f for members in THEMES.values() for f, _ in members} & set(V2)


def test_processing_ranks_seven_and_only_scales_the_conditioning_variable(md, full_features):
    raw, X = full_features
    assert list(X.columns) == V1_ORDER + V2 and not X.isna().any().any()
    d = X["days_since_earnings"]
    assert d.min() >= 2 / 126 - 1e-12 and d.max() <= 1.0
    elig = md.eligibility()
    t = md.dates[300]
    names = elig.columns[elig.loc[t]]
    want = (raw["days_since_earnings"].loc[t, names] / 126.0).fillna(1.0)
    pd.testing.assert_series_equal(d.xs(t, level="date").reindex(names), want, check_names=False)
    # the other seven are rank-Gaussian per date: symmetric, bounded, mean ~ 0
    for f in set(V2) - {"days_since_earnings"}:
        x = X[f].xs(t, level="date")
        assert abs(x.mean()) < 0.2 and x.abs().max() < 3.8, f
    # a name with no announcement yet gets the cap (1.0), not 0 ("just announced")
    none = dataclasses.replace(md, earnings_dates=md.earnings_dates.iloc[:0])
    fb = FeatureBuilder(AlphaConfig(feature_set="v2"))
    Xn = fb.process(fb.build(none), none)
    assert (Xn["days_since_earnings"] == 1.0).all() and (Xn["ear_3d"] == 0.0).all()


def test_v1_is_bitwise_unchanged_by_the_earnings_tables(raw_md):
    """Same panels with and without the tables; v1 features do not read them; the
    data hash without the tables is the hash the generator had before they existed."""
    plain = generate(SyntheticSpec(n_stocks=60, n_days=420, seed=5, ic_true=0.08, n_splits=6))
    assert plain.earnings_dates is None and plain.earnings_fundamentals is None
    for f in dataclasses.fields(plain):
        a, b = getattr(plain, f.name), getattr(raw_md, f.name)
        if isinstance(a, pd.DataFrame) and f.name not in MarketData.DATED_TABLES:
            assert a.equals(b), f.name
    tables = frozenset(MarketData.DATED_TABLES)
    assert hash_market_data(raw_md, exclude=tables) == hash_market_data(plain)
    assert hash_market_data(raw_md) != hash_market_data(plain)
    fb = FeatureBuilder(AlphaConfig())
    X0, X1 = fb.build_and_process(plain), fb.build_and_process(raw_md)
    assert list(X0.columns) == V1_ORDER and X0.equals(X1)
    # and the v1 columns of a v2 frame are the v1 frame, bit for bit
    X2 = FeatureBuilder(AlphaConfig(feature_set="v2")).build_and_process(raw_md)
    assert X2[V1_ORDER].equals(X0)


def test_v2_without_its_inputs_fails_naming_the_command(raw_md, tmp_path, capsys):
    plain = dataclasses.replace(raw_md, earnings_dates=None, earnings_fundamentals=None)
    fb = FeatureBuilder(AlphaConfig(feature_set="v2"))
    with pytest.raises(v2.FeatureSetDataError, match="qts data sharadar derive-earnings --store"):
        fb.build(plain)
    cfg = SystemConfig().with_overrides({"alpha.feature_set": "v2"})
    with pytest.raises(v2.FeatureSetDataError, match="earnings_dates / earnings_fundamentals"):
        ResearchPipeline(cfg, use_ml=False, runs_root=None).fit(plain)  # before any fit
    # a store synced before SF1 kept `ncfo`: accruals cannot be built
    no_cf = dataclasses.replace(
        raw_md, earnings_fundamentals=raw_md.earnings_fundamentals.assign(ncfo=np.nan)
    )
    with pytest.raises(v2.FeatureSetDataError, match="sync --store <store> --tables SF1 --from-files"):
        fb.build(no_cf)
    # the CLI stops with exit code 2 before fitting and names the store
    from qts.cli import main

    store = tmp_path / "store"
    ParquetStore(str(store)).write_market_data(plain)
    rc = main(["run", "--store", str(store), "--set", "alpha.feature_set=v2", "--no-save"])
    err = capsys.readouterr().err
    assert rc == 2 and f"derive-earnings --store {store}" in err


def test_store_round_trip_keeps_the_earnings_tables(raw_md, tmp_path):
    store = ParquetStore(str(tmp_path / "store"))
    store.write_market_data(raw_md)
    back = store.as_market_data()
    for name, col in MarketData.DATED_TABLES.items():
        a = getattr(raw_md, name)
        b = getattr(back, name)
        by = [c for c in ("ticker", "dimension", col, "calendardate") if c in a.columns]
        a = a.sort_values(by, kind="mergesort").reset_index(drop=True)
        pd.testing.assert_frame_equal(b[a.columns], a, check_dtype=False)
    fb = FeatureBuilder(AlphaConfig(feature_set="v2"))
    r0, r1 = fb.build(raw_md), fb.build(back)
    for k in V2:
        pd.testing.assert_frame_equal(r1[k], r0[k], check_freq=False, check_names=False)
    # only the loaded ids and rows up to `end` are read
    sub = store.as_market_data(universe_tickers=list(raw_md.tickers[:7]), end="2019-01-31")
    assert set(sub.earnings_dates["ticker"]) <= set(raw_md.tickers[:7])
    assert sub.earnings_dates["announce_date"].max() <= pd.Timestamp("2019-01-31")
    assert sub.earnings_fundamentals["datekey"].max() <= pd.Timestamp("2019-01-31")
    assert sub.earnings_fundamentals["datekey"].min() < sub.dates[0]  # pre-sample quarters kept


def test_stage_cache_key_follows_the_feature_set_on_the_alpha_fit_only(raw_md, tmp_path):
    def keys(md_, **over):
        cfg = SystemConfig().with_overrides({"cache.enabled": True, "cache.root": str(tmp_path), **over})
        pipe = ResearchPipeline(cfg, use_ml=False, runs_root=None)
        cache = pipe._stage_cache()
        full = hash_market_data(
            md_, n_chars=64, exclude=frozenset({"delisting_reasons", *MarketData.DATED_TABLES})
        )
        rk = pipe._cache_key(cache, "risk_fit", md_, full)
        return rk, pipe._cache_key(cache, "alpha_fit", md_, full, parent=rk)

    plain = dataclasses.replace(raw_md, earnings_dates=None, earnings_fundamentals=None)
    r1, a1 = keys(raw_md)
    r2, a2 = keys(raw_md, **{"alpha.feature_set": "v2"})
    assert r1 == r2 and a1 != a2  # the config value never touches the risk fit
    # adding / re-deriving the tables leaves the risk fit and a v1 alpha fit valid ...
    assert keys(plain) == (r1, a1)
    # ... and re-keys a v2 alpha fit
    other = dataclasses.replace(raw_md, earnings_dates=raw_md.earnings_dates.iloc[:-5])
    r3, a3 = keys(other, **{"alpha.feature_set": "v2"})
    assert r3 == r1 and a3 != a2
    from qts.runs.cache import STAGE_EXCLUDED_DATA

    assert set(MarketData.DATED_TABLES) <= STAGE_EXCLUDED_DATA


def test_v2_alpha_model_records_the_new_features(md):
    """Feature IC table and the refit importances read the live feature list."""
    cfg = AlphaConfig(
        feature_set="v2", horizon_days=10, train_min_days=150, refit_frequency_days=120, valid_days=40
    )
    am = AlphaModel(cfg, ml_factory=lambda: RidgeModel()).fit(md)
    assert list(am.X_.columns) == V1_ORDER + V2
    tab = am.feature_ic_table()
    assert list(tab["feature"]) == V1_ORDER + V2 and (tab.set_index("feature").loc[V2, "theme"] == "").all()
    # (net_issuance is constant on the synthetic market -- share counts never change -- so it has no IC)
    assert tab.set_index("feature").loc[[f for f in V2 if f != "net_issuance"], "n"].min() > 100
    assert am.refit_records_ and all(set(V2) <= set(r.importance["ridge"].index) for r in am.refit_records_)
    assert am.trainer.folds_[0].extra["feature_hash"] == __import__(
        "qts.alpha", fromlist=["x"]
    ).feature_signature(V1_ORDER + V2)
    # the composite does not read the new features: identical to the v1 composite
    v1 = AlphaModel(dataclasses.replace(cfg, feature_set="v1"), use_ml=False).fit(md)
    assert am.composite_score_.equals(v1.composite_score_)
    out = am.as_of(md.dates[-1])
    assert out.score.notna().all() and len(out.score) > 20


@pytest.mark.slow
def test_v2_pipeline_end_to_end_with_the_ml_sleeve(tmp_path):
    md_ = generate(SyntheticSpec(n_stocks=120, n_days=1000, seed=1, with_earnings=True, n_splits=5))
    cfg = SystemConfig().with_overrides({"alpha.feature_set": "v2", "alpha.model_seeds": 1})
    res = ResearchPipeline(cfg, use_ml=True, runs_root=str(tmp_path)).run(md_)
    assert np.isfinite(res.backtest.summary()["sharpe_net"])

    run_dir = next(p for p in tmp_path.iterdir() if (p / "alpha_refits.json").exists())
    txt = (run_dir / "alpha_refits.json").read_text()
    assert all(f in txt for f in V2)
    assert "feature_set: v2" in (run_dir / "config.yaml").read_text()
    assert set(V2) <= set(pd.read_parquet(run_dir / "alpha_refits.parquet")["feature"])


# ============================================================================= point in time
FAST = settings(
    max_examples=12,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
    database=None,
)
cut_pos = st.integers(min_value=40, max_value=418)


@FAST
@given(p=cut_pos)
def test_v2_features_pit_under_truncation(md, full_features, p):
    """Features computed on ``md.slice_until(t)`` equal rows <= t of the full build, exactly."""
    t = md.dates[p]
    raw_full, X_full = full_features
    fb = FeatureBuilder(AlphaConfig(feature_set="v2"))
    cut = md.slice_until(t)
    assert cut.earnings_dates["announce_date"].max() <= t and cut.earnings_fundamentals["datekey"].max() <= t
    raw_cut = fb.build(cut)
    assert list(raw_cut) == list(raw_full)
    for name in V2:
        pd.testing.assert_frame_equal(
            raw_cut[name],
            raw_full[name].loc[:t],
            check_exact=True,
            check_freq=False,
            obj=f"{name} cut at {t.date()}",
        )
    X_cut = fb.process(raw_cut, cut, valid=cut.eligibility())
    X_ref = X_full.loc[X_full.index.get_level_values("date") <= t]
    pd.testing.assert_frame_equal(X_cut, X_ref, check_exact=True, check_freq=False)


@FAST
@given(p=cut_pos, seed=st.integers(min_value=0, max_value=10_000))
def test_perturbing_anything_after_t_never_changes_a_feature_at_or_before_t(md, full_features, p, seed):
    """Prices, volume, market cap and both earnings tables are scrambled after t
    (values, dates and extra rows); every v2 panel at rows <= t is unchanged."""
    t = md.dates[p]
    rng = np.random.default_rng(seed)
    after = np.asarray(md.dates > t)

    def scramble(panel: pd.DataFrame) -> pd.DataFrame:
        out = panel.copy()
        out.loc[after] = out.loc[after].to_numpy() * rng.uniform(
            0.3, 3.0, size=(int(after.sum()), panel.shape[1])
        )
        return out

    ed, ef = md.earnings_dates.copy(), md.earnings_fundamentals.copy()
    late = ed["announce_date"] > t
    # later announcements move (staying after t) and new ones appear
    ed.loc[late, "announce_date"] = ed.loc[late, "announce_date"] + pd.to_timedelta(
        rng.integers(0, 20, late.sum()), "D"
    )
    extra = pd.DataFrame(
        {"ticker": rng.choice(md.tickers, 30), "announce_date": t + pd.to_timedelta(rng.integers(1, 60, 30), "D"),
         "source": "events"}
    )  # fmt: skip
    ed = pd.concat([ed, extra], ignore_index=True)
    latef = ef["datekey"] > t
    for c in ("eps", "netinc", "ncfo", "assets", "split_cum"):
        ef.loc[latef, c] = ef.loc[latef, c] * rng.uniform(0.2, 5.0, latef.sum())
    # a later restatement of an already-filed quarter
    dup = ef[~latef].tail(40).assign(datekey=t + pd.Timedelta(days=7), eps=123.0, assets=1.0)
    ef = pd.concat([ef, dup], ignore_index=True)
    pert = dataclasses.replace(
        md,
        close=scramble(md.close),
        close_raw=scramble(md.close_raw),
        volume=scramble(md.volume),
        market_cap=scramble(md.market_cap),
        earnings_dates=ed,
        earnings_fundamentals=ef,
    )
    raw_full, _ = full_features
    raw = FeatureBuilder(AlphaConfig(feature_set="v2")).build(pert)
    changed_later = False
    for name in V2:
        pd.testing.assert_frame_equal(
            raw[name].loc[:t],
            raw_full[name].loc[:t],
            check_exact=True,
            check_freq=False,
            obj=f"{name} <= {t.date()}",
        )
        changed_later |= not raw[name].loc[t:].iloc[1:].equals(raw_full[name].loc[t:].iloc[1:])
    assert changed_later or p >= 415  # the perturbation is real: later rows do move


def test_on_a_and_a_plus_one_every_earnings_feature_shows_the_previous_quarter(md, full_features):
    """Explicit timing check on the synthetic market for every announcement of five names."""
    raw, _ = full_features
    dates = md.dates
    ed = md.earnings_dates
    checked = 0
    for name in md.tickers[:5]:
        col = raw["ear_3d"][name].to_numpy()
        vol = raw["abn_volume_earn"][name].to_numpy()
        days = raw["days_since_earnings"][name].to_numpy()
        ps = np.unique(dates.searchsorted(ed.loc[ed["ticker"] == name, "announce_date"].to_numpy()))
        ps = ps[(ps >= 3) & (ps + 2 < len(dates))]
        for p in ps:
            if any(q in ps for q in (p - 2, p - 1, p + 1, p + 2)):
                continue  # two announcements within two days: covered by the property test
            for arr in (col, vol):
                assert _same(arr[p], arr[p - 1]) and _same(arr[p + 1], arr[p - 1])  # previous value
            assert days[p + 2] == 2  # the clock restarts at the close of a+2 ...
            assert not (days[p] == 0 or days[p + 1] == 1)  # ... never on a or a+1
            if np.isfinite(days[p - 1]):
                assert days[p] == min(days[p - 1] + 1, 126) and days[p + 1] == min(days[p - 1] + 2, 126)
            checked += 1
    assert checked > 10
    # filings: on the datekey itself sue / accruals still show the previous filing
    ef = md.earnings_fundamentals
    for name in md.tickers[:5]:
        for dim, feat in (("ARQ", "sue"), ("ART", "accruals")):
            panel = raw[feat][name]
            dks = ef.loc[(ef["ticker"] == name) & (ef["dimension"] == dim), "datekey"]
            for dk in dks[(dks > dates[5]) & (dks < dates[-5])]:
                pos = int(dates.searchsorted(dk, side="right")) - 1  # last trading day <= datekey
                assert _same(panel.iloc[pos], panel.iloc[pos - 1]), (name, feat, dk)


def _same(a: float, b: float) -> bool:
    return (np.isnan(a) and np.isnan(b)) or a == b


# ============================================================================= Sharadar store
@pytest.fixture(scope="module")
def synced(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("earnings_store")
    rep = sh.sync(
        str(root / "store"),
        raw_dir=str(root / "raw"),
        from_files=str(FIX),
        tables=[*sh.DEFAULT_TABLES, "EVENTS"],
    )
    assert rep.earnings == {"announcements": 4, "from_events": 3, "from_datekey": 1}
    assert "earnings_dates: announcements 4" in rep.format()
    return root / "store"


def test_sync_derives_the_earnings_tables_from_sf1_events_and_actions(synced):
    store = ParquetStore(str(synced))
    ed = store.read_long("earnings_dates").sort_values(["ticker", "date"]).reset_index(drop=True)
    got = [(r.ticker, r.date.date().isoformat(), r.source) for r in ed.itertuples()]
    assert got == [
        ("AAA", "2019-10-30", "events"),  # 8-K item 2.02 six days before the 10-Q
        ("AAA", "2020-01-10", "datekey"),  # no event before the Q4 filing -> filing date
        ("BBB", "2019-11-20", "events"),
        ("DDD.100005", "2020-01-27", "events"),  # reused ticker resolved to the holder at that date
    ]
    assert ed.loc[0, "datekey"] == pd.Timestamp("2019-11-05") and ed.loc[0, "calendardate"] == pd.Timestamp(
        "2019-09-30"
    )
    ef = store.read_long("earnings_fundamentals")
    assert set(ef["dimension"]) == {"ARQ", "ART"} and len(ef) == 11
    row = ef[(ef["ticker"] == "AAA") & (ef["dimension"] == "ART") & (ef["date"] == "2020-01-10")].iloc[0]
    assert (row["netinc"], row["ncfo"], row["assets"]) == (25e6, 30e6, 1.2e9)
    # the 2-for-1 split of AAA (2020-01-15) is after every filing: split_cum is 1 for all of them
    assert (ef["split_cum"] == 1.0).all()
    assert "ncfo" in store.read_long("sharadar_sf1").columns
    md_ = store.as_market_data()
    assert list(md_.earnings_dates.columns[:3]) == ["ticker", "announce_date", "source"]
    assert {"datekey", "eps", "split_cum", "ncfo"} <= set(md_.earnings_fundamentals.columns)
    v2.require_v2_inputs(md_)


def test_split_ratio_is_confirmed_against_the_stored_prices(synced, tmp_path):
    import shutil

    dst = tmp_path / "store"
    shutil.copytree(synced, dst)
    store = ParquetStore(str(dst))
    splits, counts = sh._stored_splits(store)
    assert counts["n"] == 1 and counts["as_given"] == 1 and counts["orientation"] == "given"
    assert splits.iloc[0].to_dict() == {"ticker": "AAA", "date": pd.Timestamp("2020-01-15"), "ratio": 2.0}
    # a filing after the split picks the factor up
    sf1 = store.read_long("sharadar_sf1").rename(columns={"date": "datekey"})
    extra = (
        sf1[(sf1["ticker"] == "AAA") & (sf1["dimension"] == "ARQ")]
        .tail(1)
        .assign(datekey=pd.Timestamp("2020-01-20"), calendardate=pd.Timestamp("2020-03-31"), eps=0.3)
    )
    f = E.build_earnings_fundamentals(pd.concat([sf1, extra]), splits)
    aaa = f[(f["ticker"] == "AAA") & (f["dimension"] == "ARQ")].set_index("datekey")["split_cum"]
    assert aaa.loc["2020-01-10"] == 1.0 and aaa.loc["2020-01-20"] == 2.0
    # the vendor value written the other way round (0.5) is turned by the prices
    ca = store.read_long("corporate_actions")
    ca.loc[ca["action"] == "split", "value"] = 0.5
    store.write_long("corporate_actions", ca, mode="replace", key=["date", "ticker", "action"])
    splits, counts = sh._stored_splits(store)
    assert counts["inverse"] == 1 and splits["ratio"].tolist() == [2.0]


def test_derive_earnings_cli_on_an_existing_store(synced, tmp_path, capsys):
    import json
    import shutil

    from qts.cli import main

    dst = tmp_path / "store"
    shutil.copytree(synced, dst)
    store = ParquetStore(str(dst))
    before = store.read_long("earnings_dates")
    for n in ("earnings_dates", "earnings_fundamentals"):
        store.drop(n)  # a store synced before the tables existed
    assert ParquetStore(str(dst)).as_market_data().earnings_dates is None
    rc = main(
        ["data", "sharadar", "derive-earnings", "--store", str(dst), "--json", str(tmp_path / "e.json")]
    )
    out = capsys.readouterr().out
    assert rc == 0 and "EVENTS code frequency" in out and "<- earnings" in out and "by year" in out
    assert "3 from EVENTS code 22, 1 filing-date fallbacks" in out
    assert "earnings_dates (4 rows)" in out and "earnings_fundamentals (11 rows)" in out
    assert "splits: 1 rows; price-confirmed as given 1" in out and "WARNING" not in out
    pd.testing.assert_frame_equal(ParquetStore(str(dst)).read_long("earnings_dates"), before)
    assert json.loads((tmp_path / "e.json").read_text())["n_filings_with_event"] == 3
    h = ParquetStore(str(dst)).snapshot_hash("earnings_dates")
    assert main(["data", "sharadar", "derive-earnings", "--store", str(dst)]) == 0  # idempotent
    assert ParquetStore(str(dst)).snapshot_hash("earnings_dates") == h
    capsys.readouterr()
    # another code: nothing matches -> every date is a filing date, and the report warns
    assert main(["data", "sharadar", "derive-earnings", "--store", str(dst), "--codes", "99"]) == 0
    out = capsys.readouterr().out
    assert "0 from EVENTS code 99, 4 filing-date fallbacks" in out and "none with code 99" in out
    # no EVENTS in the store: says so, proceeds on filing dates, tells how to add it
    store.drop("sharadar_events")
    assert main(["data", "sharadar", "derive-earnings", "--store", str(dst)]) == 0
    out = capsys.readouterr().out
    assert "EVENTS is NOT in this store" in out and f"--store {dst} --tables EVENTS --from-files" in out
    assert (ParquetStore(str(dst)).read_long("earnings_dates")["source"] == "datekey").all()
    # no SF1 at all
    rc = main(["data", "sharadar", "derive-earnings", "--store", str(tmp_path / "empty")])
    assert rc == 1 and "--tables SF1" in capsys.readouterr().err


def test_a_store_synced_before_ncfo_was_kept_warns_and_sf1_is_read_again(tmp_path, capsys):
    import shutil

    from qts.cli import main

    files = tmp_path / "files"
    shutil.copytree(FIX, files)
    store_root = str(tmp_path / "store")
    sh.sync(store_root, raw_dir=str(tmp_path / "raw"), from_files=str(files))
    # emulate the old store: SF1 without ncfo, state recorded with the old field list
    store = ParquetStore(store_root)
    sf1 = store.read_long("sharadar_sf1").drop(columns=["ncfo"])
    store.write_long("sharadar_sf1", sf1, mode="replace", key=["date", "ticker", "dimension", "calendardate"])
    state = sh._load_state(store_root)
    state["SF1"].pop("fields")
    sh._save_state(store_root, state)
    assert main(["data", "sharadar", "derive-earnings", "--store", store_root]) == 0
    out = capsys.readouterr().out
    assert (
        "WARNING" in out and "ncfo" in out and f"sync --store {store_root} --tables SF1 --from-files" in out
    )
    with pytest.raises(v2.FeatureSetDataError, match="ncfo"):
        v2.require_v2_inputs(ParquetStore(store_root).as_market_data())
    # the same raw file: the step is not skipped, because the kept field list changed
    rep = sh.sync(store_root, raw_dir=str(tmp_path / "raw"), from_files=str(files), tables=["SF1"])
    assert not rep.tables["SF1"].skipped
    assert ParquetStore(store_root).read_long("earnings_fundamentals")["ncfo"].notna().any()
    v2.require_v2_inputs(ParquetStore(store_root).as_market_data())
    rep = sh.sync(store_root, raw_dir=str(tmp_path / "raw"), from_files=str(files), tables=["SF1"])
    assert rep.tables["SF1"].skipped  # and now it is


# ============================================================================= diagnostics / demo
def _script(name: str):
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module(name)


def test_diagnose_signal_coverage_only(capsys):
    mod = _script("diagnose_signal")
    rc = mod.main(["--synthetic", "--set", "alpha.feature_set=v2", "--coverage-only"])
    out = capsys.readouterr().out
    assert rc == 0 and "feature set: v2" in out
    assert "coverage per year: feature set 2" in out and all(f in out for f in V2)
    assert "earnings announcements per year by source" in out and "share_events" in out
    assert "rank IC" not in out  # nothing was fitted
    # v1: the table of the v1 features only
    assert mod.main(["--synthetic", "--coverage-only"]) == 0
    out = capsys.readouterr().out
    assert "coverage per year: v1 features" in out and "feature set 2" not in out


def test_feature_coverage_by_year_counts_eligible_cells():
    mod = _script("diagnose_signal")
    idx = pd.to_datetime(["2020-06-01", "2020-06-02", "2021-06-01", "2021-06-02"])
    elig = pd.DataFrame(
        [[True, True], [True, False], [True, True], [False, False]], index=idx, columns=["A", "B"]
    )
    x = pd.DataFrame([[1.0, np.nan], [1.0, 5.0], [np.nan, 2.0], [3.0, 3.0]], index=idx, columns=["A", "B"])
    cov = mod.feature_coverage_by_year({"x": x}, elig)
    assert cov["x"].tolist() == pytest.approx([2 / 3, 1 / 2, 3 / 5]) and list(cov.index) == [
        "2020",
        "2021",
        "all",
    ]


def test_demo_generates_the_earnings_tables_only_for_v2(monkeypatch, capsys):
    import qts.cli as cli

    seen = {}

    def fake_run(md_, cfg, a, out=None):
        seen["md"], seen["cfg"] = md_, cfg
        return 0

    monkeypatch.setattr(cli, "_run_pipeline", fake_run)
    base = ["demo", "--stocks", "30", "--days", "300"]
    assert cli.main(base) == 0
    assert seen["md"].earnings_dates is None and seen["cfg"].alpha.feature_set == "v1"
    h1 = hash_market_data(seen["md"])
    assert cli.main([*base, "--set", "alpha.feature_set=v2"]) == 0
    assert seen["md"].earnings_dates is not None and seen["cfg"].alpha.feature_set == "v2"
    assert hash_market_data(seen["md"], exclude=frozenset(MarketData.DATED_TABLES)) == h1
    assert cli.main([*base, "--set", "alpha.feature_set=v9"]) == 2
