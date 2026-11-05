"""Fast offline tests for qts.data (store, universe, loaders)."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from qts.core.config import UniverseConfig
from qts.core.types import MarketData
from qts.data import (
    ParquetStore,
    apply_mask,
    build_universe_mask,
    ff48_from_sic_series,
    parse_finra_short_interest,
    parse_ken_french_csv,
    parse_ken_french_zip,
    pit_forward_fill,
    sharadar_sf1_to_pit_fundamentals,
    sic_to_ff48,
    survivorship_check,
    to_short_interest_ratio,
)
from qts.data.synthetic import SyntheticSpec, generate
from qts.data.universe import raw_eligibility


@pytest.fixture(scope="module")
def md() -> MarketData:
    return generate(SyntheticSpec(n_stocks=80, n_days=300, seed=1))


# --------------------------------------------------------------------------- store
def test_store_wide_round_trip_and_duckdb(tmp_path, md):
    st = ParquetStore(tmp_path / "store")
    close = md.close.copy()
    close.iloc[5, 3] = np.nan  # sparse cell survives as NaN
    st.write_wide("prices_close", close)
    back = st.read_wide("prices_close")
    pd.testing.assert_frame_equal(back, close, check_names=False, check_freq=False)
    # year partitions exist and date filtering works
    assert any(p.name.startswith("year=") for p in (tmp_path / "store" / "prices_close").iterdir())
    sub = st.read_wide("prices_close", start="2018-06-01", end="2018-09-30")
    assert sub.index.min() >= pd.Timestamp("2018-06-01") and sub.index.max() <= pd.Timestamp("2018-09-30")
    rows = st.duckdb_query(
        "SELECT ticker, count(*) AS n FROM prices_close GROUP BY ticker ORDER BY ticker LIMIT 3"
    )
    assert len(rows) == 3 and rows["n"].iloc[0] == len(close)
    h1 = st.snapshot_hash("prices_close")
    st.write_wide("prices_close", close * 1.01)
    assert st.snapshot_hash("prices_close") != h1 and len(h1) == 64
    # long table without date column
    st.write_long("industry", pd.DataFrame({"ticker": ["A", "B"], "industry": ["X", "Y"]}))
    assert st.read_long("industry").shape == (2, 2)


def test_store_as_market_data(tmp_path, md):
    st = ParquetStore(tmp_path / "s2")
    st.write_market_data(md)
    assert set(st.datasets()) >= {"prices_close", "prices_volume", "market_cap", "industry", "spread_bps"}
    md2 = st.as_market_data()
    md2.validate()
    assert md2.close.shape == md.close.shape
    np.testing.assert_allclose(md2.close.values, md.close.values)
    np.testing.assert_allclose(md2.short_interest_ratio.values, md.short_interest_ratio.values)
    assert (md2.industry.reindex(md.industry.index) == md.industry).all()
    sub = st.as_market_data(universe_tickers=list(md.tickers[:10]))
    assert list(sub.tickers) == list(md.tickers[:10])
    assert sub.book_to_price.shape == (len(md.dates), 10)


# --------------------------------------------------------------------------- universe
def test_universe_mask_shapes_and_bottom_quantile(md):
    cfg = UniverseConfig(
        top_n_by_mcap=50, min_price=1.0, min_adv_usd=0.0, adv_window=20, exclude_bottom_size_quantile=0.2
    )
    mask = build_universe_mask(md, cfg)
    assert mask.shape == md.close.shape and mask.dtypes.eq(bool).all()
    assert mask.index.equals(md.dates)
    n_in = mask.sum(axis=1)
    # hysteresis may grandfather a few names beyond top_n; raw eligibility is capped at top_n
    assert 0 < n_in.iloc[-1] <= 60
    # bottom-quantile exclusion: on a refresh date the smallest eligible names are out
    d = md.dates[42]
    raw = raw_eligibility(md, cfg).loc[d]
    adv_ok = (md.dollar_volume.rolling(20, min_periods=10).mean().loc[d] >= 0) & (md.close.loc[d] >= 1.0)
    mc = md.market_cap.loc[d][adv_ok]
    cut = mc.quantile(0.2)
    smallest = mc[mc < cut].index
    assert not raw[smallest].any()
    # top-N by size respected
    assert raw.sum() <= 50


def test_universe_hysteresis_constructed():
    dates = pd.bdate_range("2020-01-01", periods=21 * 8)
    tick = pd.Index(["A", "B", "C", "D"], name="ticker")
    close = pd.DataFrame(10.0, index=dates, columns=tick)
    volume = pd.DataFrame(1e6, index=dates, columns=tick)
    mcap = pd.DataFrame(1e9, index=dates, columns=tick)
    # (day 0 has no ADV history yet -> first eligible refresh is day 21)
    # A drops below min price from day 42 for two refresh periods, then recovers -> should stay
    close.loc[dates[42:84], "A"] = 1.0
    # B drops below min price from day 42 permanently -> leaves after 3 failed refreshes (day 84)
    close.loc[dates[42:], "B"] = 1.0
    # C delisted (NaN) from day 100 -> leaves immediately regardless of hysteresis
    close.loc[dates[100:], "C"] = np.nan
    md = MarketData(close=close, volume=volume, market_cap=mcap, industry=pd.Series("X", index=tick))
    cfg = UniverseConfig(
        top_n_by_mcap=10, min_price=5.0, min_adv_usd=0.0, adv_window=5, exclude_bottom_size_quantile=0.0
    )
    mask = build_universe_mask(md, cfg, refresh_days=21, exit_after_n_fails=3)
    assert not mask.iloc[:21].any().any()
    live = mask.loc[dates[21:]]
    # A: fails at refreshes day 42 and 63, passes at 84 -> never leaves
    assert live["A"].all()
    # B: fails at 42, 63, 84 -> out from day 84 onwards, in before
    assert mask.loc[dates[21:84], "B"].all()
    assert not mask.loc[dates[84:], "B"].any()
    # C: out as soon as price is NaN
    assert mask.loc[dates[21:100], "C"].all() and not mask.loc[dates[100:], "C"].any()
    assert live["D"].all()
    # membership is constant between refreshes
    assert (mask.iloc[21:42].nunique() == 1).all()
    m2 = apply_mask(md, mask)
    assert np.isnan(m2.market_cap.loc[dates[90], "B"]) and m2.close.loc[dates[90], "B"] == 1.0


def test_survivorship_check(md):
    with pytest.warns(UserWarning):
        res = survivorship_check(md)
    assert res["suspect"] and res["n_delisted"] == 0
    close = md.close.copy()
    close.iloc[-50:, 0] = np.nan
    md2 = MarketData(close=close, volume=md.volume, market_cap=md.market_cap, industry=md.industry)
    res2 = survivorship_check(md2)
    assert not res2["suspect"] and res2["n_delisted"] == 1


# --------------------------------------------------------------------------- Ken French
FF5_TEXT = """This file was created by CMPT_ME_BEME_OP_INV_RETS_DAILY using the 202409 CRSP database.
The Tbill return is the simple daily rate that, over the number of trading days in the month, compounds to 1-month TBill rate.

,Mkt-RF,SMB,HML,RMW,CMA,RF
19630701,-0.67,0.02,-0.35,0.03,0.13,0.012
19630702,0.79,-0.28,0.28,-0.08,-0.21,0.012
19630703,0.63,-0.18,-0.10,0.13,-0.25,0.012
19630705,-99.99,0.09,0.04,-0.01,0.05,0.012
"""
MOM_TEXT = """This file was created by CMPT_ME_PRIOR_RETS_DAILY using the 202409 CRSP database.
Missing data are indicated by -99.99 or -999.

,Mom
19261103,    0.56
19261104,   -0.49
19261105,    0.04
"""


def test_parse_ken_french_text_and_zip():
    ff5 = parse_ken_french_csv(FF5_TEXT)
    assert list(ff5.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert ff5.index[0] == pd.Timestamp("1963-07-01") and len(ff5) == 4
    assert ff5.loc["1963-07-01", "Mkt-RF"] == pytest.approx(-0.0067)
    assert ff5.loc["1963-07-01", "RF"] == pytest.approx(0.00012)
    assert np.isnan(ff5.loc["1963-07-05", "Mkt-RF"])
    mom = parse_ken_french_csv(MOM_TEXT)
    assert list(mom.columns) == ["Mom"] and mom.iloc[1, 0] == pytest.approx(-0.0049)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("F-F_Momentum_Factor_daily.CSV", MOM_TEXT)
    mom2 = parse_ken_french_zip(buf.getvalue())
    pd.testing.assert_frame_equal(mom, mom2)
    assert parse_ken_french_zip(FF5_TEXT.encode()).shape == (4, 6)


# --------------------------------------------------------------------------- FINRA
FINRA_TEXT = """accountingYearMonthNumber|symbolCode|issueName|issuerServicesGroupExchangeCode|marketClassCode|currentShortPositionQuantity|previousShortPositionQuantity|stockSplitFlag|averageDailyVolumeQuantity|daysToCoverQuantity|revisionFlag|changePercent|changePreviousNumber|settlementDate
202401|AAPL|APPLE INC|Q|NNM|120000000|110000000||55000000|2.18||9.09|10000000|20240112
202401|MSFT|MICROSOFT CORP|Q|NNM|40000000|42000000||25000000|1.6||-4.76|-2000000|20240112
202401|AAPL|APPLE INC|Q|NNM|130000000|120000000||50000000|2.6||8.33|10000000|20240131
"""


def test_parse_finra_and_pit_lag():
    si = parse_finra_short_interest(FINRA_TEXT.encode())
    assert list(si.columns[:5]) == [
        "settlement_date",
        "symbol",
        "short_interest",
        "avg_daily_volume",
        "days_to_cover",
    ]
    assert len(si) == 3 and si["short_interest"].iloc[0] == 120_000_000
    assert si["settlement_date"].iloc[0] == pd.Timestamp("2024-01-12")
    assert si["days_to_cover"].iloc[1] == pytest.approx(1.6)
    dates = pd.bdate_range("2024-01-02", "2024-03-01")
    shares = pd.DataFrame(1.0e9, index=dates, columns=["AAPL", "MSFT", "OTHER"])
    ratio = to_short_interest_ratio(si, shares, lag_business_days=8)
    pub1 = pd.Timestamp("2024-01-12") + pd.offsets.BDay(8)  # 2024-01-24
    assert pub1 == pd.Timestamp("2024-01-24")
    assert np.isnan(ratio.loc["2024-01-12", "AAPL"])  # settlement date: not yet public
    assert np.isnan(ratio.loc["2024-01-23", "AAPL"])  # day before publication
    assert ratio.loc["2024-01-24", "AAPL"] == pytest.approx(0.12)
    pub2 = pd.Timestamp("2024-01-31") + pd.offsets.BDay(8)
    assert ratio.loc[pub2 - pd.offsets.BDay(1), "AAPL"] == pytest.approx(0.12)
    assert ratio.loc[pub2, "AAPL"] == pytest.approx(0.13)
    assert ratio.loc["2024-02-20", "MSFT"] == pytest.approx(0.04)
    assert "OTHER" not in ratio.columns
    # lag=0 makes it visible on settlement date (documented as NOT PIT)
    r0 = to_short_interest_ratio(si, shares, lag_business_days=0)
    assert r0.loc["2024-01-12", "AAPL"] == pytest.approx(0.12)


# --------------------------------------------------------------------------- FF48
def test_sic_to_ff48_spot_checks():
    assert sic_to_ff48(2834) == "Drugs"
    assert sic_to_ff48(6020) == "Banks"
    assert sic_to_ff48(7372) == "BusSv"
    assert sic_to_ff48(7373) == "Comps"
    assert sic_to_ff48(3674) == "Chips"
    assert sic_to_ff48(4911) == "Util"
    assert sic_to_ff48(3711) == "Autos"
    assert sic_to_ff48(5411) == "Rtail"
    assert sic_to_ff48(6798) == "Fin"
    assert sic_to_ff48(1311) == "Oil"
    assert sic_to_ff48(9999) == "Other"
    assert sic_to_ff48(None) == "Other"
    s = ff48_from_sic_series(pd.Series([2834, 6020, np.nan, "7372", 3674]))
    assert list(s) == ["Drugs", "Banks", "Other", "BusSv", "Chips"]
    from qts.data.loaders import FF48_NAMES

    assert len(FF48_NAMES) == 48 and len(set(FF48_NAMES)) == 48


# --------------------------------------------------------------------------- Sharadar
def test_sharadar_sf1_pit_uses_datekey():
    sf1 = pd.DataFrame(
        {
            "ticker": ["ABC", "ABC", "ABC", "ABC", "ABC", "XYZ"],
            "dimension": ["ARQ"] * 5 + ["MRQ"],
            "calendardate": [
                "2022-12-31",
                "2023-03-31",
                "2023-06-30",
                "2023-09-30",
                "2023-12-31",
                "2023-03-31",
            ],
            "datekey": ["2023-02-15", "2023-05-10", "2023-08-09", "2023-11-08", "2024-02-20", "2023-05-01"],
            "bvps": [10.0, 11.0, 12.0, 13.0, 14.0, 99.0],
            "pb": [2.0, 2.2, 2.4, 2.6, 2.8, 9.0],
            "eps": [1.0, 1.1, 1.2, 1.3, 1.4, 9.0],
            "gp": [50.0, 55.0, 60.0, 65.0, 70.0, 9.0],
            "assets": [1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 9.0],
        }
    )
    dates = pd.bdate_range("2023-01-02", "2024-03-29")
    out = sharadar_sf1_to_pit_fundamentals(sf1, dimension="ARQ", dates=dates, lag_days=0)
    bvps = out["bvps"]
    assert "XYZ" not in bvps.columns  # MRQ rows filtered
    assert np.isnan(bvps.loc["2023-01-31", "ABC"])  # before first filing
    assert bvps.loc["2023-02-15", "ABC"] == 10.0
    # Q1-2023 (calendardate 2023-03-31) must NOT be visible until datekey 2023-05-10
    assert bvps.loc["2023-03-31", "ABC"] == 10.0
    assert bvps.loc["2023-05-09", "ABC"] == 10.0
    assert bvps.loc["2023-05-10", "ABC"] == 11.0
    assert out["book_to_price"].loc["2023-05-10", "ABC"] == pytest.approx(1 / 2.2)
    assert out["gross_profitability"].loc["2023-05-10", "ABC"] == pytest.approx(55.0 / 1050.0)
    # asset growth: Q4-2023 vs Q4-2022 = 1200/1000 - 1, visible from 2024-02-20
    ag = out["asset_growth"]
    assert np.isnan(ag.loc["2024-02-19", "ABC"])
    assert ag.loc["2024-02-20", "ABC"] == pytest.approx(0.2)
    # lag_days shifts availability
    out_l = sharadar_sf1_to_pit_fundamentals(sf1, dimension="ARQ", dates=dates, lag_days=5)
    assert out_l["bvps"].loc["2023-05-12", "ABC"] == 10.0
    assert out_l["bvps"].loc["2023-05-15", "ABC"] == 11.0
    # default lag_days=1: a filing dated d is first usable on the next day
    out_d = sharadar_sf1_to_pit_fundamentals(sf1, dimension="ARQ", dates=dates)
    assert out_d["bvps"].loc["2023-05-10", "ABC"] == 10.0
    assert out_d["bvps"].loc["2023-05-11", "ABC"] == 11.0
    # price-based ratios use the as-traded close
    close_raw = pd.DataFrame(20.0, index=dates, columns=["ABC"])
    out_p = sharadar_sf1_to_pit_fundamentals(
        sf1, dimension="ARQ", dates=dates, lag_days=0, close_raw=close_raw
    )
    assert out_p["book_to_price"].loc["2023-05-10", "ABC"] == pytest.approx(11.0 / 20.0)
    assert out_p["earnings_yield"].loc["2023-05-10", "ABC"] == pytest.approx(1.1 / 20.0)
    # an adjusted close is refused unless the caller asserts it is raw
    with pytest.raises(ValueError, match="unadjusted"):
        sharadar_sf1_to_pit_fundamentals(sf1, dimension="ARQ", dates=dates, close=close_raw)
    out_c = sharadar_sf1_to_pit_fundamentals(
        sf1, dimension="ARQ", dates=dates, lag_days=0, close=close_raw, price_is_adjusted=False
    )
    pd.testing.assert_frame_equal(out_c["book_to_price"], out_p["book_to_price"])


def test_sharadar_ratios_use_raw_close_across_split():
    """A 2:1 split after the filing: the adjusted close halves the pre-split
    price, so bvps / adjusted close would double B/P; the raw close does not."""
    sf1 = pd.DataFrame(
        {
            "ticker": ["ABC"],
            "dimension": ["ART"],
            "calendardate": ["2023-03-31"],
            "datekey": ["2023-05-10"],
            "bvps": [10.0],
            "eps": [2.0],
        }
    )
    dates = pd.bdate_range("2023-05-01", "2023-07-31")
    split = pd.Timestamp("2023-06-15")
    raw = pd.DataFrame(40.0, index=dates, columns=["ABC"])
    raw.loc[split:, "ABC"] = 20.0
    adjusted = pd.DataFrame(20.0, index=dates, columns=["ABC"])
    out = sharadar_sf1_to_pit_fundamentals(sf1, dates=dates, close_raw=raw)  # default ART, lag 1
    bp = out["book_to_price"]["ABC"]
    assert np.isnan(bp.loc["2023-05-10"]) and bp.loc["2023-05-11"] == pytest.approx(10.0 / 40.0)
    # after the split the as-reported bvps is on the old share basis (until the next filing)
    assert bp.loc["2023-06-14"] == pytest.approx(0.25)
    assert out["earnings_yield"]["ABC"].loc["2023-05-11"] == pytest.approx(2.0 / 40.0)
    wrong = sharadar_sf1_to_pit_fundamentals(sf1, dates=dates, close=adjusted, price_is_adjusted=False)
    assert wrong["book_to_price"]["ABC"].loc["2023-05-11"] == pytest.approx(0.5)  # the bug being guarded


def test_sharadar_asset_growth_ignores_later_restatements():
    """A restated prior-year row filed *after* the current row's datekey must not
    change the asset growth visible at that datekey (audit finding 6)."""
    sf1 = pd.DataFrame(
        {
            "ticker": ["ABC"] * 3,
            "dimension": ["ARQ"] * 3,
            "calendardate": ["2019-12-31", "2020-12-31", "2019-12-31"],
            "datekey": ["2020-02-14", "2021-02-16", "2021-06-01"],  # last row: 10-K/A restating FY2019
            "assets": [1000.0, 1200.0, 1500.0],
        }
    )
    dates = pd.bdate_range("2020-01-01", "2021-12-31")
    ag = sharadar_sf1_to_pit_fundamentals(sf1, dimension="ARQ", dates=dates, lag_days=0)["asset_growth"]
    assert ag.loc["2021-02-16", "ABC"] == pytest.approx(0.2)  # 1200/1000 - 1 with the original filing
    assert ag.loc["2021-05-28", "ABC"] == pytest.approx(0.2)
    # the restated FY2019 row has no FY2018 row of its own, so it contributes no growth value
    # (pit_forward_fill carries the last *valid* value); the leaked -0.2 must never appear
    assert (ag["ABC"].dropna().round(6) == 0.2).all()


def test_pit_forward_fill_never_leaks():
    dates = pd.bdate_range("2024-01-01", periods=10)
    long = pd.DataFrame({"available_date": [dates[3], dates[7]], "ticker": ["A", "A"], "value": [1.0, 2.0]})
    w = pit_forward_fill(long, dates)
    assert w["A"].iloc[:3].isna().all()
    assert (w["A"].iloc[3:7] == 1.0).all() and (w["A"].iloc[7:] == 2.0).all()


# --------------------------------------------------------------------------- store upsert (#33)
def _panel(dates, tickers, base):
    return pd.DataFrame(
        base + np.arange(len(dates) * len(tickers), dtype=float).reshape(len(dates), -1),
        index=dates,
        columns=pd.Index(tickers, name="ticker"),
    )


def test_store_append_one_day_keeps_rest_of_year(tmp_path):
    st = ParquetStore(tmp_path / "s")
    dates = pd.bdate_range("2021-01-04", "2021-03-31")
    tick = ["A", "B", "C"]
    full = _panel(dates, tick, 100.0)
    st.write_wide("px", full.iloc[:-1])  # everything but the last day
    st.write_wide("px", full.iloc[-1:])  # append one day
    back = st.read_wide("px")
    pd.testing.assert_frame_equal(back, full, check_names=False, check_freq=False)
    # long tables upsert too
    lg = pd.DataFrame({"date": dates[:3], "ticker": "A", "x": [1.0, 2.0, 3.0]})
    st.write_long("lg", lg)
    st.write_long("lg", pd.DataFrame({"date": [dates[3]], "ticker": ["A"], "x": [4.0]}))
    assert st.read_long("lg")["x"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_store_rewrite_day_overwrites_values_and_nan_deletes(tmp_path):
    st = ParquetStore(tmp_path / "s")
    dates = pd.bdate_range("2021-12-27", "2022-01-07")  # spans two year partitions
    full = _panel(dates, ["A", "B"], 1.0)
    st.write_wide("px", full)
    fix = full.loc[["2021-12-30", "2022-01-03"]].copy()
    fix.loc[:, "A"] = -1.0
    fix.loc["2022-01-03", "B"] = np.nan  # NaN deletes the stored cell
    st.write_wide("px", fix)
    back = st.read_wide("px")
    assert back.shape == full.shape
    assert back.loc["2021-12-30", "A"] == -1.0 and back.loc["2022-01-03", "A"] == -1.0
    assert back.loc["2021-12-30", "B"] == full.loc["2021-12-30", "B"]
    assert np.isnan(back.loc["2022-01-03", "B"])
    fixed = pd.DatetimeIndex(["2021-12-30", "2022-01-03"])
    pd.testing.assert_frame_equal(
        back.drop(index=fixed), full.drop(index=fixed), check_names=False, check_freq=False
    )
    # explicit partition rewrite keeps only the new rows of the touched year(s)
    st.write_wide("px", full.loc[["2022-01-04"]], mode="replace_partition")
    back2 = st.read_wide("px")
    assert list(back2.loc["2022"].index) == [pd.Timestamp("2022-01-04")]
    assert len(back2.loc["2021"]) == 5  # 2021 partition untouched
    with pytest.warns(DeprecationWarning):
        st.write_wide("px", full.loc[["2022-01-05"]], mode="overwrite_partitions")
    with pytest.raises(ValueError):
        st.write_wide("px", full, mode="bogus")


def test_store_write_is_atomic(tmp_path, monkeypatch):
    import qts.data.store as store_mod

    st = ParquetStore(tmp_path / "s")
    dates = pd.bdate_range("2021-01-04", periods=5)
    full = _panel(dates, ["A"], 1.0)
    st.write_wide("px", full)
    leftovers = [p for p in (tmp_path / "s").rglob("*") if p.is_file() and ".tmp-" in p.name]
    assert leftovers == []
    real = store_mod.pq.write_table

    def boom(table, where, **kw):
        real(table, where, **kw)  # temp file is fully written ...
        raise OSError("disk full")  # ... then the process "dies" before os.replace

    monkeypatch.setattr(store_mod.pq, "write_table", boom)
    with pytest.raises(OSError):
        st.write_wide("px", full * 2.0)
    monkeypatch.setattr(store_mod.pq, "write_table", real)
    # the previous partition is intact and no temp file is left behind
    pd.testing.assert_frame_equal(st.read_wide("px"), full, check_names=False, check_freq=False)
    assert [p for p in (tmp_path / "s").rglob("*") if p.is_file() and ".tmp-" in p.name] == []


def test_store_round_trips_raw_prices_and_industry_history(tmp_path):
    md = generate(SyntheticSpec(n_stocks=12, n_days=80, seed=5, n_splits=2))
    ih = pd.DataFrame(index=md.dates[[0, 40]], columns=md.tickers, dtype=object)
    ih.iloc[0] = md.industry.reindex(md.tickers).to_numpy()
    ih.iloc[1, 0] = "IND99"
    md.industry_history = ih
    st = ParquetStore(tmp_path / "s")
    st.write_market_data(md)
    assert {"prices_close_raw", "shares_outstanding", "industry_history"} <= set(st.datasets())
    md2 = st.as_market_data()
    np.testing.assert_allclose(md2.close_raw.to_numpy(), md.close_raw.to_numpy())
    np.testing.assert_allclose(md2.shares_outstanding.to_numpy(), md.shares_outstanding.to_numpy())
    t0 = md.tickers[0]
    assert md2.industry_as_of(md.dates[39])[t0] == md.industry[t0]
    assert md2.industry_as_of(md.dates[40])[t0] == "IND99"
    # a start date after the reclassification still sees it (no start cut on labels)
    md3 = st.as_market_data(start=md.dates[50])
    assert md3.industry_as_of(md3.dates[0])[t0] == "IND99"


# --------------------------------------------------------------------------- raw prices (#34)
def test_synthetic_splits_raw_vs_adjusted():
    spec = {"n_stocks": 40, "n_days": 260, "seed": 3}
    base = generate(SyntheticSpec(**spec))
    md = generate(SyntheticSpec(**spec, n_splits=3))
    # adjusted close and market cap are bit-identical; no-split default has close_raw == close
    pd.testing.assert_frame_equal(md.close, base.close)
    pd.testing.assert_frame_equal(md.market_cap, base.market_cap)
    pd.testing.assert_frame_equal(base.close_raw, base.close)
    events = md.extra["true_splits"].stack().dropna()
    assert len(events) == 3 and set(events.unique()) == {2.0, 3.0}
    for (d, t), k in events.items():
        pos = md.dates.get_loc(d)
        raw, adj = md.close_raw[t], md.close[t]
        assert raw.iloc[pos - 1] == pytest.approx(k * adj.iloc[pos - 1])
        assert raw.iloc[pos] == pytest.approx(adj.iloc[pos])
        # as-traded price drops by ~k on the split day; adjusted return is ordinary
        assert raw.iloc[pos] / raw.iloc[pos - 1] < 0.75
        assert abs(md.returns[t].iloc[pos]) < 0.3
    # dollar volume and market cap are invariant to the share basis
    np.testing.assert_allclose(md.dollar_volume.to_numpy(), (base.close * base.volume).to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(
        (md.close_raw * md.shares_outstanding).to_numpy(), md.market_cap.to_numpy(), rtol=1e-12
    )
    # style factor returns: all four, date x factor
    fr = md.extra["true_style_factor_returns"]
    assert list(fr.columns) == ["size", "value", "quality", "investment"] and fr.index.equals(md.dates)
    np.testing.assert_array_equal(fr["size"].to_numpy(), md.extra["true_style_returns"].iloc[:, 0].to_numpy())


def test_market_data_trade_price_and_slice():
    dates = pd.bdate_range("2020-01-01", periods=10)
    tick = pd.Index(["A", "B"], name="ticker")
    adj = pd.DataFrame(10.0, index=dates, columns=tick)
    raw = adj * 2.0
    vol = pd.DataFrame(100.0, index=dates, columns=tick)
    md = MarketData(close=adj, volume=vol, market_cap=adj * 1e6, industry=pd.Series("X", index=tick))
    assert md.trade_price() is md.close
    md2 = MarketData(
        close=adj,
        volume=vol,
        market_cap=adj * 1e6,
        industry=pd.Series("X", index=tick),
        close_raw=raw,
        shares_outstanding=vol * 5,
    )
    md2.validate()
    assert (md2.dollar_volume == 2000.0).all().all()
    s = md2.slice_until(dates[4])
    assert len(s.close_raw) == 5 and len(s.shares_outstanding) == 5 and s.industry is md2.industry
    bad = MarketData(
        close=adj, volume=vol, market_cap=adj, industry=pd.Series("X", index=tick), close_raw=raw.iloc[:-1]
    )
    with pytest.raises(ValueError, match="close_raw"):
        bad.validate()


def test_price_screen_uses_raw_price():
    """A name whose split-adjusted close is below $5 but that traded above $5
    (pre-split) passes the price screen."""
    dates = pd.bdate_range("2020-01-01", periods=60)
    tick = pd.Index(["A", "B"], name="ticker")
    adj = pd.DataFrame(4.0, index=dates, columns=tick)
    raw = adj.copy()
    raw["A"] = 12.0  # 3:1 split after the sample
    vol = pd.DataFrame(1e6, index=dates, columns=tick)
    cfg = UniverseConfig(
        top_n_by_mcap=10, min_price=5.0, min_adv_usd=5e6, adv_window=5, exclude_bottom_size_quantile=0.0
    )
    md_adj = MarketData(close=adj, volume=vol, market_cap=adj * 1e8, industry=pd.Series("X", index=tick))
    assert not raw_eligibility(md_adj, cfg).any().any()
    md_raw = MarketData(
        close=adj, volume=vol, market_cap=adj * 1e8, industry=pd.Series("X", index=tick), close_raw=raw
    )
    el = raw_eligibility(md_raw, cfg)
    assert el["A"].iloc[10:].all() and not el["B"].any()  # A: $12 x 1e6 sh = $12M ADV; B: $4 < $5


def test_universe_refresh_settings_from_config(md):
    cfg = UniverseConfig(
        top_n_by_mcap=30,
        min_price=1.0,
        min_adv_usd=0.0,
        adv_window=20,
        exclude_bottom_size_quantile=0.0,
        refresh_days=5,
        exit_after_n_fails=1,
    )
    m_cfg = build_universe_mask(md, cfg)
    m_kw = build_universe_mask(
        md,
        UniverseConfig(
            top_n_by_mcap=30, min_price=1.0, min_adv_usd=0.0, adv_window=20, exclude_bottom_size_quantile=0.0
        ),
        refresh_days=5,
        exit_after_n_fails=1,
    )
    pd.testing.assert_frame_equal(m_cfg, m_kw)
    assert not m_cfg.equals(build_universe_mask(md, cfg, refresh_days=21))  # kwargs override cfg
    assert UniverseConfig().refresh_days == 21 and UniverseConfig().exit_after_n_fails == 3


def test_apply_mask_handles_new_fields():
    md = generate(SyntheticSpec(n_stocks=30, n_days=120, seed=2, n_splits=2))
    md.industry_history = pd.DataFrame(
        [md.industry.reindex(md.tickers).to_numpy()], index=md.dates[:1], columns=md.tickers
    )
    mask = pd.DataFrame(True, index=md.dates, columns=md.tickers)
    mask.iloc[:, :5] = False
    mask.iloc[:60, 5] = False
    m2 = apply_mask(md, mask)
    keep = md.tickers[5:]
    assert list(m2.tickers) == list(keep)
    pd.testing.assert_frame_equal(m2.close_raw, md.close_raw[keep])  # prices intact
    assert (
        m2.shares_outstanding.iloc[:60, 0].isna().all() and m2.shares_outstanding.iloc[60:, 0].notna().all()
    )
    assert list(m2.industry_history.columns) == list(keep)
    fr = m2.extra["true_style_factor_returns"]  # not a ticker panel
    pd.testing.assert_frame_equal(fr, md.extra["true_style_factor_returns"])


# --------------------------------------------------------------------------- dated industry (#43)
def test_industry_as_of_is_point_in_time():
    dates = pd.bdate_range("2020-01-01", periods=30)
    tick = pd.Index(["A", "B", "C"], name="ticker")
    px = pd.DataFrame(10.0, index=dates, columns=tick)
    static = pd.Series({"A": "Tech", "B": "Banks", "C": "Oil"})
    hist = pd.DataFrame(
        {"A": ["Tech", "Comms"], "B": ["Banks", np.nan], "C": [np.nan, np.nan]},
        index=pd.DatetimeIndex([dates[0], dates[20]]),
    )
    md = MarketData(close=px, volume=px, market_cap=px, industry=static, industry_history=hist)
    md.validate()
    assert md.industry_as_of(dates[19]).to_dict() == {"A": "Tech", "B": "Banks", "C": "Oil"}
    assert md.industry_as_of(dates[20])["A"] == "Comms"
    assert md.industry_as_of(dates[25])["B"] == "Banks"  # NaN later row -> carried forward
    assert np.isnan(md.industry_as_of(dates[25], fallback=False)["C"])  # never classified
    cut = md.slice_until(dates[10])
    assert len(cut.industry_history) == 1 and cut.industry_as_of(dates[29])["A"] == "Tech"
    md_static = MarketData(close=px, volume=px, market_cap=px, industry=static)
    pd.testing.assert_series_equal(md_static.industry_as_of(dates[5]), static.reindex(tick))


# --------------------------------------------------------------------------- shares outstanding (#44)
def test_shares_outstanding_panel_refuses_adjusted_close():
    from qts.data import shares_outstanding_panel

    md = generate(SyntheticSpec(n_stocks=10, n_days=40, seed=1, n_splits=1))
    assert shares_outstanding_panel(md) is md.shares_outstanding
    md_raw_only = MarketData(
        close=md.close,
        volume=md.volume,
        market_cap=md.market_cap,
        industry=md.industry,
        close_raw=md.close_raw,
    )
    np.testing.assert_allclose(
        shares_outstanding_panel(md_raw_only).to_numpy(), md.shares_outstanding.to_numpy()
    )
    md_adj = MarketData(close=md.close, volume=md.volume, market_cap=md.market_cap, industry=md.industry)
    with pytest.raises(ValueError, match="split-adjusted"):
        shares_outstanding_panel(md_adj)
    with pytest.raises(TypeError):
        to_short_interest_ratio(
            pd.DataFrame(columns=["settlement_date", "symbol", "short_interest"]),
            md.shares_outstanding.iloc[0],
        )


def test_universe_price_screen_uses_as_traded_price_across_split():
    """#34: a stock at $8 as traded but $2 split-adjusted before a 4:1 split passes the
    $5 screen until the split (it really traded at $8) and fails after it (it trades
    at $2).  Screening on the adjusted close would wrongly exclude it throughout."""
    dates = pd.bdate_range("2021-01-04", periods=40)
    tick = pd.Index(["SPLT", "BIG"], name="ticker")
    split_pos = 20
    adj = pd.DataFrame({"SPLT": 2.0, "BIG": 50.0}, index=dates)
    adj.columns = tick
    raw = adj.copy()
    raw.iloc[:split_pos, 0] = 8.0  # 4:1 split on dates[split_pos]
    shares_traded = pd.DataFrame({"SPLT": 1e6, "BIG": 1e6}, index=dates)
    shares_traded.columns = tick
    shares_traded.iloc[split_pos:, 0] = 4e6  # actual shares x4 after the split
    so = pd.DataFrame(1e8, index=dates, columns=tick)
    so.iloc[split_pos:, 0] = 4e8
    mcap = raw * so
    md = MarketData(
        close=adj,
        volume=shares_traded,
        market_cap=mcap,
        industry=pd.Series("X", index=tick),
        close_raw=raw,
        shares_outstanding=so,
    )
    md.validate()
    cfg = UniverseConfig(
        top_n_by_mcap=10, min_price=5.0, min_adv_usd=1e6, adv_window=10, exclude_bottom_size_quantile=0.0
    )
    el = raw_eligibility(md, cfg)
    warm = 5  # min_periods of the ADV window
    assert el["SPLT"].iloc[warm:split_pos].all()  # $8 as traded >= $5
    assert not el["SPLT"].iloc[split_pos:].any()  # $2 as traded < $5
    assert el["BIG"].iloc[warm:].all()
    # dollar ADV is unchanged by the split ($8 x 1m = $2 x 4m)
    assert (md.dollar_volume["SPLT"] == 8e6).all()
    # legacy behaviour (adjusted close only) excludes SPLT on every date
    legacy = MarketData(close=adj, volume=shares_traded, market_cap=mcap, industry=pd.Series("X", index=tick))
    assert not raw_eligibility(legacy, cfg)["SPLT"].any()
