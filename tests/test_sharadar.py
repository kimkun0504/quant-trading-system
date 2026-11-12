"""Sharadar bundle loader (``qts/data/sharadar.py``): offline, against fixture CSVs
that mimic the real table layouts (``tests/fixtures/sharadar``), a mocked HTTP
layer, and an end-to-end synthetic market written in Sharadar format, synced into a
store and run through the pipeline."""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from qts.data import sharadar as sh
from qts.data.loaders import FF48_DEFINITIONS, FF48_NAMES
from qts.data.store import ParquetStore

FIX = Path(__file__).parent / "fixtures" / "sharadar"
FIX_INC = Path(__file__).parent / "fixtures" / "sharadar_increment"
KEY = "sk-TEST-SECRET-0123456789"


# ----------------------------------------------------------------------------- helpers
def _tickers() -> pd.DataFrame:
    return pd.read_csv(FIX / "SHARADAR_TICKERS.csv")


def _master() -> sh.SecurityMaster:
    return sh.build_security_master(_tickers())


def _zip_bytes(table: str, src: Path = FIX) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"SHARADAR_{table}_2_abcdef.csv", (src / f"SHARADAR_{table}.csv").read_bytes())
    return buf.getvalue()


class _Resp:
    def __init__(self, status=200, js=None, body=b"", text=""):
        self.status_code, self._js, self._body, self.text = status, js, body, text

    def json(self):
        return self._js

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


class FakeSession:
    """Export poll (``creating`` -> ``fresh``) then a zip download; records every call."""

    def __init__(self, fail_table: str | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.polls: dict[str, int] = {}
        self.fail_table = fail_table

    def get(self, url, params=None, timeout=None, stream=False):
        self.calls.append((url, dict(params or {})))
        if url.startswith(sh.API_BASE):
            table = url.rsplit("/", 1)[1].replace(".json", "")
            assert params["api_key"] == KEY and params["qopts.export"] == "true"
            if table == self.fail_table:
                return _Resp(status=403, text=f"forbidden for api_key={params['api_key']}")
            n = self.polls.get(table, 0)
            self.polls[table] = n + 1
            status = "creating" if n == 0 else "fresh"
            link = f"https://s3.example.com/export/{table}.zip?X-Amz-Signature=sig"
            return _Resp(
                js={
                    "datatable_bulk_download": {
                        "file": {"link": link, "status": status, "data_snapshot_time": "2026-11-09T01:00:00"},
                        "datatable": {"last_refreshed_time": "2026-11-09"},
                    }
                }
            )
        table = url.rsplit("/", 1)[1].split(".zip")[0]
        return _Resp(body=_zip_bytes(table))


@pytest.fixture(scope="module")
def synced(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("sharadar_store")
    rep = sh.sync(str(root / "store"), raw_dir=str(root / "raw"), from_files=str(FIX))
    assert rep.mode == "full" and not rep.dry_run
    return root / "store"


# ----------------------------------------------------------------------------- key handling
def test_parse_dotenv_and_load_api_key(tmp_path, monkeypatch):
    txt = "# comment\nexport FOO='bar baz'\nNASDAQ_DATA_LINK_API_KEY=\"abc123\"  \nX=1 # trailing\n\nBROKEN\n"
    assert sh.parse_dotenv(txt) == {"FOO": "bar baz", "NASDAQ_DATA_LINK_API_KEY": "abc123", "X": "1"}
    monkeypatch.delenv(sh.API_KEY_ENV, raising=False)
    assert sh.load_api_key(env_file=tmp_path / "none") is None
    (tmp_path / ".env").write_text(txt)
    assert sh.load_api_key(env_file=tmp_path / ".env") == "abc123"
    monkeypatch.setenv(sh.API_KEY_ENV, "fromenv")
    assert sh.load_api_key(env_file=tmp_path / ".env") == "fromenv"
    assert sh.load_api_key("explicit ", env_file=tmp_path / ".env") == "explicit"


def test_redact_removes_key_and_query_value():
    s = sh.redact(f"GET https://x/y.json?qopts.export=true&api_key={KEY}&z=1 ({KEY})", KEY)
    assert KEY not in s and "api_key=<redacted>&z=1" in s


# ----------------------------------------------------------------------------- master / ids
def test_master_filters_ids_and_industry():
    m = _master()
    assert m.ids == ["AAA", "BBB", "CCC", "DDD.100004", "DDD.100005"]
    f = m.frame.set_index("id")
    # excluded: OTC (EEE), ETF on NYSEARCA (SPY), ADR (FFF), secondary class (HHH), SF1-only (GGG)
    assert not set(f["ticker"]) & {"EEE", "SPY", "FFF", "HHH", "GGG"}
    # TICKERS has one row per table: the SF1 row of AAA does not duplicate it
    assert (f["ticker"] == "AAA").sum() == 1
    assert f.loc["AAA", "industry_ff48"] == "Banks"  # "Banking" (famaindustry long name)
    assert f.loc["BBB", "industry_ff48"] == "BusSv"
    assert f.loc["CCC", "industry_ff48"] == "Meals"  # Sharadar spelling "Restaraunts Hotels Motels"
    assert f.loc["DDD.100004", "industry_ff48"] == "Drugs" and f.loc["DDD.100004", "industry_source"] == "sic"
    assert (
        f.loc["DDD.100005", "industry_ff48"] == "Other" and f.loc["DDD.100005", "industry_source"] == "other"
    )
    dd = m.delisting_table().set_index("ticker")["delisting_date"]
    assert (
        pd.isna(dd["AAA"])
        and dd["CCC"] == pd.Timestamp("2020-01-21")
        and dd["DDD.100004"] == pd.Timestamp("2020-01-10")
    )
    assert set(dd.index) == set(m.ids)  # every kept id is covered (NaT = alive)
    s = m.summary()
    assert s["n_reused_tickers"] == 1 and s["n_delisted"] == 2 and s["n_kept"] == 5
    # configurable filters: keep OTC and ADRs
    m2 = sh.build_security_master(
        _tickers(),
        sh.MasterFilters(
            categories=sh.DEFAULT_CATEGORIES | {"ADR Common Stock"}, exchanges=sh.DEFAULT_EXCHANGES | {"OTC"}
        ),
    )
    assert {"EEE", "FFF"} <= set(m2.ids)


def test_famaindustry_table_covers_every_ff48_name():
    long_names = pd.Series([d[1] for d in FF48_DEFINITIONS])
    lab, src = sh.map_famaindustry(long_names)
    assert list(lab) == FF48_NAMES and (src == "famaindustry").all()
    lab, src = sh.map_famaindustry(
        pd.Series(["Restaraunts Hotels Motels", "Steel Works Etc", "", None, "Nope"]),
        pd.Series([np.nan, np.nan, 2834, 3571, np.nan]),
    )
    assert list(lab) == ["Meals", "Steel", "Drugs", "Comps", "Other"]
    assert list(src) == ["famaindustry", "famaindustry", "sic", "sic", "other"]


def test_resolve_ids_reuse_gap_and_counts():
    m = _master()
    rows = pd.DataFrame(
        {
            "ticker": ["DDD", "DDD", "DDD", "DDD", "AAA", "ZZZ", "ddd "],
            "date": pd.to_datetime(
                [
                    "2020-01-03",
                    "2020-01-10",
                    "2020-01-15",
                    "2020-01-27",
                    "1999-01-01",
                    "2020-01-03",
                    "2020-01-31",
                ]
            ),
        }
    )
    out, c = sh.resolve_ids(rows, m.ranges)
    assert list(out["id"]) == ["DDD.100004", "DDD.100004", "DDD.100005", "AAA", "DDD.100005"]
    assert (c.n_rows, c.n_kept, c.n_not_in_master, c.n_out_of_range) == (7, 5, 1, 1)
    # unique tickers are never date-restricted (AAA in 1999 kept); the gap print is dropped
    out2, c2 = sh.resolve_ids(rows, m.ranges, gap_policy="next")
    assert c2.n_out_of_range == 0 and c2.n_gap_assigned == 1
    assert out2.set_index("date").loc["2020-01-15", "id"] == "DDD.100005"


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.lists(
        st.tuples(st.sampled_from(["AAA", "DDD", "CCC", "QQQ"]), st.integers(min_value=0, max_value=40)),
        min_size=1,
        max_size=40,
    )
)
def test_resolve_ids_counts_add_up(rows):
    m = _master()
    df = pd.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "date": pd.Timestamp("2019-12-20") + pd.to_timedelta([r[1] for r in rows], unit="D"),
        }
    )
    out, c = sh.resolve_ids(df, m.ranges)
    assert c.n_kept + c.n_not_in_master + c.n_out_of_range == c.n_rows == len(df)
    assert len(out) == c.n_kept and out["id"].isin(m.ids).all()
    rg = m.ranges.set_index("id")
    for r in out.itertuples():
        assert rg.loc[r.id, "start"] <= r.date <= rg.loc[r.id, "end"]


# ----------------------------------------------------------------------------- transforms
def test_sep_split_volume_unadjust_and_guards():
    m = _master()
    sep = pd.read_csv(FIX / "SHARADAR_SEP.csv")
    out, c = sh.transform_sep_chunk(sep, m.ranges)
    a = out[out["ticker"] == "AAA"].set_index("date")
    # before the 2:1 split (2020-01-15) SEP volume is split-adjusted (2x) and close = closeunadj / 2
    assert a.loc["2020-01-14", "volume"] == pytest.approx(1080.0)  # vendor 2160 x 0.5
    assert a.loc["2020-01-15", "volume"] == pytest.approx(1090.0)  # vendor 1090 x 1.0
    assert a.loc["2020-01-14", "close_raw"] == pytest.approx(108.0) and a.loc[
        "2020-01-14", "close"
    ] == pytest.approx(54.0)
    assert a.loc["2020-01-15", "close_raw"] == pytest.approx(54.5)
    # dividend-adjusted close is closeadj (BBB: 0.99 x close before the ex-date)
    b = out[out["ticker"] == "BBB"].set_index("date")
    assert b.loc["2020-01-21", "close"] == pytest.approx(55.44) and b.loc[
        "2020-01-21", "close_raw"
    ] == pytest.approx(56.0)
    # missing closeunadj: close_raw NaN, vendor volume kept, counted
    d = out[out["ticker"] == "DDD.100004"].set_index("date")
    assert np.isnan(d.loc["2020-01-08", "close_raw"]) and d.loc["2020-01-08", "volume"] == 3000
    assert c.n_missing_closeunadj == 1 and c.n_not_in_master == 15 and c.n_out_of_range == 1
    assert set(out["ticker"]) == set(m.ids)


def test_daily_units_and_derived_caps():
    m = _master()
    daily, _ = sh.transform_daily_chunk(pd.read_csv(FIX / "SHARADAR_DAILY.csv"), m.ranges)
    assert daily.set_index(["ticker", "date"]).loc[("AAA", "2020-01-02"), "marketcap"] == pytest.approx(
        1e9
    )  # 1000 musd
    sep, _ = sh.transform_sep_chunk(pd.read_csv(FIX / "SHARADAR_SEP.csv"), m.ranges)
    sf1, _ = sh.transform_sf1(pd.read_csv(FIX / "SHARADAR_SF1.csv"), m.ranges)
    sb = sf1.rename(columns={"date": "datekey"})[["datekey", "ticker", "sharesbas"]]
    caps = sh.derive_caps(sep, daily, sb).set_index(["ticker", "date"])
    assert caps.loc[("AAA", "2020-01-14"), "shares_outstanding"] == pytest.approx(10_000_000)
    assert caps.loc[("AAA", "2020-01-15"), "shares_outstanding"] == pytest.approx(20_000_000)
    assert caps.loc[("AAA", "2020-01-15"), "cap_source"] == "daily"
    # BBB has no DAILY rows 01-13..01-17: SF1 sharesbas (50M) carried from its datekey
    assert caps.loc[("BBB", "2020-01-14"), "shares_outstanding"] == pytest.approx(50_000_000)
    assert caps.loc[("BBB", "2020-01-14"), "market_cap"] == pytest.approx(54.0 * 50_000_000)
    assert caps.loc[("BBB", "2020-01-14"), "cap_source"] == "sf1"
    # the new DDD holder has no DAILY at all: sharesbas from its own (gap-assigned) filing
    assert caps.loc[("DDD.100005", "2020-01-27"), "shares_outstanding"] == pytest.approx(8_000_000)
    # DAILY cap kept even where closeunadj is missing; shares then come from SF1 sharesbas
    assert caps.loc[("DDD.100004", "2020-01-08"), "market_cap"] == pytest.approx(24e6)
    assert caps.loc[("DDD.100004", "2020-01-08"), "shares_outstanding"] == pytest.approx(1_000_000)
    assert caps.loc[("DDD.100004", "2020-01-08"), "cap_source"] == "daily"
    # no DAILY, no SF1, no raw close -> nothing
    caps2 = sh.derive_caps(sep, None, None)
    assert caps2["market_cap"].isna().all() and (caps2["cap_source"] == "none").all()


def test_sf1_transform_dimensions_and_gap_policy():
    m = _master()
    out, c = sh.transform_sf1(pd.read_csv(FIX / "SHARADAR_SF1.csv"), m.ranges)
    assert set(out["dimension"]) == {"ART", "ARQ"} and c.n_other_dimension == 1  # MRT dropped
    assert c.n_not_in_master == 1  # GGG
    d = out[out["ticker"].str.startswith("DDD")].set_index("date")["ticker"]
    assert d[pd.Timestamp("2019-11-01")] == "DDD.100004"  # filed before the first holder's first price
    assert d[pd.Timestamp("2020-01-20")] == "DDD.100005"  # filed in the gap -> next holder
    assert c.n_gap_assigned == 2


# ----------------------------------------------------------------------------- sync
def test_sync_builds_a_consumable_store(synced):
    store = ParquetStore(str(synced))
    for n in (
        "prices_close",
        "prices_close_raw",
        "prices_volume",
        "market_cap",
        "shares_outstanding",
        "industry",
        "delisting_dates",
        "delisting_reasons",
        "security_master",
        "sharadar_sf1",
        "sharadar_ev",
        "corporate_actions",
        "sp500_membership",
        "book_to_price",
        "earnings_yield",
        "gross_profitability",
        "asset_growth",
    ):
        assert store.exists(n), n
    md = store.as_market_data()
    assert list(md.tickers) == ["AAA", "BBB", "CCC", "DDD.100004", "DDD.100005"]
    assert len(md.dates) == 21
    assert md.close.loc["2020-01-02", "AAA"] == 50.0 and md.close_raw.loc["2020-01-02", "AAA"] == 100.0
    assert md.volume.loc["2020-01-02", "AAA"] == 1000.0
    # delisting: final row on the delisting date, NaN after; the table covers every id
    assert md.close.loc["2020-01-21", "CCC"] == 4.0 and np.isnan(md.close.loc["2020-01-22", "CCC"])
    assert md.delisting_dates["CCC"] == pd.Timestamp("2020-01-21") and pd.isna(md.delisting_dates["AAA"])
    assert set(md.delisting_dates.index) == set(md.tickers)
    pos = md.delisting_positions()
    assert np.isinf(pos[list(md.tickers).index("AAA")]) and pos[list(md.tickers).index("CCC")] == 12
    # reused ticker: disjoint histories under two ids
    assert md.close["DDD.100004"].last_valid_index() == pd.Timestamp("2020-01-10")
    assert md.close["DDD.100005"].first_valid_index() == pd.Timestamp("2020-01-27")
    assert md.industry.to_dict() == {
        "AAA": "Banks",
        "BBB": "BusSv",
        "CCC": "Meals",
        "DDD.100004": "Drugs",
        "DDD.100005": "Other",
    }
    # market cap: DAILY millions -> USD; shares = cap / raw close, split-consistent
    assert md.market_cap.loc["2020-01-02", "AAA"] == pytest.approx(1e9)
    assert md.shares_outstanding.loc["2020-01-14", "AAA"] == pytest.approx(1e7)
    assert md.shares_outstanding.loc["2020-01-15", "AAA"] == pytest.approx(2e7)
    assert md.industry_history is None  # TICKERS has no dated classification


def test_sync_fundamentals_are_pit_and_split_consistent(synced):
    store = ParquetStore(str(synced))
    md = store.as_market_data()
    gp = md.gross_profitability["AAA"]
    # filing dated Fri 2020-01-10 (gp/assets = 400/1200): invisible on its datekey, first
    # visible on the next trading day (datekey + 1 = Sat 01-11 -> Mon 01-13)
    assert gp["2020-01-10"] == pytest.approx(350 / 1100)
    assert gp["2020-01-13"] == pytest.approx(400 / 1200)
    assert md.asset_growth.loc["2020-01-13", "AAA"] == pytest.approx(0.2)  # 1200 / 1000 - 1 (prior-year row)
    assert np.isnan(md.asset_growth.loc["2020-01-10", "AAA"])  # the 2019-09-30 row has no prior-year row
    # B/P = bvps x shares(filing) / market_cap(t): a filing before the panel start uses its
    # own sharesbas, and the 2:1 split on 01-15 leaves the ratio continuous
    bp = md.book_to_price["AAA"]
    assert bp["2020-01-02"] == pytest.approx(21 * 1e7 / 1e9)
    assert bp["2020-01-14"] == pytest.approx(22 * 1e7 / 1.08e9)
    assert bp["2020-01-15"] == pytest.approx(22 * 1e7 / 1.09e9)
    assert md.earnings_yield.loc["2020-01-31", "AAA"] == pytest.approx(2.5 * 1e7 / 1.2e9)
    # MR* rows were not loaded; ART and ARQ are both in the long table
    sf1 = store.read_long("sharadar_sf1")
    assert set(sf1["dimension"]) == {"ART", "ARQ"}


def test_sync_reference_tables_and_sp500_panel(synced):
    store = ParquetStore(str(synced))
    ca = store.read_long("corporate_actions")
    assert set(ca["action"]) == {
        "split",
        "dividend",
        "delisted",
        "listed",
        "acquisitionby",
        "acquisitionof",
        "bankruptcyliquidation",
    }
    assert ca.loc[ca["action"] == "split", "id"].tolist() == ["AAA"]
    assert ca.loc[(ca["action"] == "delisted") & (ca["ticker"] == "DDD"), "id"].tolist() == ["DDD.100004"]
    assert ca.loc[ca["ticker"] == "SPY", "id"].isna().all()  # outside the kept master, kept for reference
    panel = sh.sp500_membership_panel(store)
    assert panel.dtypes.eq(bool).all() and list(panel.columns) == ["AAA", "BBB"]
    assert (
        panel.loc["2020-01-02", "AAA"] and not panel.loc["2020-01-02", "BBB"]
    )  # added 2019, before the panel
    assert panel.loc["2020-01-10", "BBB"] and panel.loc["2020-01-17", "AAA"]
    assert not panel.loc["2020-01-21", "AAA"]  # removed on 01-21 -> out from that date
    ids_panel = sh.sp500_membership_panel(store, ids=["BBB", "CCC"])
    assert list(ids_panel.columns) == ["BBB", "CCC"] and not ids_panel["CCC"].any()
    # not a universe rule: the market data carries no membership
    assert "sp500_membership" not in sh.FUNDAMENTAL_PANELS


def test_sync_is_idempotent_and_resumable(synced):
    store = ParquetStore(str(synced))
    h0 = store.snapshot_hash_all()
    rep = sh.sync(str(synced), raw_dir=str(synced.parent / "raw"), from_files=str(FIX))
    assert all(t.skipped for n, t in rep.tables.items() if n not in ("TICKERS",))
    assert rep.tables["TICKERS"].note == "unchanged (state)"
    assert store.snapshot_hash_all() == h0
    rep2 = sh.sync(str(synced), raw_dir=str(synced.parent / "raw"), from_files=str(FIX), force=True)
    assert not any(t.skipped for t in rep2.tables.values())
    assert store.snapshot_hash_all() == h0  # a forced full rebuild reproduces the same bytes
    state = json.loads((synced / "_sharadar" / "state.json").read_text())
    assert state["SEP"]["done"] and state["SEP"]["mode"] == "full"
    assert (synced / "_sharadar" / "last_sync.json").exists()
    assert "_sharadar" not in store.datasets()


def test_incremental_sync_upserts_only_new_rows(synced, tmp_path):
    root = tmp_path / "store"
    shutil.copytree(synced, root)
    store = ParquetStore(str(root))
    before = store.read_wide("prices_close")
    rep = sh.sync(str(root), raw_dir=str(tmp_path / "raw"), since="2020-02-01", from_files=str(FIX_INC))
    assert rep.mode == "incremental" and rep.tables["SEP"].rows_kept == 7
    after = store.read_wide("prices_close")
    assert len(after) == 23 and after.index[-1] == pd.Timestamp("2020-02-04")
    # every old row except the vendor correction is unchanged
    old = after.loc[before.index]
    assert old.loc["2020-01-31", "AAA"] == 59.5 and before.loc["2020-01-31", "AAA"] == 60.0
    chk = before.copy()
    chk.loc["2020-01-31", "AAA"] = 59.5
    pd.testing.assert_frame_equal(old, chk)
    assert after.loc["2020-02-04", "DDD.100005"] == 37.0
    # derived panels extended: caps on the new days, new BBB filing (02-03) visible from 02-04
    mc = store.read_wide("market_cap")
    assert mc.loc["2020-02-04", "AAA"] == pytest.approx(61.0 * 2e7)
    bp = store.read_wide("book_to_price")["BBB"]
    assert bp["2020-02-03"] == pytest.approx(10 * 5e7 / (61.0 * 5e7))
    assert bp["2020-02-04"] == pytest.approx(11 * 5e7 / (62.0 * 5e7))
    md = store.as_market_data()
    md.validate()
    assert md.delisting_dates["CCC"] == pd.Timestamp("2020-01-21")


def test_sync_from_zip_exports_and_table_file_discovery(tmp_path):
    d = tmp_path / "exports"
    d.mkdir()
    for t in sh.DEFAULT_TABLES:
        (d / f"SHARADAR_{t}_2_f1a2b3.zip").write_bytes(_zip_bytes(t))
    assert [p.name for p in sh.find_table_files(d, "sep")] == ["SHARADAR_SEP_2_f1a2b3.zip"]
    assert sh.find_table_files(d, "SFP") == []
    rep = sh.sync(str(tmp_path / "store"), raw_dir=str(tmp_path / "raw"), from_files=str(d))
    assert rep.tables["SEP"].rows_kept == 67
    assert ParquetStore(str(tmp_path / "store")).as_market_data().close.shape == (21, 5)
    # raw-cache layout: newest full + later increments, oldest first
    c = tmp_path / "cache" / "SEP"
    c.mkdir(parents=True)
    for n in (
        "20260101_full.zip",
        "20260301_full.zip",
        "20260215_since-2026-02-01.zip",
        "20260315_since-2026-03-10.zip",
    ):
        (c / n).write_bytes(b"PK")
    assert [p.name for p in sh.find_table_files(tmp_path / "cache", "SEP")] == [
        "20260301_full.zip",
        "20260315_since-2026-03-10.zip",
    ]


def test_from_files_discovery_mixed_naming_styles(tmp_path, caplog):
    import gzip
    import time

    d = tmp_path / "11-09-26 Sharadar Bulk"  # spaces in the path, as downloaded from the website
    (d / "fundamentals").mkdir(parents=True)
    (d / "ACTIONS").mkdir()
    (d / "SHARADAR_SEP_2f3a9c1b.zip").write_bytes(_zip_bytes("SEP"))
    (d / "sharadar-tickers.zip").write_bytes(_zip_bytes("TICKERS"))
    shutil.copy(FIX / "SHARADAR_DAILY.csv", d / "DAILY.csv")
    with gzip.open(d / "fundamentals" / "sf1.csv.gz", "wb") as f:
        f.write((FIX / "SHARADAR_SF1.csv").read_bytes())
    (d / "ACTIONS" / "export (1).csv").write_bytes(
        (FIX / "SHARADAR_ACTIONS.csv").read_bytes()
    )  # table sub-folder
    old_sp = d / "SHARADAR_SP500_old.csv"
    old_sp.write_text("date,action,ticker,name,contraticker,contraname,note\n2001-01-02,added,CCC,Gamma,,,\n")
    shutil.copy(FIX / "SHARADAR_SP500.csv", d / "sharadar_sp500_new.csv")
    t_old = time.time() - 3600
    os.utime(old_sp, (t_old, t_old))
    (d / "README.txt").write_text("not data")
    (d / "notes_sep_2026.md").write_text("not data either")
    with caplog.at_level(logging.WARNING):
        found = sh.resolve_sources(d, sh.DEFAULT_TABLES)
    assert {t: [p.name for p in f] for t, f in found.items()} == {
        "TICKERS": ["sharadar-tickers.zip"],
        "SEP": ["SHARADAR_SEP_2f3a9c1b.zip"],
        "DAILY": ["DAILY.csv"],
        "SF1": ["sf1.csv.gz"],
        "ACTIONS": ["export (1).csv"],
        "SP500": ["sharadar_sp500_new.csv"],
    }
    assert "2 files match" in caplog.text and "SHARADAR_SP500_old.csv" in caplog.text
    txt = sh.format_source_map(found)
    assert "table -> file" in txt and "sf1.csv.gz" in txt
    rep = sh.sync(str(tmp_path / "store"), raw_dir=str(tmp_path / "raw"), from_files=str(d))
    assert rep.tables["SEP"].rows_kept == 67 and rep.tables["SF1"].rows_kept == 11
    assert rep.tables["SP500"].rows_kept == 5  # the newer file
    md = ParquetStore(str(tmp_path / "store")).as_market_data()
    assert md.close.shape == (21, 5)
    # an optional table without a file is skipped with a warning; a required one fails helpfully
    (d / "sharadar_sp500_new.csv").unlink()
    os.remove(old_sp)
    rep2 = sh.sync(str(tmp_path / "store"), raw_dir=str(tmp_path / "raw"), from_files=str(d), force=True)
    assert rep2.tables["SP500"].skipped and any("SP500" in w for w in rep2.warnings)
    (d / "DAILY.csv").unlink()
    with pytest.raises(sh.SharadarError) as ei:
        sh.resolve_sources(d, sh.DEFAULT_TABLES)
    msg = str(ei.value)
    assert "['DAILY']" in msg and "SHARADAR_SEP_2f3a9c1b.zip" in msg and "fundamentals/sf1.csv.gz" in msg
    with pytest.raises(sh.SharadarError, match="not a directory"):
        sh.resolve_sources(tmp_path / "nope", ["SEP"])


def test_sync_without_tickers_uses_stored_master(synced, tmp_path):
    root = tmp_path / "store"
    shutil.copytree(synced, root)
    rep = sh.sync(str(root), raw_dir=str(tmp_path / "raw"), from_files=str(FIX), tables=["SEP"], force=True)
    assert rep.master["n_ids"] == 5 and "TICKERS" not in rep.tables
    with pytest.raises(sh.SharadarError, match="security_master"):
        sh.sync(str(tmp_path / "empty"), raw_dir=str(tmp_path / "raw"), from_files=str(FIX), tables=["SEP"])


def test_optional_table_is_stored_verbatim(tmp_path):
    d = tmp_path / "files"
    d.mkdir()
    for t in ("TICKERS", "SEP"):
        shutil.copy(FIX / f"SHARADAR_{t}.csv", d)
    (d / "SHARADAR_EVENTS.csv").write_text(
        "ticker,date,eventcodes\nAAA,2020-01-03,21|33\nDDD,2020-01-28,13\nSPY,2020-01-03,1\n"
    )
    rep = sh.sync(
        str(tmp_path / "store"),
        raw_dir=str(tmp_path / "raw"),
        from_files=str(d),
        tables=["TICKERS", "SEP", "EVENTS"],
    )
    ev = ParquetStore(str(tmp_path / "store")).read_long("sharadar_events")
    assert rep.tables["EVENTS"].rows_kept == 2 and list(ev["ticker"]) == ["AAA", "DDD.100005"]
    assert list(ev["sharadar_ticker"]) == ["AAA", "DDD"] and ev["eventcodes"].tolist() == ["21|33", "13"]
    with pytest.raises(sh.SharadarError, match="unknown tables"):
        sh.sync(str(tmp_path / "s2"), raw_dir=str(tmp_path / "raw"), from_files=str(d), tables=["NOPE"])


# ----------------------------------------------------------------------------- download layer
def test_download_layer_polls_until_fresh_and_caches(tmp_path, caplog):
    sess = FakeSession()
    client = sh.NasdaqDataLinkClient(KEY, session=sess, poll_interval=0.0)
    with caplog.at_level(logging.DEBUG):
        rep = sh.sync(str(tmp_path / "store"), raw_dir=str(tmp_path / "raw"), client=client)
    assert all(v == 2 for v in sess.polls.values())  # "creating" then "fresh" per table
    zips = sorted(p.relative_to(tmp_path / "raw").as_posix() for p in (tmp_path / "raw").rglob("*.zip"))
    day = pd.Timestamp.today().strftime("%Y%m%d")
    assert zips == [f"{t}/{day}_full.zip" for t in sorted(sh.DEFAULT_TABLES)]
    assert rep.tables["SEP"].rows_kept == 67
    assert KEY not in caplog.text and KEY not in rep.format() and KEY not in json.dumps(rep.to_dict())
    # downloads are not repeated: a second sync reuses the cached zips without any HTTP call
    n_calls = len(sess.calls)
    sh.sync(str(tmp_path / "store"), raw_dir=str(tmp_path / "raw"), client=client)
    assert len(sess.calls) == n_calls
    # incremental export passes the filter and lands in a since-file for the incremental tables only
    sh.sync(
        str(tmp_path / "store"),
        raw_dir=str(tmp_path / "raw"),
        client=client,
        since="2026-03-01",
        tables=["TICKERS", "SEP"],
    )
    sep_calls = [p for u, p in sess.calls if u.endswith("/SEP.json")]
    assert sep_calls[-1]["lastupdated.gte"] == "2026-03-01"
    assert (tmp_path / "raw" / "SEP" / f"{day}_since-2026-03-01.zip").exists()
    assert not (tmp_path / "raw" / "TICKERS" / f"{day}_since-2026-03-01.zip").exists()


def test_download_errors_never_leak_the_key(tmp_path, monkeypatch):
    client = sh.NasdaqDataLinkClient(KEY, session=FakeSession(fail_table="SEP"), poll_interval=0.0)
    with pytest.raises(sh.SharadarDownloadError) as ei:
        client.export_table("SEP", tmp_path / "x.zip")
    assert "HTTP 403" in str(ei.value) and KEY not in str(ei.value) and "<redacted>" in str(ei.value)

    class Boom:
        def get(self, url, params=None, timeout=None, stream=False):
            raise ConnectionError(f"connection reset url={url}?api_key={params['api_key']}")

    with pytest.raises(sh.SharadarDownloadError) as ei:
        sh.NasdaqDataLinkClient(KEY, session=Boom()).export_status("DAILY")
    assert KEY not in str(ei.value) and ei.value.__cause__ is None
    with pytest.raises(sh.SharadarDownloadError, match="no API key"):
        sh.NasdaqDataLinkClient("")
    # no key anywhere -> a clear error before any network use
    monkeypatch.delenv(sh.API_KEY_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(sh.SharadarDownloadError, match="from-files"):
        sh.sync(str(tmp_path / "store"), raw_dir=str(tmp_path / "raw"), tables=["TICKERS"])


def test_default_session_is_requests_with_key_from_env(tmp_path, monkeypatch):
    pytest.importorskip("requests")
    created: list[FakeSession] = []

    def factory():
        s = FakeSession()
        created.append(s)
        return s

    monkeypatch.setattr(sh.requests, "Session", factory)
    monkeypatch.setattr(sh.time, "sleep", lambda s: None)  # the default poll interval is 5 s
    monkeypatch.setenv(sh.API_KEY_ENV, KEY)
    rep = sh.sync(str(tmp_path / "store"), raw_dir=str(tmp_path / "raw"), tables=["TICKERS", "ACTIONS"])
    assert created and created[0].polls == {"TICKERS": 2, "ACTIONS": 2}
    assert rep.tables["ACTIONS"].rows_kept == 9
    assert rep.delisting_reasons == {"acquisition": 1, "performance": 1, "voluntary": 0, "unknown": 0}


def test_dry_run_writes_nothing(tmp_path):
    rep = sh.sync(str(tmp_path / "store"), raw_dir=str(tmp_path / "raw"), from_files=str(FIX), dry_run=True)
    assert rep.dry_run and "DRY RUN" in rep.format() and "SHARADAR_SEP.csv" in rep.format()
    assert ParquetStore(str(tmp_path / "store")).datasets() == []
    # API mode without a key still plans (and names the would-be cache files) without network
    rep2 = sh.sync(
        str(tmp_path / "store2"),
        raw_dir=str(tmp_path / "raw"),
        dry_run=True,
        tables=["SEP"],
        api_key="x",
        client=sh.NasdaqDataLinkClient("x", session=FakeSession()),
    )
    assert "(to download)" in rep2.format()


# ----------------------------------------------------------------------------- delisting reasons
def _dr_actions(rows):
    return pd.DataFrame(rows, columns=["date", "action", "ticker", "id"])


def _dr_master(rows):
    return pd.DataFrame(rows, columns=["id", "ticker", "lastpricedate", "isdelisted"])


def test_derive_delisting_reasons_mapping_precedence_and_window():
    master = _dr_master(
        [
            ("ACQ", "ACQ", "2020-03-10", True),  # acquisitionby on the day
            ("BKR", "BKR", "2020-03-10", True),  # bankruptcy + acquisition: performance wins
            ("VOL", "VOL", "2020-03-10", True),  # voluntary, dated 5 business days later
            ("PLN", "PLN", "2020-03-10", True),  # plain delisted row only
            ("NON", "NON", "2020-03-10", True),  # no action row at all
            ("FAR", "FAR", "2020-03-10", True),  # mapped action 6 business days away: outside
            ("MRG", "MRG", "2020-03-10", True),  # mergerto, 3 business days before
            ("BUY", "BUY", "2020-03-10", True),  # acquisitionof (it bought something): not mapped
            ("LIV", "LIV", "2020-03-10", False),  # alive: no row
            ("RE.1", "RE", "2020-03-10", True),  # reused ticker, old holder
            ("RE.2", "RE", "2021-06-01", True),  # reused ticker, later holder
        ]
    )
    actions = _dr_actions(
        [
            ("2020-03-10", "acquisitionby", "ACQ", "ACQ"),
            ("2020-03-10", "delisted", "ACQ", "ACQ"),
            ("2020-03-11", "acquisitionby", "BKR", "BKR"),
            ("2020-03-12", "bankruptcyliquidation", "BKR", "BKR"),
            ("2020-03-17", "voluntarydelisting", "VOL", None),  # +5 business days, id unresolved
            ("2020-03-10", "delisted", "PLN", "PLN"),
            ("2020-03-18", "regulatorydelisting", "FAR", "FAR"),  # +6 business days
            ("2020-03-05", "mergerto", "MRG", "MRG"),
            ("2020-03-10", "acquisitionof", "BUY", "BUY"),
            ("2020-03-10", "delisted", "BUY", "BUY"),
            ("2020-03-10", "bankruptcyliquidation", "LIV", "LIV"),
            ("2020-03-10", "regulatorydelisting", "RE", "RE.1"),
            ("2021-06-01", "acquisitionby", "RE", "RE.2"),
            ("2021-06-02", "bankruptcyliquidation", "RE", "RE.1"),  # resolved to the other holder: ignored
            ("2020-03-10", "acquisitionby", "ZZZ", None),  # ticker outside the master
        ]
    )
    out, rep = sh.derive_delisting_reasons(actions, master)
    assert list(out.columns) == list(sh.DELISTING_REASON_COLUMNS)
    got = out.set_index("id")
    assert got["reason"].to_dict() == {
        "ACQ": "acquisition",
        "BKR": "performance",
        "VOL": "voluntary",
        "PLN": "unknown",
        "NON": "unknown",
        "FAR": "unknown",
        "MRG": "acquisition",
        "BUY": "unknown",
        "RE.1": "performance",
        "RE.2": "acquisition",
    }
    assert got["source_action"].to_dict() == {
        "ACQ": "acquisitionby",
        "BKR": "bankruptcyliquidation",
        "VOL": "voluntarydelisting",
        "PLN": "delisted",
        "NON": "",
        "FAR": "",
        "MRG": "mergerto",
        "BUY": "delisted",
        "RE.1": "regulatorydelisting",
        "RE.2": "acquisitionby",
    }
    assert got.loc["RE.2", "delisting_date"] == pd.Timestamp("2021-06-01")
    assert rep.n_delisted == 10
    assert rep.by_reason == {"acquisition": 3, "performance": 2, "voluntary": 1, "unknown": 4}
    assert rep.by_year == {
        2020: {"acquisition": 2, "performance": 2, "voluntary": 1, "unknown": 4},
        2021: {"acquisition": 1, "performance": 0, "voluntary": 0, "unknown": 0},
    }
    # unknowns: what was seen next to them, and the mapped action just outside the window
    assert rep.unknown_actions == {"delisted": 2, "(no action row)": 2, "acquisitionof": 1}
    assert rep.unknown_mapped_in_hint_window == {"regulatorydelisting": 1}
    assert (
        rep.actions_near_delistings["acquisitionby"] == 3
        and rep.actions_near_delistings["acquisitionof"] == 1
    )
    txt = rep.format()
    assert (
        "by reason" in txt
        and "by year" in txt
        and "(no action row)" in txt
        and "acquisitionby->acquisition" in txt
    )
    # a corrected mapping (e.g. after reading the real counts) is one argument away
    out2, _ = sh.derive_delisting_reasons(actions, master, mapping={"acquisitionof": "acquisition"})
    assert out2.set_index("id")["reason"]["BUY"] == "acquisition"
    with pytest.raises(ValueError, match="not delisting reasons"):
        sh.derive_delisting_reasons(actions, master, mapping={"delisted": "bankrupt"})
    # a wider window picks up the far action
    out3, _ = sh.derive_delisting_reasons(actions, master, window_days=6)
    assert out3.set_index("id")["reason"]["FAR"] == "performance"


def test_sync_writes_delisting_reasons_and_market_data_loads_them(synced):
    store = ParquetStore(str(synced))
    dr = store.read_long("delisting_reasons").set_index("id")
    # CCC: acquisitionby the day after its last price; DDD (old holder): the bankruptcy row is
    # dated after its price range, so its id is unresolved and it is matched by ticker
    assert dr["reason"].to_dict() == {"CCC": "acquisition", "DDD.100004": "performance"}
    assert dr["source_action"].to_dict() == {"CCC": "acquisitionby", "DDD.100004": "bankruptcyliquidation"}
    assert dr.loc["CCC", "delisting_date"] == pd.Timestamp("2020-01-21")
    md = store.as_market_data()
    assert md.delisting_reasons is not None
    assert md.delisting_reasons.to_dict() == {"CCC": "acquisition", "DDD.100004": "performance"}
    from qts.core.config import CostConfig
    from qts.core.delisting import reasons_by_name, returns_by_name

    by = dict(zip(md.tickers, reasons_by_name(md), strict=True))
    assert by["CCC"] == "acquisition" and by["DDD.100004"] == "performance" and by["AAA"] == "unknown"
    r = dict(zip(md.tickers, returns_by_name(md, CostConfig().delisting_returns()), strict=True))
    assert r["CCC"] == 0.0 and r["DDD.100004"] == -0.30 and r["AAA"] == -0.30
    # point in time: before CCC's delisting date its reason is not known
    cut = md.slice_until(pd.Timestamp("2020-01-15"))
    assert cut.delisting_reasons.to_dict() == {"CCC": "unknown", "DDD.100004": "performance"}


def test_derive_from_an_existing_store_without_download(synced, tmp_path, capsys):
    from qts.cli import main

    dst = tmp_path / "store"
    shutil.copytree(synced, dst)
    store = ParquetStore(str(dst))
    before = store.read_long("delisting_reasons")
    store.drop("delisting_reasons")  # a store synced before the table existed
    assert ParquetStore(str(dst)).as_market_data().delisting_reasons is None
    rc = main(["data", "sharadar", "derive", "--store", str(dst), "--json", str(tmp_path / "d.json")])
    out = capsys.readouterr().out
    assert rc == 0 and "by reason" in out and "acquisition" in out and "by year" in out
    assert "2020" in out and "delisting_reasons (2 rows)" in out
    pd.testing.assert_frame_equal(ParquetStore(str(dst)).read_long("delisting_reasons"), before)
    assert json.loads((tmp_path / "d.json").read_text())["by_reason"]["performance"] == 1
    h = ParquetStore(str(dst)).snapshot_hash("delisting_reasons")
    assert main(["data", "sharadar", "derive", "--store", str(dst)]) == 0  # idempotent
    assert ParquetStore(str(dst)).snapshot_hash("delisting_reasons") == h
    capsys.readouterr()
    rc = main(["data", "sharadar", "derive", "--store", str(tmp_path / "empty")])
    assert rc == 1 and "sync" in capsys.readouterr().err


# ----------------------------------------------------------------------------- validate / CLI
def test_validate_report_checks(synced):
    rep = sh.validate(str(synced), sample_size=50)
    by = {c.name: c for c in rep.checks}
    assert by["required_datasets"].status == "PASS"
    assert by["duplicates"].status == "PASS"
    assert by["pit_fundamentals"].status == "PASS" and by["pit_fundamentals"].detail["violations"] == []
    assert by["splits"].status == "PASS" and by["splits"].detail["sample"][0][:2] == ("AAA", "2020-01-15")
    assert by["delisting_vs_actions"].status == "PASS" and by["delisting_vs_actions"].detail["n_matched"] == 2
    assert by["sp500_current_vs_panel"].status == "PASS"
    assert by["market_cap_identity"].detail["median_ratio"] == pytest.approx(1.0)
    assert by["market_cap_identity"].status == "WARN"  # toy caps are far below the real large-cap range
    assert by["missing_close_raw"].status == "WARN" and by["missing_close_raw"].detail["missing"] == 1
    assert by["coverage_by_year"].detail["by_year"] == {2020: {"names": 5, "delisted": 2}}
    assert rep.datasets["prices_close"]["rows"] == 67 and rep.datasets["prices_close"]["n_tickers"] == 5
    txt = rep.format()
    assert "[PASS] pit_fundamentals" in txt and "checks" in txt
    json.dumps(rep.to_dict(), default=str)


def test_validate_flags_a_leaky_store(synced, tmp_path):
    root = tmp_path / "store"
    shutil.copytree(synced, root)
    store = ParquetStore(str(root))
    # make the 01-10 filing of AAA visible on its datekey -> PIT violation
    gp = store.read_wide("gross_profitability")
    gp.loc["2020-01-10", "AAA"] = 400 / 1200
    store.write_wide("gross_profitability", gp)
    # and a duplicate price row
    dup = pd.DataFrame(
        {"date": [pd.Timestamp("2020-01-02")] * 2, "ticker": ["AAA", "AAA"], "value": [1.0, 2.0]}
    )
    store.write_long("prices_close", dup, mode="upsert", kind="wide")
    by = {c.name: c for c in sh.validate(str(root)).checks}
    assert (
        by["pit_fundamentals"].status == "WARN" and by["pit_fundamentals"].detail["violations"][0][0] == "AAA"
    )
    assert by["duplicates"].status == "WARN" and by["duplicates"].detail["prices_close"] == 1


def test_cli_sync_and_validate(tmp_path, capsys):
    from qts.cli import main

    rc = main(
        [
            "data",
            "sharadar",
            "sync",
            "--store",
            str(tmp_path / "store"),
            "--raw",
            str(tmp_path / "raw"),
            "--from-files",
            str(FIX),
            "--tables",
            "TICKERS,SEP,DAILY,SF1,ACTIONS,SP500",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0 and "prices_close (67 rows)" in out and "security_master" in out
    assert "table -> file" in out  # the mapping is printed before the sync runs
    rc = main(
        [
            "data",
            "sharadar",
            "validate",
            "--store",
            str(tmp_path / "store"),
            "--json",
            str(tmp_path / "v.json"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0 and "[PASS] duplicates" in out and (tmp_path / "v.json").exists()
    rc = main(
        [
            "data",
            "sharadar",
            "sync",
            "--store",
            str(tmp_path / "s2"),
            "--from-files",
            str(tmp_path / "nowhere"),
        ]
    )
    assert rc == 1 and "not a directory" in capsys.readouterr().err
    (tmp_path / "half").mkdir()
    shutil.copy(FIX / "SHARADAR_SEP.csv", tmp_path / "half")
    rc = main(
        ["data", "sharadar", "sync", "--store", str(tmp_path / "s3"), "--from-files", str(tmp_path / "half")]
    )
    err = capsys.readouterr().err
    assert rc == 1 and "required table" in err and "SHARADAR_SEP.csv" in err
    rc = main(["data", "sharadar", "validate", "--store", str(tmp_path / "empty")])
    assert rc == 1 and "missing" in capsys.readouterr().out


# ----------------------------------------------------------------------------- end to end
def write_sharadar_files_from_market(md, out: Path, ff48_names=FF48_DEFINITIONS) -> None:
    """Write a synthetic ``MarketData`` as Sharadar-format CSVs (TICKERS, SEP, DAILY,
    SF1, ACTIONS, SP500) so the loader can be checked against known panels."""
    out.mkdir(parents=True, exist_ok=True)
    tickers = list(md.tickers)
    first = md.close.apply(lambda s: s.first_valid_index())
    last = md.close.apply(lambda s: s.last_valid_index())
    dd = md.delisting_dates
    rows = []
    for i, t in enumerate(tickers):
        delisted = dd is not None and t in dd.index and pd.notna(dd[t])
        rows.append(
            {
                "table": "SEP",
                "permaticker": 200000 + i,
                "ticker": t,
                "name": f"Synthetic {t}",
                "exchange": "NYSE" if i % 2 else "NASDAQ",
                "isdelisted": "Y" if delisted else "N",
                "category": "Domestic Common Stock",
                "cusips": f"{i:09d}",
                "siccode": "",
                "sicsector": "",
                "sicindustry": "",
                "famasector": "",
                "famaindustry": ff48_names[int(str(md.industry[t])[-2:]) % 47][1],
                "sector": "",
                "industry": "",
                "scalemarketcap": "",
                "scalerevenue": "",
                "relatedtickers": "",
                "currency": "USD",
                "location": "U.S.A",
                "lastupdated": "2026-01-01",
                "firstadded": "2018-01-01",
                "firstpricedate": first[t].date(),
                "lastpricedate": last[t].date(),
                "firstquarter": "",
                "lastquarter": "",
                "secfilings": "",
                "companysite": "",
            }
        )
    pd.DataFrame(rows).to_csv(out / "SHARADAR_TICKERS.csv", index=False)
    close, raw, vol, mcap, sh_ = md.close, md.close_raw, md.volume, md.market_cap, md.shares_outstanding
    # SEP: close/volume split-adjusted (ratio close/raw), closeadj = close, closeunadj = raw
    long = (
        pd.DataFrame(
            {
                "close": close.stack(future_stack=True),
                "closeunadj": raw.stack(future_stack=True),
                "volume_actual": vol.stack(future_stack=True),
            }
        )
        .dropna(subset=["close"])
        .reset_index()
    )
    long.columns = ["date", "ticker", "close", "closeunadj", "volume_actual"]
    long["volume"] = (long["volume_actual"] * long["closeunadj"] / long["close"]).round(6)
    sep = pd.DataFrame(
        {
            "ticker": long["ticker"],
            "date": long["date"].dt.date,
            "open": long["close"],
            "high": long["close"],
            "low": long["close"],
            "close": long["close"],
            "volume": long["volume"],
            "closeadj": long["close"],
            "closeunadj": long["closeunadj"],
            "lastupdated": "2026-01-01",
        }
    )
    sep.to_csv(out / "SHARADAR_SEP.csv", index=False)
    mc = mcap.stack(future_stack=True).dropna().rename("marketcap").reset_index()
    mc.columns = ["date", "ticker", "marketcap"]
    daily = pd.DataFrame(
        {
            "ticker": mc["ticker"],
            "date": mc["date"].dt.date,
            "lastupdated": "2026-01-01",
            "ev": mc["marketcap"] / 1e6 * 1.1,
            "evebit": 10.0,
            "evebitda": 8.0,
            "marketcap": mc["marketcap"] / 1e6,
            "pb": 1.5,
            "pe": 15.0,
            "ps": 2.0,
        }
    )
    daily.to_csv(out / "SHARADAR_DAILY.csv", index=False)
    # SF1: quarterly ART/ARQ filings 45 days after each calendar quarter end
    qends = pd.date_range(md.dates[0] - pd.offsets.QuarterEnd(2), md.dates[-1], freq="QE")
    sf1 = []
    for t in tickers:
        for q in qends:
            dk = q + pd.Timedelta(days=45)
            pos = md.dates.searchsorted(dk, side="right") - 1
            if pos < 0 or pos >= len(md.dates):
                continue
            d = md.dates[pos]
            if pd.isna(raw.loc[d, t]) or pd.isna(sh_.loc[d, t]):
                continue
            price, shares = float(raw.loc[d, t]), float(sh_.loc[d, t])
            assets = 2.0 * price * shares
            for dim, k in (("ART", 1.0), ("ARQ", 0.25)):
                sf1.append(
                    {
                        "ticker": t,
                        "dimension": dim,
                        "calendardate": q.date(),
                        "datekey": dk.date(),
                        "reportperiod": q.date(),
                        "lastupdated": "2026-01-01",
                        "bvps": 0.6 * price,
                        "eps": 0.05 * price * k,
                        "gp": 0.3 * assets * k,
                        "assets": assets,
                        "equity": 0.6 * price * shares,
                        "sharesbas": shares,
                        "netinc": 0.05 * price * shares * k,
                        "revenue": assets * k,
                        "pb": 1.0 / 0.6,
                        "pe": 20.0,
                    }
                )
    pd.DataFrame(sf1).to_csv(out / "SHARADAR_SF1.csv", index=False)
    acts = []
    ratio = (raw / raw.shift(1)).where(close.notna() & close.shift(1).notna())
    for t in tickers:
        r = ratio[t]
        for d in r.index[(r < 0.6) | (r > 1.67)]:
            acts.append(
                {
                    "date": d.date(),
                    "action": "split",
                    "ticker": t,
                    "name": t,
                    "value": round(1 / r[d], 2),
                    "contraticker": "",
                    "contraname": "",
                }
            )
        if dd is not None and t in dd.index and pd.notna(dd[t]):
            acts.append(
                {
                    "date": dd[t].date(),
                    "action": "delisted",
                    "ticker": t,
                    "name": t,
                    "value": "",
                    "contraticker": "",
                    "contraname": "",
                }
            )
    pd.DataFrame(
        acts, columns=["date", "action", "ticker", "name", "value", "contraticker", "contraname"]
    ).to_csv(out / "SHARADAR_ACTIONS.csv", index=False)
    top = mcap.iloc[0].nlargest(10).index
    sp = [
        {
            "date": "2000-01-03",
            "action": "added",
            "ticker": t,
            "name": t,
            "contraticker": "",
            "contraname": "",
            "note": "",
        }
        for t in top
    ]
    sp += [
        {
            "date": md.dates[-1].date(),
            "action": "current",
            "ticker": t,
            "name": t,
            "contraticker": "",
            "contraname": "",
            "note": "",
        }
        for t in top
    ]
    pd.DataFrame(sp).to_csv(out / "SHARADAR_SP500.csv", index=False)


@pytest.fixture(scope="module")
def synthetic_store(tmp_path_factory):
    from qts.data.synthetic import SyntheticSpec, generate

    md = generate(SyntheticSpec(n_stocks=40, n_days=700, seed=3, n_splits=4, n_delistings=4, n_halts=1))
    root = tmp_path_factory.mktemp("synthetic_sharadar")
    write_sharadar_files_from_market(md, root / "files")
    rep = sh.sync(
        str(root / "store"), raw_dir=str(root / "raw"), from_files=str(root / "files"), chunksize=5_000
    )
    return md, root / "store", rep


def test_end_to_end_synthetic_market_round_trips(synthetic_store):
    md0, store_root, rep = synthetic_store
    assert rep.tables["SEP"].dropped_not_in_master == 0 and rep.tables["SEP"].dropped_out_of_range == 0
    store = ParquetStore(str(store_root))
    md = store.as_market_data()
    assert list(md.tickers) == list(md0.tickers) and md.dates.equals(md0.dates)
    kw = {"check_names": False, "check_freq": False, "check_index_type": False}  # ns store vs us synthetic
    pd.testing.assert_frame_equal(md.close, md0.close, **kw)
    pd.testing.assert_frame_equal(md.close_raw, md0.close_raw, **kw)
    np.testing.assert_allclose(md.volume.to_numpy(), md0.volume.to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(md.market_cap.to_numpy(), md0.market_cap.to_numpy(), rtol=1e-9)
    so = md.shares_outstanding.where(md0.close_raw.notna())
    np.testing.assert_allclose(
        so.to_numpy(), md0.shares_outstanding.where(md0.close_raw.notna()).to_numpy(), rtol=1e-9
    )
    dd_store = md.delisting_dates.reindex(md0.tickers).astype("datetime64[ns]")
    dd_true = md0.delisting_dates.reindex(md0.tickers).astype("datetime64[ns]")
    pd.testing.assert_series_equal(dd_store, dd_true, check_names=False)
    assert md.industry.isin(FF48_NAMES).all() and (md.industry != "Other").all()
    for n in ("book_to_price", "earnings_yield", "gross_profitability", "asset_growth"):
        p = getattr(md, n)
        assert p is not None and p.shape == md.close.shape and p.notna().to_numpy().mean() > 0.5, n
    # B/P(t) = bvps x sharesbas (latest filing visible at t) / market_cap(t), split-consistent:
    # recompute it from the written SF1 rows for the last date
    sf1 = pd.read_csv(store_root.parent / "files" / "SHARADAR_SF1.csv", parse_dates=["datekey"])
    sf1 = sf1[(sf1["dimension"] == "ART") & (sf1["datekey"] + pd.Timedelta(days=1) <= md.dates[-1])]
    latest = sf1.sort_values("datekey").groupby("ticker").last()
    expect = (latest["bvps"] * latest["sharesbas"]) / md.market_cap.iloc[-1].reindex(latest.index)
    got = md.book_to_price.iloc[-1].reindex(latest.index)
    ok = expect.notna()
    assert ok.sum() > 20
    np.testing.assert_allclose(got[ok].to_numpy(), expect[ok].to_numpy(), rtol=1e-9)
    # names delisted before the last date keep their last ratio (market cap is carried
    # forward by the SF1 transform); they have no close, so they are never eligible
    assert md.eligibility().iloc[-1].reindex(latest.index[~ok]).eq(False).all()
    panel = sh.sp500_membership_panel(store)
    assert panel.shape[0] == len(md.dates) and panel.iloc[-1].sum() == 10


def test_end_to_end_store_feeds_universe_and_pipeline(synthetic_store):
    from qts.core.config import SystemConfig
    from qts.data.universe import build_universe_mask, survivorship_check
    from qts.pipeline import ResearchPipeline

    _, store_root, _ = synthetic_store
    md = ParquetStore(str(store_root)).as_market_data()
    cfg = SystemConfig().with_overrides({"universe.top_n_by_mcap": 30})
    mask = build_universe_mask(md, cfg.universe)
    assert mask.shape == md.close.shape and mask.sum(axis=1).iloc[-1] > 10
    chk = survivorship_check(md, warn=False)
    assert chk["n_delisted"] >= 4 and not chk["suspect"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = ResearchPipeline(cfg, use_ml=False, ledger_path=None, runs_root=None)
        res = pipe.run(md, max_days=4)
    assert len(res.backtest.returns) == 4 and np.isfinite(res.summary["sharpe_net"])
    assert res.backtest.diagnostics["n_failed_solves"] == 0


def test_store_prefilter_by_peak_market_cap(synthetic_store):
    from qts.cli import main

    md0, store_root, _ = synthetic_store
    store = ParquetStore(str(store_root))
    peak = md0.market_cap.max()
    top = store.tickers_by_market_cap(max_names=12)
    assert top == list(peak.nlargest(12).index)
    thr = float(peak.median())
    sel = store.tickers_by_market_cap(min_mcap_ever=thr)
    assert set(sel) == set(peak.index[peak >= thr])
    sub = store.tickers_by_market_cap(start=md0.dates[300], end=md0.dates[400], max_names=5)
    assert sub == list(md0.market_cap.iloc[300:401].max().nlargest(5).index)
    md = store.as_market_data(universe_tickers=top)
    assert list(md.tickers) == sorted(top) and md.close.shape[1] == 12
    pd.testing.assert_frame_equal(
        md.close, md0.close[sorted(top)], check_names=False, check_freq=False, check_index_type=False
    )
    # read_wide pushes the ticker filter into the parquet reader
    assert store.read_long("prices_close", tickers=top[:2])["ticker"].isin(top[:2]).all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = main(
            [
                "run",
                "--store",
                str(store_root),
                "--max-names",
                "20",
                "--max-days",
                "2",
                "--no-save",
                "--ledger",
                "",
            ]
        )
    assert rc == 0


@pytest.mark.slow
def test_prefilter_memory_on_a_large_store(tmp_path):
    """5,000 ids x 2,000 dates: peak RSS of a pre-filtered load (reported, bounded loosely)."""
    import resource

    import pyarrow as pa
    import pyarrow.parquet as pq

    n_ids, n_days = 5000, 2000
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    ids = np.array([f"S{i:05d}" for i in range(n_ids)])
    rng = np.random.default_rng(0)
    store = ParquetStore(str(tmp_path / "store"))
    for name in ("prices_close", "prices_close_raw", "prices_volume", "market_cap"):
        for y, idx in dates.groupby(dates.year).items():
            d = np.repeat(idx.to_numpy(), n_ids)
            t = np.tile(ids, len(idx))
            v = rng.lognormal(3 if name != "market_cap" else 21, 1, size=len(d))
            path = Path(store._dir(name)) / f"year={y}" / "data.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({"date": d, "ticker": t, "value": v}), path, compression="zstd")
        store._write_meta(name, {"kind": "wide", "columns": ["date", "ticker", "value"], "partitioned": True})
    store.write_long("industry", pd.DataFrame({"ticker": ids, "industry": "Other"}), mode="replace")
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    top = store.tickers_by_market_cap(max_names=1000)
    md = store.as_market_data(universe_tickers=top)
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert md.close.shape == (n_days, 1000)
    peak_gb = (rss1 - rss0) / (1024**2 if os.uname().sysname == "Linux" else 1024**3)
    print(f"\nprefilter 1000/5000 ids x {n_days} days: peak RSS growth {peak_gb:.2f} GB")
    assert peak_gb < 6.0
