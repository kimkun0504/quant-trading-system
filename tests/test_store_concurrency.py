"""ParquetStore locking and NaN-in-upsert semantics (audit 2 #9), dated-industry
back-fill (#13) and the ``delisting_dates`` dataset."""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pandas as pd

from qts.data.store import ParquetStore
from qts.data.synthetic import SyntheticSpec, generate

DATES = pd.bdate_range("2024-01-02", periods=40)


def _writer(root: str, ticker: str) -> None:
    st = ParquetStore(root)
    for i, d in enumerate(DATES):
        st.write_wide("px", pd.DataFrame({ticker: [float(i)]}, index=pd.DatetimeIndex([d])))


def _reader(root: str, n: int, q) -> None:
    st = ParquetStore(root)
    errors = 0
    for _ in range(n):
        try:
            if st.exists("px"):
                st.read_wide("px")
        except FileNotFoundError:
            pass
        except Exception:  # any deserialisation error is the failure
            errors += 1
    q.put(errors)


def test_concurrent_upserts_keep_every_row_and_readers_see_snapshots(tmp_path):
    root = str(tmp_path / "store")
    ParquetStore(root)
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    procs = [ctx.Process(target=_writer, args=(root, t)) for t in ("X", "Y")]
    procs.append(ctx.Process(target=_reader, args=(root, 150, q)))
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)
        assert p.exitcode == 0
    assert q.get(timeout=5) == 0
    back = ParquetStore(root).read_wide("px")
    assert back.notna().sum().sum() == 80  # 2 writers x 40 rows, none lost
    np.testing.assert_array_equal(back["X"].to_numpy(), np.arange(40.0))


def test_sparse_partial_redelivery_does_not_delete(tmp_path):
    st = ParquetStore(tmp_path / "s")
    d = pd.bdate_range("2024-01-02", periods=3)
    st.write_wide("si", pd.DataFrame({"X": [1.0, 2.0, 3.0], "Y": [10.0, 20.0, 30.0]}, index=d))
    # re-delivery of one day with the full ticker set but no Y value (sparse panel)
    st.write_wide("si", pd.DataFrame({"X": [5.0], "Y": [np.nan]}, index=d[:1]))
    back = st.read_wide("si")
    assert back.loc[d[0], "X"] == 5.0 and back.loc[d[0], "Y"] == 10.0  # kept (default keep_existing)
    st.write_wide("si", pd.DataFrame({"X": [np.nan], "Y": [np.nan]}, index=d[1:2]), nan_policy="delete")
    back = st.read_wide("si").reindex(d)
    assert np.isnan(back.loc[d[1], "Y"]) and np.isnan(back.loc[d[1], "X"]) and back.loc[d[2], "Y"] == 30.0


def test_store_industry_history_unknown_before_first_label_and_delisting_dates(tmp_path):
    md = generate(SyntheticSpec(n_stocks=12, n_days=60, seed=3, n_delistings=2))
    first = md.dates[30]
    ih = pd.DataFrame([md.industry.reindex(md.tickers).to_numpy()], index=[first], columns=md.tickers)
    ih.iloc[0, 1] = np.nan  # S0001 never classified
    md.industry_history = ih
    st = ParquetStore(tmp_path / "s")
    st.write_market_data(md)
    back = st.as_market_data()
    early = back.industry_as_of(md.dates[5])
    assert (early == "UNKNOWN").all()  # no current label back-filled before the first dated row
    late = back.industry_as_of(md.dates[40])
    assert late["S0000"] == md.industry["S0000"] and late["S0001"] == "UNKNOWN"
    pd.testing.assert_series_equal(
        back.delisting_dates.sort_index(),
        md.delisting_dates.sort_index(),
        check_names=False,
        check_freq=False,
    )
