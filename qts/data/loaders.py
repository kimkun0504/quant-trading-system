"""Offline-testable parsers and optional downloaders for external data.

Everything here is written against *documented file layouts* and unit-tested
with inline fixtures; nothing in the test-suite touches the network.  The
download helpers (``download_ken_french``, ``yfinance_to_market_data``) are
thin wrappers that the user must run locally — the build sandbox could not
reach ``mba.tuck.dartmouth.edu`` or Yahoo.

Contents
--------
* Ken French library: ``parse_ken_french_zip`` / ``parse_ken_french_csv`` for
  the daily 5-factor (``F-F_Research_Data_5_Factors_2x3_daily``) and momentum
  (``F-F_Momentum_Factor_daily``) files, plus ``download_ken_french``.
* FINRA equity short interest: ``parse_finra_short_interest`` and the PIT
  ``to_short_interest_ratio`` (settlement -> publication lag).
* ``yfinance_to_market_data`` — prototyping only; survivorship-biased.
* Fama-French 48 industries from SIC: ``sic_to_ff48``, ``ff48_from_sic_series``.
* Sharadar SF1 -> PIT fundamentals keyed on ``datekey`` (filing date):
  ``sharadar_sf1_to_pit_fundamentals``.
* Generic PIT helper ``pit_forward_fill`` (value visible from its
  availability date onward, never before).
"""
from __future__ import annotations

import io
import os
import re
import warnings
import zipfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd

from qts.core.types import MarketData, Wide

BytesOrPath = Union[bytes, str, "os.PathLike[str]"]

# =============================================================================
# Generic point-in-time forward fill
# =============================================================================
def pit_forward_fill(long: pd.DataFrame, dates: pd.DatetimeIndex, value_col: str = "value",
                     date_col: str = "available_date", ticker_col: str = "ticker") -> Wide:
    """Wide panel on ``dates`` where each (ticker, available_date, value) becomes
    visible on the first date >= ``available_date`` and is forward-filled until
    superseded.  Nothing is visible before its availability date.

    If several rows share (ticker, available_date) the *last* one wins (sort
    the input accordingly, e.g. by calendardate).
    """
    dates = pd.DatetimeIndex(dates).sort_values()
    df = long[[date_col, ticker_col, value_col]].dropna(subset=[date_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.drop_duplicates([ticker_col, date_col], keep="last")
    # snap each availability date to the first panel date >= it
    pos = np.searchsorted(dates.values, df[date_col].values, side="left")
    ok = pos < len(dates)
    df = df[ok]
    df["_snap"] = dates[pos[ok]]
    df = df.drop_duplicates([ticker_col, "_snap"], keep="last")
    wide = df.pivot(index="_snap", columns=ticker_col, values=value_col)
    wide = wide.reindex(dates).ffill()
    wide.index.name = None
    wide.columns.name = "ticker"
    return wide


# =============================================================================
# Ken French data library
# =============================================================================
KEN_FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
KEN_FRENCH_DATASETS = {
    "F-F_Research_Data_5_Factors_2x3_daily": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "F-F_Research_Data_Factors_daily": "F-F_Research_Data_Factors_daily_CSV.zip",
    "F-F_Momentum_Factor_daily": "F-F_Momentum_Factor_daily_CSV.zip",
    "F-F_ST_Reversal_Factor_daily": "F-F_ST_Reversal_Factor_daily_CSV.zip",
    "F-F_LT_Reversal_Factor_daily": "F-F_LT_Reversal_Factor_daily_CSV.zip",
}
_KF_MISSING = (-99.99, -999.0, -99.0)


def parse_ken_french_csv(text: str, block: int = 0) -> pd.DataFrame:
    """Parse the CSV inside a Ken French zip.

    Layout: free-text preamble lines, then a header row whose first field is
    empty (``,Mkt-RF,SMB,HML,RMW,CMA,RF``), then rows ``YYYYMMDD,v1,v2,...``
    in percent.  Monthly files contain several blocks (monthly, annual)
    separated by blank lines / titles; ``block`` selects which (0 = first).
    Values are returned as decimals with a tz-naive ``DatetimeIndex``;
    ``-99.99`` / ``-999`` sentinels become NaN.  Column names are stripped.
    """
    lines = text.splitlines()
    blocks: List[Tuple[List[str], List[str]]] = []
    header: Optional[List[str]] = None
    rows: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            if header is not None and rows:
                blocks.append((header, rows))
            header, rows = None, []
            continue
        first = s.split(",")[0].strip()
        if re.fullmatch(r"\d{6}|\d{8}|\d{4}", first):
            if header is None:
                header = [""] + [f"c{i}" for i in range(len(s.split(",")) - 1)]
            rows.append(s)
        elif s.startswith(",") or first.lower() in ("date", ""):
            if header is not None and rows:
                blocks.append((header, rows))
                rows = []
            header = [c.strip() for c in s.split(",")]
        # else: preamble / title line -> ignore
    if header is not None and rows:
        blocks.append((header, rows))
    if not blocks:
        raise ValueError("no data block found in Ken French CSV")
    hdr, body = blocks[block]
    df = pd.read_csv(io.StringIO("\n".join(body)), header=None, index_col=0)
    df.columns = hdr[1:len(df.columns) + 1]
    df.columns = [c.strip() for c in df.columns]
    idx = df.index.astype(str)
    if len(idx[0]) == 8:
        df.index = pd.to_datetime(idx, format="%Y%m%d")
    elif len(idx[0]) == 6:
        df.index = pd.to_datetime(idx, format="%Y%m") + pd.offsets.MonthEnd(0)
    else:
        df.index = pd.to_datetime(idx, format="%Y") + pd.offsets.YearEnd(0)
    df.index.name = "date"
    df = df.apply(pd.to_numeric, errors="coerce")
    for m in _KF_MISSING:
        df = df.mask(np.isclose(df, m))
    return df / 100.0


def parse_ken_french_zip(bytes_or_path: BytesOrPath, block: int = 0) -> pd.DataFrame:
    """Open a ``*_CSV.zip`` from the French library (bytes or path) and parse its CSV.
    A plain ``.csv`` path / CSV bytes is also accepted."""
    raw = bytes_or_path if isinstance(bytes_or_path, (bytes, bytearray)) else open(bytes_or_path, "rb").read()
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")] or z.namelist()
            text = z.read(names[0]).decode("latin-1")
    else:
        text = raw.decode("latin-1")
    return parse_ken_french_csv(text, block=block)


def download_ken_french(dataset_name: str, timeout: int = 60) -> pd.DataFrame:
    """Download and parse a French-library dataset (needs network; not used in tests).

    The build sandbox could not reach mba.tuck.dartmouth.edu — run this on your
    own machine and persist the result with ``ParquetStore.write_long``.
    """
    import requests  # local import: optional dependency
    fname = KEN_FRENCH_DATASETS.get(dataset_name, dataset_name if dataset_name.endswith(".zip") else f"{dataset_name}_CSV.zip")
    r = requests.get(KEN_FRENCH_BASE + fname, timeout=timeout)
    r.raise_for_status()
    return parse_ken_french_zip(r.content)


def ken_french_factor_panel(ff5: pd.DataFrame, mom: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Join 5 factors + momentum into one daily frame (columns Mkt-RF, SMB, HML, RMW, CMA, RF[, Mom])."""
    out = ff5.copy()
    if mom is not None:
        m = mom.copy()
        m.columns = ["Mom" if c.lower().startswith("mom") else c for c in m.columns]
        out = out.join(m, how="left")
    return out


# =============================================================================
# FINRA equity short interest
# =============================================================================
# FINRA publishes consolidated equity short interest twice a month as a
# pipe-delimited text file (one per settlement date; also via the Query API).
# Expected columns (names as published; matching is case-insensitive and a
# few aliases are accepted):
#
#   accountingYearMonthNumber | symbolCode | issueName |
#   issuerServicesGroupExchangeCode | marketClassCode |
#   currentShortPositionQuantity | previousShortPositionQuantity |
#   stockSplitFlag | averageDailyVolumeQuantity | daysToCoverQuantity |
#   revisionFlag | changePercent | changePreviousNumber | settlementDate
#
# ``settlementDate`` is YYYYMMDD (or ISO).  Data for settlement date S is
# *published* roughly 8 business days later (FINRA's dissemination schedule),
# so a PIT panel must not reveal it before S + lag.
_FINRA_ALIASES = {
    "settlement_date": ("settlementdate", "settlement_date", "settlement date", "settlementdt"),
    "symbol": ("symbolcode", "symbol", "ticker", "issuesymbolidentifier"),
    "short_interest": ("currentshortpositionquantity", "shortinterest", "short_interest", "currentshortposition",
                       "current short position"),
    "avg_daily_volume": ("averagedailyvolumequantity", "avgdailyvolume", "averagedailyvolume", "avg_daily_volume"),
    "days_to_cover": ("daystocoverquantity", "daystocover", "days_to_cover"),
    "previous_short_interest": ("previousshortpositionquantity", "previousshortposition"),
    "market_class": ("marketclasscode", "market_class"),
    "issue_name": ("issuename", "issue_name", "name"),
}


def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9_ ]", "", str(c).strip().lower())


def parse_finra_short_interest(csv_path_or_bytes: BytesOrPath, sep: Optional[str] = None) -> pd.DataFrame:
    """Parse a FINRA equity short interest file into a long DataFrame with
    columns ``settlement_date, symbol, short_interest, avg_daily_volume,
    days_to_cover`` (+ ``previous_short_interest, market_class, issue_name``
    when present).  Delimiter is sniffed (``|`` preferred, then ``,``/tab)."""
    raw = csv_path_or_bytes if isinstance(csv_path_or_bytes, (bytes, bytearray)) else open(csv_path_or_bytes, "rb").read()
    text = raw.decode("utf-8-sig", errors="replace")
    if sep is None:
        head = text.splitlines()[0] if text else ""
        sep = "|" if "|" in head else ("\t" if "\t" in head else ",")
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
    norm_cols = {_norm(c): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for target, aliases in _FINRA_ALIASES.items():
        src = next((norm_cols[a] for a in aliases if a in norm_cols), None)
        if src is not None:
            out[target] = df[src]
    missing = [c for c in ("settlement_date", "symbol", "short_interest") if c not in out.columns]
    if missing:
        raise ValueError(f"FINRA file missing columns {missing}; found {list(df.columns)}")
    sd = out["settlement_date"].astype(str).str.strip()
    out["settlement_date"] = pd.to_datetime(sd, format="%Y%m%d", errors="coerce").fillna(
        pd.to_datetime(sd, errors="coerce"))
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    for c in ("short_interest", "avg_daily_volume", "days_to_cover", "previous_short_interest"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", ""), errors="coerce")
    for c in ("avg_daily_volume", "days_to_cover"):
        if c not in out.columns:
            out[c] = np.nan
    cols = ["settlement_date", "symbol", "short_interest", "avg_daily_volume", "days_to_cover"]
    cols += [c for c in out.columns if c not in cols]
    return out[cols].dropna(subset=["settlement_date", "symbol"]).sort_values(["settlement_date", "symbol"]).reset_index(drop=True)


def finra_publication_date(settlement_date, lag_business_days: int = 8) -> pd.Timestamp:
    ts = pd.Timestamp(settlement_date)
    if pd.isna(ts):
        raise ValueError("settlement_date is NaT")
    return cast(pd.Timestamp, ts) + pd.offsets.BDay(int(lag_business_days))


def to_short_interest_ratio(long_df: pd.DataFrame, shares_outstanding: Wide, lag_business_days: int = 8,
                            dates: Optional[pd.DatetimeIndex] = None) -> Wide:
    """PIT short-interest ratio = short interest / shares outstanding.

    Each settlement-date observation becomes visible on
    ``settlement_date + lag_business_days`` (default 8, FINRA's publication
    lag) and is forward-filled until the next publication.  Before its
    publication date the value is NaN (or the previous published value).

    ``shares_outstanding`` must be the **actual share count** per date (wide
    panel; e.g. ``MarketData.shares_outstanding`` or SF1 ``sharesbas`` made PIT),
    on the same basis as FINRA's share counts.  Do not derive it as
    ``market_cap / close`` with a split-adjusted close: across a later split
    that understates the historical share count by the split factor and
    inflates the ratio.  The ratio uses the shares outstanding *as of each
    date* (forward-filled).  See :func:`shares_outstanding_panel`.
    """
    if not isinstance(shares_outstanding, pd.DataFrame):
        raise TypeError("shares_outstanding must be a wide DataFrame (date x ticker) of actual share counts")
    idx = pd.DatetimeIndex(dates) if dates is not None else shares_outstanding.index
    df = long_df[["settlement_date", "symbol", "short_interest"]].rename(columns={"symbol": "ticker", "short_interest": "value"}).copy()
    df["available_date"] = pd.to_datetime(df["settlement_date"]) + pd.offsets.BDay(int(lag_business_days))
    df = df.sort_values(["ticker", "settlement_date"])
    si = pit_forward_fill(df, idx)
    cols = si.columns.intersection(shares_outstanding.columns)
    so = shares_outstanding.reindex(index=idx, columns=cols).ffill()
    return si[cols] / so.replace(0, np.nan)


def shares_outstanding_panel(md: MarketData) -> Wide:
    """Actual shares outstanding for :func:`to_short_interest_ratio`.

    ``md.shares_outstanding`` if present, else ``market_cap / close_raw``.
    Raises if neither is available: deriving share counts from the
    split-adjusted ``close`` is wrong across splits (finding #44).
    """
    if md.shares_outstanding is not None:
        return md.shares_outstanding
    if md.close_raw is not None:
        return md.market_cap / md.close_raw.replace(0.0, np.nan)
    raise ValueError("MarketData has neither shares_outstanding nor close_raw; refusing to derive share "
                     "counts from the split-adjusted close")


def short_interest_long_to_wide(long_df: pd.DataFrame, dates: pd.DatetimeIndex, lag_business_days: int = 8,
                                value_col: str = "short_interest") -> Wide:
    """PIT wide panel of a raw short-interest column (shares) on ``dates``."""
    df = long_df[["settlement_date", "symbol", value_col]].rename(columns={"symbol": "ticker", value_col: "value"}).copy()
    df["available_date"] = pd.to_datetime(df["settlement_date"]) + pd.offsets.BDay(int(lag_business_days))
    return pit_forward_fill(df.sort_values(["ticker", "settlement_date"]), dates)


# =============================================================================
# yfinance (prototyping only)
# =============================================================================
def yfinance_to_market_data(tickers: Sequence[str], start: str, end: Optional[str] = None,
                            industry_map: Optional[Dict[str, str]] = None, fetch_shares: bool = True,
                            fetch_industry: bool = False) -> MarketData:
    """Build a MarketData from Yahoo Finance via ``yfinance`` (lazy import).

    **Prototyping only.**  Yahoo data is survivorship-biased (delisted names
    are absent), adjusted closes are revised retroactively, shares outstanding
    are *current* (so historical market cap is close x today's shares — not
    PIT), and ``industry`` comes from ``Ticker.info`` which is current-view.
    Run :func:`qts.data.universe.survivorship_check` on the result and expect
    it to warn.  Never report a Sharpe from this source.
    """
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install yfinance  (optional dependency 'data')") from e
    tickers = list(tickers)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="column", threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
        volume = raw["Volume"].copy()
    else:  # single ticker
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": tickers[0]})
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    volume.index = close.index
    close = close.reindex(columns=tickers)
    volume = volume.reindex(columns=tickers).astype(float)
    shares = pd.Series(np.nan, index=tickers, dtype=float)
    industry = pd.Series(industry_map or {}, dtype=object).reindex(tickers)
    if fetch_shares or fetch_industry:
        for t in tickers:
            try:
                tk = yf.Ticker(t)
                if fetch_shares:
                    fi = getattr(tk, "fast_info", None)
                    s = None
                    if fi is not None:
                        s = fi.get("shares", None) if hasattr(fi, "get") else getattr(fi, "shares", None)
                    if s is None:
                        s = (tk.info or {}).get("sharesOutstanding")
                    shares[t] = float(s) if s else np.nan
                if fetch_industry and pd.isna(industry[t]):
                    industry[t] = (tk.info or {}).get("industry", "UNKNOWN")
            except Exception as e:  # pragma: no cover - network
                warnings.warn(f"yfinance metadata failed for {t}: {e}", stacklevel=2)
    market_cap = close.mul(shares, axis=1)
    industry = industry.fillna("UNKNOWN").astype(str)
    industry.index.name = "ticker"
    close.columns.name = volume.columns.name = market_cap.columns.name = "ticker"
    md = MarketData(close=close, volume=volume, market_cap=market_cap, industry=industry)
    md.validate()
    return md


# =============================================================================
# Fama-French 48 industries (Siccodes48, Ken French data library)
# =============================================================================
# Each entry: (short name, long name, [(lo, hi), ...]) inclusive SIC ranges.
FF48_DEFINITIONS: List[Tuple[str, str, List[Tuple[int, int]]]] = [
    ("Agric", "Agriculture", [(100, 299), (700, 799), (910, 919), (2048, 2048)]),
    ("Food", "Food Products", [(2000, 2046), (2050, 2063), (2070, 2079), (2090, 2092), (2095, 2095), (2098, 2099)]),
    ("Soda", "Candy & Soda", [(2064, 2068), (2086, 2087), (2096, 2097)]),
    ("Beer", "Beer & Liquor", [(2080, 2080), (2082, 2085)]),
    ("Smoke", "Tobacco Products", [(2100, 2199)]),
    ("Toys", "Recreation", [(920, 999), (3650, 3652), (3732, 3732), (3930, 3931), (3940, 3949)]),
    ("Fun", "Entertainment", [(7800, 7833), (7840, 7841), (7900, 7900), (7910, 7911), (7920, 7933), (7940, 7949),
                              (7980, 7980), (7990, 7999)]),
    ("Books", "Printing and Publishing", [(2700, 2749), (2770, 2771), (2780, 2799)]),
    ("Hshld", "Consumer Goods", [(2047, 2047), (2391, 2392), (2510, 2519), (2590, 2599), (2840, 2844),
                                 (3160, 3161), (3170, 3172), (3190, 3199), (3229, 3229), (3260, 3260),
                                 (3262, 3263), (3269, 3269), (3230, 3231), (3630, 3639), (3750, 3751),
                                 (3800, 3800), (3860, 3861), (3870, 3873), (3910, 3911), (3914, 3915),
                                 (3960, 3962), (3991, 3991), (3995, 3995)]),
    ("Clths", "Apparel", [(2300, 2390), (3020, 3021), (3100, 3111), (3130, 3131), (3140, 3151), (3963, 3965)]),
    ("Hlth", "Healthcare", [(8000, 8099)]),
    ("MedEq", "Medical Equipment", [(3693, 3693), (3840, 3851)]),
    ("Drugs", "Pharmaceutical Products", [(2830, 2831), (2833, 2836)]),
    ("Chems", "Chemicals", [(2800, 2829), (2850, 2879), (2890, 2899)]),
    ("Rubbr", "Rubber and Plastic Products", [(3031, 3031), (3041, 3041), (3050, 3053), (3060, 3099)]),
    ("Txtls", "Textiles", [(2200, 2284), (2290, 2295), (2297, 2299), (2393, 2395), (2397, 2399)]),
    ("BldMt", "Construction Materials", [(800, 899), (2400, 2439), (2450, 2459), (2490, 2499), (2660, 2661),
                                         (2950, 2952), (3200, 3200), (3210, 3211), (3240, 3241), (3250, 3259),
                                         (3261, 3261), (3264, 3264), (3270, 3275), (3280, 3281), (3290, 3293),
                                         (3295, 3299), (3420, 3433), (3440, 3442), (3446, 3446), (3448, 3452),
                                         (3490, 3499), (3996, 3996)]),
    ("Cnstr", "Construction", [(1500, 1511), (1520, 1549), (1600, 1799)]),
    ("Steel", "Steel Works Etc", [(3300, 3300), (3310, 3317), (3320, 3325), (3330, 3341), (3350, 3357),
                                  (3360, 3379), (3390, 3399)]),
    ("FabPr", "Fabricated Products", [(3400, 3400), (3443, 3444), (3460, 3479)]),
    ("Mach", "Machinery", [(3510, 3536), (3538, 3538), (3540, 3569), (3580, 3582), (3585, 3586), (3589, 3599)]),
    ("ElcEq", "Electrical Equipment", [(3600, 3600), (3610, 3613), (3620, 3621), (3623, 3629), (3640, 3646),
                                       (3648, 3649), (3660, 3660), (3690, 3692), (3699, 3699)]),
    ("Autos", "Automobiles and Trucks", [(2296, 2296), (2396, 2396), (3010, 3011), (3537, 3537), (3647, 3647),
                                         (3694, 3694), (3700, 3700), (3710, 3711), (3713, 3716), (3790, 3792),
                                         (3799, 3799)]),
    ("Aero", "Aircraft", [(3720, 3721), (3723, 3725), (3728, 3729)]),
    ("Ships", "Shipbuilding, Railroad Equipment", [(3730, 3731), (3740, 3743)]),
    ("Guns", "Defense", [(3760, 3769), (3795, 3795), (3480, 3489)]),
    ("Gold", "Precious Metals", [(1040, 1049)]),
    ("Mines", "Non-Metallic and Industrial Metal Mining", [(1000, 1039), (1050, 1119), (1400, 1499)]),
    ("Coal", "Coal", [(1200, 1299)]),
    ("Oil", "Petroleum and Natural Gas", [(1300, 1300), (1310, 1339), (1370, 1382), (1389, 1389),
                                          (2900, 2912), (2990, 2999)]),
    ("Util", "Utilities", [(4900, 4900), (4910, 4911), (4920, 4925), (4930, 4932), (4939, 4942)]),
    ("Telcm", "Communication", [(4800, 4800), (4810, 4813), (4820, 4822), (4830, 4841), (4880, 4892),
                                (4899, 4899)]),
    ("PerSv", "Personal Services", [(7020, 7021), (7030, 7033), (7200, 7200), (7210, 7212), (7214, 7217),
                                    (7219, 7221), (7230, 7231), (7240, 7241), (7250, 7251), (7260, 7299),
                                    (7395, 7395), (7500, 7500), (7510, 7515), (7520, 7549), (7600, 7600),
                                    (7620, 7620), (7622, 7623), (7629, 7631), (7640, 7641), (7690, 7699),
                                    (8100, 8499), (8600, 8699), (8800, 8899)]),
    ("BusSv", "Business Services", [(2750, 2759), (3993, 3993), (7218, 7218), (7300, 7300), (7310, 7342),
                                    (7349, 7353), (7359, 7372), (7374, 7385), (7389, 7394), (7396, 7397),
                                    (7399, 7399), (7519, 7519), (8700, 8700), (8710, 8713), (8720, 8721),
                                    (8730, 8734), (8740, 8748), (8900, 8911), (8920, 8999), (4220, 4229)]),
    ("Comps", "Computers", [(3570, 3579), (3680, 3689), (3695, 3695), (7373, 7373)]),
    ("Chips", "Electronic Equipment", [(3622, 3622), (3661, 3666), (3669, 3679), (3810, 3810), (3812, 3812)]),
    ("LabEq", "Measuring and Control Equipment", [(3811, 3811), (3820, 3827), (3829, 3839)]),
    ("Paper", "Business Supplies", [(2520, 2549), (2600, 2639), (2670, 2699), (2760, 2761), (3950, 3955)]),
    ("Boxes", "Shipping Containers", [(2440, 2449), (2640, 2659), (3220, 3221), (3410, 3412)]),
    ("Trans", "Transportation", [(4000, 4013), (4040, 4049), (4100, 4100), (4110, 4121), (4130, 4131),
                                 (4140, 4142), (4150, 4151), (4170, 4173), (4190, 4200), (4210, 4219),
                                 (4230, 4231), (4240, 4249), (4400, 4700), (4710, 4712), (4720, 4749),
                                 (4780, 4780), (4782, 4785), (4789, 4789)]),
    ("Whlsl", "Wholesale", [(5000, 5000), (5010, 5015), (5020, 5023), (5030, 5060), (5063, 5065),
                            (5070, 5078), (5080, 5088), (5090, 5094), (5099, 5100), (5110, 5113),
                            (5120, 5122), (5130, 5172), (5180, 5182), (5190, 5199)]),
    ("Rtail", "Retail", [(5200, 5200), (5210, 5231), (5250, 5251), (5260, 5261), (5270, 5271), (5300, 5300),
                         (5310, 5311), (5320, 5320), (5330, 5331), (5334, 5334), (5340, 5349), (5390, 5400),
                         (5410, 5412), (5420, 5469), (5490, 5500), (5510, 5579), (5590, 5700), (5710, 5722),
                         (5730, 5736), (5750, 5799), (5900, 5900), (5910, 5912), (5920, 5932), (5940, 5990),
                         (5992, 5995), (5999, 5999)]),
    ("Meals", "Restaurants, Hotels, Motels", [(5800, 5829), (5890, 5899), (7000, 7000), (7010, 7019),
                                              (7040, 7049), (7213, 7213)]),
    ("Banks", "Banking", [(6000, 6000), (6010, 6036), (6040, 6062), (6080, 6082), (6090, 6100), (6110, 6113),
                          (6120, 6179), (6190, 6199)]),
    ("Insur", "Insurance", [(6300, 6300), (6310, 6331), (6350, 6351), (6360, 6361), (6370, 6379), (6390, 6411)]),
    ("RlEst", "Real Estate", [(6500, 6500), (6510, 6510), (6512, 6515), (6517, 6532), (6540, 6541),
                              (6550, 6553), (6590, 6599), (6610, 6611)]),
    ("Fin", "Trading", [(6200, 6299), (6700, 6700), (6710, 6726), (6730, 6733), (6740, 6779), (6790, 6795),
                        (6798, 6799)]),
    ("Other", "Almost Nothing", [(4950, 4961), (4970, 4971), (4990, 4991)]),
]

FF48_NAMES = [d[0] for d in FF48_DEFINITIONS]
FF48_LONG_NAMES = {d[0]: d[1] for d in FF48_DEFINITIONS}


def _build_sic_lookup() -> np.ndarray:
    table = np.full(10000, -1, dtype=np.int16)
    for i, (_, _, ranges) in enumerate(FF48_DEFINITIONS):
        for lo, hi in ranges:
            # earlier (more specific) assignments win only where unassigned,
            # but the French table is non-overlapping by construction
            seg = table[lo:hi + 1]
            seg[seg < 0] = i
    return table


_SIC_LOOKUP = _build_sic_lookup()


def sic_to_ff48(sic_code, default: str = "Other") -> str:
    """Map a 4-digit SIC code to its Fama-French 48 industry short name.

    SIC codes not covered by any of the 47 named industries fall into
    ``default`` (``"Other"``, which is how French treats unlisted codes, e.g.
    6xxx SICs outside Banks/Insur/RlEst/Fin and all 9xxx public-admin codes).
    """
    try:
        s = int(sic_code)
    except (TypeError, ValueError):
        return default
    if not 0 <= s < 10000:
        return default
    i = int(_SIC_LOOKUP[s])
    return FF48_NAMES[i] if i >= 0 else default


def ff48_from_sic_series(series: pd.Series, default: str = "Other") -> pd.Series:
    """Vectorised :func:`sic_to_ff48` over a Series of SIC codes (NaN -> default)."""
    s = pd.Series(pd.to_numeric(series, errors="coerce"), index=series.index)
    codes = s.fillna(-1).astype(int).to_numpy()
    valid = (codes >= 0) & (codes < 10000)
    idx = np.where(valid, _SIC_LOOKUP[np.clip(codes, 0, 9999)], -1)
    names = np.array(FF48_NAMES + [default], dtype=object)
    out = names[np.where(idx >= 0, idx, len(FF48_NAMES))]
    return pd.Series(out, index=series.index, name="ff48")


# =============================================================================
# Sharadar SF1 -> point-in-time fundamentals
# =============================================================================
SF1_DEFAULT_FIELDS = ("bvps", "pb", "eps", "epsusd", "gp", "assets", "equity", "revenue", "netinc",
                      "sharesbas", "shareswa", "marketcap", "pe", "ev", "ebitda", "fcf", "debt", "liabilities")


def sharadar_sf1_to_pit_fundamentals(sf1_df: pd.DataFrame, dimension: str = "ART", lag_days: int = 1,
                                     dates: Optional[pd.DatetimeIndex] = None,
                                     fields: Optional[Iterable[str]] = None,
                                     close: Optional[Wide] = None,
                                     close_raw: Optional[Wide] = None,
                                     price_is_adjusted: bool = True) -> Dict[str, Wide]:
    """Turn Sharadar SF1 rows into PIT wide panels keyed on ``datekey``.

    Sharadar SF1 columns used: ``ticker, dimension, calendardate, datekey``
    plus any numeric fields.  ``datekey`` is the **filing date** (when the
    numbers became public); ``calendardate`` is the normalised period end.
    Using ``calendardate`` would leak the report ~30-90 days early, so every
    panel here becomes visible at ``datekey + lag_days`` (calendar days,
    snapped forward to the next panel date).  Default ``lag_days=1``: a filing
    dated ``d`` may be released after the close (8-K/10-Q filed at 16:30), so
    it is first usable at the close of the next day; ``lag_days=0`` assumes
    every filing was public before the close of its datekey (optimistic).  Use
    ~90 for sources that only carry period-end dates.

    Dimensions (``dimension``):

    * ``ART`` (default) -- as-reported **trailing twelve months**.  Flow items
      (``eps``, ``netinc``, ``revenue``, ``gp``) are TTM sums, so ``eps/price``
      is an annual earnings yield and ``gp/assets`` an annual gross
      profitability, comparable across firms regardless of fiscal calendar
      and free of seasonality.  Use for valuation / profitability ratios.
    * ``ARQ`` -- as-reported **quarterly**: flow items are single-quarter
      values (``eps/price`` is a *quarterly* yield, ~1/4 of the TTM figure,
      and seasonal); stock items (``assets``, ``equity``, ``bvps``) equal ART's.
      Use when you need quarter-on-quarter changes.
    * ``ARY`` -- as-reported annual (10-K only; slowest to update).
    * ``MR*`` dimensions are *most recent reported* and restate history -- not
      PIT; a warning is issued if requested.

    Prices: ``book_to_price = bvps / P`` and ``earnings_yield = eps / P`` divide
    *as-reported per-share* values by the price, so ``P`` must be the
    **as-traded (unadjusted) close** on the same share basis.  Dividing by a
    split-adjusted close inflates historical ratios by the cumulative split
    factor (a 2:1 split after the filing doubles the pre-split B/P).  Pass the
    raw close as ``close_raw`` (preferred; e.g. ``MarketData.close_raw``).
    ``close`` is accepted only with ``price_is_adjusted=False`` (i.e. you
    assert it is the raw close); with the default ``price_is_adjusted=True`` a
    ``close`` argument raises ``ValueError``.  With no price at all the
    ratios fall back to Sharadar's ``1/pb`` and ``1/pe`` (computed by the
    vendor with the contemporaneous price).

    Returns a dict of wide panels on ``dates`` (default: union of
    availability dates) containing the requested ``fields`` plus derived
    panels when inputs are present: ``book_to_price``, ``earnings_yield``,
    ``gross_profitability`` (``gp/assets``), ``asset_growth`` (assets vs the
    row with ``calendardate`` one year earlier, same dimension, as known at the
    current filing date).
    """
    df = sf1_df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    need = {"ticker", "datekey", "calendardate"}
    if not need <= set(df.columns):
        raise ValueError(f"SF1 frame needs columns {sorted(need)}")
    if "dimension" in df.columns:
        if dimension.upper().startswith("MR"):
            warnings.warn("Sharadar MR* dimensions restate history and are not point-in-time", UserWarning,
                          stacklevel=2)
        df = df[df["dimension"].str.upper() == dimension.upper()]
    df["datekey"] = pd.to_datetime(df["datekey"])
    df["calendardate"] = pd.to_datetime(df["calendardate"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.sort_values(["ticker", "datekey", "calendardate"])
    # among rows filed the same day keep the most recent period
    df = df.drop_duplicates(["ticker", "datekey"], keep="last")

    fields = list(fields) if fields is not None else [f for f in SF1_DEFAULT_FIELDS if f in df.columns]
    fields = [f for f in fields if f in df.columns]
    # asset growth: YoY on calendardate within ticker.  The prior-year value is
    # taken from the latest filing for that period whose *filing date* is on or
    # before the current row's datekey (merge_asof backward), so a restatement
    # filed later (10-K/A, restated comparatives) cannot leak into an earlier
    # row (audit finding 6).
    if "assets" in df.columns:
        prev = df[["ticker", "calendardate", "datekey", "assets"]].rename(
            columns={"calendardate": "_prev_cd", "datekey": "_prev_dk", "assets": "_assets_prev"})
        prev = prev.dropna(subset=["_assets_prev"]).sort_values("_prev_dk")
        cur = df.assign(_prev_cd=df["calendardate"] - pd.DateOffset(years=1), _row=np.arange(len(df)))
        cur = cur.sort_values("datekey")
        merged = pd.merge_asof(cur, prev, left_on="datekey", right_on="_prev_dk", by=["ticker", "_prev_cd"],
                               direction="backward", allow_exact_matches=True)
        merged = merged.sort_values("_row")
        df = merged.drop(columns=["_prev_cd", "_prev_dk", "_row"]).reset_index(drop=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["asset_growth"] = df["assets"] / df["_assets_prev"].replace(0, np.nan) - 1.0
        fields.append("asset_growth")
    if "gp" in df.columns and "assets" in df.columns:
        df["gross_profitability"] = df["gp"] / df["assets"].replace(0, np.nan)
        fields.append("gross_profitability")

    df["available_date"] = df["datekey"] + pd.Timedelta(days=int(lag_days))
    if dates is None:
        dates = pd.DatetimeIndex(sorted(df["available_date"].unique()))
    dates = pd.DatetimeIndex(dates)

    out: Dict[str, Wide] = {}
    for f in dict.fromkeys(fields):
        sub = df[["available_date", "ticker", f]].rename(columns={f: "value"})
        sub = sub.assign(value=pd.to_numeric(sub["value"], errors="coerce"))
        out[f] = pit_forward_fill(sub, dates)

    # derived valuation ratios (price-dependent ones use the as-traded close when given)
    if close is not None:
        if close_raw is not None:
            raise ValueError("pass either close_raw or close, not both")
        if price_is_adjusted:
            raise ValueError("B/P and E/P need the as-traded (unadjusted) close: pass close_raw=..., or "
                             "close=... with price_is_adjusted=False if that frame is the raw close")
        close_raw = close
    if close_raw is not None:
        cl = close_raw.reindex(index=dates).ffill()
        if "bvps" in out:
            cols = out["bvps"].columns.intersection(cl.columns)
            out["book_to_price"] = out["bvps"][cols] / cl[cols]
        if "eps" in out:
            cols = out["eps"].columns.intersection(cl.columns)
            out["earnings_yield"] = out["eps"][cols] / cl[cols]
    else:
        if "pb" in out:
            out["book_to_price"] = 1.0 / out["pb"].replace(0, np.nan)
        if "pe" in out:
            out["earnings_yield"] = 1.0 / out["pe"].replace(0, np.nan)
    return out
