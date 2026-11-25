"""Acquisition and loading of the public financial data used in the paper.

Sources (all free, no registration):
  * FRED, Federal Reserve Bank of St. Louis - macro-financial series;
  * Yahoo Finance chart API - daily prices of the S&P 500 index and stocks;
  * UCI Machine Learning Repository - "Default of credit card clients".

Raw files are cached under ``data/raw``; every loader first looks at the
cache, so after the first ``python -m dmfin.download`` the whole project
runs offline.
"""
from __future__ import annotations

import io
import time
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from . import config as C

UA = {"User-Agent": "Mozilla/5.0 (educational data-mining course; GSU)"}

# Publication delays (calendar days) of series that are released after the date they refer to.  Market prices and
# yields are known at the close of their date; these constructed series are used only after their publication.
PUBLICATION_LAG_DAYS = {
    "STLFSI4": 7,       # St. Louis Fed stress index: week ending Friday, released on the following Thursday
    "NFCI": 7,          # Chicago Fed financial conditions index: week ending Friday, released the next Wednesday
    "USEPUINDXD": 1,    # daily news-based uncertainty index, released the next day
    "DTWEXBGS": 7,      # broad dollar index: daily values released weekly (H.10, Mondays)
    "DFF": 1,           # effective federal funds rate: released on the next business day
}


def available_at(s: pd.Series, index: pd.DatetimeIndex, lag_days: int = 0,
                 tolerance_days: int | None = None) -> pd.Series:
    """Value of ``s`` known at each date of ``index``: the last observation dated at least ``lag_days``
    calendar days earlier (and, if given, not more than ``tolerance_days`` before that); never a later one."""
    s = s.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    tol = None if tolerance_days is None else pd.Timedelta(days=tolerance_days)
    v = s.reindex(pd.DatetimeIndex(index) - pd.Timedelta(days=lag_days), method="ffill", tolerance=tol)
    return pd.Series(v.to_numpy(), index=index)

# ----------------------------------------------------------------------------
# Stock universe for Lab 1: 110 large US companies, 10 per GICS sector.
# Sector labels are the GICS sectors as of 2026; they are used only for
# *external* validation of clusters, never as model inputs.
# ----------------------------------------------------------------------------
UNIVERSE = {
    "Information Technology": ["AAPL", "MSFT", "NVDA", "ORCL", "CSCO", "ADBE", "INTC", "TXN", "QCOM", "IBM"],
    "Health Care": ["JNJ", "UNH", "PFE", "MRK", "ABT", "TMO", "LLY", "AMGN", "MDT", "BMY"],
    "Financials": ["JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "BLK", "USB", "PNC"],
    "Consumer Discretionary": ["AMZN", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "F", "GM"],
    "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST", "CL", "MDLZ", "MO", "KMB", "GIS"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY", "PSX", "VLO", "MPC", "HAL"],
    "Industrials": ["HON", "UPS", "CAT", "DE", "LMT", "RTX", "UNP", "MMM", "FDX", "GD"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "PEG"],
    "Real Estate": ["PLD", "AMT", "SPG", "EQIX", "PSA", "O", "WELL", "AVB", "EQR", "CCI"],
    "Materials": ["LIN", "APD", "SHW", "ECL", "NEM", "FCX", "NUE", "IP", "VMC", "MLM"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "EA", "CHTR"],
}
# Reserve names, used in order when a primary ticker has no full history
# on the public endpoint (delisting, merger, re-listing under a new id).
SPARES = {
    "Information Technology": ["AMD", "CRM", "AMAT", "MU"],
    "Health Care": ["GILD", "CVS", "SYK", "ISRG"],
    "Financials": ["SCHW", "MET", "CB", "MMC"],
    "Consumer Discretionary": ["ROST", "MAR", "YUM", "ORLY"],
    "Consumer Staples": ["KR", "SYY", "HSY", "CLX"],
    "Energy": ["KMI", "WMB", "OKE", "DVN"],
    "Industrials": ["EMR", "ITW", "CSX", "NSC"],
    "Utilities": ["AEE", "WEC", "ES", "ETR"],
    "Real Estate": ["DLR", "VTR", "ESS", "MAA"],
    "Materials": ["PPG", "ALB", "BALL", "MOS"],
    "Communication Services": ["OMC", "TTWO", "IPG", "LUMN"],
}
SECTOR_OF = {t: s for s, ts in UNIVERSE.items() for t in ts}  # nominal universe


def _epoch(d: str) -> int:
    return int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


# ------------------------------- download -----------------------------------
def download_fred(series: str, force: bool = False) -> pd.Series:
    path = C.RAW_DIR / "fred" / f"{series}.csv"
    if force or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        path.write_text(r.text)
    df = pd.read_csv(path, na_values=["."])
    df.columns = ["date", series]
    s = df.set_index(pd.to_datetime(df["date"]))[series].astype(float)
    return s


def download_yahoo(ticker: str, start: str = C.START, end: str = C.END,
                   force: bool = False) -> pd.DataFrame:
    """Daily OHLCV + split/dividend-adjusted close from the Yahoo chart API."""
    fname = ticker.replace("^", "IDX_")
    path = C.RAW_DIR / "yahoo" / f"{fname}.csv"
    if force or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
               f"?period1={_epoch(start)}&period2={_epoch(end_excl)}"
               f"&interval=1d&events=div%2Csplits&includeAdjustedClose=true")
        for attempt in range(6):
            try:
                r = requests.get(url, headers=UA, timeout=60)
            except requests.exceptions.ConnectionError:
                time.sleep(5 + 5 * attempt)
                continue
            if r.status_code == 200:
                break
            time.sleep(2 + 3 * attempt)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
        idx = pd.to_datetime(res["timestamp"], unit="s", utc=True)
        # trading date in the exchange's own time zone (a Tokyo session that opens at
        # 09:00 JST is still the previous calendar day in New York)
        tz = res.get("meta", {}).get("exchangeTimezoneName") or "America/New_York"
        idx = idx.tz_convert(tz).tz_localize(None).normalize()
        df = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"],
                           "close": q["close"], "adj_close": adj,
                           "volume": q["volume"]}, index=idx)
        df.index.name = "date"
        df = df[~df.index.duplicated(keep="last")].dropna(subset=["close"])
        df.to_csv(path)
        time.sleep(0.4)  # be polite to the public endpoint
    return pd.read_csv(path, index_col="date", parse_dates=True)


def download_credit(force: bool = False) -> pd.DataFrame:
    """UCI 'Default of credit card clients' (Yeh & Lien, 2009), 30 000 rows."""
    path = C.RAW_DIR / "uci" / "default_of_credit_card_clients.csv"
    if force or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
        r = requests.get(url, headers=UA, timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            name = [n for n in z.namelist() if n.endswith(".xls")][0]
            raw = pd.read_excel(io.BytesIO(z.read(name)), header=1)
        raw.to_csv(path, index=False)
    return pd.read_csv(path)


# ------------------------------- loaders ------------------------------------
def load_fred_panel() -> pd.DataFrame:
    return pd.concat({s: download_fred(s) for s in C.FRED_SERIES}, axis=1)


def load_market_panel() -> pd.DataFrame:
    """S&P 500 OHLCV joined with FRED series on S&P 500 trading days.

    FRED values are aligned *backwards only*: on day t we use the latest
    observation dated <= t (forward fill, at most 5 days).  This rules out
    look-ahead bias caused by holidays that differ between the bond and the
    stock market.
    """
    cache = C.PROC_DIR / "market_panel.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col="date", parse_dates=True)
    spx = download_yahoo("^GSPC")
    spx = spx.loc[C.START:C.END]
    fred = load_fred_panel()
    daily = fred.drop(columns=["USREC"])
    daily = daily.reindex(daily.index.union(spx.index)).sort_index().ffill(limit=5)
    panel = spx.join(daily.reindex(spx.index))
    # monthly recession flag -> daily, by calendar month
    rec = fred["USREC"].dropna()
    panel["USREC"] = rec.reindex(panel.index.to_period("M").to_timestamp()).to_numpy()
    panel.index.name = "date"
    panel.to_csv(cache)
    return panel


def load_stock_prices() -> pd.DataFrame:
    """Adjusted close prices (columns = tickers) for the Lab 1 universe.

    Ten names per sector are taken from ``UNIVERSE``; a name without the
    complete 10-year history is replaced by the next one from ``SPARES``.
    The realised universe is stored in ``processed/universe.csv``.
    """
    cache = C.PROC_DIR / "stock_prices.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col="date", parse_dates=True)
    first_ok = pd.Timestamp(C.STOCKS_START) + pd.Timedelta(days=10)
    cols, rows = {}, []
    for sector, names in UNIVERSE.items():
        got = 0
        for t in names + SPARES[sector]:
            if got == 10:
                break
            try:
                df = download_yahoo(t, start=C.STOCKS_START, end=C.END)
            except requests.exceptions.HTTPError:
                print(f"  {t}: no data on the endpoint, replaced by a spare")
                continue
            if df.index.min() > first_ok or df.index.max() < pd.Timestamp(C.END) - pd.Timedelta(days=10):
                print(f"  {t}: incomplete history, replaced by a spare")
                continue
            cols[t] = df["adj_close"]
            rows.append((t, sector))
            got += 1
    px = pd.DataFrame(cols).loc[C.STOCKS_START:C.END]
    px.index.name = "date"
    px.to_csv(cache)
    pd.DataFrame(rows, columns=["ticker", "sector"]).to_csv(C.PROC_DIR / "universe.csv", index=False)
    return px


def load_universe() -> pd.Series:
    """Ticker -> GICS sector for the realised Lab 1 universe."""
    load_stock_prices()
    u = pd.read_csv(C.PROC_DIR / "universe.csv")
    return u.set_index("ticker")["sector"]


def load_credit() -> pd.DataFrame:
    df = download_credit()
    df = df.rename(columns={"default payment next month": "default", "PAY_0": "PAY_1"})
    return df.drop(columns=["ID"])


if __name__ == "__main__":  # python -m dmfin.data
    p = load_market_panel(); print("market panel", p.shape, p.index.min(), p.index.max())
    s = load_stock_prices(); print("stocks", s.shape, s.isna().sum().sum())
    c = load_credit(); print("credit", c.shape, c["default"].mean())
