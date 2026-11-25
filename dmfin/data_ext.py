"""Extended domain data (iteration 2): many series and many features.

Sources (all public and free):
  D1  Yahoo Finance - daily prices of further equity indices, commodities,
      bond and currency instruments (multi-asset panel);
  D2  Yahoo Finance and FRED - implied-volatility indices and financial
      stress / uncertainty indices;
  D3  Kenneth R. French Data Library - daily Fama-French factors, momentum
      and the returns of 49 industry portfolios;
  D4  FRED-MD (McCracken & Ng) - 127 monthly macroeconomic series, a dated
      vintage, with the recommended stationarity transformations;
  D5  SEC EDGAR - company metadata (CIK, SIC industry, exchange) for the
      Lab 1 universe, and the SIC definition of the 49 industries;
  D6  Federal Reserve Board - dates of the scheduled FOMC meetings;
  D7  UCI - Statlog (German) credit data with its official cost matrix;
  D8  Hugging Face - Twitter Financial News: tweets of financial news accounts labelled bearish, bullish or
      neutral (MIT licence), a fixed revision.

The loaders of ``dmfin.data`` (iteration 1) are not changed.  Raw files are
cached under ``data/raw``; after the first run everything works offline.
"""
from __future__ import annotations

import io
import json
import os
import re
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from . import config as C
from .data import UA, download_fred, download_yahoo, load_universe

# ------------------------------------------------------------------ D1
MULTI_ASSET = {
    "^GSPC": "S&P 500", "^IXIC": "Nasdaq Composite", "^RUT": "Russell 2000", "^FTSE": "FTSE 100",
    "^N225": "Nikkei 225", "^STOXX50E": "Euro Stoxx 50", "GC=F": "Gold futures", "CL=F": "WTI crude oil futures",
    "TLT": "US Treasury 20+ years ETF", "IEF": "US Treasury 7-10 years ETF", "HYG": "US high-yield bond ETF",
    "EEM": "Emerging-markets equity ETF", "DX-Y.NYB": "US dollar index",
}
# ------------------------------------------------------------------ D2
VOL_YAHOO = {"^VIX9D": "VIX 9-day", "^VVIX": "Volatility of VIX", "^SKEW": "CBOE SKEW index"}
VOL_FRED = {"VXVCLS": "VIX 3-month", "OVXCLS": "Oil volatility index", "GVZCLS": "Gold volatility index",
            "STLFSI4": "St. Louis Fed financial stress index (weekly)",
            "NFCI": "Chicago Fed national financial conditions index (weekly)",
            "USEPUINDXD": "Economic policy uncertainty index (daily, news-based)"}
FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FREDMD_VINTAGE = "2026-09"          # data up to August 2026; fixed for reproducibility
SEC_CONTACT_VAR = "DMFIN_SEC_CONTACT"   # the user's own name and e-mail, required by the SEC for downloads


def _sec_headers() -> dict:
    """User-Agent for SEC EDGAR, which asks every automated client to declare its own contact."""
    contact = os.environ.get(SEC_CONTACT_VAR, "").strip()
    if not contact:
        raise RuntimeError(f"SEC EDGAR asks automated clients to identify themselves: set the environment variable "
                           f"{SEC_CONTACT_VAR} to your name and e-mail address, for example\n"
                           f"  export {SEC_CONTACT_VAR}='Jane Doe jane.doe@example.org'")
    return {"User-Agent": f"dmfin data-mining course {contact}"}


def _get(url: str, headers=UA, timeout=120, tries=5) -> requests.Response:
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
        except requests.exceptions.RequestException:
            pass
        time.sleep(3 + 4 * attempt)
    r.raise_for_status()
    return r


def load_multi_asset(start: str = C.START, end: str = C.END) -> dict[str, pd.DataFrame]:
    """Ticker -> daily OHLC frame on the ticker's own trading calendar.

    Non-positive prices are set to missing: on 20 April 2020 the expiring WTI
    futures contract settled at a negative price, for which log-returns and
    range-based volatility are undefined.
    """
    out = {}
    for t in MULTI_ASSET:
        d = download_yahoo(t).loc[start:end, ["open", "high", "low", "close", "adj_close", "volume"]].copy()
        bad = (d[["high", "low", "close", "adj_close"]] <= 0).any(axis=1)
        d.loc[bad, ["open", "high", "low", "close", "adj_close"]] = np.nan
        out[t] = d
    return out


def load_vol_indices(start: str = C.START, end: str = C.END) -> pd.DataFrame:
    """Implied-volatility and stress indices as daily series (values dated by the source)."""
    cols = {k: download_yahoo(k)["close"] for k in VOL_YAHOO}
    cols.update({k: download_fred(k) for k in VOL_FRED})
    return pd.DataFrame(cols).sort_index().loc[start:end]


# ------------------------------------------------------------------ D3
def _ff_zip(name: str) -> str:
    path = C.RAW_DIR / "french" / f"{name}.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(FF_BASE + f"{name}.zip").content)
    with zipfile.ZipFile(path) as z:
        return z.read(z.namelist()[0]).decode("latin-1")


def _ff_table(text: str, title: str | None = None) -> pd.DataFrame:
    """First (or titled) daily table of a Ken French CSV; percent -> decimal."""
    lines = text.replace("\r", "").split("\n")
    i0 = 0
    if title:
        i0 = next(i for i, l in enumerate(lines) if title in l) + 1
    hdr = next(i for i in range(i0, len(lines)) if lines[i].startswith(","))
    end = next((i for i in range(hdr + 1, len(lines)) if not re.match(r"\s*\d{8}", lines[i])), len(lines))
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:end])), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
    df.columns = [c.strip() for c in df.columns]
    df = df.replace([-99.99, -999.0], np.nan) / 100.0
    df.index.name = "date"
    return df


def load_ff_factors(start: str = C.START, end: str = C.END) -> pd.DataFrame:
    f5 = _ff_table(_ff_zip("F-F_Research_Data_5_Factors_2x3_daily_CSV"))
    mom = _ff_table(_ff_zip("F-F_Momentum_Factor_daily_CSV"))
    return f5.join(mom, how="left").loc[start:end]


def load_ff_industries(start: str = C.START, end: str = C.END) -> pd.DataFrame:
    """Daily value-weighted returns of the 49 Fama-French industry portfolios."""
    txt = _ff_zip("49_Industry_Portfolios_daily_CSV")
    return _ff_table(txt, title="Average Value Weighted Returns -- Daily").loc[start:end]


def load_ff49_sic() -> pd.DataFrame:
    """SIC ranges of the 49 industries (Siccodes49): columns no, short, name, lo, hi."""
    txt = _ff_zip("Siccodes49")
    rows, cur = [], None
    for line in txt.replace("\r", "").split("\n"):
        m = re.match(r"\s*(\d+)\s+(\S+)\s+(.+)", line)
        r = re.match(r"\s*(\d{4})-(\d{4})", line)
        if r and cur:
            rows.append((*cur, int(r.group(1)), int(r.group(2))))
        elif m and not r:
            cur = (int(m.group(1)), m.group(2), m.group(3).strip())
    return pd.DataFrame(rows, columns=["no", "short", "name", "lo", "hi"])


def sic_to_ff49(sic: int | None, table: pd.DataFrame | None = None) -> str | None:
    if sic is None or not np.isfinite(sic):
        return None
    t = load_ff49_sic() if table is None else table
    hit = t[(t["lo"] <= sic) & (t["hi"] >= sic)]
    return hit["short"].iloc[0] if len(hit) else "Other"


# ------------------------------------------------------------------ D4
FREDMD_TCODE = {
    1: lambda x: x,
    2: lambda x: x.diff(),
    3: lambda x: x.diff().diff(),
    4: lambda x: np.log(x),
    5: lambda x: np.log(x).diff(),
    6: lambda x: np.log(x).diff().diff(),
    7: lambda x: (x / x.shift(1) - 1).diff(),
}


def load_fredmd(vintage: str = FREDMD_VINTAGE) -> tuple[pd.DataFrame, pd.Series]:
    """FRED-MD vintage: (levels, transformation codes).  Index = first day of the month."""
    path = C.RAW_DIR / "fredmd" / f"{vintage}-md.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = ("https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
               f"fred-md/monthly/{vintage}-md.csv")
        path.write_bytes(_get(url).content)
    raw = pd.read_csv(path)
    tcode = raw.iloc[0, 1:].astype(int)
    df = raw.iloc[1:].dropna(subset=["sasdate"]).copy()
    df.index = pd.to_datetime(df.pop("sasdate"), format="%m/%d/%Y")
    df.index.name = "month"
    return df.astype(float), tcode


def transform_fredmd(levels: pd.DataFrame, tcode: pd.Series) -> pd.DataFrame:
    """Stationarity transformations recommended by McCracken & Ng (codes 1-7)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {c: FREDMD_TCODE[int(tcode[c])](levels[c].where(levels[c] > 0) if int(tcode[c]) >= 4 else levels[c])
               for c in levels}
    return pd.DataFrame(out).replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------------------------ D5
def load_edgar_meta(tickers=None) -> pd.DataFrame:
    """CIK, name, SIC code and description, exchanges and state for the Lab 1 universe.

    The SEC asks automated clients to declare a User-Agent with a contact (taken from the
    environment variable ``DMFIN_SEC_CONTACT``) and to stay below 10 requests per second;
    responses are cached, so the variable is needed only for the first download.
    """
    tickers = list(load_universe().index) if tickers is None else list(tickers)
    base = C.RAW_DIR / "edgar"
    base.mkdir(parents=True, exist_ok=True)
    tpath = base / "company_tickers.json"
    if not tpath.exists():
        tpath.write_bytes(_get("https://www.sec.gov/files/company_tickers.json", headers=_sec_headers()).content)
    cik_of = {v["ticker"]: int(v["cik_str"]) for v in json.loads(tpath.read_text()).values()}
    rows = []
    for t in tickers:
        cik = cik_of.get(t)
        if cik is None:
            rows.append({"ticker": t}); continue
        p = base / f"CIK{cik:010d}.json"
        if not p.exists():
            p.write_bytes(_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", headers=_sec_headers()).content)
            time.sleep(0.15)
        s = json.loads(p.read_text())
        rows.append({"ticker": t, "cik": cik, "name": s.get("name"), "sic": int(s["sic"]) if s.get("sic") else np.nan,
                     "sic_description": s.get("sicDescription"), "exchanges": ";".join(s.get("exchanges") or []),
                     "state": s.get("stateOfIncorporation"), "fiscal_year_end": s.get("fiscalYearEnd")})
    return pd.DataFrame(rows).set_index("ticker")


# ------------------------------------------------------------------ D6
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                      "September", "October", "November", "December"], start=1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})      # "Jan/Feb" headings


def _fed_page(name: str) -> str:
    path = C.RAW_DIR / "fomc" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_get(f"https://www.federalreserve.gov/monetarypolicy/{name}").text)
    return path.read_text()


def load_fomc_dates(first_year: int = 2000, last_date: str = C.END) -> pd.DataFrame:
    """Last day of every scheduled FOMC meeting (the day of the policy statement).

    Unscheduled meetings and conference calls are excluded: they are reactions to
    events and would confound an event study.
    """
    rows = []
    for y in range(first_year, 2021):
        for title in re.findall(r"<h5[^>]*>([^<]*)</h5>", _fed_page(f"fomchistorical{y}.htm")):
            if "unscheduled" in title.lower() or "Meeting" not in title:
                continue
            m = re.match(r"\s*([A-Za-z]+)\s+(\d+)-([A-Za-z]+)\s+(\d+)\s+Meeting\s+-\s+(\d{4})", title)
            if m:                                   # e.g. "January 31-February 1 Meeting - 2006"
                rows.append(pd.Timestamp(int(m.group(5)), MONTHS[m.group(3)[:3]], int(m.group(4))))
                continue
            m = re.match(r"\s*([A-Za-z]+)(?:/([A-Za-z]+))?\s+(\d+)(?:-(\d+))?\s+Meeting\s+-\s+(\d{4})", title)
            if m:
                mon = m.group(2) or m.group(1)
                rows.append(pd.Timestamp(int(m.group(5)), MONTHS[mon[:3]], int(m.group(4) or m.group(3))))
    cal = _fed_page("fomccalendars.htm")
    for panel in re.split(r'<div class="panel panel-default">', cal)[1:]:
        ym = re.search(r"(\d{4}) FOMC Meetings", panel)
        if not ym or int(ym.group(1)) < 2021:
            continue
        year = int(ym.group(1))
        for mon, date in re.findall(r'fomc-meeting__month[^>]*><strong>([^<]+)</strong>.*?'
                                    r'fomc-meeting__date[^>]*>([^<]+)<', panel, re.S):
            if "unscheduled" in date.lower() or "notation" in date.lower():
                continue
            last_mon = mon.split("/")[-1].strip()
            d = re.findall(r"\d+", date)
            if last_mon[:3] in MONTHS and d:
                rows.append(pd.Timestamp(year, MONTHS[last_mon[:3]], int(d[-1])))
    s = pd.Series(sorted(set(rows)), name="fomc_date")
    return s[s <= pd.Timestamp(last_date)].reset_index(drop=True).to_frame()


# ------------------------------------------------------------------ D7
GERMAN_COLS = ["checking", "duration", "history", "purpose", "amount", "savings", "employment",
               "installment_rate", "personal_status", "other_debtors", "residence_since", "property",
               "age", "other_plans", "housing", "existing_credits", "job", "dependents", "telephone",
               "foreign", "class"]
GERMAN_COST = {"fn": 5.0, "fp": 1.0}   # documentation: rows actual, columns predicted; bad -> good costs 5
# value labels from the documentation of the data set (german.doc)
GERMAN_LABELS = {
    "A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM", "A14": "no account",
    "A30": "no credits or all paid", "A31": "all paid at this bank", "A32": "existing paid duly", "A33": "past delays",
    "A34": "critical account", "A40": "new car", "A41": "used car", "A42": "furniture", "A43": "radio/TV",
    "A44": "appliances", "A45": "repairs", "A46": "education", "A47": "vacation", "A48": "retraining", "A49": "business",
    "A410": "other", "A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">= 1000 DM",
    "A65": "unknown or none", "A71": "unemployed", "A72": "< 1 year", "A73": "1-4 years", "A74": "4-7 years",
    "A75": ">= 7 years", "A91": "male, divorced", "A92": "female, div./married", "A93": "male, single",
    "A94": "male, married/widowed", "A95": "female, single", "A101": "none", "A102": "co-applicant",
    "A103": "guarantor", "A121": "real estate", "A122": "savings agreement", "A123": "car or other",
    "A124": "none", "A141": "bank", "A142": "stores", "A143": "none", "A151": "rent", "A152": "own",
    "A153": "for free", "A171": "unskilled non-resident", "A172": "unskilled resident", "A173": "skilled",
    "A174": "management or self-employed", "A191": "none", "A192": "yes", "A201": "yes", "A202": "no"}


def load_german_credit() -> pd.DataFrame:
    """Statlog (German credit): 1 000 applicants, 20 attributes, target ``bad`` (1 = bad risk)."""
    path = C.RAW_DIR / "uci" / "german.data"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = _get("https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip")
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            path.write_bytes(z.read("german.data"))
            (path.parent / "german.doc").write_bytes(z.read("german.doc"))   # value labels and cost matrix
    df = pd.read_csv(path, sep=r"\s+", header=None, names=GERMAN_COLS)
    df["bad"] = (df.pop("class") == 2).astype(int)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].map(GERMAN_LABELS).fillna(df[c]).astype(str)
    return df


# ------------------------------------------------------------------ D8
TWEETS = "zeroshot/twitter-financial-news-sentiment"
TWEETS_REV = "ccbe24de388e287beb92dd393a335c376b350ac3"   # revision of 2024-02-23, fixed for reproducibility
TWEET_LABELS = ["bearish", "bullish", "neutral"]          # label codes 0, 1, 2


def load_fin_tweets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Twitter Financial News (sentiment): the published training (9 543) and validation (2 388) tweets,
    columns ``text`` and ``label`` (0 bearish, 1 bullish, 2 neutral)."""
    d = C.RAW_DIR / "fin_tweets"
    files = ("sent_train.csv", "sent_valid.csv")
    for f in ("README.md",) + files:                      # README: data card with the labels and the licence
        if not (d / f).exists():
            d.mkdir(parents=True, exist_ok=True)
            (d / f).write_bytes(_get(f"https://huggingface.co/datasets/{TWEETS}/resolve/{TWEETS_REV}/{f}").content)
    return tuple(pd.read_csv(d / f) for f in files)


def inventory() -> pd.DataFrame:
    """One row per extended source: kind, number of series, first and last date, missing share."""
    rows = []
    ma = load_multi_asset()
    for t, d in ma.items():
        rows.append(("D1", MULTI_ASSET[t], "daily", 1, d.index.min(), d.index.max(), d["adj_close"].isna().mean()))
    for c, name in {**VOL_YAHOO, **VOL_FRED}.items():          # each series on its own calendar
        raw = (download_yahoo(c)["close"] if c in VOL_YAHOO else download_fred(c)).loc[C.START:C.END]
        s = raw.dropna()
        rows.append(("D2", name, "weekly" if c in ("STLFSI4", "NFCI") else "daily", 1, s.index.min(), s.index.max(),
                     raw.loc[s.index.min():].isna().mean()))
    ff, ind = load_ff_factors(), load_ff_industries()
    rows.append(("D3", "Fama-French 5 factors + momentum", "daily", ff.shape[1], ff.index.min(), ff.index.max(),
                 ff.isna().mean().mean()))
    rows.append(("D3", "49 industry portfolios (value-weighted)", "daily", ind.shape[1], ind.index.min(),
                 ind.index.max(), ind.isna().mean().mean()))
    md, tc = load_fredmd()
    md = md.loc[C.START:]
    rows.append(("D4", f"FRED-MD vintage {FREDMD_VINTAGE}", "monthly", md.shape[1], md.index.min(), md.index.max(),
                 md.isna().mean().mean()))
    fomc = load_fomc_dates()
    rows.append(("D6", "Scheduled FOMC meetings", "events", len(fomc), fomc["fomc_date"].min(),
                 fomc["fomc_date"].max(), 0.0))
    g = load_german_credit()
    rows.append(("D7", "Statlog German credit", "cross-section", g.shape[1] - 1, pd.NaT, pd.NaT, 0.0))
    tw = load_fin_tweets()
    rows.append(("D8", "Financial news tweets, sentiment", "texts", 1, pd.NaT, pd.NaT,
                 float(np.mean([t["text"].isna().mean() for t in tw]))))
    return pd.DataFrame(rows, columns=["id", "source", "frequency", "series", "first", "last", "missing"])


if __name__ == "__main__":     # python -m dmfin.data_ext
    inv = inventory()
    print(inv.to_string())
    e = load_edgar_meta(); print(e.head(), e["sic"].isna().sum())
    load_ff49_sic(); download_yahoo("^VIX")      # SIC ranges of Lab 0 and the VIX of the Lab 7 extension
