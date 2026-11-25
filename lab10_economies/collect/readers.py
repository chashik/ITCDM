"""Readers of the raw files of D9 (no side effects): used by build.py, the stages of Lab 10 and the mathematical chapter.
The files are untrusted downloads: they are only parsed (pdftotext for the PDF info blocks of the EEC)."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import RAW

RU_MONTHS = {m: i for i, m in enumerate(["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август",
                                         "сентябрь", "октябрь", "ноябрь", "декабрь"], 1)}


def eurostat(path: Path) -> pd.Series:
    d = json.loads(path.read_text())
    t = d["dimension"]["time"]["category"]["index"]
    inv = {v: k for k, v in t.items()}
    return pd.Series({inv[int(i)]: v for i, v in d["value"].items()}).sort_index()


def step_month_mean(dates, values) -> pd.Series:
    """A rate valid from given dates -> its mean over the days of every month (the rate in force on each day): the rate
    at which a deposit earned interest in the month."""
    s = pd.Series(np.asarray(values, float), index=pd.to_datetime(list(dates))).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    days = pd.date_range("2020-12-01", "2025-12-31", freq="D")
    d = s.reindex(s.index.union(days)).ffill().loc[days]
    return d.groupby(d.index.to_period("M")).mean()


def step_to_month_end(dates, values) -> pd.Series:
    """A rate valid from given dates -> its value at the end of every month."""
    s = pd.Series(values, index=pd.to_datetime(dates)).sort_index()
    ends = pd.date_range("2020-12-31", "2025-12-31", freq="ME")
    return pd.Series(s.reindex(s.index.union(ends)).ffill().loc[ends].to_numpy(), index=ends.to_period("M"))


def imf_series(path: Path, country: str, indicator: str) -> pd.Series:
    s = path.read_text(encoding="utf-8")
    for attrs, body in re.findall(r"<Series ([^>]*)>(.*?)</Series>", s, re.S):
        a = dict(re.findall(r'(\w+)="([^"]*)"', attrs))
        if a.get("COUNTRY") == country and a.get("INDICATOR", "").startswith(indicator):
            obs = re.findall(r'TIME_PERIOD="(\d{4})-M(\d\d)" OBS_VALUE="([^"]+)"', body)
            return pd.Series({f"{y}-{m}": float(v) for y, m, v in obs})
    return pd.Series(dtype=float)


def html_table(path: Path) -> list[list[str]]:
    import html as h
    s = path.read_text(encoding="utf-8")
    t = s[s.find("<table"):s.find("</table>") + 8]
    return [[re.sub(r"<[^>]+>|\s+", " ", h.unescape(c)).strip() for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S)]
            for r in re.findall(r"<tr.*?</tr>", t, re.S)]


def num(v: str) -> float:
    v = str(v).replace("\xa0", "").replace(" ", "").replace(",", ".").replace("−", "-")
    return float(v) if re.fullmatch(r"-?\d+(\.\d+)?", v) else np.nan


def daily(path: Path) -> pd.Series:
    """CBR table 'date | value' (newest first) -> daily series."""
    rows = [r for r in html_table(path) if len(r) >= 2 and re.fullmatch(r"\d\d\.\d\d\.\d{4}", r[0])]
    return pd.Series({pd.to_datetime(r[0], dayfirst=True): num(r[1]) for r in rows}).sort_index()


def eec_series(path: Path, country: str = "Россия") -> pd.Series:
    """EEC Series_CPI file: blocks '2022 год' followed by one row per country, January ... December."""
    d = pd.read_excel(path, header=None)
    out, year = {}, None
    for _, r in d.iterrows():
        m = re.match(r"(\d{4}) год", str(r[0]).strip())
        if m:
            year = int(m.group(1))
        elif year and str(r[0]).strip() == country:
            out.update({f"{year}-{mo:02d}": num(r[mo]) for mo in range(1, 13)})
    return pd.Series(out)


def eec_items(year: int) -> dict:
    """EEC December info block: growth December over December of the previous year, % (Rosstat CPI of the item),
    for Russia (last of the five country columns) and Belarus (second, used as a check against Belstat)."""
    import subprocess
    txt = subprocess.run(["pdftotext", "-layout", str(RAW / "eec" / f"InfoCPI{year}12.pdf"), "-"],
                         capture_output=True, text=True).stdout
    items = {"flour": r"мука", "milk_dairy": r"молоко и молочные продукты", "sugar": r"сахар",
             "petrol": r"бензин автомобильный"}
    out = {}
    for key, pat in items.items():
        for line in txt.splitlines():
            if re.match(rf"\s*{pat}\s", line, re.I):
                v = [num(x) for x in re.findall(r"-?\d+,\d", line)]
                if len(v) == 5:
                    out[key] = {"RU": v[4], "BY": v[1]}
                    break
    return out
