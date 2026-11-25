"""Realized yields of the Polish retail savings bonds sold 2021-01 ... 2025-12, held until 2025-12-31 or maturity.

Rules (obligacjeskarbowe.pl, issue pages):
  EDO, COI   1st year: the published rate; later years: margin + CPI (y/y, GUS, the month two months before the start
             of the year; deflation counts as 0). EDO capitalizes yearly (compound yield), COI pays yearly (mean rate).
  TOS        fixed rate, capitalized yearly.
  ROR, DOR   1st month: the published rate; later months: NBP reference rate at the start of the month + margin.
Before tax (19 %) and without early-redemption fees.  CPI: OECD (national CPI = GUS); reference rate: NBP archive.

Run (from code/):  python -m lab10_economies.collect.pl_bond_yields
Output: data/processed/d9/pl_retail_bonds_yields.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import DATA as _DATA, RAW as _RAW
END = pd.Period("2025-12", "M")                       # holding ends on 2025-12-31

o = pd.read_csv(_RAW / "oecd" / "cpi_pol.csv")
CPI = pd.Series(o["OBS_VALUE"].to_numpy(), index=pd.PeriodIndex(o["TIME_PERIOD"], freq="M")).sort_index()
xml = (_RAW / "nbp" / "stopy_archiwum.xml").read_text(encoding="utf-8-sig")
ref = sorted((pd.Timestamp(m.group(1)), float(r.group(1).replace(",", ".")))
             for m in re.finditer(r'obowiazuje_od="([\d-]+)">(.*?)</pozycje>', xml, re.S)
             for r in [re.search(r'id="ref"\s+oprocentowanie="([\d,]+)"', m.group(2))] if r)
REF = pd.Series([v for _, v in ref], index=[d for d, _ in ref])


def ref_rate(month: pd.Period) -> float:
    return float(REF[REF.index <= month.to_timestamp()].iloc[-1])


def months_held(start: pd.Period, years: int) -> int:
    return min(years * 12, (END - start).n + 1)


def yearly_coupons(row, start, years):
    """Coupon rate of each started year of an EDO/COI issue, with the share of the year held."""
    n = months_held(start, years)
    out = []
    for k in range((n + 11) // 12):
        frac = min(12, n - 12 * k) / 12
        if k == 0:
            r = row.first_rate
        else:
            cpi = CPI.get(start + 12 * k - 2, np.nan)
            r = row.margin + max(cpi, 0.0)
        out.append((r, frac))
    return out


def realized(row) -> dict:
    start = pd.Period(row.month, "M")
    if row.type in ("EDO", "COI"):
        yrs = 10 if row.type == "EDO" else 4
        cps = yearly_coupons(row, start, yrs)
        T = sum(f for _, f in cps)
        if row.type == "EDO":                               # yearly capitalization, simple interest inside a year
            v = np.prod([1 + r / 100 * f for r, f in cps])
            y = (v ** (1 / T) - 1) * 100
        else:
            y = sum(r * f for r, f in cps) / T
        return {"years_held": T, "realized_rate": y, "coupons": " ".join(f"{r:.2f}" for r, _ in cps)}
    if row.type == "TOS":
        return {"years_held": months_held(start, 3) / 12, "realized_rate": row.first_rate, "coupons": f"{row.first_rate:.2f}"}
    if row.type in ("ROR", "DOR"):
        n = months_held(start, 1 if row.type == "ROR" else 2)
        rates = [row.first_rate] + [ref_rate(start + k) + row.margin for k in range(1, n)]
        return {"years_held": n / 12, "realized_rate": float(np.mean(rates)), "coupons": f"{len(rates)} monthly"}
    return {"years_held": np.nan, "realized_rate": np.nan, "coupons": "WIBOR 6M not collected"}


def main():
    d = pd.read_csv(_DATA / "pl_retail_bonds_issues.csv")
    d = d[d["page_found"]].copy()
    d = pd.concat([d.reset_index(drop=True), pd.DataFrame([realized(r) for r in d.itertuples()])], axis=1)
    d.to_csv(_DATA / "pl_retail_bonds_yields.csv", index=False)
    piv = d.pivot(index="month", columns="type", values="realized_rate")
    print(d.groupby("type").agg(issues=("issue", "size"), first=("first_rate", "mean"), realized=("realized_rate", "mean")).round(2))
    print(piv.iloc[[0, 12, 24, 36, 48, 59]].round(2))


if __name__ == "__main__":
    main()
