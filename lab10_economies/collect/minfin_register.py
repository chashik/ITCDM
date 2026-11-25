"""Belarus: bonds of banks in circulation, monthly 2021-2025, from the Ministry of Finance registers.

Input: raw/minfin_by/register/obl_obr_MMYY.xlsx ("Выпуски облигаций ... находящиеся в обращении по состоянию на
1 число месяца", all sheets: ordinary, exchange-traded and depositary bonds).  Issuers whose short name contains
"БАНК" are banks.  For every register date and currency (USD, BYN): number of interest-bearing issues, their volume
and the volume-weighted average coupon ("Размер дохода" of issues with income type "Проценты"; non-numeric rates such
as floating formulas are counted but left out of the average).  The register of 1 Feb describes January's end, so
the date is shifted back by one month.

Output: data/processed/d9/by_bank_bonds.csv.  Run (from code/):  python -m lab10_economies.collect.minfin_register
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import DATA as _DATA, RAW as _RAW
SRC = _RAW / "minfin_by" / "register"
CUR = {"доллар США": "usd", "белорусский рубль": "byn"}


def read(path: Path) -> pd.DataFrame:
    frames = []
    for sh, df in pd.read_excel(path, sheet_name=None, header=None, dtype=str).items():
        df = df.fillna("")
        hdr = next((i for i in range(min(10, len(df))) if df.iloc[i, 0].startswith("Краткое наименование")), None)
        if hdr is None:
            continue
        cols = [re.sub(r"\s+", " ", c).strip() for c in df.iloc[hdr]]
        d = df.iloc[hdr + 1:].copy()
        d.columns = cols
        d = d[d[cols[0]].str.strip() != ""]
        pick = {c: k for c in cols for k in ("Валюта", "Объем выпуска", "Вид дохода", "Размер дохода") if c.startswith(k)}
        d = d.rename(columns=pick)
        d["issuer"] = d[cols[0]]
        d["sheet"] = sh
        frames.append(d[["issuer", "Валюта", "Объем выпуска", "Вид дохода", "Размер дохода", "sheet"]])
    return pd.concat(frames, ignore_index=True)


def main():
    rows = []
    for f in sorted(SRC.glob("obl_obr_*.xlsx")):
        mm, yy = f.stem[-4:-2], f.stem[-2:]
        month = (pd.Period(f"20{yy}-{mm}", "M") - 1)
        d = read(f)
        d = d[d["issuer"].str.upper().str.contains("БАНК") & d["Валюта"].isin(CUR)]
        d["volume"] = pd.to_numeric(d["Объем выпуска"].str.replace(" ", "").str.replace(",", "."), errors="coerce")
        d["rate"] = pd.to_numeric(d["Размер дохода"].str.replace(",", "."), errors="coerce") * 100
        for cur, g in d.groupby(d["Валюта"].map(CUR)):
            pct = g[g["Вид дохода"].str.startswith("Процент")]
            ok = pct.dropna(subset=["rate", "volume"])
            rows.append({"month": str(month), "currency": cur, "issues": len(g), "interest_issues": len(pct),
                         "volume": g["volume"].sum(), "rate_numeric_issues": len(ok),
                         "coupon_wavg": float(np.average(ok["rate"], weights=ok["volume"])) if len(ok) else np.nan,
                         "coupon_median": float(ok["rate"].median()) if len(ok) else np.nan,
                         "banks": g["issuer"].nunique()})
    out = pd.DataFrame(rows)
    out = out[out["month"].between("2021-01", "2025-12")].sort_values(["currency", "month"])
    out.to_csv(_DATA / "by_bank_bonds.csv", index=False)
    print(out.groupby("currency").agg(months=("month", "size"), issues=("issues", "mean"), banks=("banks", "mean"),
                                      coupon=("coupon_wavg", "mean")).round(2))
    print(out.pivot(index="month", columns="currency", values="coupon_wavg").iloc[::12].round(2))


if __name__ == "__main__":
    main()
