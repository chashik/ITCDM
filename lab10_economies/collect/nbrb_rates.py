"""NBRB: average interest rates of the credit-deposit market in national currency, monthly 2021-2025.

Input: raw/manual/nbrb/www.nbrb.by/statistics/financialmarkets/avgintratesdyn/YYYY_ru.xlsx, sheet "НВ"
(национальная валюта); every file holds the previous and the current year.  Rows used, for individuals
("физическим лицам" / "физических лиц"):
  4  new bank deposits                        -> deposits_new_indiv_byn
  7  new loans (without interbank loans)      -> loans_new_indiv_byn
  9  new loans without preferential loans     -> loans_new_indiv_byn_nonpref
Housing loans have no rate series of their own at NBRB; in Belarus most of them are preferential state loans, so the
pair 7/9 shows the range.  Output: data/processed/d9/by_nbrb_rates.csv (long format; for each month the latest file wins).

Run (from code/):  python -m lab10_economies.collect.nbrb_rates
"""
import re
from pathlib import Path

import pandas as pd

from ..paths import DATA as _DATA, RAW as _RAW
SRC = _RAW / "manual" / "nbrb" / "www.nbrb.by" / "statistics" / "financialmarkets" / "avgintratesdyn"
ITEMS = {4: "deposits_new_indiv_byn", 7: "loans_new_indiv_byn", 9: "loans_new_indiv_byn_nonpref"}


def parse(path: Path) -> list[tuple]:
    df = pd.read_excel(path, sheet_name="НВ", header=None, dtype=str).fillna("")
    out, item, group = [], None, None
    for _, r in df.iterrows():
        label = re.sub(r"\s+", " ", str(r[0])).strip()
        m = re.match(r"^(\d+)\.\s", label)
        if m:
            item, group = int(m.group(1)), "total"
        elif label:
            group = "indiv" if label.startswith("физическ") else "other"
        year = str(r[1]).strip()
        if item in ITEMS and group == "indiv" and re.fullmatch(r"20\d\d", year):
            for k in range(12):
                v = str(r[2 + k]).replace(",", ".").strip()
                if re.fullmatch(r"-?\d+(\.\d+)?", v):
                    out.append((path.stem[:4], ITEMS[item], f"{year}-{k + 1:02d}", float(v)))
    return out


def main():
    rows = [r for f in sorted(SRC.glob("20*_ru.xlsx")) for r in parse(f)]
    df = pd.DataFrame(rows, columns=["file", "indicator", "month", "value"])
    df = df.sort_values("file").drop_duplicates(["indicator", "month"], keep="last")
    df = df[df["month"].between("2021-01", "2025-12")].sort_values(["indicator", "month"])
    df.to_csv(_DATA / "by_nbrb_rates.csv", index=False)
    print(df.groupby("indicator")["month"].agg(["count", "min", "max"]))
    print(df.pivot(index="month", columns="indicator", values="value").iloc[[0, 11, 23, 35, 47, 59]].round(2))


if __name__ == "__main__":
    main()
