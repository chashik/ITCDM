"""Belstat: monthly average retail prices (BYN) of wheat flour, milk, sugar and petrol, Republic of Belarus.

Input: raw/belstat/goods/goods_YYYY-MM.xls ("Средние цены на товары, реализуемые в розничной сети", one workbook
per month, downloaded from belstat.gov.by). Output: data/processed/d9/by_belstat_prices.csv (long format).
Column 1 of each workbook is the Republic of Belarus; the regional columns are kept as a check of dispersion.

Run (untrusted input, isolated interpreter):  python -m lab10_economies.collect.belstat_prices
"""
import re
from pathlib import Path

import pandas as pd

from ..paths import DATA as _DATA, RAW as _RAW
ITEMS = {  # indicator: (pattern of the row label, unit)
    "flour_wheat": (r"^Мука пшеничная\s*$", "BYN/kg"),
    "milk": (r"^Молоко(?: цельное[^(]*)?$", "BYN/l"),   # 2022: label "Молоко"
    "sugar": (r"^Сахар(?:-песок)?\s*$", "BYN/kg"),
    "petrol": (r"^Бензин автомобильный", "BYN/l"),
}
REGIONS = ["Беларусь", "Брестская", "Витебская", "Гомельская", "Гродненская", "г. Минск", "Минская", "Могилевская"]


def parse(path: Path) -> list[dict]:
    df = pd.read_excel(path, header=None, dtype=str).fillna("")
    month = re.search(r"(\d{4}-\d\d)", path.name).group(1)
    label = df.iloc[:, 0].str.replace(r"\s+", " ", regex=True).str.strip()
    rows = []
    for ind, (pat, unit) in ITEMS.items():
        hit = label[label.str.contains(pat, regex=True, case=False)]   # 2020 files: labels in capitals
        if len(hit) != 1:
            raise ValueError(f"{path.name}: {ind}: {len(hit)} rows match {pat!r}: {hit.tolist()}")
        r = df.loc[hit.index[0]]
        vals = [float(str(v).replace(",", ".")) if str(v).strip() not in ("", "-", "…") else None for v in r.iloc[1:9]]
        rows.append({"country": "BY", "indicator": ind, "unit": unit, "month": month, "source": "belstat",
                     "value": vals[0], "label": hit.iloc[0],
                     **{f"reg_{i}": v for i, v in enumerate(vals[1:], 1)}})
    return rows


def main():
    files = sorted((_RAW / "belstat" / "goods").glob("goods_*.xls*"))
    out = pd.DataFrame([r for f in files for r in parse(f)])
    out.to_csv(_DATA / "by_belstat_prices.csv", index=False)
    piv = out.pivot(index="month", columns="indicator", values="value")
    print(f"{len(files)} months, {len(out)} values; labels used:", sorted(out["label"].unique()))
    print(piv.iloc[[0, 11, 23, 35, 47, 59]])


if __name__ == "__main__":
    main()
