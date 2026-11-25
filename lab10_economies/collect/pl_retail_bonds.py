"""Polish retail treasury savings bonds: the terms of every monthly issue 2021-01 ... 2025-12.

For EDO (10 years, inflation + margin, capitalized), COI (4 years, inflation + margin, paid yearly), TOS (3 years,
fixed), ROR (1 year, NBP reference rate + margin, paid monthly) and DOR (2 years, reference rate + margin) the page of
each issue on obligacjeskarbowe.pl gives the rate of the first interest period, the margin and the prices.
Step 1 (download, once):  python -m lab10_economies.collect.pl_retail_bonds download
Step 2 (parse, untrusted pages, isolated):  python -m lab10_economies.collect.pl_retail_bonds parse
Output: data/processed/d9/pl_retail_bonds_issues.csv
"""
import html
import re
import subprocess
import sys
import time
from pathlib import Path

from ..paths import DATA as _DATA, RAW as _RAW
RAW = _RAW / "obligacje" / "issues"
TYPES = {"EDO": ("obligacje-10-letnie-edo", 10), "COI": ("obligacje-4-letnie-coi", 4), "TOS": ("obligacje-3-letnie-tos", 3),
         "ROR": ("obligacje-roczne-ror", 1), "DOR": ("obligacje-2-letnie-dor", 2),
         "TOZ": ("obligacje-3-letnie-toz", 3)}   # 3 years, floating (WIBOR 6M + margin), sold until 2022 before TOS


def code(t, y, m):
    """Symbol = type + month and year of redemption (e.g. EDO0131 sold in January 2021)."""
    years = TYPES[t][1]
    return f"{t}{m:02d}{(y + years) % 100:02d}"


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    for t, (path, _) in TYPES.items():
        for y in range(2021, 2026):
            for m in range(1, 13):
                c = code(t, y, m)
                f = RAW / f"{c}_{y}-{m:02d}.html"
                if f.exists():
                    continue
                url = f"https://www.obligacjeskarbowe.pl/oferta-obligacji/{path}/{c.lower()}/"
                subprocess.run(["curl", "-sL", "-A", "Mozilla/5.0", "-o", str(f), url], check=False)
                time.sleep(0.3)
    print(len(list(RAW.glob("*.html"))), "pages")


def num(s):
    return float(s.replace(",", ".")) if s else None


def parse():
    import pandas as pd
    rows = []
    for f in sorted(RAW.glob("*.html")):
        c, ym = f.stem.split("_")
        t = html.unescape(re.sub(r"\s+", " ", re.sub("<[^>]+>", " ", re.sub(r"<script.*?</script>|<style.*?</style>", "",
                                                                                  f.read_text(errors="ignore"), flags=re.S))))
        first = (re.search(r"Oprocentowanie:\s*([\d,]+)\s*%", t)
                 or re.search(r"pierwszym[^%]{0,80}?wynosi\s+([\d,]+)\s*%", t)
                 or re.search(r"oprocentowanie wynosi\s+([\d,]+)\s*%", t))
        margin = re.search(r"marża\s*([\d,]+)\s*%|NBP\s*\+\s*([\d,]+)\s*%|WIBOR\s*6M\s*\+\s*([\d,]+)\s*%", t)
        mult = re.search(r"mnożnik\s*([\d,]+)\s*x\s*WIBOR", t)
        fixed = bool(re.search(r"stałe przez cały", t))
        price = re.search(r"Cena sprzedaży jednej obligacji:\s*([\d,]+)", t)
        swap = re.search(r"Cena zamiany jednej obligacji:\s*([\d,]+)", t)
        sale = re.search(r"Sprzedaż:\s*([\d.]+)\s*-\s*([\d.]+)", t)
        rows.append({"type": c[:3], "issue": c, "month": ym, "first_rate": num(first.group(1)) if first else None,
                     "margin": num(next(g for g in margin.groups() if g)) if margin else None, "fixed": fixed, "wibor_multiplier": num(mult.group(1)) if mult else None, "price": num(price.group(1)) if price else None,
                     "swap_price": num(swap.group(1)) if swap else None, "sale": " - ".join(sale.groups()) if sale else None,
                     "page_found": "Cena sprzedaży" in t})
    df = pd.DataFrame(rows).sort_values(["type", "month"])
    df.to_csv(_DATA / "pl_retail_bonds_issues.csv", index=False)
    print(df.groupby("type").agg(pages=("issue", "size"), found=("page_found", "sum"), first_rate_ok=("first_rate", "count"),
                                 margin_ok=("margin", "count")))
    print(df[df.month.isin(["2021-01", "2023-01", "2025-12"])].to_string(index=False))


if __name__ == "__main__":
    {"download": download, "parse": parse}[sys.argv[1]]()
