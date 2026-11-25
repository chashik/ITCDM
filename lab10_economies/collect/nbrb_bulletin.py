"""NBRB statistical bulletins (stat_bulletin_YYYY_MM.pdf): monthly series for Belarus from two tables.

  Table 3.7   average interest rates on new loans in national currency: item 7 (all new loans) and item 9 (without
              loans on preferential terms), to individuals ("физическим лицам")
  Table 3.39 (3.38 in 2022-2024)  government securities on the domestic market: 1.3 yield of short-term bonds (ГКО), 1.7 yield of
              long-term bonds (ГДО) at placement, in national currency, US dollars and euros ("…" = no placement)

Every bulletin shows the previous year in full and the current year so far; for each month the latest bulletin wins.
Input: raw/manual/nbrb/www.nbrb.by/publications/bulletin/*.pdf (downloaded by fetch_blocked.sh); pdftotext -layout.
Output: data/processed/d9/by_nbrb_bulletin.csv (long format).

Run (from code/):  python -m lab10_economies.collect.nbrb_bulletin
"""
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from ..paths import DATA as _DATA, RAW as _RAW
BUL = _RAW / "manual" / "nbrb" / "www.nbrb.by" / "publications" / "bulletin"
NUM = r"…|-?\d{1,3}(?: \d{3})+(?:,\d+)?|-?\d+(?:,\d+)?"   # "2 943" (space as thousands separator) is one number
YEAR_LINE = re.compile(r"\s(20\d\d)\s+((?:(?:…|-?\d+(?:[ ,]\d+)*)\s*)+)$")


def text_of(pdf: Path) -> list[str]:
    with tempfile.NamedTemporaryFile(suffix=".txt") as t:
        subprocess.run(["pdftotext", "-layout", str(pdf), t.name], check=True)
        return Path(t.name).read_text(encoding="utf-8", errors="ignore").split("\n")


def block(lines: list[str], table: str, title: str | None = None) -> list[str]:
    """Lines from 'Таблица <table>' (or from the table whose title contains ``title``) to the next 'Таблица'."""
    if title:                                   # table numbers change between years (3.38 in 2022-2024, 3.39 later)
        t = next(i for i, l in enumerate(lines) if title in l)
        start = max(i for i in range(t + 1) if re.search(r"Таблица\s+\d", lines[i]))
    else:
        start = next(i for i, l in enumerate(lines) if re.search(rf"Таблица\s+{re.escape(table)}\s*$", l))
    end = next((i for i in range(start + 1, len(lines)) if re.search(r"Таблица\s+\d", lines[i])), len(lines))
    out = []
    for l in lines[start:end]:
        if not out or l != out[-1]:            # page breaks repeat lines in the text layer
            out.append(l)
    return out


def values(line: str):
    m = YEAR_LINE.search(line)
    if not m:
        return None, []
    toks = re.findall(NUM, m.group(2))
    return int(m.group(1)), [None if t == "…" else float(t.replace(" ", "").replace(",", ".")) for t in toks]


def table_37(lines):
    """Items 7 and 9, subgroup 'физическим лицам' (third pair of year lines of each item)."""
    out, item, pairs, prev_year = [], None, [], None
    for l in block(lines, "3.7"):
        m = re.match(r"^\s*(\d+)\.\s", l)
        if m:
            item, pairs, prev_year = int(m.group(1)), [], None
        y, v = values(l)
        if item not in (7, 9) or y is None:
            continue
        if prev_year is None or y < prev_year or not pairs:
            pairs.append([])
        pairs[-1].append((y, v))
        prev_year = y
        if len(pairs) == 3:                        # individuals
            for yy, vv in pairs[2][-1:]:
                name = "loans_new_indiv_byn" if item == 7 else "loans_new_indiv_byn_nonpref"
                out += [(name, f"{yy}-{k + 1:02d}", x) for k, x in enumerate(vv[:12]) if x is not None]
    return out


def table_339(lines):
    out, item, cur = [], None, None
    for l in block(lines, "", title="Динамика показателей государственных ценных бумаг на внутреннем рынке"):
        m = re.match(r"^\s*1\.(\d)\.", l)
        if m:
            item = int(m.group(1))
        elif re.match(r"^\s*дней\s*$", l):               # unit line of 1.4 / 1.8 (average term, days): the heading
            item = None                                    # "1.8." itself is lost in some bulletins' text layer
        for key, name in (("национальной", "byn"), ("российских", "rub"), ("долларах", "usd"), ("евро", "eur")):
            if key in l:
                cur = name
        y, v = values(l)
        if y is None or item not in (3, 7) or cur in (None, "rub"):
            continue
        name = f"gov_bond_{'short' if item == 3 else 'long'}_yield_{cur}"
        bad = [x for x in v[:12] if x is not None and not 0 <= x <= 50]
        if bad:                                            # a yield in % p.a. above 50: a misread row, not data
            print(f"  skipped implausible {name} {y}: {bad}")
        out += [(name, f"{y}-{k + 1:02d}", x) for k, x in enumerate(v[:12]) if x is not None and 0 <= x <= 50]
    return out


def main():
    rows = []
    for pdf in sorted(BUL.glob("stat_bulletin_*.pdf")):
        stamp = re.search(r"(\d{4})_(\d\d)", pdf.name).group(0)
        try:
            lines = text_of(pdf)
        except subprocess.CalledProcessError:
            print(f"{pdf.name}: unreadable (incomplete download?) - skipped")
            continue
        for fn in (table_37, table_339):
            try:
                rows += [(stamp, *r) for r in fn(lines)]
            except StopIteration:
                print(f"{pdf.name}: table of {fn.__name__} not found")
    df = pd.DataFrame(rows, columns=["bulletin", "indicator", "month", "value"])
    df = df.sort_values("bulletin").drop_duplicates(["indicator", "month"], keep="last").sort_values(["indicator", "month"])
    df = df[df["month"].between("2021-01", "2025-12")]
    df.to_csv(_DATA / "by_nbrb_bulletin.csv", index=False)
    print(df.groupby("indicator")["month"].agg(["count", "min", "max"]))


if __name__ == "__main__":
    main()
