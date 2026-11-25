"""Monthly series 2020-12 ... 2025-12 for Belarus (BY), Poland (PL), Russia (RU) and Lithuania (LT) from the raw
downloads in data/raw/d9 (collect/download.py; collect/fetch_blocked.sh for NBRB, BCSE, Rosstat).  December 2020 is
the base of growth rates; Lab 10 reports 2021-2025.  The parsers of collect/build_all.py run first.

Output
  data/processed/d9/monthly_long.csv     one row per country, indicator, month and source (value as published or derived)
  data/processed/d9/monthly_summary.csv  per country, indicator and month: mean over the sources, min, max, number and list
  data/processed/d9/sources.csv          the source of every (country, indicator, source) series with unit and remarks

Run (from code/):  python -m lab10_economies.collect.build
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import DATA as _DATA, RAW as _RAW
from .readers import (RU_MONTHS, daily, eec_items, eec_series, eurostat, html_table, imf_series, num,
                      step_month_mean, step_to_month_end)
RAW, OUT = _RAW, _DATA
MONTHS = pd.period_range("2020-12", "2025-12", freq="M")   # December 2020: base of the growth rates of 2021
ROWS, SOURCES = [], {}


def add(country, indicator, unit, source, series: pd.Series, note=""):
    """series: index = pandas Period (M) or 'YYYY-MM' strings; only 2020-12 ... 2025-12 is kept."""
    s = series.copy()
    s.index = pd.PeriodIndex([pd.Period(str(i)[:7], "M") for i in s.index])
    s = s[~s.index.duplicated(keep="last")].reindex(MONTHS)
    for m, v in s.items():
        if pd.notna(v):
            ROWS.append({"country": country, "indicator": indicator, "unit": unit, "month": str(m),
                         "source": source, "value": float(v)})
    SOURCES[(country, indicator, source)] = {"unit": unit, "months": int(s.notna().sum()), "note": note}


# ------------------------------------------------------------------ Belarus: prices
bp = pd.read_csv(OUT / "by_belstat_prices.csv")          # from belstat_prices.py
for ind, g in bp.groupby("indicator"):
    add("BY", f"price_{ind}", g["unit"].iloc[0], "Belstat", g.set_index("month")["value"],
        "Средние цены в розничной сети, Республика Беларусь (belstat.gov.by, ежемесячные таблицы)")
fao = json.loads((RAW / "fao" / "series_9b5697fb-4ad4-4a15-a6e8-23ad23157ab3.json").read_text())["datapoints"]
add("BY", "price_flour_wheat", "BYN/kg", "FAO FPMA", pd.Series({p["date"]: p["price_value"] for p in fao}),
    "FAO GIEWS FPMA, national average retail; source of FAO: Belstat (republication, not independent)")

# ------------------------------------------------------------------ Belarus: CPI
x = pd.read_excel(RAW / "belstat" / "IPC-0926.xlsx", sheet_name="2010 г. = 100", header=None, dtype=str)
lev = {}
for _, r in x.iterrows():
    if re.fullmatch(r"\d{4}", str(r[0]).strip()):
        for mo in range(1, 13):
            v = str(r[mo]).replace(",", ".")
            if re.fullmatch(r"[\d.]+", v):
                lev[pd.Period(f"{int(r[0])}-{mo:02d}", "M")] = float(v)
lev = pd.Series(lev).sort_index()
add("BY", "cpi_yoy", "%", "Belstat", (lev / lev.shift(12) - 1) * 100, "ИПЦ, 2010 = 100 (IPC-0926.xlsx), к соотв. месяцу пред. года")
add("BY", "cpi_mom", "%", "Belstat", (lev / lev.shift(1) - 1) * 100, "ИПЦ к предыдущему месяцу")


for c, cc in (("BY", "BLR"), ("PL", "POL")):
    add(c, "cpi_yoy", "%", "IMF CPI", imf_series(RAW / "imf" / "cpi.json", cc, ""), "IMF STA CPI, YoY, % (api.imf.org)")

# ------------------------------------------------------------------ Belarus: rates
nb = json.loads((RAW / "nbrb_api" / "refinancingrate.json").read_text())
add("BY", "policy_rate_avg", "% p.a.", "NBRB API", step_month_mean([r["Date"] for r in nb], [r["Value"] for r in nb]),
    "Refinancing rate of the NBRB, mean over the days of the month (api.nbrb.by/refinancingrate)")
add("BY", "policy_rate", "% p.a.", "NBRB API", step_to_month_end([r["Date"] for r in nb], [r["Value"] for r in nb]),
    "Ставка рефинансирования НБРБ на конец месяца (api.nbrb.by/refinancingrate)")
ir = RAW / "imf" / "mfs_ir.xml"
add("BY", "policy_rate", "% p.a.", "IMF MFS", imf_series(ir, "BLR", "MFS166"), "IMF MFS_IR: monetary policy-related rate")
add("BY", "lending_rate_all", "% p.a.", "IMF MFS", imf_series(ir, "BLR", "MFS162"),
    "IMF MFS_IR: lending rate, all loans (not mortgages) — placeholder until NBRB mortgage statistics")
add("BY", "deposit_rate_all", "% p.a.", "IMF MFS", imf_series(ir, "BLR", "MFS135"), "IMF MFS_IR: deposit rate")

# ------------------------------------------------------------------ Belarus: NBRB rates and bonds (raw/manual, fetch_blocked.sh)
nr = OUT / "by_nbrb_rates.csv"                           # nbrb_rates.py
if nr.exists():
    d = pd.read_csv(nr)
    for ind, name, note in (("loans_new_indiv_byn", "mortgage_rate", "NBRB: new BYN loans to individuals, all (incl. preferential); "
                             "no housing-only series at NBRB - proxy"),
                            ("loans_new_indiv_byn_nonpref", "mortgage_rate_nonpref", "NBRB: new BYN loans to individuals without "
                             "preferential loans - proxy"),
                            ("deposits_new_indiv_byn", "deposit_rate_new_indiv", "NBRB: new BYN deposits of individuals")):
        g = d[d.indicator == ind].set_index("month")["value"]
        add("BY", name, "% p.a.", "NBRB xlsx", g, note + " (avgintratesdyn/YYYY_ru.xlsx)")
nb_b = OUT / "by_nbrb_bulletin.csv"                      # nbrb_bulletin.py
if nb_b.exists():
    d = pd.read_csv(nb_b)
    for ind, name in (("loans_new_indiv_byn", "mortgage_rate"), ("loans_new_indiv_byn_nonpref", "mortgage_rate_nonpref")):
        add("BY", name, "% p.a.", "NBRB bulletin", d[d.indicator == ind].set_index("month")["value"],
            "NBRB statistical bulletin, table 3.7 (same data as the xlsx, rounded)")
    for ind in sorted(set(d.indicator)):
        if ind.startswith("gov_bond"):
            add("BY", ind + "_placement", "% p.a.", "NBRB bulletin", d[d.indicator == ind].set_index("month")["value"],
                "NBRB bulletin, table 3.38/3.39: yield of government bonds at placement in the month (gaps = no placement)")

bb = OUT / "by_bank_bonds.csv"                           # minfin_register.py
if bb.exists():
    d = pd.read_csv(bb)
    for cur in ("usd", "byn"):
        g = d[d.currency == cur].set_index("month")
        add("BY", f"bond_bank_{cur}_coupon", "% p.a.", "Minfin register", g["coupon_wavg"],
            f"Bonds of banks in circulation in {cur.upper()}: volume-weighted coupon of interest-bearing issues "
            "(Minfin register of bonds in circulation, 1st of the following month)")

# ------------------------------------------------------------------ Poland: CPI
add("PL", "cpi_yoy", "%", "Eurostat HICP", eurostat(RAW / "eurostat" / "prc_hicp_manr_PL.json"), "HICP (not national CPI), annual rate")
add("PL", "cpi_mom", "%", "Eurostat HICP", eurostat(RAW / "eurostat" / "prc_hicp_mmor_PL.json"), "HICP, monthly rate")
o = pd.read_csv(RAW / "oecd" / "cpi_pol.csv")
add("PL", "cpi_yoy", "%", "OECD CPI", o.set_index("TIME_PERIOD")["OBS_VALUE"].sort_index(), "OECD, national CPI, growth over 1 year")

# ------------------------------------------------------------------ Poland: rates
xml = (RAW / "nbp" / "stopy_archiwum.xml").read_text(encoding="utf-8-sig")
ref = [(m.group(1), float(r.group(1).replace(",", "."))) for m in re.finditer(r'obowiazuje_od="([\d-]+)">(.*?)</pozycje>', xml, re.S)
       for r in [re.search(r'id="ref"\s+oprocentowanie="([\d,]+)"', m.group(2))] if r]
add("PL", "policy_rate", "% p.a.", "NBP", step_to_month_end(*zip(*ref)), "Stopa referencyjna NBP na koniec miesiąca (static.nbp.pl)")
add("PL", "policy_rate_avg", "% p.a.", "NBP", step_month_mean(*zip(*ref)),
    "NBP reference rate, mean over the days of the month (static.nbp.pl)")
b = pd.read_csv(RAW / "bis" / "cbpol.csv")
add("PL", "policy_rate", "% p.a.", "BIS", b[b.REF_AREA == "PL"].set_index("TIME_PERIOD")["OBS_VALUE"], "BIS WS_CBPOL, end of period")
add("PL", "policy_rate", "% p.a.", "IMF MFS", imf_series(ir, "POL", "MFS166"), "IMF MFS_IR: monetary policy-related rate")
m = pd.read_excel(RAW / "nbp" / "mir" / "stopy_proc_pl_srdW.xlsx", "4 OPN2PLN", header=None)
dates = pd.to_datetime(m.iloc[1], errors="coerce", format="mixed")
row = pd.to_numeric(m[m.iloc[:, 3].astype(str).str.startswith("na nieruchomości mieszkani")].iloc[0], errors="coerce")
ok = dates.notna() & row.notna()
add("PL", "mortgage_rate", "% p.a.", "NBP MIR", pd.Series(row[ok].to_numpy() * 100, index=dates[ok].dt.to_period("M")),
    "NBP MIR: average rate on new PLN loans to households for housing real estate")
e = pd.read_csv(RAW / "ecb" / "mir_pl.csv")
add("PL", "mortgage_rate", "% p.a.", "ECB MIR", e.set_index("TIME_PERIOD")["OBS_VALUE"],
    "ECB MIR PL, new business, housing loans to households (compiled by NBP: same data, other publisher)")
add("PL", "gov_bond_10y", "% p.a.", "Eurostat", eurostat(RAW / "eurostat" / "irt_lt_mcby_m_PL.json"), "EMU convergence criterion bond yield, 10 years")
add("PL", "gov_bond_10y", "% p.a.", "IMF MFS", imf_series(ir, "POL", "S13BOND"), "IMF MFS_IR: government bonds rate")

# ------------------------------------------------------------------ Poland: prices
ANNUAL = {"flour_wheat": ["4953", "1749185"], "milk": ["4975"], "sugar": ["4994"], "petrol": ["196398"]}
HICP = {"flour_wheat": "CP01112", "milk": "CP01141", "sugar": "CP01181", "petrol": "CP07222"}
UNIT = {"flour_wheat": "PLN/kg", "milk": "PLN/l", "sugar": "PLN/kg", "petrol": "PLN/l"}
for ind, vids in ANNUAL.items():
    ann = {}
    for vid in vids:                                   # a later variable (new definition) overrides an older one
        d = json.loads((RAW / "gus" / f"annual_{vid}.json").read_text())
        for r in d.get("results", []):
            for v in r["values"]:
                ann[int(v["year"])] = v["val"]
    idx = eurostat(RAW / "eurostat" / f"midx_{HICP[ind]}.json")
    idx.index = pd.PeriodIndex(idx.index, freq="M")
    est = pd.Series({p: ann[p.year] * idx[p] / idx[idx.index.year == p.year].mean() for p in idx.index if p.year in ann})
    add("PL", f"price_{ind}", UNIT[ind], "GUS annual x HICP", est,
        f"Derived: GUS annual average price (BDL var {','.join(vids)}) distributed over months by Eurostat HICP {HICP[ind]}; "
        "annual means equal GUS exactly")
    add("PL", f"price_{ind}_annual", UNIT[ind], "GUS annual", pd.Series({f"{y}-06": v for y, v in ann.items()}),
        "GUS annual average price (stored at June of each year)")
h = pd.read_excel(RAW / "ec_oil" / "history.xlsx", "Prices with taxes", header=None, skiprows=3, usecols=[0, 172, 173])
h.columns = ["date", "eur_per_pln", "eur_1000l"]
h["date"] = pd.to_datetime(h["date"], errors="coerce")
h = h.dropna()
h["pln_l"] = h["eur_1000l"] / h["eur_per_pln"] / 1000
add("PL", "price_petrol", "PLN/l", "EC Oil Bulletin", h.groupby(h["date"].dt.to_period("M"))["pln_l"].mean(),
    "EU Weekly Oil Bulletin, Euro-super 95 with taxes, monthly mean of weekly prices, converted with the bulletin's rate")

# ------------------------------------------------------------------ Poland: retail savings bonds (pl_bond_yields.py)
by = _DATA / "pl_retail_bonds_yields.csv"
if by.exists():
    yb = pd.read_csv(by)
    for t, g in yb.groupby("type"):
        g = g.set_index("month")
        add("PL", f"bond_{t}_first_rate", "% p.a.", "obligacjeskarbowe.pl", g["first_rate"],
            f"{t}: rate of the first interest period of the issue sold in the month; price 100 PLN (exchange 99.90 PLN)")
        add("PL", f"bond_{t}_realized", "% p.a.", "computed", g["realized_rate"],
            f"{t}: realized yield of the issue held until 2025-12-31 or maturity, before tax (pl_bond_yields.py)")


# ------------------------------------------------------------------ Russia and Lithuania (ru_lt_download.py)
# Rosstat and Statistics Lithuania do not answer from the analysis session; their figures come through official
# republishers.  Price series of Lithuania are HICP item indices (Statistics Lithuania, via Eurostat) except petrol.
# Russia: prices.  Flour and milk: Rosstat weekly monitoring of average retail prices, republished by FAO FPMA.
for g, unit in (("flour", "RUB/kg"), ("milk", "RUB/l")):
    fp = pd.DataFrame(json.loads((RAW / "fao" / f"series_RUS_{g}.json").read_text())["datapoints"])
    fp = fp[fp["periodicity"] == "weekly"].assign(date=lambda d: pd.to_datetime(d["date"])).sort_values("date")
    # Weekly values more than 20 % away from the median of the surrounding half year are errors of the republication
    # (milk, 2022-06-03 ... 2022-07-29: about 98 RUB/l between 72.9 and 72.8); their months stay empty.
    med = fp["price_value"].rolling(27, center=True, min_periods=9).median()
    bad = (fp["price_value"] / med - 1).abs() > 0.2
    fp = fp[~bad]
    m = fp.groupby(fp["date"].dt.to_period("M"))["price_value"].mean()
    m = m.reindex(pd.period_range(m.index.min(), m.index.max(), freq="M"))
    add("RU", "price_flour_wheat" if g == "flour" else "price_milk", unit, "FAO FPMA", m,
        f"FAO GIEWS FPMA, national average retail price ({'wheat flour' if g == 'flour' else 'pasteurised milk'}), "
        f"monthly mean of the weekly prices; source of FAO: Rosstat; {int(bad.sum())} weekly values dropped as errors "
        "(more than 20 % from the median of the surrounding half year)")
# Sugar and petrol (and flour, milk and dairy as a check): Rosstat CPI of the item, December over December, from the
# December info blocks of the EEC, chained into an index with December 2020 = 100 (values in December only).
items = {y: eec_items(y) for y in range(2021, 2026)}
for key in ("flour", "milk_dairy", "sugar", "petrol"):
    level, idx = 100.0, {"2020-12": 100.0}
    for y in range(2021, 2026):
        if key in items[y]:
            level *= 1 + items[y][key]["RU"] / 100
            idx[f"{y}-12"] = level
        else:
            break                                       # a missing year breaks the chain
    note = ("Rosstat CPI of the item, December over December, chained (Dec 2020 = 100; December values only); "
            "EEC info blocks InfoCPIyyyy12.pdf")
    add("RU", f"cpi_item_{key}", "index", "EEC (Rosstat)", pd.Series(idx), note)
    if key in ("sugar", "petrol"):
        add("RU", f"price_{key}", "index Dec 2020 = 100", "EEC (Rosstat)", pd.Series(idx),
            note + "; monthly Rosstat prices pending (fetch_blocked.sh)")
    add("BY", f"cpi_item_{key}_eec", "% Dec/Dec", "EEC (Belstat)",
        pd.Series({f"{y}-12": items[y][key]["BY"] for y in range(2021, 2026) if key in items[y]}),
        "Belstat CPI of the item, December over December (EEC): check of the Belstat average prices")

# Russia: inflation (Rosstat; EEC with one decimal, the Bank of Russia with two)
add("RU", "cpi_mom", "%", "EEC (Rosstat)", (eec_series(RAW / "eec" / "CPI_1.xls") - 100).round(4),
    "Rosstat CPI to the previous month (EEC Series_CPI/CPI_1.xls)")
add("RU", "cpi_yoy", "%", "EEC (Rosstat)", (eec_series(RAW / "eec" / "CPI_3.xls") - 100).round(4),
    "Rosstat CPI to the same month of the previous year (EEC Series_CPI/CPI_3.xls)")
inf = [r for r in html_table(RAW / "cbr" / "infl.html") if len(r) == 4 and re.fullmatch(r"\d\d\.\d{4}", r[0])]
add("RU", "cpi_yoy", "%", "Bank of Russia", pd.Series({f"{r[0][3:]}-{r[0][:2]}": num(r[2]) for r in inf}),
    "Inflation, % y/y (Rosstat), Bank of Russia table 'Inflation and key rate' (cbr.ru/hd_base/infl)")
# Russia: rates
kr = daily(RAW / "cbr" / "keyrate.html")
add("RU", "policy_rate_avg", "% p.a.", "Bank of Russia", step_month_mean(kr.index, kr.to_numpy()),
    "Key rate of the Bank of Russia, mean over the days of the month (cbr.ru/hd_base/KeyRate)")
add("RU", "policy_rate", "% p.a.", "Bank of Russia", kr.groupby(kr.index.to_period("M")).last(),
    "Key rate of the Bank of Russia, end of month (cbr.ru/hd_base/KeyRate)")
add("RU", "policy_rate", "% p.a.", "Bank of Russia (monthly table)", pd.Series({f"{r[0][3:]}-{r[0][:2]}": num(r[1]) for r in inf}),
    "Key rate in the table 'Inflation and key rate' (cbr.ru/hd_base/infl)")
for fname, ind, what in (("02_13_Rates_mortgage.xlsx", "mortgage_rate", "mortgage (ипотечные) housing loans"),
                         ("02_08_Rates_housing.xlsx", "housing_loan_rate", "all housing loans")):
    x = pd.read_excel(RAW / "cbr" / fname, sheet_name="ставка в рублях", header=None)
    head = x.iloc[2]
    row = x[x[0].astype(str).str.strip() == "РОССИЙСКАЯ ФЕДЕРАЦИЯ"].iloc[0]
    s = {}
    for j, h in head.items():
        m = re.fullmatch(r"(\w+) (\d{4})", str(h).strip())
        if m and m.group(1).lower() in RU_MONTHS:
            s[f"{m.group(2)}-{RU_MONTHS[m.group(1).lower()]:02d}"] = num(row[j])
    add("RU", ind, "% p.a.", "Bank of Russia", pd.Series(s),
        f"Weighted average rate on {what} to residents in roubles, issued in the month (cbr.ru, {fname}); "
        "includes subsidized programmes")
zr = [r for r in html_table(RAW / "cbr" / "zcyc.html") if len(r) == 13 and re.fullmatch(r"\d\d\.\d\d\.\d{4}", r[0])]
z10 = pd.Series({pd.to_datetime(r[0], dayfirst=True): num(r[9]) for r in zr}).sort_index()   # column of 10 years
add("RU", "gov_bond_10y", "% p.a.", "Bank of Russia", z10.groupby(z10.index.to_period("M")).mean(),
    "Zero-coupon yield of government bonds (OFZ), 10 years, monthly mean of daily values (cbr.ru/hd_base/zcyc_params)")
z5 = pd.Series({pd.to_datetime(r[0], dayfirst=True): num(r[7]) for r in zr}).sort_index()    # column of 5 years
add("RU", "gov_bond_5y", "% p.a.", "Bank of Russia", z5.groupby(z5.index.to_period("M")).mean(),
    "Zero-coupon yield of government bonds (OFZ), 5 years, monthly mean of daily values (cbr.ru/hd_base/zcyc_params)")
tr = pd.concat([pd.read_csv(f, sep=";", skiprows=2) for f in sorted((RAW / "moex").glob("rgbitr_*.csv"))])
tr = tr.dropna().drop_duplicates("TRADEDATE")
tr = tr.set_index(pd.to_datetime(tr["TRADEDATE"]))["CLOSE"].sort_index()
add("RU", "gov_bond_tr_index", "index", "Moscow Exchange", tr.groupby(tr.index.to_period("M")).mean(),
    "Total-return index of federal loan bonds RGBITR (coupons and price changes), monthly mean of daily closes "
    "(iss.moex.com)")

# Lithuania: prices (HICP item indices, 2015 = 100; petrol also as a price)
for coicop, ind in (("CP01112", "price_flour_wheat"), ("CP01141", "price_milk"), ("CP01181", "price_sugar"),
                    ("CP07222", "price_petrol_index")):
    add("LT", ind, "index 2015 = 100", "Eurostat HICP", eurostat(RAW / "eurostat" / f"midx_{coicop}_LT.json"),
        f"HICP item index {coicop} (Statistics Lithuania via Eurostat); Statistics Lithuania's average prices "
        "(osp.stat.gov.lt) are behind a browser check")
h = pd.read_excel(RAW / "ec_oil" / "history.xlsx", "Prices with taxes", header=None, skiprows=3, usecols=[0, 136])
h.columns = ["date", "eur_1000l"]
h["date"] = pd.to_datetime(h["date"], errors="coerce")
h = h.dropna()
add("LT", "price_petrol", "EUR/l", "EC Oil Bulletin",
    h.groupby(h["date"].dt.to_period("M"))["eur_1000l"].mean() / 1000,
    "EU Weekly Oil Bulletin, Euro-super 95 with taxes, monthly mean of weekly prices")
# Lithuania: inflation.  Statistics Lithuania and the Bank of Lithuania report inflation by the HICP; the national CPI
# (OECD) is kept for reference.
add("LT", "cpi_yoy", "%", "Eurostat HICP", eurostat(RAW / "eurostat" / "prc_hicp_manr_LT.json"),
    "HICP, annual rate: the inflation measure reported by Statistics Lithuania and the Bank of Lithuania")
add("LT", "cpi_mom", "%", "Eurostat HICP", eurostat(RAW / "eurostat" / "prc_hicp_mmor_LT.json"), "HICP, monthly rate")
o = pd.read_csv(RAW / "oecd" / "cpi_ltu.csv")
add("LT", "cpi_national_yoy", "%", "OECD CPI", o.set_index("TIME_PERIOD")["OBS_VALUE"],
    "National CPI of Lithuania, growth over 12 months (OECD): reference only")
# Lithuania: rates (euro area since 2015: the ECB sets the policy rate)
for f, ind, note in (("dfr.csv", "policy_rate", "ECB deposit facility rate, end of month (the rate that steers "
                      "money-market rates since 2015)"),
                     ("mro.csv", "policy_rate_mro", "ECB main refinancing operations rate, end of month")):
    e = pd.read_csv(RAW / "ecb" / f)
    add("LT", ind, "% p.a.", "ECB", step_to_month_end(e["TIME_PERIOD"].tolist(), e["OBS_VALUE"].tolist()), note)
    if ind == "policy_rate":
        add("LT", "policy_rate_avg", "% p.a.", "ECB", step_month_mean(e["TIME_PERIOD"].tolist(), e["OBS_VALUE"].tolist()),
            "ECB deposit facility rate, mean over the days of the month")
e = pd.read_csv(RAW / "ecb" / "mir_lt.csv")
add("LT", "mortgage_rate", "% p.a.", "ECB MIR", e.set_index("TIME_PERIOD")["OBS_VALUE"],
    "Bank interest rates, new loans to households for house purchase, Lithuania (compiled by the Bank of Lithuania)")
add("LT", "gov_bond_10y", "% p.a.", "Eurostat", eurostat(RAW / "eurostat" / "irt_lt_mcby_m_LT.json"),
    "EMU convergence criterion bond yield, 10 years")
add("LT", "gov_bond_10y_ea", "% p.a.", "Eurostat", eurostat(RAW / "eurostat" / "irt_lt_mcby_m_EA.json"),
    "EMU convergence criterion bond yield, 10 years, euro area (the Lithuanian series is that of the last issue)")


def main():
    long = pd.DataFrame(ROWS)
    long.to_csv(OUT / "monthly_long.csv", index=False)
    g = long.groupby(["country", "indicator", "unit", "month"])
    summ = g["value"].agg(["mean", "min", "max", "count"]).reset_index()
    summ["sources"] = g["source"].agg(lambda s: "; ".join(sorted(s))).to_numpy()
    summ.to_csv(OUT / "monthly_summary.csv", index=False)
    for c in ("BY", "PL", "RU", "LT"):           # one column per indicator: mean over the sources
        wide = summ[summ.country == c].pivot(index="month", columns="indicator", values="mean")
        wide.to_csv(OUT / f"monthly_wide_{c}.csv")
    src = pd.DataFrame([{"country": c, "indicator": i, "source": s, **v} for (c, i, s), v in SOURCES.items()])
    src.to_csv(OUT / "sources.csv", index=False)
    cov = src.groupby(["country", "indicator"]).agg(sources=("source", lambda s: ", ".join(s)), months=("months", "min"))
    print(cov.to_string())
    print(f"\n{len(long)} values in data/monthly_long.csv")


if __name__ == "__main__":
    main()
