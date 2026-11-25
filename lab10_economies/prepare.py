"""Analysis panels of Lab 10 (Belarus, Poland, Russia, Lithuania): the collected monthly series,
exchange rates and derived features.

Input
  data/processed/d9/monthly_wide_{BY,PL,RU,LT}.csv   collection 2020-12..2025-12 (collect/build.py, mean over sources)
  data/processed/d9/monthly_long.csv           the Belstat CPI (official inflation of Belarus) by source
  data/raw/d9/oecd/cpi_pol.csv            the official inflation of Poland: GUS CPI, growth over 12 months as published
                                        (republished by the OECD, DSD_PRICES@DF_PRICES_ALL, transformation GY)
  data/raw/d9/oecd/cpi_pol_IX__Z.csv      the GUS CPI index 2015 = 100 (POL.M.N.CPI.IX._T.N._Z): monthly changes
  data/raw/d9/imf/er_{BLR,POL,RUS}.xml   IMF ER: national currency per US dollar, period average (XDC_USD, PA_RT)
  data/raw/d9/ecb/exr_usd.csv            ECB: US dollars per euro, monthly average (Lithuania)
Inflation is the official figure of each country: Belstat's CPI, GUS's CPI, Rosstat's CPI (via the EEC and the Bank of
Russia) and, for Lithuania, the HICP that Statistics Lithuania and the Bank of Lithuania report. (The collection's
averages for Poland also contain the IMF series and the HICP, which differ from the official CPI by up to 2 p.p.; the
HICP is kept as cpi_hicp_yoy for reference.)
Output
  data/processed/d9/panel_{BY,PL,RU,LT}.csv with, in addition to the collected levels:
    fx_usd             national currency per US dollar, monthly average
    cpi_index          consumer prices, 2020-12 = 100, chained from the official month-on-month changes and
                       benchmarked to the official December rates over 12 months (benchmark)
    policy_rate        the rate in force on each day, averaged over the month (policy_rate_end: end of month)
    <x>_yoy            growth over 12 months, % (from 2021-12, because the collection starts in 2020-12)
    <x>_cum            growth since December 2020 (the base of the study period 2021-2025), %
    <x>_rel            price relative to consumer prices, 2020-12 = 100 (above 100: the good became dearer than
                       the average basket)
    real_<rate>        rate or yield minus CPI inflation over 12 months, p.p. (ex-post real rate)
  Key series of each country are renamed to common names (bond_gov, bond_retail, ...), see KEYS.

Run (from code/):  python -m lab10_economies.prepare
"""
import re

import numpy as np
import pandas as pd

from .paths import DATA, RAW
PRICES = ["price_flour_wheat", "price_milk", "price_sugar", "price_petrol"]
BASE_MONTH = "2020-12"            # base of growth (first row of the collection): the study period is 2021-2025
# common names of the rate series used in the comparison; the collection keeps the original names
# Where several sources publish an indicator, the panel takes the figure of its producer (the others stay in the
# collection as checks): averaging sources mixes definitions and roundings.
PRIMARY = {"BY": {"mortgage_rate": "NBRB xlsx", "price_flour_wheat": "Belstat"},
           "PL": {"mortgage_rate": "NBP MIR", "gov_bond_10y": "Eurostat", "price_petrol": "GUS annual x HICP"},
           "RU": {}, "LT": {}}
KEYS = {
    "BY": {"policy_rate": "policy_rate_avg",                  # NBRB refinancing rate, mean over the days of the month
           "mortgage_rate": "mortgage_rate",                  # new loans to individuals in BYN (no housing-only series)
           "bond_gov": "gov_bond_long_yield_usd_placement",   # long-term government bonds in USD, at placement
           "bond_gov_local": "gov_bond_long_yield_byn_placement",
           "bond_bank": "bond_bank_byn_coupon"},              # coupons of bank bonds in BYN in circulation
    "RU": {"policy_rate": "policy_rate_avg",                  # key rate of the Bank of Russia, mean over the month
           "mortgage_rate": "mortgage_rate",                  # new mortgage loans in RUB (incl. subsidized programmes)
           "bond_gov": "gov_bond_10y",                        # OFZ zero-coupon yield, 10 years, monthly mean
           "bond_gov_local": "gov_bond_10y",
           "bond_market": "gov_bond_10y",                     # market yield: the price of the bonds a holder has
           "bond_gov_5y": "gov_bond_5y",
           "bond_index": "gov_bond_tr_index"},                # Moscow Exchange total-return index RGBITR
    "LT": {"policy_rate": "policy_rate_avg",                  # ECB deposit facility rate, mean over the month
           "mortgage_rate": "mortgage_rate",                  # new housing loans to households (ECB MIR)
           "bond_gov": "gov_bond_10y",                        # 10-year yield, last primary-market issue (Eurostat)
           "bond_gov_local": "gov_bond_10y",
           "bond_market": "gov_bond_10y_ea"},                 # no market yield of its own: the euro-area 10-year yield
    "PL": {"policy_rate": "policy_rate_avg",                  # NBP reference rate, mean over the month
           "mortgage_rate": "mortgage_rate",                  # new housing loans to households in PLN (NBP MIR)
           "bond_gov": "gov_bond_10y",                        # 10-year government bond yield, secondary market
           "bond_gov_local": "gov_bond_10y",
           "bond_market": "gov_bond_10y",
           "bond_retail": "bond_EDO_first_rate",              # 10-year savings bond, first-year rate
           "bond_retail_realized": "bond_EDO_realized"},      # realized yield of the issue of that month
}


def fx(country: str) -> pd.Series:
    """National currency per US dollar, monthly average (IMF ER; Lithuania, in the euro since 2015: ECB reference rate)."""
    if country == "LTU":
        e = pd.read_csv(RAW / "ecb" / "exr_usd.csv")         # US dollars per euro
        return 1 / e.set_index("TIME_PERIOD")["OBS_VALUE"].sort_index()
    s = (RAW / "imf" / f"er_{country}.xml").read_text(encoding="utf-8")
    for attrs, body in re.findall(r"<Series ([^>]*)>(.*?)</Series>", s, re.S):
        a = dict(re.findall(r'(\w+)="([^"]*)"', attrs))
        if a.get("INDICATOR") == "XDC_USD" and a.get("TYPE_OF_TRANSFORMATION") == "PA_RT":
            obs = {}
            for o in re.findall(r"<Obs ([^>]*)/>", body):
                d = dict(re.findall(r'(\w+)="([^"]*)"', o))
                obs[d["TIME_PERIOD"].replace("-M", "-")] = float(d["OBS_VALUE"])
            return pd.Series(obs).sort_index()
    raise ValueError(f"no XDC_USD series for {country}")


def official_cpi(country: str) -> tuple[pd.Series, pd.Series]:
    """Official inflation: (growth over 12 months, growth over 1 month), %, months as 'YYYY-MM'."""
    if country == "BY":
        l = pd.read_csv(DATA / "monthly_long.csv")
        b = l[(l.country == "BY") & (l.source == "Belstat")]
        get = lambda ind: b[b.indicator == ind].set_index("month")["value"]
        return get("cpi_yoy"), get("cpi_mom")
    if country == "RU":           # Rosstat: the rate over 12 months with two decimals (Bank of Russia), the monthly
        l = pd.read_csv(DATA / "monthly_long.csv")      # rate with one (EEC); the EEC's rounded rate over 12 months is a check
        r = l[l.country == "RU"]
        get = lambda ind, src: r[(r.indicator == ind) & (r.source == src)].set_index("month")["value"]
        return get("cpi_yoy", "Bank of Russia"), get("cpi_mom", "EEC (Rosstat)")
    if country == "LT":                                       # HICP of Lithuania (Eurostat)
        w = pd.read_csv(DATA / f"monthly_wide_{country}.csv", index_col=0)
        return w["cpi_yoy"], w["cpi_mom"]
    # GUS publishes the rate over 12 months directly (basket of the current year); chaining its monthly index across a
    # change of weights gives a different number (February 2023: 19.2 % instead of the published 18.4 %). The published
    # rate is the official inflation; the index serves the monthly changes and the cumulative growth.
    g = pd.read_csv(RAW / "oecd" / "cpi_pol.csv")
    yoy = g[g["TRANSFORMATION"] == "GY"].set_index("TIME_PERIOD")["OBS_VALUE"].sort_index()
    o = pd.read_csv(RAW / "oecd" / "cpi_pol_IX__Z.csv")
    idx = o.set_index("TIME_PERIOD")["OBS_VALUE"].sort_index()
    return yoy, 100 * (idx / idx.shift(1) - 1)


def benchmark(index: pd.Series, yoy: pd.Series) -> pd.Series:
    """Consumer prices chained from the official monthly changes, scaled within every calendar year so that each December
    matches the official rate over 12 months (the factor of a year is spread geometrically over its months).  Monthly
    changes are published rounded (Russia, Lithuania: one decimal) and the Polish basket is re-weighted every year, so
    the chained index alone drifts from the compound of the published yearly rates (by 0.3 p.p. over five years)."""
    out, carry = index.copy(), 1.0
    for y in range(int(BASE_MONTH[:4]) + 1, int(index.index[-1][:4]) + 1):
        k = (1 + yoy[f"{y}-12"] / 100) / (index[f"{y}-12"] / index[f"{y - 1}-12"])   # correction of the year
        months = [m for m in index.index if m.startswith(f"{y}-")]
        for j, m in enumerate(months, 1):
            out[m] = index[m] * carry * k ** (j / 12)
        carry *= k
    return out


def fill_short_gaps(s: pd.Series, max_len: int = 2) -> pd.Series:
    """Linear interpolation of runs of at most max_len missing months between two values (Russian milk: two months of
    2022 dropped as a republication error); longer gaps (Russian sugar and petrol, December values only) stay empty."""
    isna = s.isna()
    size = isna.groupby((isna != isna.shift()).cumsum()).transform("sum")
    fill = isna & (size <= max_len) & s.ffill().notna() & s.bfill().notna()
    return s.where(~fill, s.interpolate(limit_area="inside"))


def panel(country: str, iso: str) -> pd.DataFrame:
    w = pd.read_csv(DATA / f"monthly_wide_{country}.csv", index_col=0)
    w.index = w.index.astype(str)
    long = pd.read_csv(DATA / "monthly_long.csv")
    for ind, src in PRIMARY[country].items():
        s = long[(long.country == country) & (long.indicator == ind) & (long.source == src)].set_index("month")["value"]
        w[ind] = s.reindex(w.index)
    e = fx(iso)
    p = w.copy()
    for c in PRICES:
        p[c] = fill_short_gaps(p[c])
    yoy, mom = official_cpi(country)
    if country == "PL":
        p["cpi_hicp_yoy"] = w["cpi_yoy"]                    # the collection's average, kept for reference
    p["cpi_yoy"], p["cpi_mom"] = yoy.reindex(p.index), mom.reindex(p.index)
    p["fx_usd"] = e.reindex(p.index)
    chained = 100 * (1 + p["cpi_mom"].where(p.index > BASE_MONTH, 0) / 100).cumprod()          # 2020-12 = 100
    p["cpi_index"] = benchmark(chained, p["cpi_yoy"])
    for c in PRICES + ["cpi_index", "fx_usd"]:
        name = c.replace("price_", "") if c.startswith("price_") else c
        p[f"{name}_cum"] = 100 * (p[c] / p.loc[BASE_MONTH, c] - 1)
        p[f"{name}_yoy"] = 100 * (p[c] / p[c].shift(12) - 1)
    p["fx_usd_yoy"] = 100 * (p["fx_usd"] / e.shift(12).reindex(p.index) - 1)  # from 2021 (IMF has 2020)
    for c in PRICES:
        name = c.replace("price_", "")
        p[f"{name}_rel"] = 100 * (p[c] / p.loc[BASE_MONTH, c]) / (p["cpi_index"] / p.loc[BASE_MONTH, "cpi_index"])
    p["policy_rate_end"] = w["policy_rate"]                 # end of month, for reference
    for common, col in KEYS[country].items():
        p[common] = w[col] if col in w else np.nan
    for common in ("policy_rate", "mortgage_rate", "bond_gov", "bond_gov_local", "bond_bank", "bond_retail"):
        if common in p:
            p[f"real_{common}"] = p[common] - p["cpi_yoy"]
    return p


def main():
    for country, iso in (("BY", "BLR"), ("PL", "POL"), ("RU", "RUS"), ("LT", "LTU")):
        p = panel(country, iso)
        p.to_csv(DATA / f"panel_{country}.csv")
        print(country, p.shape, "| fx", p["fx_usd"].iloc[0].round(3), "->", p["fx_usd"].iloc[-1].round(3),
              "| cpi index Dec 2025:", p["cpi_index"].iloc[-1].round(1))


if __name__ == "__main__":
    main()
