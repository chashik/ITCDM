"""Part 1, chapter 2 (mathematical foundations), section M7. Index numbers as functional equations, on the data set D9 (Lab 10).

1. Chain-linking and circularity.  A price index P(a, b) is circular if P(a, c) = P(a, b) P(b, c) for all periods —
   a cocycle condition.  The official Polish CPI (GUS) re-weights its basket every year; the growth over 12 months
   computed from the chained monthly index therefore differs from the published one (CPI Manual 2020; ch. 16 of the
   2004 Manual for the tests).  Measured month by month over 2021-2025.
2. Elementary indices of four goods (Belstat average prices of flour, milk, sugar, petrol; Decembers 2020-2025):
   Jevons (geometric mean of price relatives) and Dutot (ratio of mean prices) are circular and pass time reversal;
   Carli (arithmetic mean of relatives) fails both: P_C(a, b) P_C(b, a) > 1 and the chained Carli drifts upwards.
3. Real interest rates: the exact rate (1 + i) / (1 + pi) - 1 against the approximation i - pi used in Lab 10's
   tables, over all country-months 2021-2025.
4. Rounding in a chained index: Russian CPI chained from month-on-month rates published with one decimal (EEC)
   against the growth over 12 months published with two decimals (Bank of Russia); bound 12 x 0.05 p.p.
Outputs: paper/generated/math7_numbers.tex.  Run (from code/):  python -m math_foundations.m7_index_numbers
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dmfin.reporting import save_numbers

GOODS = ["price_flour_wheat", "price_milk", "price_sugar", "price_petrol"]


def jevons(p0, p1):
    return float(np.exp(np.mean(np.log(np.asarray(p1) / np.asarray(p0)))))


def carli(p0, p1):
    return float(np.mean(np.asarray(p1) / np.asarray(p0)))


def dutot(p0, p1):
    return float(np.mean(p1) / np.mean(p0))


def elementary_tests(P: np.ndarray) -> dict:
    """P: periods x goods.  Time reversal P(a,b) P(b,a) - 1 and chain drift (chained / direct - 1) per formula."""
    out = {}
    for name, f in (("jevons", jevons), ("carli", carli), ("dutot", dutot)):
        rev = max(abs(f(P[t], P[t + 1]) * f(P[t + 1], P[t]) - 1) for t in range(len(P) - 1))
        chained = np.prod([f(P[t], P[t + 1]) for t in range(len(P) - 1)])
        out[f"{name}_reversal_max"] = rev
        out[f"{name}_chain_drift_pct"] = 100 * (chained / f(P[0], P[-1]) - 1)
    return out


def main():
    from lab10_economies.collect.readers import eec_series, html_table, num as tonum
    from lab10_economies.common import COUNTRIES, panel
    from lab10_economies.paths import DATA, RAW
    num = {}
    # 1. chained index against the published rate: official Polish CPI
    o = pd.read_csv(RAW / "oecd" / "cpi_pol.csv")
    pub = o.set_index("TIME_PERIOD")["OBS_VALUE"].sort_index()
    ix = pd.read_csv(RAW / "oecd" / "cpi_pol_IX__Z.csv").set_index("TIME_PERIOD")["OBS_VALUE"].sort_index()
    chained = 100 * (ix / ix.shift(12) - 1)
    d = (chained - pub).dropna().loc["2021-01":"2025-12"]
    k = d.abs().idxmax()
    num.update({"pl_months": len(d), "pl_maxdiff": float(d.abs().max()), "pl_maxdiff_month": k,
                "pl_published": float(pub[k]), "pl_chained": float(chained[k]), "pl_meanabs": float(d.abs().mean()),
                "pl_feb23_published": float(pub["2023-02"]), "pl_feb23_chained": float(chained["2023-02"])})
    # 2. elementary indices of four Belarusian goods, Decembers 2020-2025
    w = pd.read_csv(DATA / "monthly_wide_BY.csv", index_col=0)
    P = w.loc[[f"{y}-12" for y in range(2020, 2026)], GOODS].to_numpy()
    num.update(elementary_tests(P))
    num["carli_reversal_example"] = carli(P[1], P[2]) * carli(P[2], P[1])      # 2021 -> 2022 -> 2021
    # 3. exact and approximate real interest rates
    diffs = []
    for c in COUNTRIES:
        p = panel(c)
        i, pi = p["policy_rate"] / 100, p["cpi_yoy"] / 100
        approx, exact = 100 * (i - pi), 100 * ((1 + i) / (1 + pi) - 1)
        diffs.append(pd.DataFrame({"country": c, "approx": approx, "exact": exact}))
    D = pd.concat(diffs).dropna()
    D["gap"] = D["approx"] - D["exact"]
    m = D["gap"].abs().idxmax()
    worst = D.loc[[m]].iloc[D.loc[[m]]["gap"].abs().argmax()]
    num.update({"real_rows": len(D), "real_maxgap": float(D["gap"].abs().max()), "real_maxgap_country": worst["country"],
                "real_maxgap_month": str(m), "real_maxgap_approx": float(worst["approx"]),
                "real_maxgap_exact": float(worst["exact"]), "real_meanabsgap": float(D["gap"].abs().mean())})
    # 4. rounding in a chained index: Russia, one-decimal monthly rates against two-decimal yearly ones
    mom = eec_series(RAW / "eec" / "CPI_1.xls") / 100                       # index to the previous month (1.006 ...)
    mom.index = pd.PeriodIndex(mom.index, freq="M")
    chain12 = 100 * (mom.sort_index().rolling(12).apply(np.prod, raw=True) - 1)
    rows = [r for r in html_table(RAW / "cbr" / "infl.html") if len(r) == 4 and r[0][2] == "."]
    yoy = pd.Series({pd.Period(f"{r[0][3:]}-{r[0][:2]}", "M"): tonum(r[2]) for r in rows}).sort_index()
    e = (chain12.reindex(yoy.index) - yoy).dropna().loc["2021-01":"2025-12"]
    num.update({"ru_months": len(e), "ru_round_maxabs": float(e.abs().max()), "ru_round_rms": float(np.sqrt((e ** 2).mean())),
                "ru_round_bound": 12 * 0.05, "ru_round_sd_theory": float(np.sqrt(12) * 0.05 / np.sqrt(3))})
    save_numbers("math7", num)
    print(pd.Series(num).to_string())


if __name__ == "__main__":
    main()
