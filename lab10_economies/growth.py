"""Question 1: growth of food and petrol prices against reported inflation, policy and mortgage rates and government
bond yields, Belarus, Poland, Russia and Lithuania, 2021-2025 (December 2020 is the base of growth).

Outputs (results/lab10; figures paper/figures/lab10_*.pdf):
  q1_cumulative.csv      growth 2020-12 -> 2025-12 of each good, consumer prices and the US dollar rate; per year and
                         relative to consumer prices
  q1_yearly.csv          December-on-December growth of each good and of consumer prices, 2021-2025
  q1_rates.csv           yearly averages of inflation, rates and yields and of the real (inflation-adjusted) rates
  q1_savings.csv         what a deposit at the policy rate, a mortgage borrower and a bond holder gained or lost in real
                         terms over 2021-2025
  q1_match.csv           does each price, and the policy rate, match official inflation: growth 2021-2025 against
                         inflation, years within 3 p.p. of it, correlation of the growth over 12 months with inflation
  q1_same_units.csv      official inflation, price growth, the policy rate, the yield of government bonds and the return
                         of their holder in one unit (% per year), 2021-2025
  q1_bond_check_RU.csv   the holder's return computed from the yields against the total-return index of the Moscow
                         Exchange (RGBITR)
  q1_prices_vs_cpi.png, q1_rates.png
Run (from code/):  python -m lab10_economies.growth
"""

import numpy as np
import pandas as pd

from .common import ALL, CURRENCY, GOODS, LABEL, P, PERIOD, md_table, panel, save_figure, save_table

YEARS = [2021, 2022, 2023, 2024, 2025]
RATES = ["policy_rate", "mortgage_rate", "bond_gov", "bond_gov_local", "bond_bank", "bond_retail"]
SPAN = "2021-2025"
# Every function takes the panel from December 2020 (the base) to December 2025: growth is measured from the first row,
# averages and counts use the 60 months 2021-01..2025-12 (rows 1:).


def cumulative(p: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    months = len(p) - 1
    cpi = p["cpi_index_cum"].iloc[-1]
    for g in GOODS + ["cpi_index", "fx_usd"]:
        c = p[f"{g}_cum"].iloc[-1]
        rows[LABEL[g]] = {f"growth {SPAN}, %": c,
                          "per year, %": 100 * ((1 + c / 100) ** (12 / months) - 1),
                          "minus consumer prices, p.p.": c - cpi if g in GOODS else np.nan,
                          "real change, %": 100 * ((1 + c / 100) / (1 + cpi / 100) - 1) if g in GOODS else np.nan}
    return pd.DataFrame(rows).T


def yearly(p: pd.DataFrame) -> pd.DataFrame:
    dec = p[p.index.month == 12]
    out = {}
    for g in GOODS + ["cpi_index"]:
        col = f"price_{g}" if g in GOODS else "cpi_index"
        out[LABEL[g]] = {y: 100 * (dec.loc[dec.index.year == y, col].iloc[0] /
                                   dec.loc[dec.index.year == y - 1, col].iloc[0] - 1) for y in YEARS}
    return pd.DataFrame(out).T


def rates(p: pd.DataFrame) -> pd.DataFrame:
    q = p.iloc[1:]
    cols = ["cpi_yoy"] + [r for r in RATES if r in q and q[r].notna().any()]
    t = q[cols].groupby(q.index.year).mean().T
    t.columns = [str(c) for c in t.columns]
    t[SPAN] = q[cols].mean()
    real = q[[f"real_{r}" for r in cols[1:]]].groupby(q.index.year).mean().T
    real.columns = [str(c) for c in real.columns]
    real[SPAN] = q[[f"real_{r}" for r in cols[1:]]].mean()
    real.index = [f"real {LABEL[r[5:]]}" for r in real.index]
    t.index = [LABEL[r] for r in t.index]
    return pd.concat([t, real])


MATURITY = 10               # the bond of the holder's return: 10 years, like the yields of the panels
BOND_HOLDER = "government bond, 10 years, holder"


def bond_return(y: pd.Series, n: int = MATURITY) -> pd.Series:
    """Monthly return of the holder of a government bond, %, from the market yield y (% per year) alone.  In month t-1
    the holder buys a new n-year bond at par, whose coupon is the yield y[t-1]; in month t the bond is worth the present
    value of its payments at the new yield y[t], one month later, and is exchanged for a new one.  Coupon and price
    change together are the return: a rise of the yield lowers the price, a fall raises it.  (A bond of constant
    maturity: the usual stand-in for a bond index where only yields are published.)"""
    r = y.to_numpy(float) / 100
    c = np.r_[np.nan, r[:-1]]                              # coupon: the yield at purchase, a month earlier
    small = np.abs(r) < 1e-9
    annuity = np.where(small, n, (1 - (1 + r) ** -n) / np.where(small, 1, r))   # value of 1 a year for n years
    price = c * annuity + (1 + r) ** -n                    # 1 (par) when the yield has not changed
    return pd.Series(100 * ((1 + r) ** (1 / 12) * price - 1), index=y.index)


def compound(m: pd.Series, year: int | None = None) -> float:
    """Monthly returns (%) compounded over a calendar year, or per year over all months."""
    if year is not None:
        return 100 * (np.prod(1 + m[m.index.year == year] / 100) - 1)
    return 100 * (np.prod(1 + m / 100) ** (12 / len(m)) - 1)


def savings(p: pd.DataFrame, country: str) -> pd.DataFrame:
    """Real value at the end of 2025 of 100 units placed at the end of 2020 at a rate (monthly compounding of the annual
    rate of each month), deflated by consumer prices.  Bonds traded on a market are valued at their market price (the
    holder's return, bond_return); bonds that are not traded (Belarus) are held to repayment at the yields of the
    issues of each month (a stand-in for reinvestment)."""
    q = p.iloc[1:]
    cpi = p["cpi_index"].iloc[-1] / p["cpi_index"].iloc[0]
    market = "bond_market" in q and q["bond_market"].notna().any()
    rows = {}
    for r in ["policy_rate", "mortgage_rate", "bond_gov_local", "bond_bank", "bond_retail"]:
        if r not in q or q[r].isna().all() or (r == "bond_gov_local" and market):
            continue
        s = q[r].interpolate(limit_direction="both")       # months without a placement: the neighbouring yields
        nominal = float(np.prod(1 + s / 1200))
        rows[LABEL[r]] = {"nominal value of 100": 100 * nominal, "real value of 100": 100 * nominal / cpi}
    if market:                                             # the holder of 10-year bonds, valued at market prices
        nominal = float(np.prod(1 + bond_return(p["bond_market"]).iloc[1:] / 100))
        rows[BOND_HOLDER] = {"nominal value of 100": 100 * nominal, "real value of 100": 100 * nominal / cpi}
    if country == "BY":                                    # US dollar bonds converted into roubles
        s = q["bond_gov"].interpolate(limit_direction="both")
        usd = float(np.prod(1 + s / 1200))
        nominal = usd * p["fx_usd"].iloc[-1] / p["fx_usd"].iloc[0]
        rows["government bond in USD, in BYN"] = {"nominal value of 100": 100 * nominal, "real value of 100": 100 * nominal / cpi}
    rows["cash (no interest)"] = {"nominal value of 100": 100.0, "real value of 100": 100 / cpi}
    return pd.DataFrame(rows).T


def match(p: pd.DataFrame) -> pd.DataFrame:
    """Does each price, and the policy rate, match official inflation?  Growth 2021-2025 against inflation over the same
    months (within 5 p.p. = "about the same"), the calendar years within 3 p.p. of inflation, and the correlation of the
    monthly growth over 12 months with inflation (only where the price is monthly)."""
    q = p.iloc[1:]
    cpi4 = p["cpi_index_cum"].iloc[-1]
    dec = p[p.index.month == 12]
    yr = lambda col, y: 100 * (dec.loc[dec.index.year == y, col].iloc[0] / dec.loc[dec.index.year == y - 1, col].iloc[0] - 1)
    rows = {}
    for g in GOODS:
        g4 = p[f"{g}_cum"].iloc[-1]
        close = sum(abs(yr(f"price_{g}", y) - yr("cpi_index", y)) <= 3 for y in YEARS)
        monthly = q[f"{g}_yoy"].notna().sum() >= 24
        rows[LABEL[g]] = {f"growth {SPAN}, %": g4, f"official inflation {SPAN}, %": cpi4, "difference, p.p.": g4 - cpi4,
                          "verdict": "about the same" if abs(g4 - cpi4) <= 5 else ("faster" if g4 > cpi4 else "slower"),
                          "years within 3 p.p. of inflation (of 5)": int(close),
                          "correlation with inflation, monthly y/y": q[f"{g}_yoy"].corr(q["cpi_yoy"]) if monthly else np.nan}
    pol, cpi = q["policy_rate"], q["cpi_yoy"]
    rows["policy rate"] = {f"growth {SPAN}, %": np.nan, f"official inflation {SPAN}, %": np.nan,
                           "difference, p.p.": (pol - cpi).mean(),
                           "verdict": f"above inflation in {int((pol >= cpi).sum())} of {len(q)} months",
                           "years within 3 p.p. of inflation (of 5)": int(sum(abs(pol[pol.index.year == y].mean() -
                                                                                   cpi[cpi.index.year == y].mean()) <= 3
                                                                               for y in YEARS)),
                           "correlation with inflation, monthly y/y": pol.corr(cpi)}
    t = pd.DataFrame(rows).T
    for c in t:
        if c != "verdict":
            t[c] = pd.to_numeric(t[c])
    return t


USD_BONDS = {"BY"}          # government bonds denominated in US dollars: their return in national currency includes the dollar rate


def same_units(p: pd.DataFrame, country: str = "") -> pd.DataFrame:
    """Inflation, price growth, the policy rate, the yield of government bonds and the return of their holder in one
    unit, % per year.  Calendar years: official inflation in December over 12 months (as published), price growth
    December to December, the policy rate averaged over the year (what a deposit at that rate earned in the year), the
    government bond yield averaged over the year (what the state promised to the buyers of that year, in the currency of
    the bonds), and the return of a holder of government bonds in the year: coupon and change of the market price
    (bond_return); dollar bonds (Belarus) are not traded and are held to repayment, so their return in national currency
    is the dollar yield and the change of the dollar rate from December to December.  Last row: December 2020 -
    December 2025 converted to a yearly rate (compound), the mean policy rate and the mean bond yield."""
    dec = p[p.index.month == 12]
    at = lambda col, y: dec.loc[dec.index.year == y, col].iloc[0]
    has_bonds = "bond_gov" in p and p["bond_gov"].notna().any()
    market = bond_return(p["bond_market"]) if "bond_market" in p and p["bond_market"].notna().any() else None

    def bond_yield(y):                                 # % per year in the currency of the bonds, NaN without placements
        s = p.loc[p.index.year == y, "bond_gov"].dropna() if has_bonds else pd.Series(dtype=float)
        return s.mean() if len(s) else np.nan

    def bond(y):                                       # the holder's return, % per year in national currency
        if country in USD_BONDS:
            return 100 * ((1 + bond_yield(y) / 100) * at("fx_usd", y) / at("fx_usd", y - 1) - 1)
        return compound(market, y) if market is not None else np.nan

    rows = {}
    for y in YEARS:
        r = {"official inflation": at("cpi_yoy", y)}
        for g in GOODS:
            r[LABEL[g]] = 100 * (at(f"price_{g}", y) / at(f"price_{g}", y - 1) - 1)
        r["policy rate"] = p.loc[p.index.year == y, "policy_rate"].mean()
        r["government bond yield"] = bond_yield(y)
        r["government bond return"] = bond(y)
        rows[str(y)] = r
    n = len(p) - 1
    yearly = lambda s: 100 * ((s.iloc[-1] / s.iloc[0]) ** (12 / n) - 1)
    b = [rows[str(y)]["government bond return"] for y in YEARS]
    rows[f"{SPAN}, per year"] = {"official inflation": yearly(p["cpi_index"]),
                                 **{LABEL[g]: yearly(p[f"price_{g}"]) for g in GOODS},
                                 "policy rate": p["policy_rate"].iloc[1:].mean(),
                                 "government bond yield": p["bond_gov"].iloc[1:].mean() if has_bonds else np.nan,
                                 "government bond return": (100 * (np.prod([1 + v / 100 for v in b]) ** (1 / len(b)) - 1)
                                                            if all(np.isfinite(b)) else np.nan)}
    return pd.DataFrame(rows).T


def bond_check(p: pd.DataFrame) -> pd.DataFrame:
    """Russia: the holder's return computed from the yields against the total-return index of federal loan bonds of the
    Moscow Exchange (RGBITR: coupons and prices of the traded bonds), by calendar year and per year over 2021-2025.  The
    bonds of the index are on average shorter than ten years, so the check is made for the 5-year yield and a 5-year
    bond; the 10-year bond of the comparison is shown beside it."""
    idx = 100 * (p["bond_index"] / p["bond_index"].shift(1) - 1)
    series = {"exchange index RGBITR": idx, "computed, 5-year bond": bond_return(p["bond_gov_5y"], 5),
              "computed, 10-year bond": bond_return(p["bond_market"], 10)}
    rows = {}
    for k, m in series.items():
        m = m.iloc[1:]
        rows[k] = {**{str(y): compound(m, y) for y in YEARS}, f"{SPAN}, per year": compound(m),
                   "monthly correlation with the index": m.corr(idx.iloc[1:])}
    return pd.DataFrame(rows).T


def years(ax):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def figures(panels):
    f, axs = P.fig(5.6, nrows=2, ncols=2, sharey=True)
    for ax, (c, p) in zip(axs.ravel(), panels.items()):
        for g, col in zip(GOODS, P.CAT):
            s = 100 + p[f"{g}_cum"].dropna()             # Russian sugar and petrol: December values only
            ax.plot(s.index.to_timestamp(), s, color=col, lw=1.2, marker="o" if len(s) < 12 else None, ms=3,
                    label=LABEL[g])
        ax.plot(p.index.to_timestamp(), 100 + p["cpi_index_cum"], color=P.INK, lw=1.8, ls="--", label=LABEL["cpi_index"])
        ax.set_title(f"{ALL[c]} ({CURRENCY[c]})", fontsize=9.5)
        ax.axhline(100, color=P.MUTED, lw=0.6)
        years(ax)
    for ax in axs[:, 0]:
        ax.set_ylabel("index, December 2020 = 100")
    axs[0, 0].legend(fontsize=9)
    f.tight_layout()
    save_figure(f, "q1_prices_vs_cpi")

    f, axs = P.fig(5.6, nrows=2, ncols=2, sharey=True)
    for ax, (c, p) in zip(axs.ravel(), panels.items()):
        x = p.index.to_timestamp()
        for r, col in zip(["cpi_yoy", "policy_rate", "mortgage_rate", "bond_gov"], [P.INK, P.BLUE, P.ORANGE, P.AQUA]):
            s = p[r]
            ax.plot(x, s, color=col, lw=1.8 if r == "cpi_yoy" else 1.2, marker="." if s.isna().any() else None,
                    ms=3, label=LABEL[r])
        ax.set_title(ALL[c], fontsize=9.5)
        ax.axhline(0, color=P.MUTED, lw=0.6)
        years(ax)
    for ax in axs[:, 0]:
        ax.set_ylabel("% per year")
    h, l = axs[0, 1].get_legend_handles_labels()
    f.legend(h, l[:3] + [l[3] + " (Belarus: USD bonds at placement)"], loc="lower center", ncol=4, fontsize=9,
             frameon=False)
    f.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(f, "q1_rates")


def main():
    panels = {c: panel(c, base=True).loc[:PERIOD[1]] for c in ALL}   # December 2020 = base row
    report = []
    for c, p in panels.items():
        cum, yr, rt, sv = cumulative(p), yearly(p), rates(p), savings(p, c)
        save_table(cum, f"q1_cumulative_{c}", 3)          # 3 digits: q5 rounds again to 1
        save_table(yr, f"q1_yearly_{c}")
        save_table(rt, f"q1_rates_{c}", 3)
        save_table(sv, f"q1_savings_{c}", 3)
        mt = match(p)
        save_table(mt, f"q1_match_{c}")
        su = same_units(p, c)
        save_table(su, f"q1_same_units_{c}", 3)
        if c == "RU":
            save_table(bond_check(p), "q1_bond_check_RU", 3)
        neg = (p["real_policy_rate"].iloc[1:] < 0)
        report += [f"## {ALL[c]}", md_table(cum.rename_axis("series")), "", md_table(yr.rename_axis("Dec/Dec, %")), "",
                   md_table(rt.rename_axis("average, %")), "", md_table(sv.rename_axis("100 held 2021-2025")), "",
                   f"months with a negative real policy rate: {neg.sum()} of {neg.notna().sum()} "
                   f"({neg.index[neg][0]}–{neg.index[neg][-1]})" if neg.any() else "no month with a negative real policy rate", "",
                   "Match with official inflation", md_table(mt.rename_axis("series")), "",
                   "Inflation, price growth, policy rate, government bond yield and return, % per year",
                   md_table(su.rename_axis("year")), ""]
    figures(panels)
    print("\n".join(report))


if __name__ == "__main__":
    main()
