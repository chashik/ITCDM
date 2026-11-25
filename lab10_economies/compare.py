"""Question 5: the four countries compared — one table assembled from the results of questions 1-4.

Output: results/q5_comparison.csv and its Markdown form results/q5_comparison.md (used in README.md).
Run after the other stages (from code/):  python -m lab10_economies.compare
"""

import pandas as pd

from .common import ALL, PERIOD, panel
from .paths import RESULTS as R


def read(name, **kw):
    return pd.read_csv(R / f"{name}.csv", **kw)


def main():
    rows = []
    f1 = lambda v: f"{v:.1f}"

    def add4(topic, item, get):                  # question 1: all four countries
        rows.append({"topic": topic, "indicator": item, **{ALL[c]: get(c) for c in ALL}})

    cum = {c: read(f"q1_cumulative_{c}", index_col=0) for c in ALL}
    for g in ["consumer prices", "wheat flour", "milk", "sugar", "petrol", "US dollar rate"]:
        add4("1 growth 2021-2025", f"{g}, %", lambda c: f1(cum[c].loc[g].iloc[0]))
    for g in ["wheat flour", "milk", "sugar", "petrol"]:
        add4("1 growth 2021-2025", f"{g}, real change (vs consumer prices), %", lambda c: f1(cum[c].loc[g, "real change, %"]))
    rt = {c: read(f"q1_rates_{c}", index_col=0) for c in ALL}
    for item, key in [("inflation, mean y/y", "inflation (CPI, y/y)"), ("policy rate, mean", "policy rate"),
                      ("mortgage rate, mean", "mortgage rate"), ("government bond yield, mean", "government bond yield"),
                      ("real policy rate, mean", "real policy rate"), ("real mortgage rate, mean", "real mortgage rate"),
                      ("real government bond yield, mean", "real government bond yield")]:
        add4("1 rates 2021-2025, % or p.p.", item, lambda c: f1(rt[c].loc[key, "2021-2025"]))
    sv = {c: read(f"q1_savings_{c}", index_col=0) for c in ALL}
    add4("1 real value of 100 after 5 years", "cash", lambda c: f1(sv[c].loc["cash (no interest)"].iloc[1]))
    add4("1 real value of 100 after 5 years", "at the policy rate", lambda c: f1(sv[c].loc["policy rate"].iloc[1]))
    add4("1 real value of 100 after 5 years", "government bonds (BY: USD bonds in BYN held; others: 10-year, market value)",
         lambda c: f1(sv[c].loc["government bond in USD, in BYN" if c == "BY" else "government bond, 10 years, holder"].iloc[1]))
    add4("1 rates 2021-2025, % or p.p.", "months with a negative real policy rate (of 60)",
         lambda c: int((panel(c).loc[PERIOD[0]:PERIOD[1], "real_policy_rate"] < 0).sum()))

    groups = read("q2_groups")
    add4("2 related series", "groups with correlation of at least 0.6 in absolute value",
         lambda c: "; ".join(x for x in groups[groups.country == c]["features"] if "," in x))
    pr = read("q2_predictability")
    add4("2 related series", "series better nowcast from the others than from last value",
         lambda c: f"{int((pr[pr.country == c]['skill with own last value'] > 0).sum())} of "
                   f"{int(pr[pr.country == c]['skill with own last value'].notna().sum())}")
    ll = read("q2_leadlag", index_col=0)
    add4("2 related series", "policy rate follows inflation by, months (negative: leads)", lambda c: -int(ll.loc[c].iloc[0]))
    gc = read("q2_granger")
    sig = gc[gc["BH-adjusted p (all tests)"] < 0.05]
    add4("2 related series", "Granger relations significant after BH correction",
         lambda c: "; ".join(f"{a} → {b}" for a, b in sig[sig.country == c][["cause", "effect"]].values) or "none")

    ep = read("q3_episode", index_col=0)
    for item in ["inflation peak, %", "peak month", "months above 10 %", "first rate hike", "policy-rate peak, %",
                 "first cut", "real policy rate positive from", "lowest real policy rate, p.p."]:
        def cell(c, item=item):
            v = ep.loc[item, c]
            try:
                return f"{float(v):.0f}" if "months" in item else f"{float(v):.2f}".rstrip("0").rstrip(".")
            except ValueError:
                return v
        add4("3 inflation episode", item, cell)
    pt = read("q3_passthrough", index_col=0)
    for item in ["after 3 months", "after 6 months", "R2"]:
        add4("3 policy → mortgage pass-through", item, lambda c: f"{pt.loc[c, item]:.2f}")
    ps = {c: read(f"q3_price_setting_{c}", index_col=0) for c in ALL}
    for g in ["petrol", "sugar", "milk"]:
        add4("3 price setting", f"{g}: months unchanged, %",
             lambda c: f1(ps[c].loc[g].iloc[0]) if g in ps[c].index else "– (December only)")
    cc = read("q3_crosscountry", index_col=0)
    for pair, r in cc.iterrows():
        a, b = pair.split("-")
        k = int(r.iloc[1])
        lead = f"{ALL[a] if k > 0 else ALL[b]} leads by {abs(k)}" if k else "same month"
        rows.append({"topic": "3 cross-country", "indicator": f"inflation, {ALL[a]} and {ALL[b]}: who leads (correlation)",
                     ALL[a]: f"{lead} ({r.iloc[2]:.2f})"})

    fc = read("q4_forecasts")
    for pn in ("short 2021-2025", "long 2013-2025"):
        for h in (1, 3):
            def best(c, pn=pn, h=h):
                q = fc[(fc.panel == pn) & (fc.horizon == h) & (fc.country == c) & (fc.model != "random walk")]
                b = q.loc[q["RMSE relative to random walk"].idxmin()]
                return f"{b['model']}: {b['RMSE relative to random walk']:.2f}" + ("*" if b["DM p vs random walk"] < 0.05 else "")
            add4("4 forecasting inflation", f"best model, {pn} panel, {h} month(s) ahead", best)
    mi = read("q4_mutual_information")
    add4("4 forecasting inflation", "features with significant dependence (cyclic-shift p < 0.05)",
         lambda c: f"{int((mi[mi.country == c]['cyclic-shift p'] < 0.05).sum())} of {int((mi.country == c).sum())}")

    t = pd.DataFrame(rows).groupby(["topic", "indicator"], sort=False).first().fillna("")
    t.to_csv(R / "q5_comparison.csv")
    md = ["| topic | indicator | " + " | ".join(ALL.values()) + " |", "|---|---|" + "---|" * len(ALL)]
    md += [f"| {a} | {b} | " + " | ".join(str(r[n]) for n in ALL.values()) + " |" for (a, b), r in t.iterrows()]
    (R / "q5_comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
