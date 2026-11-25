"""Lab 10: tables (paper/generated/lab10_*.tex) and numbers (\\LabTen... macros) for Part 3 of the guide, assembled from
the results of the stages in results/lab10 and the data set D9.
Run after the stages (from code/):  python -m lab10_economies.report   (run.py calls it last)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dmfin.reporting import fmt, save_numbers, save_table

from .common import COUNTRIES
from .paths import DATA, RESULTS

C4 = list(COUNTRIES)                                   # BY, PL, RU, LT
NAMES = [COUNTRIES[c] for c in C4]
GOODS = ["wheat flour", "milk", "sugar", "petrol"]
f1 = lambda v: fmt(float(v), 1)


def read(name, **kw):
    return pd.read_csv(RESULTS / f"{name}.csv", **kw)


def data_set(num):
    long = pd.read_csv(DATA / "monthly_long.csv")
    src = pd.read_csv(DATA / "sources.csv")
    num.update({"values": len(long), "series": len(src), "sources": src["source"].nunique(),
                "months": long["month"].nunique()})
    for c in C4:
        num[f"values_{c}"] = int((long.country == c).sum())
        num[f"sources_{c}"] = int(src[src.country == c]["source"].nunique())
    show = {"price_flour_wheat": "wheat flour", "price_milk": "milk", "price_sugar": "sugar", "price_petrol": "petrol",
            "cpi_yoy": "inflation, y/y", "policy_rate": "policy rate", "mortgage_rate": "mortgage rate",
            "gov_bond_10y": "government bond yield", "gov_bond_tr_index": "government bond return"}
    alias = {("BY", "gov_bond_10y"): "gov_bond_long_yield_usd_placement", ("LT", "gov_bond_tr_index"): "gov_bond_10y_ea"}
    rows = {}
    for ind, label in show.items():
        r = {}
        for c in C4:
            s = src[(src.country == c) & (src.indicator == alias.get((c, ind), ind))]
            r[COUNTRIES[c]] = "; ".join(f"{a.replace(' x ', ' $\\times$ ')} ({m})" for a, m in zip(s["source"], s["months"])) if len(s) else "--"
        rows[label] = r
    t = pd.DataFrame(rows).T.rename_axis("Indicator")
    save_table("lab10_sources", t, col_format="p{2.3cm}p{3.1cm}p{3.1cm}p{3.1cm}p{3.1cm}")


def growth(num):
    rows = []
    for c in C4:
        t = read(f"q1_same_units_{c}", index_col=0)
        for i, (y, r) in enumerate(t.iterrows()):
            rows.append([COUNTRIES[c] if i == 0 else "", "per year" if "per year" in y else y,
                         *[float(v) for v in r.values]])
    t = pd.DataFrame(rows, columns=["Country", "Year", "Inflation", "Flour", "Milk", "Sugar", "Petrol", "Policy rate",
                                    "Bond yield", "Bond return"])
    save_table("lab10_same_units", t, nd=1, index=False, col_format="llcccccccc")
    for c in C4:                                       # government bonds: yield promised, return of the holder
        u = read(f"q1_same_units_{c}", index_col=0)
        su = u["government bond return"]
        num[f"bond_py_{c}"] = f1(su.iloc[-1])                   # one decimal, as in the tables
        num[f"bond_yield_py_{c}"] = f1(u["government bond yield"].iloc[-1])
        for y, v in su.iloc[:-1].items():
            num[f"bond_{c}_{y}"] = float(v)
        num[f"bond_above_infl_{c}"] = int((su.iloc[:-1] > u["official inflation"].iloc[:-1]).sum())
        num[f"bond_yield_above_ret_{c}"] = int((u["government bond yield"].iloc[:-1] > su.iloc[:-1]).sum())
    bc = read("q1_bond_check_RU", index_col=0)         # the computed return against the exchange's index (Russia)
    span = "2021-2025, per year"
    num["ofz_index_py"] = f1(bc.loc["exchange index RGBITR", span])
    num["ofz_five_py"] = f1(bc.loc["computed, 5-year bond", span])
    num["ofz_ten_py"] = f1(bc.loc["computed, 10-year bond", span])
    num["ofz_five_corr"] = fmt(float(bc.loc["computed, 5-year bond", "monthly correlation with the index"]), 2)
    num["ofz_five_maxdiff"] = f1((bc.loc["computed, 5-year bond", [str(y) for y in range(2021, 2026)]] -
                                  bc.loc["exchange index RGBITR", [str(y) for y in range(2021, 2026)]]).abs().max())
    for y in (2021, 2022, 2024, 2025):
        num[f"ofz_index_{y}"] = f1(bc.loc["exchange index RGBITR", str(y)])
    t = bc.rename(index={"exchange index RGBITR": "Moscow Exchange index RGBITR", "computed, 5-year bond":
                         "computed from the 5-year yield", "computed, 10-year bond": "computed from the 10-year yield"},
                  columns={span: "per year", "monthly correlation with the index": "corr."})
    save_table("lab10_bond_check", t.rename_axis(""), nd={**{k: 1 for k in t.columns}, "corr.": 2}, col_format="lccccccc")
    from .common import panel as _panel
    ru, lt = _panel("RU", base=True), _panel("LT", base=True)
    num["ofz_yield_start"] = f1(ru["bond_gov"].iloc[0])           # December 2020
    num["ofz_yield_max"] = f1(ru["bond_gov"].max())
    num["ofz_yield_max_month"] = ru["bond_gov"].idxmax().strftime("%B %Y")
    num["ofz_yield_end"] = f1(ru["bond_gov"].iloc[-1])
    num["ea_yield_start"] = f1(lt["bond_market"].iloc[0])
    num["ea_yield_end"] = f1(lt["bond_market"].iloc[-1])
    from .common import panel as _panel                # savers' instruments beside the 10-year bond
    pl, by = _panel("PL"), _panel("BY")
    num["edo_first"] = float(pl["bond_retail"].mean())            # Polish inflation-indexed savings bond EDO, first-year rate
    num["edo_realized"] = float(pl["bond_retail_realized"].mean())  # realized per year, held until 2025-12 or maturity
    num["edo_jan2021"] = f1(pl.loc[pl.index == "2021-01", "bond_retail_realized"].iloc[0])   # held through the whole period
    num["by_byn_bonds"] = float(by["bond_gov_local"].mean())       # Belarusian government bonds in roubles, at placement
    num["by_byn_bonds_months"] = int(by["bond_gov_local"].notna().sum())
    num["by_usd_bonds_months"] = int(by["bond_gov"].notna().sum())
    cum = {c: read(f"q1_cumulative_{c}", index_col=0) for c in C4}
    m = {c: read(f"q1_match_{c}", index_col=0) for c in C4}
    rows = {"inflation, Dec 2020 $\\to$ Dec 2025": {COUNTRIES[c]: f"+{f1(cum[c].loc['consumer prices'].iloc[0])}\\,\\%" for c in C4}}
    for g in GOODS:
        rows[g] = {COUNTRIES[c]: f"{m[c].loc[g, 'verdict']} ({fmt(float(m[c].loc[g, 'difference, p.p.']), 1)})" for c in C4}
    rows["policy rate above inflation, months of 60"] = {COUNTRIES[c]: m[c].loc["policy rate", "verdict"].split(" in ")[1].split(" of")[0] for c in C4}
    save_table("lab10_match", pd.DataFrame(rows).T.rename_axis(""), col_format="p{3.4cm}" + ">{\\centering\\arraybackslash}p{2.7cm}" * 4)
    for c in C4:
        num[f"cpi_{c}"] = float(cum[c].loc["consumer prices"].iloc[0])
        num[f"cpi_py_{c}"] = f1(cum[c].loc["consumer prices", "per year, %"])   # one decimal, as in the tables
        for g in GOODS:
            k = g.split()[-1]
            num[f"{k}_{c}"] = float(cum[c].loc[g].iloc[0])
            num[f"{k}_gap_{c}"] = float(m[c].loc[g, "difference, p.p."])
            num[f"{k}_years_{c}"] = int(m[c].loc[g, "years within 3 p.p. of inflation (of 5)"])
        num[f"above_{c}"] = int(m[c].loc["policy rate", "verdict"].split(" in ")[1].split(" of")[0])
        num[f"usd_{c}"] = float(cum[c].loc["US dollar rate"].iloc[0])
    num["below_min"] = 60 - max(num[f"above_{c}"] for c in C4)     # months with the policy rate below inflation
    num["below_max"] = 60 - min(num[f"above_{c}"] for c in C4)
    num["level_pairs"] = sum(m[c].loc[g, "verdict"] == "about the same" for c in C4 for g in GOODS)
    rt = {c: read(f"q1_rates_{c}", index_col=0) for c in C4}
    keys = [("inflation (CPI, y/y)", "inflation"), ("policy rate", "policy rate"), ("mortgage rate", "mortgage rate"),
            ("government bond yield", "government bond yield"), ("real policy rate", "real policy rate"),
            ("real mortgage rate", "real mortgage rate"), ("real government bond yield", "real government bond yield")]
    t = pd.DataFrame({COUNTRIES[c]: {lab: float(rt[c].loc[k, "2021-2025"]) for k, lab in keys} for c in C4}).rename_axis("Mean 2021--2025")
    save_table("lab10_rates", t, nd=1, col_format="lcccc")
    for c in C4:
        for k, lab in keys:
            num[f"mean_{lab.replace(' ', '_')}_{c}"] = float(rt[c].loc[k, "2021-2025"])
    sv = {c: read(f"q1_savings_{c}", index_col=0) for c in C4}
    bond = lambda c: "government bond in USD, in BYN" if c == "BY" else "government bond, 10 years, holder"
    t = pd.DataFrame({COUNTRIES[c]: {"cash": float(sv[c].loc["cash (no interest)", "real value of 100"]),
                                     "at the policy rate": float(sv[c].loc["policy rate", "real value of 100"]),
                                     "in government bonds": float(sv[c].loc[bond(c), "real value of 100"])} for c in C4})
    save_table("lab10_savings", t.rename_axis("Real value of 100"), nd=1, col_format="lcccc")
    for c in C4:
        num[f"save_cash_{c}"] = float(sv[c].loc["cash (no interest)", "real value of 100"])
        num[f"save_policy_{c}"] = float(sv[c].loc["policy rate", "real value of 100"])
        num[f"save_bond_{c}"] = float(sv[c].loc[bond(c), "real value of 100"])


def relations(num):
    g = read("q2_groups")
    t = pd.DataFrame({"Groups (correlation at least 0.6)": {COUNTRIES[c]: "; ".join("\\{" + x + "\\}" for x in g[g.country == c]["features"]
                                                                                    if "," in x) for c in C4}}).rename_axis("Country")
    save_table("lab10_groups", t, col_format="lp{13cm}")
    pr = read("q2_predictability")
    rows = {}
    for c in C4:
        q = pr[pr.country == c]
        ok = q[q["skill with own last value"] > 0]
        rows[COUNTRIES[c]] = {"series": len(q), "$R^2$ in sample, min": float(q["R2 from the others (in sample)"].min()),
                              "$R^2$ in sample, max": float(q["R2 from the others (in sample)"].max()),
                              "better than last value": int(len(ok)),
                              "which (skill)": ", ".join(f"{a} ({fmt(float(b), 2)})" for a, b in zip(ok["target"], ok["skill with own last value"])) or "--"}
        num[f"nowcast_better_{c}"], num[f"nowcast_series_{c}"] = int(len(ok)), int(len(q))
    save_table("lab10_predict", pd.DataFrame(rows).T.rename_axis("Country"), nd=2, col_format="lcccc p{5.4cm}")
    gc = read("q2_granger")
    gc["test"] = gc["cause"] + " $\\to$ " + gc["effect"]
    t = gc.pivot(index="test", columns="country", values="BH-adjusted p (all tests)").reindex(columns=C4)
    order = list(dict.fromkeys(gc["test"]))
    t = t.reindex(order).rename(columns=COUNTRIES)
    t.columns = [f"p ({x})" for x in t.columns]
    save_table("lab10_granger", t.rename_axis("Test (monthly changes)"), nd=3, col_format="p{6.2cm}cccc")
    num["granger_tests"] = int(gc["BH-adjusted p (all tests)"].notna().sum())
    num["granger_sig"] = int((gc["BH-adjusted p (all tests)"] < 0.05).sum())
    num["granger_sig_lag"] = int((gc["BH-adjusted p, lag choice corrected"] < 0.05).sum())
    ll = read("q2_leadlag", index_col=0)
    for c in C4:
        num[f"lag_{c}"] = int(-ll.loc[c].iloc[0])
        num[f"lagcorr_{c}"] = float(ll.loc[c].iloc[1])


def patterns(num):
    ep = read("q3_episode", index_col=0)
    keep = ["inflation peak, %", "peak month", "months above 10 %", "first month below 5 % after the peak", "first rate hike",
            "policy-rate peak, %", "rate peak reached", "first cut", "real policy rate positive from",
            "lowest real policy rate, p.p."]
    t = ep.loc[keep, C4].rename(columns=COUNTRIES)
    t = t.map(lambda v: (fmt(float(v), 1) if "." in str(v) and str(v).replace(".", "").replace("-", "").isdigit() else str(v)))
    save_table("lab10_episode", t.rename_axis(""), col_format="p{5.2cm}cccc")
    for c in C4:
        num[f"peak_{c}"] = float(ep.loc["inflation peak, %", c]); num[f"peak_month_{c}"] = ep.loc["peak month", c]
        num[f"above10_{c}"] = int(ep.loc["months above 10 %", c]); num[f"first_hike_{c}"] = ep.loc["first rate hike", c]
        num[f"rate_peak_{c}"] = float(ep.loc["policy-rate peak, %", c]); num[f"first_cut_{c}"] = ep.loc["first cut", c]
        num[f"rate_peak_month_{c}"] = pd.Period(ep.loc["rate peak reached", c], "M").strftime("%B %Y")
        num[f"real_pos_{c}"] = ep.loc["real policy rate positive from", c]
        num[f"lowest_real_{c}"] = float(ep.loc["lowest real policy rate, p.p.", c])
    pt = read("q3_passthrough", index_col=0).loc[C4].rename(index=COUNTRIES)
    pt.columns = ["same month", "after 3 months", "after 6 months", "$R^2$", "spread, mean p.p.", "spread s.d."]
    save_table("lab10_passthrough", pt.rename_axis("Country"), nd=2)
    for c, r in read("q3_passthrough", index_col=0).loc[C4].iterrows():
        num[f"pt3_{c}"], num[f"pt6_{c}"], num[f"ptr2_{c}"] = float(r["after 3 months"]), float(r["after 6 months"]), float(r["R2"])
        num[f"spread_{c}"] = float(r["mortgage - policy spread, mean p.p."])
        num[f"spread_sd_{c}"] = f1(r["spread s.d."])
    ps = {c: read(f"q3_price_setting_{c}", index_col=0) for c in C4}
    rows = {}
    for g in GOODS + ["consumer prices"]:
        rows[g] = {COUNTRIES[c]: (f"{fmt(float(ps[c].loc[g].iloc[0]), 1)} ({int(ps[c].loc[g].iloc[1])})" if g in ps[c].index else "--")
                   for c in C4}
    save_table("lab10_price_setting", pd.DataFrame(rows).T.rename_axis("Unchanged months, \\% (jumps $>$ 5\\,\\%)"), col_format="lcccc")
    for c in C4:
        for g in ps[c].index:
            k = g.split()[-1] if g != "consumer prices" else "cpi"
            num[f"unchanged_{k}_{c}"] = float(ps[c].loc[g].iloc[0])
            num[f"jumps_{k}_{c}"] = int(ps[c].loc[g].iloc[1])
            num[f"top3_{k}_{c}"] = float(ps[c].loc[g].iloc[4])
    un = read("q3_regimes_uncapped", index_col=0)
    num["seg_uncapped_min"], num["seg_uncapped_max"] = int(un["segments, BIC up to 15"].min()), int(un["segments, BIC up to 15"].max())
    num["infl_acf_min"] = float(un["lag-1 autocorrelation of inflation"].min())
    rg = read("q3_regimes")
    for c in C4:                                       # the regime of the highest inflation and the last regime
        g = rg[rg.country == c]
        hi, last = g.loc[g["mean inflation, %"].idxmax()], g.iloc[-1]
        num.update({f"reg_hi_from_{c}": hi["from"], f"reg_hi_to_{c}": hi["to"],
                    f"reg_hi_infl_{c}": f1(hi["mean inflation, %"]), f"reg_hi_real_{c}": f1(hi["mean real policy rate, p.p."]),
                    f"reg_last_real_{c}": f1(last["mean real policy rate, p.p."])})
    rg["country"] = rg["country"].map(COUNTRIES)
    rg.columns = ["Country", "From", "To", "Inflation, %", "Policy rate, %", "Real policy rate, p.p."]
    rg["Country"] = [c if i == 0 or c != rg["Country"].iloc[i - 1] else "" for i, c in enumerate(rg["Country"])]
    save_table("lab10_regimes", rg, nd=1, index=False, col_format="lccccc")
    cc = read("q3_crosscountry", index_col=0)
    rows = {}
    for pair, r in cc.iterrows():
        a, b = pair.split("-")
        k = int(r.iloc[1])
        rows[f"{COUNTRIES[a]} -- {COUNTRIES[b]}"] = {"same month": float(r.iloc[0]),
                                                    "who leads": (f"{COUNTRIES[a] if k > 0 else COUNTRIES[b]}, {abs(k)} months" if k else "same month"),
                                                    "correlation at that lag": float(r.iloc[2])}
        num[f"cc_{a}{b}_lag"], num[f"cc_{a}{b}_corr"] = abs(k), float(r.iloc[2])
    save_table("lab10_cross", pd.DataFrame(rows).T.rename_axis("Pair"), nd=2, col_format="lccc")


def forecasting(num):
    fc = read("q4_forecasts")
    star = lambda r: fmt(float(r["RMSE relative to random walk"]), 2) + ("*" if r["DM p vs random walk"] < 0.05 else "")
    order = ["AR (own lags)", "ridge", "lasso", "random forest", "gradient boosting", "k nearest neighbours",
             "pooled ridge", "pooled random forest"]
    for pn, tag in (("short 2021-2025", "short"), ("long 2013-2025", "long")):
        q = fc[(fc.panel == pn) & (fc.model != "random walk")]
        t = pd.DataFrame({f"{COUNTRIES[c]}, {h}": {m: star(q[(q.country == c) & (q.horizon == h) & (q.model == m)].iloc[0])
                                                    for m in order} for c in C4 for h in (1, 3)})
        t.columns = [f"{c.split(',')[0][:2]}{c.split(',')[1].strip()}" if False else c for c in t.columns]
        save_table(f"lab10_fc_{tag}", t.rename_axis("Model"), col_format="l" + "c" * 8,
                   header=["Model"] + [f"{n[:3]}.\\ {h}" for n in NAMES for h in (1, 3)])
        for c in C4:
            for h in (1, 3):
                g = q[(q.country == c) & (q.horizon == h)]
                b = g.loc[g["RMSE relative to random walk"].idxmin()]
                num[f"best_{tag}_{h}_{c}"] = b["model"]
                num[f"best_{tag}_{h}_{c}_rel"] = float(b["RMSE relative to random walk"])
                num[f"best_{tag}_{h}_{c}_sig"] = "yes" if b["DM p vs random walk"] < 0.05 else "no"
                for m in order:                        # as in the tables: * when different from "no change"
                    num[f"rel_{tag}_{h}_{c}_{m.replace(' ', '')}"] = star(g[g.model == m].iloc[0])
        num[f"forecasts_{tag}"] = int(q["forecasts"].max())
    mi = read("q4_mutual_information")
    for c in C4:
        g = mi[mi.country == c]
        num[f"mi_sig_{c}"], num[f"mi_feat_{c}"] = int((g["cyclic-shift p"] < 0.05).sum()), len(g)


def fuzzy(num):
    th = read("q6_threshold", index_col=0)
    sel = [1, 2, 3, 5, 8, 10, 15]
    t = pd.DataFrame({"Threshold, p.p.": sel, "Pairs ``about the same'' (of 16)": [int(th.loc[s].iloc[0]) for s in sel]})
    save_table("lab10_threshold", t.set_index("Threshold, p.p.").T.rename_axis(""), col_format="l" + "c" * len(sel))
    for s in sel:
        num[f"thr_{s}"] = int(th.loc[s].iloc[0])
    mf = read("q6_match_fuzzy")
    mf["country"] = mf["country"].map(COUNTRIES)
    t = mf[["country", "good", "about equal, mean over the years", "gap per year 2021-2025, p.p.", "5 years: slower",
            "5 years: about equal", "5 years: faster", "crisp verdict at 3 / 5 / 8 p.p."]].copy()
    t["country"] = [c if i == 0 or c != t["country"].iloc[i - 1] else "" for i, c in enumerate(t["country"])]
    t.columns = ["Country", "Good", "about equal, mean of years", "gap per year, p.p.", "slower", "about equal", "faster",
                 "crisp verdict at 3 / 5 / 8 p.p."]
    save_table("lab10_fuzzy_match", t, nd=2, index=False, col_format="llcccccp{3.4cm}")
    raw = read("q6_match_fuzzy")
    for _, r in raw.iterrows():
        k = r["good"].split()[-1]
        num[f"fz_mean_{k}_{r['country']}"] = float(r["about equal, mean over the years"])
    sb = read("q6_summaries_best")
    sm = read("q6_summaries")
    sb["sentence"] = [(f"In {r['quantifier Q']} of the " +
                       ("good-years" if "good" in r["qualifier W"] else "months") +
                       ("" if r["qualifier W"].startswith("(") else f" with {r['qualifier W'].replace('5-10', '5--10')}") + ", "
                       + r["summarizer S"])
                      for _, r in sb.iterrows()]
    pick = sb.sort_values(["country", "score T1 x T3"], ascending=[True, False]).groupby("country").head(3)
    pick = pick.set_index("country").loc[C4].reset_index()
    t = pd.DataFrame({"Country": [COUNTRIES[c] for c in pick["country"]], "Period": pick["period"].str.replace("-", "--"),
                      "Summary": pick["sentence"], "$T_1$": pick["T1 truth"], "$T_2$": pick["T2 imprecision (high = precise S)"],
                      "$T_3$": pick["T3 covering"]})
    t["Country"] = [c if i == 0 or c != t["Country"].iloc[i - 1] else "" for i, c in enumerate(t["Country"])]
    save_table("lab10_summaries", t, nd=2, index=False, col_format="llp{8.6cm}ccc")
    full = sm[sm["T1 truth"] >= 0.99]
    num.update({"sum_all": len(sm), "sum_true": len(full), "sum_trivial": int((full["T3 covering"] < 0.15).sum()),
                "sum_frames": int(sm.groupby(["country", "period", "qualifier W", "summarizer S"]).ngroups)})
    rules = read("q6_tsk_rules")
    rules["rule"] = rules["rule"].str.replace("IF inflation ", "").str.replace(" THEN rate next month is about c", "")
    rules["cell"] = [fmt(float(c), 1) if s == "yes" else f"({fmt(float(c), 1)})" for c, s in zip(rules["c, %"], rules["supported"])]
    t = rules.pivot(index="rule", columns="bank (inflation used)", values="cell")
    t = t.reindex(list(dict.fromkeys(rules["rule"]))).reindex(columns=["BY", "PL", "RU", "LT", "LT (euro-area inflation)"])
    t.columns = ["NBRB", "NBP", "Bank of Russia", "ECB / Lithuanian inflation", "ECB / euro-area inflation"]
    save_table("lab10_tsk_rules", t.rename_axis("Inflation, its 3-month change"), col_format="lccccc")
    for _, r in read("q6_tsk_rules").iterrows():
        a = r["rule"].replace("IF inflation ", "").replace(" THEN rate next month is about c", "").replace(", ", "")
        num[f"tsk_{r['bank (inflation used)'].split()[0]}{'ea' if 'euro' in r['bank (inflation used)'] else ''}_{a}"] = float(r["c, %"])
    fit = read("q6_tsk_fit", index_col=0)
    fit.index = ["NBRB", "NBP", "Bank of Russia", "ECB (Lithuanian inflation)", "ECB (euro-area inflation)"]
    fit = fit.drop(columns="months")
    fit.columns = ["$R^2$ linear", "$R^2$ TS-0", "$R^2$ TS-1", "RMSE no change", "RMSE linear", "RMSE TS-0", "RMSE TS-1"]
    save_table("lab10_tsk_fit", fit.rename_axis("Bank (inflation)"), nd=2, col_format="l" + "c" * 7)
    raw = read("q6_tsk_fit", index_col=0)
    for k, r in raw.iterrows():
        kk = k.split()[0] + ("ea" if "euro" in k else "")
        num[f"tskr2lin_{kk}"], num[f"tskr2_{kk}"], num[f"tskr2one_{kk}"] = (float(r.iloc[1]), float(r.iloc[2]), float(r.iloc[3]))
        num[f"tskrmse_nochange_{kk}"], num[f"tskrmse_{kk}"] = float(r.iloc[4]), float(r.iloc[6])
    va = read("q6_fcm_validity", index_col=0)
    save_table("lab10_fcm_validity", va.rename_axis("Clusters $c$").rename(columns={"partition coefficient": "$PC$",
               "partition entropy": "$PE$", "Xie-Beni": "$XB$"}), nd=3, col_format="lccc")
    ce = read("q6_fcm_centres", index_col=0)
    save_table("lab10_fcm_centres", ce.rename_axis("Regime").rename(columns={"inflation": "inflation, %",
               "change over 3 months": "3-month change, p.p.", "real policy rate": "real policy rate, p.p."}), nd=1)
    tr = read("q6_fcm_transitions", index_col=0)
    num.update({"fcm_pc2": float(va.loc[2, "partition coefficient"]), "fcm_pc3": float(va.loc[3, "partition coefficient"]),
                "fcm_xb2": float(va.loc[2, "Xie-Beni"]), "fcm_xb3": float(va.loc[3, "Xie-Beni"])})
    for k in range(3):
        for col, tag in (("inflation", "infl"), ("change over 3 months", "chg"), ("real policy rate", "real")):
            num[f"fcm_c{k + 1}_{tag}"] = float(ce.iloc[k][col])
    for c in C4:
        num[f"fcm_near_{c}"] = float(tr.loc[c].iloc[1]); num[f"fcm_else_{c}"] = float(tr.loc[c].iloc[2])


COMPARE = [("1 growth 2021-2025", "consumer prices, %", "consumer prices, growth 2021--2025, \\%"),
           ("1 growth 2021-2025", "wheat flour, real change (vs consumer prices), %", "wheat flour, real change, \\%"),
           ("1 growth 2021-2025", "milk, real change (vs consumer prices), %", "milk, real change, \\%"),
           ("1 growth 2021-2025", "sugar, real change (vs consumer prices), %", "sugar, real change, \\%"),
           ("1 growth 2021-2025", "petrol, real change (vs consumer prices), %", "petrol, real change, \\%"),
           ("1 rates 2021-2025, % or p.p.", "real policy rate, mean", "real policy rate, mean, p.p."),
           ("1 rates 2021-2025, % or p.p.", "months with a negative real policy rate (of 60)", "months with a negative real policy rate (of 60)"),
           ("1 real value of 100 after 5 years", "at the policy rate", "real value of 100 at the policy rate"),
           ("2 related series", "policy rate follows inflation by, months (negative: leads)", "policy rate follows inflation by, months (negative: leads)"),
           ("2 related series", "series better nowcast from the others than from last value", "series nowcast better from the others than from their last value"),
           ("3 inflation episode", "inflation peak, %", "inflation peak, \\%"),
           ("3 inflation episode", "peak month", "peak month"),
           ("3 inflation episode", "real policy rate positive from", "real policy rate positive from"),
           ("3 policy → mortgage pass-through", "after 3 months", "pass-through to mortgages after 3 months"),
           ("3 price setting", "milk: months unchanged, %", "milk: months unchanged, \\%"),
           ("4 forecasting inflation", "best model, short 2021-2025 panel, 1 month(s) ahead", "best forecast 1 month ahead, 2021--2025 panel"),
           ("4 forecasting inflation", "best model, long 2013-2025 panel, 1 month(s) ahead", "best forecast 1 month ahead, 2013--2025 panel"),
           ("4 forecasting inflation", "features with significant dependence (cyclic-shift p < 0.05)", "features with significant dependence")]


def _cell(v: str) -> str:
    """A cell of the comparison table: numbers with one decimal and a true minus sign, text as it is."""
    try:
        x = float(v)
    except ValueError:
        return v
    if "." not in v:
        return fmt(int(x)).replace("-", "\u2212")
    return fmt(x, 1 if abs(x) >= 10 else min(len(v.split(".")[1]), 2))      # the decimals of question 5, at most 2


def compare(num):
    q5 = pd.read_csv(RESULTS / "q5_comparison.csv").set_index(["topic", "indicator"])
    rows = {}
    for topic, ind, label in COMPARE:
        r = q5.loc[(topic, ind)]
        rows[label] = {COUNTRIES[c]: _cell(str(r[COUNTRIES[c]])) for c in C4}
        if label.startswith("real policy rate"):        # the bond return beside the policy rate (stage 1, per year)
            su = {c: read(f"q1_same_units_{c}", index_col=0).iloc[-1] for c in C4}
            rows["government bonds: yield / holder's return / inflation, \\% per year"] = {
                COUNTRIES[c]: f"{fmt(float(su[c]['government bond yield']), 1)} / "
                              f"{fmt(float(su[c]['government bond return']), 1)} / "
                              f"{fmt(float(su[c]['official inflation']), 1)}" for c in C4}
    save_table("lab10_compare", pd.DataFrame(rows).T.rename_axis(""),
               col_format="p{4.0cm}" + ">{\\centering\\arraybackslash}p{2.6cm}" * 4)


def further(num):
    """Stage 6: the further methods of chapter 4."""
    import json
    ex = json.loads((RESULTS / "q7_extra.json").read_text())
    month = lambda s: f"{COUNTRIES[s.split()[0]]}, {s.split()[1]}"
    sil = read("q7_kmedoids", index_col=0)
    save_table("lab10_kmedoids", sil.T.rename_axis("Clusters"), nd=2, col_format="l" + "c" * len(sil))
    num.update({"fur_regime_months": int(ex["regime_months"]), "fur_pamk_k": int(ex["pamk_k"]), "fur_sil_pam_2": float(sil.loc[2, "PAM"]), "fur_sil_km_2": float(sil.loc[2, "k-means"]),
                "fur_sil_pam_3": float(sil.loc[3, "PAM"]), "fur_sil_km_3": float(sil.loc[3, "k-means"])})
    md = read("q7_medoids", index_col=0)
    t = pd.DataFrame({"PAM medoid": [month(s) for s in md["PAM medoid"]],
                      "$\\pi$": md["medoid: inflation"], "$\\Delta_3\\pi$": md["medoid: change over 3 months"],
                      "real rate": md["medoid: real policy rate"],
                      "centre $\\pi$": md["centre: inflation"], "centre $\\Delta_3\\pi$": md["centre: change over 3 months"],
                      "centre real rate": md["centre: real policy rate"],
                      "nearest month (distance)": [f"{month(s)} ({fmt(float(d), 2)})" for s, d in
                                                   zip(md["k-means centre: nearest country-month"], md["k-means centre: its distance (s.d.)"])]},
                     index=md.index.rename("Regime"))
    save_table("lab10_medoids", t, nd=1, col_format="lp{2.2cm}ccccccp{2.7cm}")
    for r, row in md.iterrows():
        num[f"fur_medoid_{r}"] = month(row["PAM medoid"])
        num[f"fur_centre_dist_{r}"] = float(row["k-means centre: its distance (s.d.)"])
    num["fur_centre_dist_max"] = float(md["k-means centre: its distance (s.d.)"].max())
    rb = read("q7_robust", index_col=0)
    save_table("lab10_robust", rb.rename_axis("Inflation multiplied by"), nd=2, col_format="lcccc")
    num["fur_bad_month"] = month(ex["bad_month"])
    for f in rb.index:
        for k in (2, 3):
            num[f"fur_rob_km_{k}_{f}"] = float(rb.loc[f, f"k-means, k = {k}"])
            num[f"fur_rob_pam_{k}_{f}"] = float(rb.loc[f, f"PAM, k = {k}"])
    cl = read("q7_clara", index_col=0)
    cl.columns = ["objects in PAM", "cost / PAM", "worst of 10", "medoids of PAM", "seconds"]
    save_table("lab10_clara", cl.rename_axis("Method"), nd={"objects in PAM": 0, "cost / PAM": 3, "worst of 10": 3,
                                                           "medoids of PAM": 1, "seconds": 3}, col_format="lccccc")
    num.update({"fur_clara_n": int(ex["clara_n"]), "fur_clara_46": float(cl.iloc[1]["cost / PAM"]),
                "fur_clara_46_worst": float(cl.iloc[1]["worst of 10"]), "fur_clara_12": float(cl.iloc[2]["cost / PAM"]),
                "fur_clara_12_one": float(cl.iloc[3]["cost / PAM"]), "fur_clara_12_one_worst": float(cl.iloc[3]["worst of 10"]),
                "fur_pam_seconds": float(cl.iloc[0]["seconds"])})
    sp = read("q7_spectral", index_col=0)
    t = pd.DataFrame({"series": sp["series"], "groups, linkage": sp["groups, average linkage"],
                      "groups, eigengap": sp["groups by eigengap"], "adj. Rand": sp["adjusted Rand, same number of groups"],
                      "Spectral groups (eigengap)": sp["spectral groups (eigengap)"]})
    t.index = [COUNTRIES[c] for c in sp.index]
    save_table("lab10_spectral", t.rename_axis("Country"), nd=2, col_format="lcccc p{6.4cm}")
    for c in sp.index:
        num[f"fur_sp_ari_{c}"] = float(sp.loc[c, "adjusted Rand, same number of groups"])
        num[f"fur_sp_k_{c}"] = int(sp.loc[c, "groups by eigengap"])
    st = read("q7_hmm_states", index_col=0)
    sh = read("q7_hmm_share", index_col=0)
    t = pd.DataFrame({"mean, %": st["mean inflation, %"], "s.d., p.p.": st["standard deviation, p.p."],
                      "stay prob.": st["stay probability"], "stay, months": st["expected stay, months"],
                      **{COUNTRIES[c]: sh.loc[c] for c in sh.index}})
    save_table("lab10_hmm", t.rename_axis("State"), nd={**{k: 1 for k in t.columns}, "stay prob.": 3}, col_format="lcccccccc")
    for s in st.index:
        num[f"fur_hmm_mean_{s}"] = float(st.loc[s, "mean inflation, %"])
        num[f"fur_hmm_stay_{s}"] = float(st.loc[s, "expected stay, months"])
    for c in sh.index:
        for s in sh.columns:
            num[f"fur_hmm_share_{c}_{s}"] = float(sh.loc[c, s])
    bic = read("q7_hmm_bic", index_col=0)
    for n_ in bic.index:
        num[f"fur_hmm_bic_{n_}"] = float(bic.loc[n_, "BIC"])
    cmp_ = read("q7_hmm_compare", index_col=0)
    for c in cmp_.index:
        num[f"fur_hmm_switch_{c}"] = int(cmp_.loc[c, "HMM state changes 2021-2025"])
        num[f"fur_hmm_high_from_{c}"] = str(cmp_.loc[c, "first month in the high state"])
        num[f"fur_hmm_high_to_{c}"] = str(cmp_.loc[c, "last month in the high state"])
    fim = read("q7_fim", index_col=0)
    t = fim.copy()
    t = t.map(lambda v: "yes" if str(v) == "True" else v)
    for col in t.columns:                                  # seconds with three decimals, counts as integers
        t[col] = [fmt(float(v), 3) if "seconds" in i else (v if v == "yes" else fmt(int(float(v)), 0)) for i, v in zip(t.index, t[col])]
    save_table("lab10_fim", t.rename_axis(""), col_format="lcc")
    for col, key in (("Lab 10 events", "ten"), ("Lab 5 credit data", "five")):
        num.update({f"fur_fim_{key}_tx": int(fim.loc["transactions", col]), f"fur_fim_{key}_items": int(fim.loc["items", col]),
                    f"fur_fim_{key}_freq": int(fim.loc["frequent itemsets", col]),
                    f"fur_fim_{key}_cand": int(fim.loc["Apriori: candidates counted", col]),
                    f"fur_fim_{key}_nodes": int(fim.loc["FP-growth: nodes of the FP-tree", col]),
                    f"fur_fim_{key}_cond": int(fim.loc["FP-growth: conditional trees", col]),
                    f"fur_fim_{key}_ta": float(fim.loc["Apriori, seconds", col]), f"fur_fim_{key}_tf": float(fim.loc["FP-growth, seconds", col])})
    num["fur_fim_speedup"] = float(fim.loc["Apriori, seconds", "Lab 5 credit data"]) / float(fim.loc["FP-growth, seconds", "Lab 5 credit data"])
    ev = read("q7_events", index_col=0)
    num["fur_events"] = len(ev)
    rl = read("q7_rules")
    t = pd.DataFrame({"Antecedent": rl["antecedent"].str.replace(" AND ", " and "), "Consequent": rl["consequent"],
                      "support": rl["support"], "confidence": rl["confidence"], "lift": rl["lift"]})
    save_table("lab10_rules", t, nd=2, index=False, col_format="p{6.4cm}p{3.0cm}ccc")
    num.update({"fur_rule_conf": float(rl["confidence"].iloc[0]), "fur_rule_lift": float(rl["lift"].iloc[0]),
                "fur_rule_sup": float(rl["support"].iloc[0])})
    tr = read("q7_trees", index_col=0)
    t = tr.copy()
    t.index = [COUNTRIES.get(i, i) for i in t.index]
    t.columns = ["months", "``hold'' (majority)", "conditional inference tree", "CART"]
    save_table("lab10_trees", t.rename_axis("Evaluation (country left out)"), nd={"months": 0, **{k: 2 for k in t.columns[1:]}},
               col_format="lcccc")
    num.update({"fur_tr_cv_hold": float(tr.iloc[-1, 1]), "fur_tr_cv_ct": float(tr.iloc[-1, 2]), "fur_tr_cv_cart": float(tr.iloc[-1, 3]),
                "fur_tr_loco_hold": float(tr.loc["all four"].iloc[1]), "fur_tr_loco_ct": float(tr.loc["all four"].iloc[2]),
                "fur_tr_loco_cart": float(tr.loc["all four"].iloc[3]), "fur_tr_months": int(tr.loc["all four"].iloc[0]),
                "fur_dec_hold": int(ex["decisions"]["hold"]), "fur_dec_raise": int(ex["decisions"]["raise"]),
                "fur_dec_cut": int(ex["decisions"]["cut"]), "fur_ct_root_p": float(ex["ct_root_p"]),
                "fur_cart_leaves": int(ex["cart_leaves"]), "fur_cart_depth": int(ex["cart_depth"])})
    cr = read("q7_ctree_rules")
    num["fur_ct_leaves"] = len(cr)
    t = pd.DataFrame({"Leaf": cr["leaf"].str.replace(" AND ", " and ").str.replace("<=", "$\\le$").str.replace(">", "$>$"),
                      "months": cr["months"], "P(cut)": cr["P(cut)"], "P(hold)": cr["P(hold)"], "P(raise)": cr["P(raise)"]})
    save_table("lab10_ctree", t, nd={"months": 0, "P(cut)": 2, "P(hold)": 2, "P(raise)": 2}, index=False, col_format="p{8.2cm}cccc")
    bs = read("q7_bias", index_col=0)
    save_table("lab10_bias", bs.rename_axis("Root variable, % of 300 permutations"), nd=1, col_format="lcc")
    num.update({"fur_bias_cart_bin": float(bs.loc["noise, binary", "CART"]), "fur_bias_ct_bin": float(bs.loc["noise, binary", "conditional inference tree"]),
                "fur_bias_cart_cont": float(bs.loc["noise, continuous", "CART"]),
                "fur_bias_ct_split": float(bs.loc["root split made, % of permutations", "conditional inference tree"]),
                "fur_bias_fair": 100 / 7})
    cv = read("q7_covering")
    t = pd.DataFrame({"Class": cv["class"], "Rule": cv["rule"].str.replace(" AND ", " and ").str.replace("<=", "$\\le$").str.replace(">", "$>$"),
                      "months covered": cv["months covered"], "of them in the class": cv["of them in the class"], "accuracy": cv["accuracy"]})
    t["Class"] = [c if i == 0 or c != t["Class"].iloc[i - 1] else "" for i, c in enumerate(t["Class"])]
    save_table("lab10_covering", t, nd={"months covered": 0, "of them in the class": 0, "accuracy": 2}, index=False,
               col_format="lp{8.6cm}ccc")
    for cls in ("raise", "cut"):
        a = cv[(cv["class"] == cls) & (cv["rule"] == "all rules together")].iloc[0]
        num[f"fur_cov_{cls}_acc"] = float(a["accuracy"]); num[f"fur_cov_{cls}_share"] = float(a["share of the class covered"])
        num[f"fur_cov_{cls}_rules"] = int(((cv["class"] == cls) & (cv["rule"] != "all rules together")).sum())
    ar = read("q7_bn_arcs")
    num.update({"fur_bn_arcs_bic": int((ar["BIC"] == "yes").sum()), "fur_bn_arcs_aic": int((ar["AIC"] == "yes").sum()),
                "fur_bn_months": int(ex["bn_months"])})
    q = read("q7_bn_queries")
    t = pd.DataFrame({"Probability": q["query"].str.replace("P(", "$P($", regex=False).str.replace(")", "$)$", regex=False)
                      .str.replace(" | ", " $\\mid$ ", regex=False),
                      "network (BIC)": q["network (BIC)"], "network (AIC)": q["network (AIC)"],
                      "share in the data": q["share in the data"], "months": q["months"]})
    save_table("lab10_bn", t, nd={"network (BIC)": 2, "network (AIC)": 2, "share in the data": 2, "months": 0}, index=False,
               col_format="p{8.0cm}cccc")
    for i, r in q.iterrows():
        num[f"fur_bn_q{i}_bic"] = float(r["network (BIC)"]); num[f"fur_bn_q{i}_aic"] = float(r["network (AIC)"])
        num[f"fur_bn_q{i}_data"] = float(r["share in the data"]); num[f"fur_bn_q{i}_n"] = int(r["months"])


def main():
    num = {}
    for f in (data_set, growth, relations, patterns, forecasting, fuzzy, further, compare):
        f(num)
    save_numbers("labTen", num)                         # macros \\LabTen... (lab10 would give \\LabOneZero...)
    print(f"{len(num)} numbers; tables lab10_*.tex written")


if __name__ == "__main__":
    main()
