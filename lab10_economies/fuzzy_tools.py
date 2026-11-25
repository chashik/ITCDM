"""Question 6: fuzzy tools on the four economies (the fuzzy additions F1-F4 of the plan).

F1  fuzzy match of price growth and official inflation. The gap (price growth minus inflation, p.p. per year) is cut by a
    Ruspini partition (memberships add up to 1) into "slower", "about equal" and "faster"; beside it, how the crisp
    verdict of question 1 ("about the same" = within tau p.p. over the five years) changes with the threshold tau.
F2  linguistic summaries (Yager 1982): "Q of the months, S" and "Q of the months with W, S". Degree of truth by Zadeh's
    calculus of quantified propositions, degree of imprecision and degree of covering (Kacprzyk and Yager 2001);
    sentences ranked by truth x covering, so that sentences true only because S rarely holds ("few months ...") drop.
F3  the reaction function of each central bank as a zero-order Takagi-Sugeno system: rules "IF inflation is A AND its
    3-month change is B THEN the rate next month is about c" (3 x 2 rules, product t-norm, normalized firing strengths,
    constants c by ridge least squares; first-order consequents a + b * inflation overfit 57 months and are kept only
    as a comparison); against a linear rule and "no change", in sample and with an expanding window (forecasts from
    2023-01). Lithuania twice: with Lithuanian and with euro-area inflation (the ECB's target).
F4  fuzzy c-means of the 232 country-months 2021-03..2025-12 (inflation, its 3-month change, real policy rate; standardized) into
    regimes; validity indices for c = 2..5; memberships near the change points of the crisp segmentation (question 3).
The fuzzy tools are those of Lab 5 (code/lab5_fuzzy/fuzzy.py).

Outputs: results/q6_match_fuzzy.csv, q6_threshold.csv, q6_summaries.csv, q6_summaries_best.csv, q6_tsk_fit.csv,
q6_tsk_rules.csv,
q6_fcm_validity.csv, q6_fcm_centres.csv, q6_fcm_transitions.csv; figures lab10_match, lab10_tsk, lab10_fcm.
Run (from code/):  python -m lab10_economies.fuzzy_tools
"""
import json
from itertools import product

import numpy as np
import pandas as pd

from lab5_fuzzy.fuzzy import Partition, fcm, partition_coefficient, partition_entropy, xie_beni

from .common import COUNTRIES, GOODS, LABEL, P, PERIOD, panel, save_figure, save_table
from .paths import RAW, RESULTS

YEARS = [2021, 2022, 2023, 2024, 2025]
S_ = lambda x, a, b: np.clip((np.asarray(x, float) - a) / (b - a), 0, 1)       # 0 below a, 1 above b
Z_ = lambda x, a, b: 1 - S_(x, a, b)                                             # 1 below a, 0 above b


def ruspini(g, inner: float, outer: float) -> dict:
    """slower / about equal / faster for a gap g: 'about equal' is 1 within +-inner and 0 beyond +-outer."""
    slower, faster = S_(-np.asarray(g, float), inner, outer), S_(g, inner, outer)
    return {"slower": slower, "about equal": 1 - slower - faster, "faster": faster}


# --------------------------------------------------------------------------------------------- F1
def f1_match(panels: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for c, p in panels.items():
        dec = p[p.index.month == 12]
        at = lambda col, y: dec.loc[dec.index.year == y, col].iloc[0]
        n = len(p) - 1
        per_year = lambda s: 100 * ((s.iloc[-1] / s.iloc[0]) ** (12 / n) - 1)
        for g in GOODS:
            gaps = [100 * (at(f"price_{g}", y) / at(f"price_{g}", y - 1) - 1) - at("cpi_yoy", y) for y in YEARS]
            mu_years = ruspini(gaps, 2, 6)["about equal"]
            gap5 = per_year(p[f"price_{g}"]) - per_year(p["cpi_index"])
            mu5 = ruspini(gap5, 1, 3)                        # per year over five years: a narrower band
            rows.append({"country": c, "good": LABEL[g], **{f"gap {y}, p.p.": v for y, v in zip(YEARS, gaps)},
                         "about equal, mean over the years": float(np.mean(mu_years)),
                         "gap per year 2021-2025, p.p.": gap5, **{f"5 years: {k}": float(v) for k, v in mu5.items()}})
    t = pd.DataFrame(rows).set_index(["country", "good"])
    # crisp verdict of question 1 against its threshold (cumulative growth gap over five years, p.p.)
    cum = {c: pd.read_csv(RESULTS / f"q1_cumulative_{c}.csv", index_col=0) for c in panels}
    gap = pd.Series({(c, LABEL[g]): cum[c].loc[LABEL[g], "minus consumer prices, p.p."] for c in panels for g in GOODS})
    thr = pd.DataFrame({"threshold, p.p.": range(1, 16)})
    thr["pairs 'about the same' (of 16)"] = [int((gap.abs() <= tau).sum()) for tau in thr["threshold, p.p."]]
    t["crisp gap 2021-2025, p.p."] = gap
    t["crisp verdict at 3 / 5 / 8 p.p."] = [" / ".join("same" if abs(v) <= tau else ("faster" if v > 0 else "slower")
                                                     for tau in (3, 5, 8)) for v in gap.reindex(t.index)]
    return t, thr.set_index("threshold, p.p.")


# --------------------------------------------------------------------------------------------- F2
QUANT = {"almost none": lambda r: Z_(r, 0.05, 0.2), "few": lambda r: np.minimum(S_(r, 0.05, 0.2), Z_(r, 0.3, 0.45)),
         "about half": lambda r: np.minimum(S_(r, 0.3, 0.45), Z_(r, 0.55, 0.7)), "most": lambda r: S_(r, 0.55, 0.8),
         "almost all": lambda r: S_(r, 0.8, 0.95)}


def policy_sets(d):                                          # d = policy rate - inflation, p.p. (Ruspini partition)
    return {"the policy rate was much lower than inflation": Z_(d, -4, -1),
            "the policy rate was about equal to inflation": 1 - Z_(d, -4, -1) - S_(d, 1, 4),
            "the policy rate was much higher than inflation": S_(d, 1, 4)}


def summaries(records: pd.DataFrame, summ: dict, qual: dict, country: str, period: str, inside: dict) -> list[dict]:
    """All sentences 'Q of the <records> [with W], S' with truth T1 (Zadeh's calculus), imprecision T2 and covering T3
    (Kacprzyk and Yager 2001, p. 142-143). inside[S] = share of the attribute's range where S > 0 (its flatness)."""
    out, n = [], len(records)
    for (wname, w), (sname, s) in product(qual.items(), summ.items()):
        w = np.ones(n) if w is None else w
        if w.sum() == 0:
            continue
        r = float(np.minimum(w, s).sum() / w.sum())
        t2 = 1 - inside[sname]                               # one attribute (m = 1): T2 = 1 - in(S); data-independent
        t3 = float(((s > 0) & (w > 0)).sum() / max((w > 0).sum(), 1))
        for q, mu in QUANT.items():
            t1 = float(mu(r))
            if t1 > 0:
                out.append({"country": country, "period": period, "qualifier W": wname, "summarizer S": sname,
                            "quantifier Q": q, "proportion r": r, "T1 truth": t1, "T2 imprecision (high = precise S)": t2,
                            "T3 covering": t3, "score T1 x T3": t1 * t3})
    return out


Q_SPECIFIC = {"almost none": 0.8, "few": 0.6, "about half": 0.6, "most": 0.55, "almost all": 0.8}   # 1 - width of support


def best_quantifier(sm: pd.DataFrame) -> pd.DataFrame:
    """One sentence per frame (country, period, W, S): the truest quantifier, ties to the more specific one."""
    sm = sm.assign(_spec=sm["quantifier Q"].map(Q_SPECIFIC))
    sm = sm.sort_values(["T1 truth", "_spec"], ascending=False)
    return sm.groupby(["country", "period", "qualifier W", "summarizer S"], sort=False).head(1).drop(columns="_spec")


# ranges of the attributes (the universes X_j of Kacprzyk and Yager): observed range over the four countries
POLICY_GRID = np.linspace(-22, 13, 3501)                   # policy rate minus inflation, p.p. (2021-2025: -21.8 .. 12.5)
GAP_GRID = np.linspace(-45, 80, 12501)                     # price growth minus inflation in a year, p.p. (-40.8 .. 75.7)
INSIDE_POLICY = {k: float((v > 0).mean()) for k, v in policy_sets(POLICY_GRID).items()}


def f2_summaries(panels: dict) -> pd.DataFrame:
    rows = []
    for c, p in panels.items():
        q = p.iloc[1:]                                       # 2021-01 .. 2025-12
        for period, sl in (("2021-2025", slice(None)), ("2024-2025", slice("2024-01", None))):
            m = q.loc[sl]
            d = (m["policy_rate"] - m["cpi_yoy"]).to_numpy()
            dpi = (m["cpi_yoy"] - m["cpi_yoy"].shift(3)).fillna(0).to_numpy()
            qual = {"(all months)": None, "high inflation (above 5-10 %)": S_(m["cpi_yoy"], 5, 10),
                    "falling inflation": Z_(dpi, -1.5, -0.25)}
            rows += summaries(m, policy_sets(d), qual, c, period, INSIDE_POLICY)
        dec = p[p.index.month == 12]                         # years x goods: price growth against inflation
        gaps = [100 * (dec[f"price_{g}"].iloc[i] / dec[f"price_{g}"].iloc[i - 1] - 1) - dec["cpi_yoy"].iloc[i]
                for g in GOODS for i in range(1, len(dec))]
        rus = ruspini(gaps, 2, 6)
        summ = {f"a good grew {'about as fast as' if k == 'about equal' else 'much ' + k + ' than'} inflation": v
                for k, v in rus.items()}
        inside = {k: float((v > 0).mean()) for k, v in zip(summ, ruspini(GAP_GRID, 2, 6).values())}
        rows += summaries(pd.DataFrame(index=range(len(gaps))), summ, {"(all good-years)": None}, c, "2021-2025", inside)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- F3
INFL = Partition(3, 8, 15)                                   # low / medium / high inflation, %
TREND = {"falling": lambda x: Z_(x, -1, 1), "rising": lambda x: S_(x, -1, 1)}   # 3-month change, p.p. (adds to 1)


def tsk_design(pi, dpi, order: int = 0):
    mu = INFL(pi)
    w = np.stack([mu[a] * TREND[b](dpi) for a, b in product(mu, TREND)], 1)
    w = w / np.clip(w.sum(1, keepdims=True), 1e-12, None)
    X = w if order == 0 else np.hstack([w, w * pi[:, None]])
    return X, [f"inflation {a}, {b}" for a, b in product(mu, TREND)]


def ridge(X, y, alpha=0.1):
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)


def f3_tsk(panels: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ea = json.loads((RAW / "eurostat" / "prc_hicp_manr_EA.json").read_text())
    inv = {v: k for k, v in ea["dimension"]["time"]["category"]["index"].items()}
    ea = pd.Series({pd.Period(inv[int(i)], "M"): v for i, v in ea["value"].items()}).sort_index()
    cases = [(c, c, None) for c in panels] + [("LT", "LT (euro-area inflation)", ea)]
    fit_rows, rule_rows, curves = [], [], {}
    for c, name, infl in cases:
        p = panel(c, base=True)
        pi = (p["cpi_yoy"] if infl is None else infl.reindex(p.index)).astype(float)
        d = pd.DataFrame({"pi": pi, "dpi": pi - pi.shift(3), "rate": p["policy_rate"],
                          "target": p["policy_rate"].shift(-1)}).loc[PERIOD[0]:].dropna()
        X, names = tsk_design(d["pi"].to_numpy(), d["dpi"].to_numpy())
        X1, _ = tsk_design(d["pi"].to_numpy(), d["dpi"].to_numpy(), order=1)
        L = np.c_[np.ones(len(d)), d[["pi", "dpi"]].to_numpy()]
        y = d["target"].to_numpy()
        th, beta, th1 = ridge(X, y), np.linalg.lstsq(L, y, rcond=None)[0], ridge(X1, y)
        r2 = lambda e: 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
        # expanding window: forecasts for 2023-01 .. end, fitted on the months known then
        err = {"no change": [], "linear rule": [], "Takagi-Sugeno": [], "Takagi-Sugeno, first order": []}
        for t in range(list(d.index).index(pd.Period("2023-01", "M")), len(d)):
            Xt, Lt, yt = X[:t - 1], L[:t - 1], y[:t - 1]               # the target of month t-1 is known at t
            err["no change"].append(d["rate"].iloc[t] - y[t])
            err["linear rule"].append(L[t] @ np.linalg.lstsq(Lt, yt, rcond=None)[0] - y[t])
            err["Takagi-Sugeno"].append(X[t] @ ridge(Xt, yt) - y[t])
            err["Takagi-Sugeno, first order"].append(X1[t] @ ridge(X1[:t - 1], yt) - y[t])
        rmse = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in err.items()}
        fit_rows.append({"bank (inflation used)": name, "months": len(d),
                         "R2 in sample, linear rule": r2(L @ beta - y), "R2 in sample, Takagi-Sugeno": r2(X @ th - y),
                         "R2 in sample, Takagi-Sugeno, first order": r2(X1 @ th1 - y),
                         **{f"RMSE 2023-2025, {k}": v for k, v in rmse.items()}})
        for j, rn in enumerate(names):
            weight = float(X[:, j].sum())                    # total firing strength: the months behind the constant
            rule_rows.append({"bank (inflation used)": name, "rule": f"IF {rn} THEN rate next month is about c",
                              "c, %": th[j], "months behind the rule (sum of firing strengths)": weight,
                              "supported": "yes" if weight >= 3 else "no (shrunk towards 0)"})
        grid = np.linspace(0, 25, 101)
        curves[name] = (d, grid, {b: tsk_design(grid, np.full_like(grid, 2.0 if b == "rising" else -2.0))[0] @ th
                                  for b in TREND}, beta)
    return pd.DataFrame(fit_rows).set_index("bank (inflation used)"), pd.DataFrame(rule_rows), curves


# --------------------------------------------------------------------------------------------- F4
def f4_fcm(panels: dict) -> tuple:
    rows = []
    for c, p in panels.items():
        q = p.iloc[1:]
        rows.append(pd.DataFrame({"country": c, "month": q.index, "inflation": q["cpi_yoy"],
                                  "change over 3 months": q["cpi_yoy"] - p["cpi_yoy"].shift(3).iloc[1:],
                                  "real policy rate": q["real_policy_rate"]}))
    D = pd.concat(rows, ignore_index=True).dropna().reset_index(drop=True)   # January, February 2021: no 3-month change
    feats = ["inflation", "change over 3 months", "real policy rate"]
    Xr = D[feats].to_numpy(float)
    mu, sd = Xr.mean(0), Xr.std(0)
    X = (Xr - mu) / sd
    val = []
    for c in range(2, 6):
        U, V, J, it = fcm(X, c, seed=0)
        val.append({"clusters": c, "partition coefficient": partition_coefficient(U),
                    "partition entropy": partition_entropy(U), "Xie-Beni": xie_beni(X, U, V)})
    U, V, J, it = fcm(X, 3, seed=0)
    order = np.argsort(V[:, 0] * sd[0] + mu[0])               # clusters ordered by inflation of the centre
    U, V = U[:, order], V[order]
    centres = pd.DataFrame(V * sd + mu, columns=feats, index=["regime 1", "regime 2", "regime 3"])
    for k in range(3):
        D[f"membership, regime {k + 1}"] = U[:, k]
    # graded transitions: the largest membership near the change points of question 3 and elsewhere
    reg = pd.read_csv(RESULTS / "q3_regimes.csv")
    tr = []
    for c in panels:
        m = D[D.country == c].set_index("month")
        top = m[[f"membership, regime {k + 1}" for k in range(3)]].max(axis=1)
        cuts = [pd.Period(x, "M") for x in reg[reg.country == c]["from"].iloc[1:]]
        near = top.index.isin([x + k for x in cuts for k in (-1, 0, 1)])
        tr.append({"country": c, "change points (question 3)": len(cuts),
                   "largest membership near a change point (+-1 month)": float(top[near].mean()),
                   "largest membership elsewhere": float(top[~near].mean())})
    return D, pd.DataFrame(val).set_index("clusters"), centres, pd.DataFrame(tr).set_index("country"), reg


# --------------------------------------------------------------------------------------------- figures
def figures(match, curves, D, reg):
    f, ax = P.fig(3.0)                                       # F1: membership of "about equal" per year
    g = np.linspace(-15, 15, 301)
    for (k, v), col in zip(ruspini(g, 2, 6).items(), [P.BLUE, P.INK, P.ORANGE]):
        ax.plot(g, v, color=col, lw=1.6, label=k)
    pts = match[[f"gap {y}, p.p." for y in YEARS]].to_numpy().ravel()
    ax.plot(pts.clip(-15, 15), np.full(len(pts), -0.06), "|", color=P.MUTED, ms=8)
    ax.axvline(0, color=P.MUTED, lw=0.6)
    ax.set_xlabel("price growth minus inflation in a year, p.p. (ticks: the 80 good-years, clipped at 15)")
    ax.set_ylabel("membership")
    ax.legend(fontsize=9)
    f.tight_layout()
    save_figure(f, "q6_match")

    f, axs = P.fig(5.8, nrows=2, ncols=3, sharex=True, sharey=True)   # F3: rules against the data
    for ax, (name, (d, grid, curve, beta)) in zip(axs.ravel(), curves.items()):
        ax.scatter(d["pi"], d["target"], s=8, color=P.MUTED, lw=0, label="months")
        for b, col in zip(TREND, [P.BLUE, P.ORANGE]):
            ax.plot(grid, curve[b], color=col, lw=1.5, label=f"Takagi-Sugeno, inflation {b}")
        ax.plot(grid, beta[0] + beta[1] * grid, color=P.INK, lw=1, ls="--", label="linear rule (inflation unchanged)")
        ax.set_title(COUNTRIES.get(name, name).replace("LT (euro-area inflation)", "Lithuania: ECB vs euro area"),
                     fontsize=9)
    axs.ravel()[-1].axis("off")
    for ax in axs[-1]:
        ax.set_xlabel("inflation, %")
    for ax in axs[:, 0]:
        ax.set_ylabel("policy rate next month, %")
    h, l = axs[0, 0].get_legend_handles_labels()
    axs.ravel()[-1].legend(h, l, loc="center", fontsize=9, frameon=False)
    f.tight_layout()
    save_figure(f, "q6_tsk")

    f, axs = P.fig(5.6, nrows=4, sharex=True)                # F4: memberships over time, change points of question 3
    for ax, c in zip(axs, COUNTRIES):
        m = D[D.country == c]
        x = m["month"].dt.to_timestamp() if hasattr(m["month"], "dt") else pd.PeriodIndex(m["month"]).to_timestamp()
        ax.stackplot(x, *[m[f"membership, regime {k + 1}"] for k in range(3)], colors=[P.BLUE, P.YELLOW, P.ORANGE],
                     alpha=0.85, labels=["regime 1 (lowest inflation)", "regime 2", "regime 3 (highest inflation)"])
        for x0 in reg[reg.country == c]["from"].iloc[1:]:
            ax.axvline(pd.Period(x0, "M").to_timestamp(), color=P.INK, lw=0.9, ls="--")
        ax.set_ylabel(COUNTRIES[c], fontsize=9)
        ax.set_ylim(0, 1)
    h, l = axs[0].get_legend_handles_labels()
    f.legend(h, l + [], fontsize=9, ncol=3, loc="lower center", frameon=False)
    f.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(f, "q6_fcm")


def main():
    panels = {c: panel(c, base=True).loc[:PERIOD[1]] for c in COUNTRIES}
    match, thr = f1_match(panels)
    save_table(match, "q6_match_fuzzy", 3); save_table(thr, "q6_threshold")
    sm = f2_summaries(panels)
    save_table(sm.set_index("country"), "q6_summaries", 3)
    fit, rules, curves = f3_tsk(panels)
    save_table(fit, "q6_tsk_fit", 3); save_table(rules.set_index("bank (inflation used)"), "q6_tsk_rules", 3)
    D, val, centres, tr, reg = f4_fcm(panels)
    save_table(val, "q6_fcm_validity", 3); save_table(centres, "q6_fcm_centres"); save_table(tr, "q6_fcm_transitions", 3)
    figures(match, curves, D, reg)

    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
    print("F1 fuzzy match\n", match.round(2).to_string(), "\n\n", thr.T.to_string(), "\n")
    best = best_quantifier(sm).sort_values("score T1 x T3", ascending=False).groupby("country").head(5)
    save_table(best.set_index("country"), "q6_summaries_best", 3)
    print("F2 best summaries: five per country, one quantifier per sentence (by truth x covering)\n",
          best.sort_values(["country", "score T1 x T3"], ascending=[True, False]).round(2).to_string(index=False), "\n")
    trivial = sm[(sm["T1 truth"] >= 0.99) & (sm["T3 covering"] < 0.15)]
    print("F2 true but uninformative (T1 >= 0.99, covering < 0.15):", len(trivial), "of", len(sm[sm["T1 truth"] >= 0.99]),
          "fully true sentences\n", trivial.head(6).round(2).to_string(index=False), "\n")
    print("F3 reaction functions\n", fit.round(3).to_string(), "\n\n", rules.round(2).to_string(index=False), "\n")
    print("F4 fuzzy c-means\n", val.round(3).to_string(), "\n\n", centres.round(2).to_string(), "\n\n", tr.round(3).to_string())


if __name__ == "__main__":
    main()
