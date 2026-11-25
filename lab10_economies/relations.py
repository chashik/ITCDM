"""Question 2: which series move together, and which can be predicted from the others (Belarus, Poland, Russia,
Lithuania).

Window 2022-01..2025-12 (48 months, four calendar years; growth over 12 months of the prices exists from 2021-12, because the
collection starts in 2020-12). Prices and the US dollar rate enter as growth over 12 months, rates and yields as levels (both are
"per year" quantities that the Fisher relation compares directly).

Outputs (results/lab10; figures paper/figures/lab10_*.pdf):
  q2_corr_<c>.csv            Spearman correlations (pairwise complete months)
  q2_groups.csv              features grouped by average-linkage clustering of 1 - |rho| (cut at |rho| = 0.6)
  q2_predictability.csv      share of each series explained by the others (in-sample R^2, OLS) and out-of-sample skill
                             of a nowcast from the others (ridge, expanding window, 2024-01..2025-12) against persistence;
                             skill > 0 means the other series improve on last month's value
  q2_granger.csv             Granger tests on monthly changes 2021-02..2025-12, lags 1-3 (smallest p and its lag;
                             Benjamini-Hochberg adjustment over all tests of all countries; a pair needs 30 months)
  q2_leadlag.csv             lag (months) at which the policy rate correlates most with inflation
  q2_correlations.png
Run (from code/):  python -m lab10_economies.relations
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import COUNTRIES, P, panel, save_figure, save_table

WINDOW = ("2022-01", "2025-12")
TEST_FROM = "2024-01"


def features(p: pd.DataFrame, country: str) -> pd.DataFrame:
    f = pd.DataFrame({"flour y/y": p["flour_wheat_yoy"], "milk y/y": p["milk_yoy"], "sugar y/y": p["sugar_yoy"],
                      "petrol y/y": p["petrol_yoy"], "inflation": p["cpi_yoy"], "USD rate y/y": p["fx_usd_yoy"],
                      "policy rate": p["policy_rate"], "mortgage rate": p["mortgage_rate"],
                      "gov. bond yield": p["bond_gov"]})
    if "bond_bank" in p:
        f["bank bond coupon"] = p["bond_bank"]
    if "bond_retail" in p:
        f["retail bond rate"] = p["bond_retail"]
    f = f.loc[WINDOW[0]:WINDOW[1]]
    return f[[c for c in f if f[c].notna().sum() >= 12]]   # Russian sugar and petrol: December values only (4)


def complete(f: pd.DataFrame) -> pd.DataFrame:
    """Columns observed in every month (Belarusian government bonds are placed only in some months)."""
    return f[[c for c in f if f[c].notna().all()]]


def groups(corr: pd.DataFrame, cut: float = 0.6) -> list[list[str]]:
    d = 1 - corr.abs().fillna(0).to_numpy(copy=True)       # pandas 3 hands out read-only arrays
    np.fill_diagonal(d, 0)
    lab = fcluster(linkage(squareform(d, checks=False), "average"), 1 - cut, "distance")
    return [sorted(corr.index[lab == k]) for k in sorted(set(lab))]


def predictability(f: pd.DataFrame) -> pd.DataFrame:
    f = complete(f)
    rows = {}
    for target in f:
        X, y = f.drop(columns=target), f[target]
        r2 = sm.OLS(y, sm.add_constant((X - X.mean()) / X.std())).fit().rsquared
        err_m, err_p, err_o = [], [], []
        for t in f.index[f.index >= pd.Period(TEST_FROM, "M")]:
            tr = f.index < t
            prev = f.index[f.index < t][-1]
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))).fit(X[tr], y[tr])
            err_m.append(float(m.predict(X.loc[[t]])[0] - y[t]))
            err_p.append(float(y[prev] - y[t]))
            Xo = X.assign(own_lag=y.shift(1))                # the others now and the series' own last value
            tro = tr & Xo["own_lag"].notna().to_numpy()
            mo = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))).fit(Xo[tro], y[tro])
            err_o.append(float(mo.predict(Xo.loc[[t]])[0] - y[t]))
        mse = lambda e: float(np.mean(np.square(e)))
        skill = lambda e: 1 - mse(e) / mse(err_p) if mse(err_p) > 0 else np.nan   # constant in the test months
        rows[target] = {"R2 from the others (in sample)": r2,
                        "nowcast skill vs last value": skill(err_m),
                        "skill with own last value": skill(err_o),
                        "RMSE of last value": np.sqrt(mse(err_p))}
    return pd.DataFrame(rows).T


def granger(p: pd.DataFrame, country: str) -> pd.DataFrame:
    """Does the past of x improve the forecast of y beyond y's own past?  Monthly changes, lags 1-3; F test."""
    d = pd.DataFrame({"inflation (m/m)": p["cpi_mom"],
                      "petrol (m/m)": 100 * p["price_petrol"].pct_change(),
                      "food (m/m)": 100 * p[["price_flour_wheat", "price_milk", "price_sugar"]].pct_change().mean(axis=1),
                      "USD rate (m/m)": 100 * p["fx_usd"].pct_change(),
                      "policy rate (change)": p["policy_rate"].diff(),
                      "mortgage rate (change)": p["mortgage_rate"].diff()}).iloc[1:]
    pairs = [("petrol (m/m)", "inflation (m/m)"), ("food (m/m)", "inflation (m/m)"), ("USD rate (m/m)", "inflation (m/m)"),
             ("policy rate (change)", "mortgage rate (change)"), ("inflation (m/m)", "policy rate (change)"),
             ("policy rate (change)", "inflation (m/m)"), ("policy rate (change)", "USD rate (m/m)")]
    rows = []
    for x, y in pairs:
        best = (1.0, 0)
        for lag in (1, 2, 3):
            lags = pd.concat({f"y{k}": d[y].shift(k) for k in range(1, lag + 1)} |
                             {f"x{k}": d[x].shift(k) for k in range(1, lag + 1)}, axis=1)
            data = pd.concat([d[y], lags], axis=1).dropna()
            if len(data) < 30:
                break
            full = sm.OLS(data[y], sm.add_constant(data.drop(columns=y))).fit()
            own = sm.OLS(data[y], sm.add_constant(data[[f"y{k}" for k in range(1, lag + 1)]])).fit()
            pval = float(full.compare_f_test(own)[1])
            best = min(best, (pval, lag))
        if best == (1.0, 0):                                 # not enough months for this pair
            continue
        rows.append({"country": country, "cause": x, "effect": y, "smallest p": best[0], "at lag": best[1]})
    return pd.DataFrame(rows)


def leadlag(p: pd.DataFrame) -> dict:
    """Correlation of the policy rate at t with inflation at t + k; k > 0: the policy rate leads."""
    c = {k: p["policy_rate"].corr(p["cpi_yoy"].shift(-k)) for k in range(-12, 13)}
    k = max(c, key=lambda j: c[j])
    return {"best lag k (policy at t, inflation at t+k)": k, "correlation at best lag": c[k], "correlation at lag 0": c[0]}


def main():
    panels = {c: panel(c) for c in COUNTRIES}
    f, axs = P.fig(7.4, nrows=2, ncols=2)
    allgroups, pred, gr, ll = [], [], [], {}
    for ax, (c, p) in zip(axs.ravel(), panels.items()):
        F = features(p, c)
        corr = F.corr(method="spearman")
        save_table(corr, f"q2_corr_{c}")
        im = ax.imshow(corr.values, cmap=P.DIV_CELL, vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=9)
        ax.set_yticks(range(len(corr)), corr.index, fontsize=9)
        for i in range(len(corr)):
            for j in range(len(corr)):
                v = corr.values[i, j]
                if i != j and not np.isnan(v):          # the diagonal is 1 by definition
                    ax.text(j, i, P.per_cent(v), ha="center", va="center", fontsize=8.5, color=P.INK)
        ax.set_title(COUNTRIES[c], fontsize=9.5)
        ax.grid(False)
        allgroups += [{"country": c, "group": i + 1, "features": ", ".join(g)} for i, g in enumerate(groups(corr))]
        pr = predictability(F)
        pr.insert(0, "country", c)
        pred.append(pr)
        gr.append(granger(p, c))
        ll[c] = leadlag(p)
    f.tight_layout()
    save_figure(f, "q2_correlations")
    g = pd.DataFrame(allgroups).set_index("country")
    pr = pd.concat(pred).rename_axis("target")
    gc = pd.concat(gr).set_index("country")
    from statsmodels.stats.multitest import multipletests      # many tests on 59 months: control the false discoveries
    gc["BH-adjusted p (all tests)"] = multipletests(gc["smallest p"], method="fdr_bh")[1]
    # the smallest of three p-values (lags 1-3) is itself optimistic: Bonferroni factor 3 for the choice of the lag
    gc["BH-adjusted p, lag choice corrected"] = multipletests(np.minimum(1, 3 * gc["smallest p"]), method="fdr_bh")[1]
    llt = pd.DataFrame(ll).T
    save_table(g, "q2_groups"); save_table(pr, "q2_predictability", 3); save_table(gc, "q2_granger", 3)
    save_table(llt, "q2_leadlag", 3)
    pd.set_option("display.width", 200)
    print(g.to_string(), "\n"); print(pr.round(3).to_string(), "\n"); print(gc.round(3).to_string(), "\n")
    print(llt.round(3).to_string())


if __name__ == "__main__":
    main()
