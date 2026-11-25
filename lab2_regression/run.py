"""Lab 2. Polynomial regression of next-month S&P 500 realised volatility.

Run:  python -m lab2_regression.run
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from statsmodels.tsa.stattools import acf

from dmfin import config as C
from dmfin import plotting as P
from dmfin.data import load_market_panel
from dmfin.features import add_targets, market_features
from dmfin.reporting import save_numbers, save_table
from dmfin.splits import chrono_split, expanding_window_cv

from .model import FEATURES, HORIZON, TARGET, cv_score, design, make_model, metrics, n_terms

ALPHAS = [0.1, 1, 10, 100, 1000, 10000]
DEGREES = [1, 2, 3, 4]


def fig_cv_scheme(n=3730, n_splits=5, gap=HORIZON):
    f, ax = P.fig(2.0)
    for k, (tr, va) in enumerate(expanding_window_cv(n, n_splits=n_splits, gap=gap)):
        y = n_splits - k
        ax.barh(y, len(tr), left=0, color=P.BLUE, height=0.55)
        ax.barh(y, gap, left=tr.max() + 1, color="#d9d8d3", height=0.55)
        ax.barh(y, len(va), left=va.min(), color=P.ORANGE, height=0.55)
    ax.set_yticks(range(1, n_splits + 1)); ax.set_yticklabels([f"fold {n_splits - i}" for i in range(n_splits)])
    ax.set_xlabel("observation number inside the training block (time order)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=P.BLUE, label="fit"), Patch(color="#d9d8d3", label=f"embargo ({gap} days)"),
                       Patch(color=P.ORANGE, label="validate")], loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax.set_ylim(0.4, n_splits + 0.9)
    P.save(f, "lab2_cv_scheme")


def main():
    feats = add_targets(market_features(load_market_panel()))
    d = design(feats)
    sp = chrono_split(d.index, horizon=HORIZON)
    tr, va, te = d.loc[sp.train], d.loc[sp.val], d.loc[sp.test]
    trva = pd.concat([tr, va])
    Xtr, ytr = tr[FEATURES].to_numpy(), tr[TARGET].to_numpy()
    Xva, yva = va[FEATURES].to_numpy(), va[TARGET].to_numpy()
    Xte, yte = te[FEATURES].to_numpy(), te[TARGET].to_numpy()
    num = {"n_train": len(tr), "n_val": len(va), "n_test": len(te), "p": len(FEATURES)}

    fig_cv_scheme(len(tr))

    # ---------------------------------------- why scaling matters for polynomials
    for deg in (2, 3):
        raw = PolynomialFeatures(deg, include_bias=True).fit_transform(Xtr)
        std = PolynomialFeatures(deg, include_bias=True).fit_transform(StandardScaler().fit_transform(Xtr))
        sci = lambda v: (lambda m, e: f"{m}\\cdot 10^{{{int(e)}}}")(*f"{v:.1e}".split("e"))
        num[f"cond_raw_d{deg}"] = sci(np.linalg.cond(raw))
        num[f"cond_std_d{deg}"] = sci(np.linalg.cond(std))
    for deg in DEGREES:
        num[f"terms_d{deg}"] = n_terms(len(FEATURES), deg)

    # ----------------------------------------------------------------- PCA
    Z = StandardScaler().fit_transform(Xtr)
    pca = PCA().fit(Z)
    evr = pca.explained_variance_ratio_
    f, ax = P.fig(2.6)
    k = np.arange(1, len(evr) + 1)
    ax.bar(k, 100 * evr, color=P.BLUE, width=0.6, label="individual")
    ax.plot(k, 100 * np.cumsum(evr), "o-", color=P.ORANGE, ms=4, label="cumulative")
    for i, v in enumerate(np.cumsum(evr)):
        ax.text(k[i], 100 * v + 3, f"{100*v:.0f}", ha="center", fontsize=8, color=P.INK2)
    ax.set_xlabel("principal component"); ax.set_ylabel("variance explained, %")
    ax.set_ylim(0, 112); ax.legend(loc="center right")
    P.save(f, "lab2_pca_scree")
    load = pd.DataFrame(pca.components_[:3].T, index=FEATURES, columns=["PC1", "PC2", "PC3"])
    load.index.name = "Feature"
    save_table("lab2_pca_loadings", load, nd=2)
    num.update({"pca_ev1": 100 * evr[0], "pca_ev2": 100 * evr[1], "pca_cum3": 100 * evr[:3].sum(),
                "pca_cum4": 100 * evr[:4].sum()})

    # ---------------------------------------------- degree selection by CV
    variants = {  # name -> factory(deg, alpha)
        "OLS": lambda dg, a: make_model(dg),
        "OLS + clip": lambda dg, a: make_model(dg, clip=True),
        "Ridge + clip": lambda dg, a: make_model(dg, alpha=a, clip=True),
        "PCA(4) + clip": lambda dg, a: make_model(dg, n_pca=4, clip=True),
    }
    rows, curves = [], {}
    for deg in DEGREES:
        for name, fac in variants.items():
            grid = ALPHAS if name.startswith("Ridge") else [None]
            best = None
            for a in grid:
                trE, vaE = cv_score(lambda: fac(deg, a), Xtr, ytr)
                if best is None or vaE.mean() < best[1].mean():
                    best = (trE, vaE, a)
            curves[(name, deg)] = best
            rows.append((name, deg, best[2], best[0].mean(), best[1].mean(),
                         best[1].std() / np.sqrt(len(best[1])), *best[1]))
    cv = pd.DataFrame(rows, columns=["model", "degree", "alpha", "cv_train", "cv_val", "cv_se",
                                     "f1", "f2", "f3", "f4", "f5"])
    cv.to_csv(C.RESULTS_DIR / "lab2_cv.csv", index=False)
    best_row = cv.loc[cv["cv_val"].idxmin()]
    # one-standard-error rule: simplest model within one SE of the minimum
    thr = best_row["cv_val"] + best_row["cv_se"]
    ose = cv[cv["cv_val"] <= thr].sort_values(["degree", "cv_val"]).iloc[0]
    num.update({"best_family": best_row["model"], "best_degree": int(best_row["degree"]),
                "best_alpha": "--" if pd.isna(best_row["alpha"]) else f"{best_row['alpha']:g}",
                "best_cv_rmse": best_row["cv_val"], "best_cv_se": best_row["cv_se"],
                "ose_family": ose["model"], "ose_degree": int(ose["degree"]), "ose_cv_rmse": ose["cv_val"],
                "ols_d1_fold2": float(cv.query("model=='OLS' and degree==1")["f2"].iloc[0]),
                "ols_d2_fold2": float(cv.query("model=='OLS' and degree==2")["f2"].iloc[0]),
                "clip_d2_fold2": float(cv.query("model=='OLS + clip' and degree==2")["f2"].iloc[0]),
                "ols_d3_fold2": float(cv.query("model=='OLS' and degree==3")["f2"].iloc[0]),
                "ols_d4_fold2": float(cv.query("model=='OLS' and degree==4")["f2"].iloc[0]),
                "ols_d1_cv": float(cv.query("model=='OLS' and degree==1")["cv_val"].iloc[0]),
                "ridge_d2_cv": float(cv.query("model=='Ridge + clip' and degree==2")["cv_val"].iloc[0])})
    folds = list(expanding_window_cv(len(tr), n_splits=5, gap=HORIZON))
    num["fold2_from"] = tr.index[folds[1][1][0]].strftime("%Y-%m-%d")
    num["fold2_to"] = tr.index[folds[1][1][-1]].strftime("%Y-%m-%d")
    tab = cv.copy()
    tab["alpha"] = tab["alpha"].map(lambda a: "--" if pd.isna(a) else f"{a:g}")
    tab["terms"] = [n_terms(4 if m.startswith("PCA") else len(FEATURES), int(dg))
                    for m, dg in zip(cv["model"], cv["degree"])]
    tab = tab.sort_values(["degree", "model"])[["model", "degree", "terms", "alpha", "cv_train", "f2", "cv_val", "cv_se"]]
    tab.columns = ["Model", "$d$", "Terms", "$\\alpha$", "Train", "Fold 2", "CV mean", "SE"]
    save_table("lab2_cv_table", tab.set_index("Model"), nd=3, col_format="lccccccc")

    f, axs = P.fig(2.9, ncols=2)
    for name, col, mk in [("OLS", P.RED, "o"), ("OLS + clip", P.ORANGE, "^"), ("Ridge + clip", P.BLUE, "s"),
                          ("PCA(4) + clip", P.AQUA, "D")]:
        vae = [curves[(name, dg)][1].mean() for dg in DEGREES]
        se = [curves[(name, dg)][1].std() / np.sqrt(5) for dg in DEGREES]
        axs[0].errorbar(DEGREES, vae, yerr=se, fmt=mk + "-", color=col, ms=4, capsize=2, label=name)
    tre = [curves[("OLS", dg)][0].mean() for dg in DEGREES]
    axs[0].plot(DEGREES, tre, "o--", color=P.MUTED, ms=3, label="OLS, training folds")
    axs[0].set_yscale("log"); axs[0].set_xticks(DEGREES)
    axs[0].set_xlabel("polynomial degree $d$"); axs[0].set_ylabel("CV RMSE (log scale)")
    axs[0].legend(fontsize=7.5, loc="upper left")
    width = 0.2
    for j, dg in enumerate([1, 2, 3]):
        fe = curves[("OLS", dg)][1]
        axs[1].bar(np.arange(1, 6) + (j - 1) * width, fe, width=width, color=[P.BLUE, P.ORANGE, P.RED][j],
                   label=f"OLS, $d={dg}$")
    axs[1].set_yscale("log"); axs[1].set_xlabel("validation fold"); axs[1].set_ylabel("fold RMSE (log scale)")
    axs[1].legend(fontsize=7.5, loc="upper right")
    f.tight_layout()
    P.save(f, "lab2_degree_cv")

    # ---------------------------------------------------- compare on validation
    a2 = curves[("Ridge + clip", 2)][2]
    a3 = curves[("Ridge + clip", 3)][2]
    num.update({"alpha_d2": f"{a2:g}", "alpha_d3": f"{a3:g}"})
    cands = {
        "Naive (last 22-day RV)": None,
        "HAR, linear (3 features)": ("har", make_model(1)),
        "Linear, 7 features": ("all", make_model(1)),
        "Polynomial $d=2$, OLS": ("all", make_model(2)),
        "Polynomial $d=2$, ridge + clip": ("all", make_model(2, alpha=a2, clip=True)),
        "Polynomial $d=3$, ridge + clip": ("all", make_model(3, alpha=a3, clip=True)),
        "PCA(4) + polynomial $d=2$ + clip": ("all", make_model(2, n_pca=4, clip=True)),
    }
    har = ["log_pk_d", "log_pk_w", "log_pk_m"]
    res, preds = [], {}
    for name, spec in cands.items():
        if spec is None:
            pv, pt = va["log_rv_22"].to_numpy(), te["log_rv_22"].to_numpy()
        else:
            cols = har if spec[0] == "har" else FEATURES
            m = spec[1].fit(tr[cols], tr[TARGET])
            pv = m.predict(va[cols])
            m2 = spec[1].fit(trva[cols], trva[TARGET])  # refit on train+validation
            pt = m2.predict(te[cols])
        preds[name] = pt
        mv, mt = metrics(yva, pv), metrics(yte, pt)
        res.append((name, mv["RMSE"], mv["R2"], mt["RMSE"], mt["MAE"], mt["R2"]))
    comp = pd.DataFrame(res, columns=["Model", "Valid. RMSE", "Valid. $R^2$", "Test RMSE",
                                      "Test MAE", "Test $R^2$"]).set_index("Model")
    save_table("lab2_compare", comp, nd=3, col_format="p{5.0cm}ccccc")
    chosen = "Polynomial $d=2$, ridge + clip"
    num.update({"val_best": comp["Valid. RMSE"].idxmin(),
                "chosen_val_rmse": comp.loc[chosen, "Valid. RMSE"],
                "chosen_test_rmse": comp.loc[chosen, "Test RMSE"], "chosen_test_r2": comp.loc[chosen, "Test $R^2$"],
                "naive_test_rmse": comp.loc["Naive (last 22-day RV)", "Test RMSE"],
                "naive_test_r2": comp.loc["Naive (last 22-day RV)", "Test $R^2$"],
                "har_test_rmse": comp.loc["HAR, linear (3 features)", "Test RMSE"],
                "har_test_r2": comp.loc["HAR, linear (3 features)", "Test $R^2$"],
                "lin_val_rmse": comp.loc["Linear, 7 features", "Valid. RMSE"],
                "lin_test_rmse": comp.loc["Linear, 7 features", "Test RMSE"],
                "lin_test_r2": comp.loc["Linear, 7 features", "Test $R^2$"],
                "p2ols_test_rmse": comp.loc["Polynomial $d=2$, OLS", "Test RMSE"],
                "p3_test_rmse": comp.loc["Polynomial $d=3$, ridge + clip", "Test RMSE"],
                "pca_test_rmse": comp.loc["PCA(4) + polynomial $d=2$ + clip", "Test RMSE"]})
    pd.DataFrame(preds, index=te.index).assign(actual=yte).to_csv(C.RESULTS_DIR / "lab2_test_preds.csv")

    # ------------------------------------------------ levels instead of logs
    lev = ["pk_d", "pk_w", "pk_m", "vix", "ret_22", "term_10y3m", "credit_baa_aaa"]
    dl = feats[lev + ["y_rv_22", "rv_22"]].dropna().copy()
    dl["vix"] = dl["vix"] / 100
    dl = dl.loc[d.index.intersection(dl.index)]
    trl, val_, tel = dl.loc[dl.index.intersection(sp.train)], dl.loc[dl.index.intersection(sp.val)], dl.loc[dl.index.intersection(sp.test)]
    trval = pd.concat([trl, val_])
    lrows = []
    for name, fac in [("Linear", lambda: make_model(1)),
                      ("Polynomial $d=2$, ridge + clip", lambda: make_model(2, alpha=1000, clip=True)),
                      ("Polynomial $d=3$, ridge + clip", lambda: make_model(3, alpha=1000, clip=True))]:
        _, cvl = cv_score(fac, trl[lev].to_numpy(), trl["y_rv_22"].to_numpy())
        pv = fac().fit(trl[lev], trl["y_rv_22"]).predict(val_[lev])
        pt = fac().fit(trval[lev], trval["y_rv_22"]).predict(tel[lev])
        # the log model, back-transformed to levels for a like-for-like comparison
        lrows.append((name, 100 * cvl.mean(), 100 * metrics(val_["y_rv_22"], pv)["RMSE"],
                      100 * metrics(tel["y_rv_22"], pt)["RMSE"]))
    for name in ["Linear, 7 features", chosen]:
        lrows.append((name.replace(", 7 features", "") + " (log target)", np.nan, np.nan,
                      100 * metrics(np.exp(yte), np.exp(preds[name]))["RMSE"]))
    levt = pd.DataFrame(lrows, columns=["Model (volatility in levels)", "CV RMSE, p.p.", "Valid. RMSE, p.p.",
                                        "Test RMSE, p.p."]).set_index("Model (volatility in levels)")
    save_table("lab2_levels", levt, nd=2, col_format="p{6.2cm}ccc")
    num.update({"lev_lin_test": levt.iloc[0, 2], "lev_p2_test": levt.iloc[1, 2], "lev_p3_test": levt.iloc[2, 2],
                "lev_lin_cv": levt.iloc[0, 0], "lev_p2_cv": levt.iloc[1, 0],
                "loglin_test_levels": levt.iloc[3, 2], "logp2_test_levels": levt.iloc[4, 2]})

    # ------------------------------------------------ time series of forecasts
    f, ax = P.fig(3.0)
    ax.plot(te.index, 100 * np.exp(yte), color=P.INK2, lw=0.9, label="realized (next 22 days)")
    ax.plot(te.index, 100 * np.exp(preds["Naive (last 22-day RV)"]), color=P.ORANGE, lw=0.9,
            label="naive: last 22 days")
    ax.plot(te.index, 100 * np.exp(preds[chosen]), color=P.BLUE, lw=1.1, label="polynomial $d=2$, ridge + clip")
    ax.set_yscale("log"); ax.set_ylabel("annualised volatility, % (log scale)")
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ax.yaxis.set_major_locator(FixedLocator([5, 10, 20, 40, 80]))
    ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="upper right")
    P.save(f, "lab2_test_series")

    # ------------------------------------------------------- residual analysis
    resid = yte - preds[chosen]
    f, axs = P.fig(2.6, ncols=3)
    axs[0].scatter(preds[chosen], resid, s=3, color=P.BLUE, alpha=0.5, lw=0)
    axs[0].axhline(0, color=P.INK2, lw=0.6)
    axs[0].set_xlabel("fitted log RV"); axs[0].set_ylabel("residual")
    (osm, osr), (sl, ic, _) = stats.probplot(resid, dist="norm")
    axs[1].scatter(osm, osr, s=3, color=P.BLUE, lw=0)
    axs[1].plot(osm, sl * osm + ic, color=P.ORANGE, lw=1)
    axs[1].set_xlabel("normal quantiles"); axs[1].set_ylabel("residual quantiles")
    ra = acf(resid, nlags=60)[1:]
    axs[2].bar(np.arange(1, 61), ra, color=P.BLUE, width=0.8)
    axs[2].axvline(HORIZON, color=P.ORANGE, lw=0.8, ls="--")
    axs[2].set_xlabel("lag, days"); axs[2].set_ylabel("residual ACF")
    f.tight_layout()
    P.save(f, "lab2_residuals")
    num.update({"resid_acf1": ra[0], "resid_acf22": ra[21], "resid_acf40": ra[39],
                "resid_skew": stats.skew(resid), "resid_kurt": stats.kurtosis(resid)})

    # ------------------------------------------- response surface (degree 2)
    m2 = make_model(2, alpha=a2, clip=True).fit(trva[FEATURES], trva[TARGET])
    med = trva[FEATURES].median()
    vix_g = np.linspace(np.log(10), np.log(60), 80)
    ret_g = np.linspace(-0.25, 0.12, 80)
    VV, RR = np.meshgrid(vix_g, ret_g)
    G = pd.DataFrame(np.tile(med.to_numpy(), (VV.size, 1)), columns=FEATURES)
    G["log_vix"] = VV.ravel(); G["ret_22"] = RR.ravel()
    Zs = 100 * np.exp(m2.predict(G)).reshape(VV.shape)
    f, ax = P.fig(3.2, w=4.8)
    cs = ax.contourf(np.exp(VV), 100 * RR, Zs, levels=14, cmap=P.SEQ)
    ax.contour(np.exp(VV), 100 * RR, Zs, levels=14, colors="white", linewidths=0.3)
    f.colorbar(cs, ax=ax, label="predicted volatility, %")
    ax.set_xlabel("VIX, points"); ax.set_ylabel("past 22-day return, %")
    ax.grid(False)
    P.save(f, "lab2_surface")

    # ----------------------------------- inference: linear model, HAC errors
    Xs = sm.add_constant(StandardScaler().fit(tr[FEATURES]).transform(tr[FEATURES]))
    ols = sm.OLS(ytr, Xs).fit(cov_type="HAC", cov_kwds={"maxlags": HORIZON})
    ols_naive = sm.OLS(ytr, Xs).fit()
    coef = pd.DataFrame({"Coefficient": ols.params, "SE (OLS)": ols_naive.bse,
                         "SE (HAC)": ols.bse, "$t$ (HAC)": ols.tvalues},
                        index=["const"] + FEATURES)
    coef.index.name = "Term"
    save_table("lab2_ols_hac", coef, nd=3)
    num.update({"ols_r2_train": ols.rsquared, "se_ratio": float(np.median(ols.bse / ols_naive.bse))})

    save_numbers("lab2", num)
    print(cv.round(4).to_string()); print(comp.round(4).to_string()); print(num)


if __name__ == "__main__":
    main()
