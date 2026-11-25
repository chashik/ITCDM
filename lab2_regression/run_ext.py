"""Lab 2 (iteration 2). Many predictors, multiple testing, prediction intervals and causal tools.

A. Monthly volatility from 126 macroeconomic series (FRED-MD): ordinary least squares
   with p close to n, principal-component regression, ridge and lasso.
B. Multiple testing: 126 predictive regressions, Bonferroni and Benjamini-Hochberg.
C. Prediction intervals for the daily model of Lab 2: split conformal prediction and
   adaptive conformal inference with delayed feedback.
D. Event study of FOMC announcement days (Mill's method of difference).
E. Granger causality between implied and realized volatility.

Run:  python -m lab2_regression.run_ext
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.api import VAR

from dmfin import config as C
from dmfin import plotting as P
from dmfin.data import load_market_panel
from dmfin.data_ext import load_fomc_dates, load_fredmd, transform_fredmd
from dmfin.features import add_targets, market_features
from dmfin.reporting import save_numbers, save_table
from dmfin.splits import chrono_split

from .model import FEATURES, HORIZON, TARGET, design, make_model

ALPHA = 0.10                  # 90 % prediction intervals


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ======================================================================= A, B
def monthly_data():
    p = load_market_panel()
    r = np.log(p["adj_close"]).diff().dropna()
    g = r.groupby(r.index.to_period("M"))
    rv = np.log(np.sqrt(252 * g.apply(lambda x: (x ** 2).mean())))
    vix = np.log(p["VIXCLS"]).groupby(p.index.to_period("M")).last()
    md, tc = load_fredmd()
    X = transform_fredmd(md, tc)
    X.index = X.index.to_period("M")
    X = X.shift(1)                      # month m-1: the latest macro data published by the end of month m
    df = pd.DataFrame({"y": rv.shift(-1), "lrv": rv, "lvix": vix}).join(X, how="inner")
    df = df.loc[pd.Period("2000-03", "M"):pd.Period(C.END[:7], "M")].dropna(subset=["y", "lrv", "lvix"])
    macro = [c for c in X.columns if c in df.columns]
    return df, macro


def block(df):
    d = df.index.to_timestamp(how="end")
    tr = d <= pd.Timestamp(C.TRAIN_END) - pd.offsets.MonthEnd(1)       # purge: target month inside the block
    va = (d > pd.Timestamp(C.TRAIN_END)) & (d <= pd.Timestamp(C.VAL_END) - pd.offsets.MonthEnd(1))
    te = d > pd.Timestamp(C.VAL_END)
    return tr, va, te


def part_ab(num):
    df, macro = monthly_data()
    tr, va, te = block(df)
    base = ["lrv", "lvix"]
    fill = df.loc[tr, macro].mean()
    Z = df[macro].fillna(fill)
    sc = StandardScaler().fit(Z[tr])
    Zs = pd.DataFrame(sc.transform(Z), index=df.index, columns=macro)
    y = df["y"]
    num.update({"md_n_train": int(tr.sum()), "md_n_val": int(va.sum()), "md_n_test": int(te.sum()), "md_p": len(macro)})

    # ---------------- PCA of the macro panel (training block)
    pca = PCA().fit(Zs[tr])
    cum = np.cumsum(pca.explained_variance_ratio_)
    num.update({"md_pc1": 100 * pca.explained_variance_ratio_[0], "md_k50": int(np.searchsorted(cum, 0.5) + 1),
                "md_k80": int(np.searchsorted(cum, 0.8) + 1), "md_pc8": 100 * cum[7]})
    load = pd.DataFrame(pca.components_[:3].T, index=macro, columns=["PC1", "PC2", "PC3"])
    top = {c: ", ".join(load[c].abs().sort_values(ascending=False).index[:6]) for c in load}
    num.update({f"md_top_{c.lower()}": v.replace("_", "\\_").replace("&", "\\&") for c, v in top.items()})

    # ---------------- models: validation chooses k (PCR), alpha (ridge, lasso)
    F = pd.DataFrame(pca.transform(Zs), index=df.index)
    def fit_eval(cols_tr, cols_all, model):
        model.fit(cols_all[tr], y[tr])
        return model, rmse(y[va], model.predict(cols_all[va]))
    B = df[base]
    res = {}
    m0, v0 = fit_eval(None, B, LinearRegression()); res["Baseline: log RV and log VIX"] = (m0, B, v0, 2)
    allX = pd.concat([B, Zs], axis=1)
    m1, v1 = fit_eval(None, allX, LinearRegression()); res["Baseline + all 126 series, OLS"] = (m1, allX, v1, allX.shape[1])
    pcr = {}
    for k in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30):
        XX = pd.concat([B, F.iloc[:, :k].add_prefix("f")], axis=1)
        pcr[k] = fit_eval(None, XX, LinearRegression()) + (XX,)
    kbest = min(pcr, key=lambda k: pcr[k][1])
    res[f"Baseline + {kbest} principal components"] = (pcr[kbest][0], pcr[kbest][2], pcr[kbest][1], 2 + kbest)
    sB = StandardScaler().fit(allX[tr]); allXs = pd.DataFrame(sB.transform(allX), index=df.index)
    rid = {a: fit_eval(None, allXs, Ridge(alpha=a)) for a in (1, 10, 30, 100, 300, 1000, 3000)}
    ab = min(rid, key=lambda a: rid[a][1]); res["Baseline + all series, ridge"] = (rid[ab][0], allXs, rid[ab][1], allX.shape[1])
    las = {a: fit_eval(None, allXs, Lasso(alpha=a, max_iter=50000)) for a in (0.003, 0.01, 0.02, 0.05, 0.1)}
    al = min(las, key=lambda a: las[a][1])
    nz = int((np.abs(las[al][0].coef_) > 1e-8).sum())
    res["Baseline + all series, lasso"] = (las[al][0], allXs, las[al][1], nz)
    rows = []
    for name, (m, XX, v, k) in res.items():
        tr_rmse = rmse(y[tr], m.predict(XX[tr]))
        m.fit(XX[tr | va], y[tr | va])                       # refit on training + validation
        rows.append((name, k, tr_rmse, v, rmse(y[te], m.predict(XX[te]))))
    tab = pd.DataFrame(rows, columns=["Model", "Predictors", "Train RMSE", "Valid. RMSE", "Test RMSE"]).set_index("Model")
    save_table("lab2_fredmd", tab, nd={"Predictors": 0, "Train RMSE": 3, "Valid. RMSE": 3, "Test RMSE": 3},
               col_format="p{6.4cm}cccc")
    num.update({"md_k_best": kbest, "md_ridge_alpha": ab, "md_lasso_alpha": al, "md_lasso_nz": nz,
                "md_ols_train": tab.iloc[1, 1], "md_ols_test": tab.iloc[1, 3], "md_base_test": tab.iloc[0, 3],
                "md_pcr_test": tab.iloc[2, 3], "md_ridge_test": tab.iloc[3, 3], "md_lasso_test": tab.iloc[4, 3],
                "md_base_val": tab.iloc[0, 2], "md_pcr_val": tab.iloc[2, 2]})
    f, axs = P.fig(2.7, ncols=2)
    axs[0].plot(np.arange(1, 41), 100 * cum[:40], "o-", ms=2.5, color=P.BLUE)
    axs[0].axhline(50, color=P.MUTED, ls=":", lw=0.8); axs[0].axhline(80, color=P.MUTED, ls=":", lw=0.8)
    axs[0].set_xlabel("number of components"); axs[0].set_ylabel("cumulative variance, %")
    ks = sorted(pcr); axs[1].plot(ks, [pcr[k][1] for k in ks], "o-", ms=3, color=P.ORANGE, label="baseline + $k$ PCs")
    axs[1].axhline(v0, color=P.MUTED, ls="--", lw=0.9, label="baseline")
    axs[1].set_xlabel("number of components $k$"); axs[1].set_ylabel("validation RMSE"); axs[1].legend(fontsize=9)
    f.tight_layout(); P.save(f, "lab2_fredmd")

    # ---------------- B: multiple testing of 126 predictive regressions (training block)
    pv, coef = {}, {}
    for c in macro:
        Xc = sm.add_constant(pd.concat([B[tr], Zs.loc[tr, c]], axis=1))
        fitc = sm.OLS(y[tr], Xc).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        pv[c] = float(fitc.pvalues[c]); coef[c] = float(fitc.params[c])
    p = pd.Series(pv).sort_values()
    m = len(p)
    raw = p < 0.05
    bonf = multipletests(p, 0.05, "bonferroni")[0]
    bh = multipletests(p, 0.05, "fdr_bh")[0]
    # out-of-sample check of the selected predictors (validation block)
    def val_with(cols):
        XX = pd.concat([B, Zs[list(cols)]], axis=1) if len(cols) else B
        return rmse(y[va], LinearRegression().fit(XX[tr], y[tr]).predict(XX[va]))
    num.update({"mt_m": m, "mt_raw": int(raw.sum()), "mt_expected": 0.05 * m, "mt_bonf": int(bonf.sum()),
                "mt_bh": int(bh.sum()), "mt_bh_names": ", ".join(p.index[bh][:8]).replace("_", "\\_").replace("&", "\\&"),
                "mt_val_raw": val_with(p.index[raw]), "mt_val_bh": val_with(p.index[bh]), "mt_val_base": v0,
                "mt_pmin": float(p.iloc[0])})
    f, ax = P.fig(2.8)
    i = np.arange(1, m + 1)
    ax.plot(i, p.to_numpy(), "o", ms=2.5, color=P.BLUE, label="sorted $p$-values")
    ax.plot(i, 0.05 * i / m, color=P.ORANGE, lw=1.1, label="Benjamini--Hochberg line $0.05\\,i/m$")
    ax.axhline(0.05, color=P.MUTED, ls=":", lw=0.9, label="unadjusted 0.05")
    ax.axhline(0.05 / m, color=P.RED, ls="--", lw=0.9, label="Bonferroni $0.05/m$")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("rank $i$"); ax.set_ylabel("$p$-value")
    ax.legend(fontsize=9)
    P.save(f, "lab2_multiple_testing")


# ======================================================================= C
def adaptive_conformal(pred, y, s_cal, h, alpha, gamma, window):
    """Adaptive conformal inference (Gibbs & Candes 2021) with delayed feedback.

    At time t the half-width of the interval is the (1 - a_t) quantile of the last
    ``window`` conformity scores that are already known.  The outcome of the forecast
    made at t is observed only at t + h; then its score joins the window and the level
    is updated, a <- a + gamma * (alpha - err), where err = 1 if the interval missed.
    """
    n = len(pred)
    widths, alphas = np.empty(n), np.empty(n)
    a, scores = alpha, list(s_cal[-window:])
    for t in range(n):
        if t >= h:
            j = t - h
            err = float(abs(y[j] - pred[j]) > widths[j])
            a += gamma * (alpha - err)
            scores = (scores + [abs(y[j] - pred[j])])[-window:]
        level = 1.0 - a
        widths[t] = np.inf if level >= 1 else (0.0 if level <= 0 else float(np.quantile(scores, level, method="higher")))
        alphas[t] = a
    return widths, alphas


def part_c(num):
    feats = add_targets(market_features(load_market_panel()))
    d = design(feats)
    sp = chrono_split(d.index, horizon=HORIZON)
    tr, va, te = d.loc[sp.train], d.loc[sp.val], d.loc[sp.test]
    m = make_model(1).fit(tr[FEATURES], tr[TARGET])
    s_cal = np.abs(va[TARGET] - m.predict(va[FEATURES])).to_numpy()          # conformity scores
    n = len(s_cal)
    q = float(np.quantile(s_cal, np.ceil((n + 1) * (1 - ALPHA)) / n, method="higher"))
    pred = m.predict(te[FEATURES]); yte = te[TARGET].to_numpy()
    cover_s = np.abs(yte - pred) <= q
    # adaptive conformal inference (Gibbs & Candes 2021) with delayed feedback
    W, gamma = 500, 0.005
    widths, alphas = adaptive_conformal(pred, yte, s_cal, HORIZON, ALPHA, gamma, W)
    cover_a = np.abs(yte - pred) <= widths
    years = te.index.year
    rows = []
    for yv in sorted(set(years)):
        mk = years == yv
        rows.append((str(yv), 100 * cover_s[mk].mean(), 2 * q, 100 * cover_a[mk].mean(), 2 * float(np.median(widths[mk][np.isfinite(widths[mk])]))))
    rows.append(("2019--2026", 100 * cover_s.mean(), 2 * q, 100 * cover_a.mean(), 2 * float(np.median(widths[np.isfinite(widths)]))))
    ct = pd.DataFrame(rows, columns=["Year", "Coverage, split, \\%", "Width, split", "Coverage, adaptive, \\%",
                                     "Median width, adaptive"]).set_index("Year")
    save_table("lab2_conformal", ct, nd={"Coverage, split, \\%": 1, "Width, split": 2, "Coverage, adaptive, \\%": 1,
                                         "Median width, adaptive": 2}, col_format="lcccc")
    num.update({"cf_q": q, "cf_width_pct": 100 * (np.exp(q) - 1), "cf_cover_split": 100 * cover_s.mean(),
                "cf_cover_adapt": 100 * cover_a.mean(), "cf_cover_split_2020": float(ct.loc["2020", "Coverage, split, \\%"]),
                "cf_cover_adapt_2020": float(ct.loc["2020", "Coverage, adaptive, \\%"]), "cf_n_cal": n,
                "cf_window": W, "cf_gamma": gamma})
    f, axs = P.fig(4.2, nrows=2, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    dt = te.index
    axs[0].plot(dt, np.exp(yte) * 100, color=P.INK2, lw=0.6, label="realized next-month volatility")
    axs[0].fill_between(dt, np.exp(pred - q) * 100, np.exp(pred + q) * 100, color=P.BLUE, alpha=0.25, lw=0,
                        label="split conformal 90 %")
    wf = np.where(np.isfinite(widths), widths, np.nan)
    axs[0].plot(dt, np.exp(pred + wf) * 100, color=P.ORANGE, lw=0.6, label="adaptive 90 % (upper bound)")
    axs[0].set_yscale("log"); axs[0].set_ylabel("volatility, %"); axs[0].legend(fontsize=9, loc="upper right")
    roll = lambda c: pd.Series(c.astype(float), index=dt).rolling(250, min_periods=60).mean() * 100
    axs[1].plot(dt, roll(cover_s), color=P.BLUE, lw=1, label="split")
    axs[1].plot(dt, roll(cover_a), color=P.ORANGE, lw=1, label="adaptive")
    axs[1].axhline(90, color=P.MUTED, ls=":", lw=0.9)
    axs[1].set_ylabel("coverage, % (250 days)"); axs[1].legend(fontsize=9, loc="lower right")
    f.tight_layout(); P.save(f, "lab2_conformal")


# ======================================================================= D
def part_d(num):
    p = load_market_panel()
    r = np.log(p["adj_close"]).diff()
    absr = r.abs() * 100
    pk = np.sqrt(np.log(p["high"] / p["low"]) ** 2 / (4 * np.log(2))) * 100
    days = p.index
    ev = [d for d in load_fomc_dates()["fomc_date"] if d in days]
    pos = np.searchsorted(days, ev)
    rows = []
    for name, lo, hi in (("2000--2014", "1999-12-31", C.TRAIN_END), ("2015--2018", C.TRAIN_END, C.VAL_END),
                         ("2019--2026", C.VAL_END, C.END)):
        sel = [i for i in pos if pd.Timestamp(lo) < days[i] <= pd.Timestamp(hi)]
        ctrl = sorted({j for i in sel for j in range(i - 10, i + 11) if j not in set(pos) and 0 <= j < len(days)})
        e, c = absr.iloc[sel].dropna(), absr.iloc[ctrl].dropna()
        ep, cp = pk.iloc[sel].dropna(), pk.iloc[ctrl].dropna()
        t, pt = stats.ttest_ind(e, c, equal_var=False)
        rng = np.random.default_rng(C.SEED)
        both = np.r_[ep.to_numpy(), cp.to_numpy()]
        obs = ep.mean() - cp.mean()
        perm = [rng.permutation(both)[:len(ep)].mean() - cp.mean() for _ in range(5000)]
        pp = float(np.mean(np.abs(np.array(perm)) >= abs(obs)))
        rows.append((name, len(sel), float(e.mean()), float(c.mean()), float(t), float(pt), float(ep.mean()),
                     float(cp.mean()), pp))
    et = pd.DataFrame(rows, columns=["Period", "Events", "$|r|$ event, \\%", "$|r|$ control, \\%", "$t$", "$p$",
                                     "Range vol. event, \\%", "Range vol. control, \\%", "$p$ (perm.)"]).set_index("Period")
    save_table("lab2_fomc", et, nd={"Events": 0, "$|r|$ event, \\%": 2, "$|r|$ control, \\%": 2, "$t$": 2, "$p$": 3,
                                    "Range vol. event, \\%": 2, "Range vol. control, \\%": 2, "$p$ (perm.)": 3},
               col_format="lcccccccc")
    num.update({"fomc_events": int(et["Events"].sum()), "fomc_p_train": float(et.iloc[0]["$p$"]),
                "fomc_p_test": float(et.iloc[2]["$p$"]), "fomc_abs_event_all": float(np.nanmean(absr.iloc[pos])),
                "fomc_abs_ctrl_all": float(absr.mean())})
    prof = pd.DataFrame({k: absr.shift(-k).iloc[pos].to_numpy() for k in range(-5, 6)})
    f, ax = P.fig(2.6)
    ax.bar(prof.columns, prof.mean().to_numpy(), color=[P.ORANGE if k == 0 else P.BLUE for k in prof.columns])
    ax.axhline(absr.mean(), color=P.MUTED, ls=":", lw=0.9, label="mean of all days")
    ax.set_xlabel("trading days relative to the FOMC statement"); ax.set_ylabel("mean $|r_t|$, %"); ax.legend(fontsize=9)
    P.save(f, "lab2_fomc")


# ======================================================================= E
def part_e(num):
    p = load_market_panel()
    pk = np.log(np.sqrt(252 * np.log(p["high"] / p["low"]) ** 2 / (4 * np.log(2))).clip(lower=1e-3))
    df = pd.DataFrame({"log_vix": np.log(p["VIXCLS"]), "log_rv": pk}).dropna()
    rows = []
    for name, lo, hi in (("2000--2014", "1999-12-31", C.TRAIN_END), ("2019--2026", C.VAL_END, C.END)):
        blk = df.loc[pd.Timestamp(lo) + pd.Timedelta(days=1):hi]
        model = VAR(blk)
        lag = int(model.select_order(10).bic)
        res = model.fit(max(lag, 1))
        for cause, effect in (("log_vix", "log_rv"), ("log_rv", "log_vix")):
            t = res.test_causality(effect, [cause], kind="f")
            rows.append((name, f"{cause} $\\to$ {effect}".replace("_", "\\_"), res.k_ar, float(t.test_statistic), float(t.pvalue)))
        if name == "2000--2014":
            irf = res.irf(20)
    gt = pd.DataFrame(rows, columns=["Period", "Direction", "Lags (BIC)", "$F$", "$p$"]).set_index("Period")
    save_table("lab2_granger", gt, nd={"Lags (BIC)": 0, "$F$": 1, "$p$": 3}, col_format="llccc")
    num.update({"gc_lags_train": int(gt.iloc[0, 1]), "gc_f_vix_rv": float(gt.iloc[0, 2]), "gc_f_rv_vix": float(gt.iloc[1, 2]),
                "gc_p_vix_rv": float(gt.iloc[0, 3]), "gc_p_rv_vix": float(gt.iloc[1, 3])})
    f, axs = P.fig(2.6, ncols=2)
    o = irf.orth_irfs
    axs[0].plot(o[:, 1, 0], color=P.BLUE); axs[0].set_title("response of log RV to a VIX shock", fontsize=9)
    axs[1].plot(o[:, 0, 1], color=P.ORANGE); axs[1].set_title("response of log VIX to an RV shock", fontsize=9)
    for ax in axs:
        ax.axhline(0, color=P.INK2, lw=0.6); ax.set_xlabel("days")
    f.tight_layout(); P.save(f, "lab2_granger")


def main():
    num = {"alpha": 100 * (1 - ALPHA)}
    part_ab(num)
    part_c(num)
    part_d(num)
    part_e(num)
    save_numbers("lab2x", num)
    print(num)


if __name__ == "__main__":
    main()
