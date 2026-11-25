"""Lab 1 (iteration 2). Model-based and density-based clustering, non-linear maps,
the 49 industry portfolios and market regimes.

* Gaussian mixtures (EM) on the risk features of the 110 stocks, number of
  components and covariance type by BIC, soft memberships;
* HDBSCAN versus DBSCAN;
* t-SNE map versus PCA (trustworthiness);
* co-movement clustering of the 49 Fama-French industries on the training
  block and its stability on the test block (no survivorship bias);
* market regimes: a Gaussian mixture of daily market-state features fitted on
  2000-2014 and applied to every later day (used by Lab 7).

Run:  python -m lab1_clustering.run_ext
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from dmfin import config as C
from dmfin import plotting as P
from dmfin.data import (PUBLICATION_LAG_DAYS, available_at, download_fred, load_market_panel, load_stock_prices,
                        load_universe)
from dmfin.data_ext import load_ff_factors, load_ff_industries
from dmfin.reporting import save_numbers, save_table

from .model import FEATURES, corr_distance, purity, stock_features

K = 6
REG = 1e-2        # covariance regularization: 110 points cannot support full 9x9 covariances


def risk_features():
    px = load_stock_prices()
    sectors = load_universe().reindex(px.columns)
    panel = load_market_panel()
    rets = np.log(px).diff().iloc[1:]
    fit = slice(C.STOCKS_START, C.STOCKS_SPLIT)
    feats = stock_features(rets.loc[fit], np.log(panel["adj_close"]).diff().loc[fit], panel["DGS10"].diff().loc[fit],
                           panel["DCOILWTICO"].pct_change(fill_method=None).clip(-0.5, 0.5).loc[fit])
    return feats, sectors, rets.loc[fit]


def kmeans_labels(feats, X):
    """The K=6 risk clusters of Lab 1, named C1..C6 by increasing market beta."""
    km = KMeans(K, n_init=100, random_state=C.SEED).fit(X)
    lab = pd.Series(km.labels_, index=feats.index)
    order = feats.groupby(lab)["beta_mkt"].mean().sort_values().index
    return lab.map({old: new for new, old in enumerate(order)})


def main():
    num = {}
    feats, sectors, R = risk_features()
    X = StandardScaler().fit_transform(feats)
    lab = kmeans_labels(feats, X)
    # co-movement clusters of Lab 1 (k-means on normalized loadings of 10 return PCs)
    Rs = (R - R.mean()) / R.std()
    pc = PCA(10).fit(Rs.fillna(0).to_numpy())
    load = pc.components_.T * np.sqrt(pc.explained_variance_)
    load = load / np.linalg.norm(load, axis=1, keepdims=True)
    fac = KMeans(11, n_init=100, random_state=C.SEED).fit(load).labels_
    out = pd.DataFrame({"sector": sectors, "risk_cluster": [f"C{i + 1}" for i in lab],
                        "comove_cluster": [f"F{i + 1}" for i in fac]}, index=feats.index)
    out.index.name = "ticker"
    out.to_csv(C.RESULTS_DIR / "lab1_clusters.csv")

    # ------------------------------------------------ Gaussian mixtures (EM) and BIC
    rows = []
    for cov in ("spherical", "diag", "tied", "full"):
        for k in range(1, 11):
            gm = GaussianMixture(k, covariance_type=cov, n_init=10, random_state=C.SEED, reg_covar=REG).fit(X)
            rows.append((cov, k, gm.bic(X), gm._n_parameters()))
    bic = pd.DataFrame(rows, columns=["cov", "k", "bic", "params"])
    best = bic.loc[bic["bic"].idxmin()]
    gm = GaussianMixture(int(best["k"]), covariance_type=best["cov"], n_init=10, random_state=C.SEED,
                         reg_covar=REG).fit(X)
    post = gm.predict_proba(X)
    gml = post.argmax(1)
    gm6 = GaussianMixture(K, covariance_type="diag", n_init=10, random_state=C.SEED, reg_covar=REG).fit(X)
    post6 = gm6.predict_proba(X)
    num.update({"gmm_best_k": int(best["k"]), "gmm_best_cov": best["cov"], "gmm_best_params": int(best["params"]),
                "gmm_full6_params": int(bic[(bic["cov"] == "full") & (bic["k"] == 6)]["params"].iloc[0]),
                "gmm_ari_kmeans": adjusted_rand_score(lab, gml), "gmm6_ari_kmeans": adjusted_rand_score(lab, post6.argmax(1)),
                "gmm6_uncertain": 100 * float((post6.max(1) < 0.9).mean()), "gmm6_maxpost_mean": float(post6.max(1).mean()),
                "gmm_ari_sector": adjusted_rand_score(sectors, gml)})
    f, axs = P.fig(2.9, ncols=2)
    for cov, col in zip(("spherical", "diag", "tied", "full"), P.CAT):
        b = bic[bic["cov"] == cov]
        axs[0].plot(b["k"], b["bic"], "o-", ms=3, color=col, label=cov)
    axs[0].set_xlabel("number of components $k$"); axs[0].set_ylabel("BIC"); axs[0].legend(fontsize=7.5)
    Z = PCA(2).fit_transform(X)
    sc = axs[1].scatter(Z[:, 0], Z[:, 1], c=post6.max(1), cmap=P.SEQ, s=18, vmin=0.4, vmax=1, edgecolor=P.INK2, lw=0.3)
    f.colorbar(sc, ax=axs[1], label="largest posterior probability")
    axs[1].set_xlabel("PC1"); axs[1].set_ylabel("PC2")
    f.tight_layout(); P.save(f, "lab1_gmm")

    # ------------------------------------------------ HDBSCAN vs DBSCAN
    hrows = []
    for mcs, ms in ((5, 5), (5, 3), (4, 2), (3, 1)):
        h = HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit(X)
        hrows.append((mcs, ms, int(h.labels_.max() + 1), int((h.labels_ == -1).sum())))
    save_table("lab1_hdbscan", pd.DataFrame(hrows, columns=["min. cluster size", "min. samples", "Clusters", "Noise points"]
                                            ).set_index("min. cluster size"), nd=0)
    hd = HDBSCAN(min_cluster_size=5).fit(X)
    d5 = np.sort(NearestNeighbors(n_neighbors=6).fit(X).kneighbors(X)[0][:, 5])
    eps = float(np.quantile(d5, 0.95))
    db = DBSCAN(eps=eps, min_samples=5).fit(X)
    noise_hd = set(feats.index[hd.labels_ == -1]); noise_db = set(feats.index[db.labels_ == -1])
    num.update({"hdb_clusters": int(hd.labels_.max() + 1), "hdb_noise": len(noise_hd),
                "db_noise": len(noise_db), "noise_common": len(noise_hd & noise_db)})

    # ------------------------------------------------ t-SNE map vs PCA
    tsne = TSNE(2, perplexity=20, init="pca", random_state=C.SEED).fit_transform(X)
    num.update({"trust_pca": trustworthiness(X, Z, n_neighbors=5), "trust_tsne": trustworthiness(X, tsne, n_neighbors=5)})
    f, axs = P.fig(3.0, ncols=2)
    for ax, E, title in ((axs[0], Z, "PCA"), (axs[1], tsne, "t-SNE (perplexity 20)")):
        for c in range(K):
            m = lab.to_numpy() == c
            ax.scatter(E[m, 0], E[m, 1], s=14, color=P.CAT[c], label=f"C{c + 1}", edgecolor="white", lw=0.3)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    axs[0].legend(fontsize=7, ncol=2)
    f.tight_layout(); P.save(f, "lab1_tsne")

    # ------------------------------------------------ 49 industries: co-movement clusters and stability
    ind = load_ff_industries()
    mkt = load_ff_factors()["Mkt-RF"]
    def residuals(block):
        m = mkt.reindex(block.index)
        beta = block.apply(lambda y: np.polyfit(m, y, 1)[0])
        return block - np.outer(m, beta)
    raw_tr = ind.loc[:C.TRAIN_END]
    tr, te = residuals(raw_tr), residuals(ind.loc[pd.Timestamp(C.VAL_END) + pd.Timedelta(days=1):])
    res = {}
    for name, block in (("train", tr), ("test", te)):
        D = corr_distance(block)
        Zl = linkage(squareform(D.to_numpy(), checks=False), "average")
        sil = {k: silhouette_score(D.to_numpy(), fcluster(Zl, k, "maxclust"), metric="precomputed") for k in range(4, 13)}
        res[name] = (D, Zl, sil)
    k_ind = max(res["train"][2], key=res["train"][2].get)
    lab_tr = fcluster(res["train"][1], k_ind, "maxclust"); lab_te = fcluster(res["test"][1], k_ind, "maxclust")
    corr_tr = raw_tr.corr().to_numpy()
    num.update({"ind_k": k_ind, "ind_sil": res["train"][2][k_ind], "ind_ari_train_test": adjusted_rand_score(lab_tr, lab_te),
                "ind_mean_corr_train": float(corr_tr[np.triu_indices(49, 1)].mean()),
                "ind_mean_resid_corr_train": float(tr.corr().to_numpy()[np.triu_indices(49, 1)].mean()),
                "ind_sizes": ", ".join(str(int(v)) for v in pd.Series(lab_tr).value_counts().sort_values(ascending=False))})
    groups = pd.Series(lab_tr, index=ind.columns)
    ind_tab = pd.DataFrame({"Cluster": [f"I{g}" for g in sorted(groups.unique())],
                            "Industries": [", ".join(groups.index[groups == g]) for g in sorted(groups.unique())],
                            "Size": [int((groups == g).sum()) for g in sorted(groups.unique())]}).set_index("Cluster")
    save_table("lab1_industry_clusters", ind_tab, nd={"Size": 0}, col_format="lp{11.6cm}c")
    order = leaves_list(res["train"][1])
    f, axs = P.fig(3.6, ncols=2)
    for ax, (name, block) in zip(axs, (("2000--2014", tr), ("2019--2026", te))):
        cc = block.corr().to_numpy()[np.ix_(order, order)]
        im = ax.imshow(cc, cmap=P.DIV, vmin=-1, vmax=1); ax.grid(False)
        ax.set_title(f"correlations of residual returns, {name}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    f.colorbar(im, ax=axs, fraction=0.03, pad=0.02)
    P.save(f, "lab1_industries")

    # ------------------------------------------------ market regimes (Gaussian mixture, fitted on 2000-2014)
    p = load_market_panel()
    st = download_fred("STLFSI4")
    S = pd.DataFrame(index=p.index)
    r = np.log(p["adj_close"]).diff()
    S["log_vix"] = np.log(p["VIXCLS"])
    S["vix_chg_5"] = S["log_vix"].diff(5)
    S["log_rv_22"] = np.log(np.sqrt(252 * (r ** 2).rolling(22).mean()))
    S["credit"] = p["DBAA"] - p["DAAA"]
    S["term"] = p["DGS10"] - p["DTB3"]
    S["stress"] = available_at(st, S.index, PUBLICATION_LAG_DAYS["STLFSI4"])     # known after publication
    S = S.dropna()
    trm = S.index <= pd.Timestamp(C.TRAIN_END)
    ssc = StandardScaler().fit(S[trm])
    Xs = ssc.transform(S)
    rb = [(k, GaussianMixture(k, covariance_type="full", n_init=5, random_state=C.SEED).fit(Xs[trm]).bic(Xs[trm]))
          for k in range(2, 9)]
    # BIC keeps falling with k on 3 700 days; the elbow of BIC is used together with interpretability
    rbic = pd.Series(dict(rb))
    k_reg = 4
    gmr = GaussianMixture(k_reg, covariance_type="full", n_init=10, random_state=C.SEED).fit(Xs[trm])
    reg = pd.Series(gmr.predict(Xs), index=S.index)
    order = list(S.groupby(reg)["log_vix"].mean().sort_values().index)
    # the two middle components have almost the same VIX and differ mainly in the slope of the yield curve:
    # the one with the steeper curve is listed first
    if S.loc[reg == order[1], "term"].mean() < S.loc[reg == order[2], "term"].mean():
        order[1], order[2] = order[2], order[1]
    reg = reg.map({o: i for i, o in enumerate(order)})
    fut = np.sqrt(252 / 22 * (r ** 2).rolling(22).sum().shift(-22)).reindex(S.index)
    names = ["calm", "moderate, steep curve", "moderate, flat curve", "crisis"]
    rows = []
    for i in range(k_reg):
        m = reg == i
        blk = lambda a, b: 100 * float(m[(S.index > pd.Timestamp(a)) & (S.index <= pd.Timestamp(b))].mean())
        stay = float(((reg.shift(-1) == i) & m).sum() / max(m.sum() - 1, 1))
        rows.append((names[i], 100 * m.mean(), float(np.exp(S.loc[m, "log_vix"]).mean()), 100 * float(fut[m].mean()),
                     float(S.loc[m, "credit"].mean()), float(S.loc[m, "term"].mean()), float(S.loc[m, "stress"].mean()), stay,
                     blk("1999-12-31", C.TRAIN_END), blk(C.TRAIN_END, C.VAL_END), blk(C.VAL_END, C.END)))
    rt = pd.DataFrame(rows, columns=["Regime", "Days, \\%", "Mean VIX", "Next-month vol., \\%", "Credit spread",
                                     "Term spread", "Stress index", "Persistence", "Train, \\%", "Valid., \\%",
                                     "Test, \\%"]).set_index("Regime")
    save_table("lab1_regimes", rt, nd={"Days, \\%": 1, "Mean VIX": 1, "Next-month vol., \\%": 1, "Credit spread": 2,
                                       "Term spread": 2, "Stress index": 2, "Persistence": 3, "Train, \\%": 1,
                                       "Valid., \\%": 1, "Test, \\%": 1},
               col_format="p{3.2cm}cccccccccc")
    pd.DataFrame({"regime": reg, **{f"p{i}": gmr.predict_proba(Xs)[:, o] for i, o in enumerate(order)}}).to_csv(
        C.RESULTS_DIR / "lab1_regimes.csv")
    num.update({"reg_k": k_reg, "reg_bic_k2": int(round(rbic[2])), "reg_bic_k4": int(round(rbic[4])),
                "reg_bic_k8": int(round(rbic[8])),
                "reg_persist_min": float(rt["Persistence"].min()),
                "reg_crisis_test_share": float(rt.loc["crisis", "Test, \\%"]),
                "reg_crisis_vol": float(rt.loc["crisis", "Next-month vol., \\%"]),
                "reg_calm_vol": float(rt.loc["calm", "Next-month vol., \\%"]),
                "reg_vix_steep": float(rt.loc[names[1], "Mean VIX"]), "reg_vix_flat": float(rt.loc[names[2], "Mean VIX"]),
                "reg_term_steep": f"{rt.loc[names[1], 'Term spread']:.2f}", "reg_term_flat": f"{rt.loc[names[2], 'Term spread']:.2f}",
                "reg_flat_test_share": float(rt.loc[names[2], "Test, \\%"]),
                "term_mean_train": f"{S.loc[: C.TRAIN_END, 'term'].mean():.2f}",
                "term_mean_test": f"{S.loc[S.index > pd.Timestamp(C.VAL_END), 'term'].mean():.2f}"})
    # the two crises of the sample: why 2020 is not a "crisis" day of the 2008 type
    for tag, a, b in (("2008", "2008-09-01", "2009-06-30"), ("2020", "2020-02-15", "2020-05-31")):
        w = S.loc[a:b]
        num.update({f"crisis{tag}_vix_max": float(np.exp(w["log_vix"]).max()), f"crisis{tag}_credit_max": float(w["credit"].max()),
                    f"crisis{tag}_stress_max": float(w["stress"].max()),
                    f"crisis{tag}_share": 100 * float((reg.loc[a:b] == 3).mean()),
                    f"crisis{tag}_flat_share": 100 * float((reg.loc[a:b] == 2).mean())})
    f, axs = P.fig(3.6, nrows=2, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    px = p["adj_close"].reindex(S.index)
    for i in range(k_reg):
        m = reg == i
        axs[0].scatter(S.index[m], px[m], s=1.2, color=[P.AQUA, P.BLUE, P.ORANGE, P.RED][i], label=names[i])
    axs[0].set_yscale("log"); axs[0].set_ylabel("S\\&P 500"); axs[0].legend(fontsize=7.5, markerscale=6, ncol=4)
    axs[1].plot(S.index, np.exp(S["log_vix"]), color=P.INK2, lw=0.5); axs[1].set_ylabel("VIX")
    P.shade_splits(axs[1], label=False)
    f.tight_layout(); P.save(f, "lab1_regimes")
    save_numbers("lab1x", num)
    print(num)


if __name__ == "__main__":
    main()
