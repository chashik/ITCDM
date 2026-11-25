"""Lab 1. Exploratory data analysis of 110 US stocks with clustering.

Run:  python -m lab1_clustering.run
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_samples, silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from dmfin import config as C
from dmfin import plotting as P
from dmfin.data import load_market_panel, load_stock_prices, load_universe
from dmfin.reporting import save_numbers, save_table

from .model import FEATURES, bootstrap_stability, corr_distance, hopkins, purity, stock_features

SECTOR_SHORT = {"Information Technology": "IT", "Health Care": "Health", "Financials": "Financ.",
                "Consumer Discretionary": "Discr.", "Consumer Staples": "Staples", "Energy": "Energy",
                "Industrials": "Indust.", "Utilities": "Util.", "Real Estate": "R.Estate",
                "Materials": "Mater.", "Communication Services": "Comm."}


def main():
    num = {}
    px = load_stock_prices()
    sectors = load_universe().reindex(px.columns)
    panel = load_market_panel()
    rets_all = np.log(px).diff().iloc[1:]
    mkt_all = np.log(panel["adj_close"]).diff()
    dy_all = panel["DGS10"].diff()
    oil_all = panel["DCOILWTICO"].pct_change(fill_method=None).clip(-0.5, 0.5)

    fit = slice(C.STOCKS_START, C.STOCKS_SPLIT)
    hold = slice(pd.Timestamp(C.STOCKS_SPLIT) + pd.Timedelta(days=1), C.END)
    R, Rh = rets_all.loc[fit], rets_all.loc[hold]
    feats = stock_features(R, mkt_all.loc[fit], dy_all.loc[fit], oil_all.loc[fit])
    feats_h = stock_features(Rh, mkt_all.loc[hold], dy_all.loc[hold], oil_all.loc[hold])
    num.update({"n_stocks": len(feats), "n_days_fit": len(R), "n_days_hold": len(Rh),
                "fit_from": R.index.min().strftime("%Y-%m-%d"), "fit_to": R.index.max().strftime("%Y-%m-%d"),
                "hold_from": Rh.index.min().strftime("%Y-%m-%d"), "hold_to": Rh.index.max().strftime("%Y-%m-%d")})

    desc = feats.describe().T[["mean", "std", "min", "50%", "max"]]
    desc.index.name = "Feature"
    save_table("lab1_features", desc, nd=3)

    sc = StandardScaler().fit(feats)
    X = sc.transform(feats)
    Xh = StandardScaler().fit_transform(feats_h)

    # --------------------------------------------------- clustering tendency
    H = np.mean([hopkins(X, seed=s) for s in range(50)])
    rng = np.random.default_rng(C.SEED)
    U = rng.uniform(X.min(0), X.max(0), size=X.shape)
    Hu = np.mean([hopkins(U, seed=s) for s in range(50)])
    num.update({"hopkins": H, "hopkins_uniform": Hu, "sqrt_rule": np.sqrt(len(X) / 2)})

    # ------------------------------------------------ number of clusters (k)
    ks = range(2, 16)
    rows = []
    for k in ks:
        km = KMeans(k, n_init=50, random_state=C.SEED).fit(X)
        rows.append((k, km.inertia_, silhouette_score(X, km.labels_), calinski_harabasz_score(X, km.labels_),
                     davies_bouldin_score(X, km.labels_), adjusted_rand_score(sectors, km.labels_)))
    kt = pd.DataFrame(rows, columns=["k", "inertia", "silhouette", "CH", "DB", "ARI"]).set_index("k")
    save_table("lab1_k_table", kt, nd={"inertia": 1, "silhouette": 3, "CH": 1, "DB": 3, "ARI": 3})
    k_sil = int(kt["silhouette"].idxmax())
    # elbow: largest second difference of the inertia curve
    sec = kt["inertia"].diff().diff()
    k_elbow = int(sec.idxmax() - 1)
    f, axs = P.fig(2.7, ncols=2)
    axs[0].plot(kt.index, kt["inertia"], "o-", ms=3.5, color=P.BLUE)
    axs[0].axvline(k_elbow, color=P.ORANGE, ls="--", lw=0.9)
    axs[0].set_xlabel("number of clusters $k$"); axs[0].set_ylabel("within-cluster sum of squares")
    axs[1].plot(kt.index, kt["silhouette"], "o-", ms=3.5, color=P.BLUE, label="silhouette")
    axs[1].plot(kt.index, kt["ARI"], "s-", ms=3.5, color=P.AQUA, label="ARI vs GICS sectors")
    axs[1].axvline(k_sil, color=P.ORANGE, ls="--", lw=0.9)
    axs[1].set_xlabel("number of clusters $k$"); axs[1].set_ylabel("score"); axs[1].legend(fontsize=9)
    for ax_ in axs:
        ax_.set_xticks(list(ks)[::2])
    f.tight_layout(); P.save(f, "lab1_k_selection")
    K = 6  # working choice, discussed in the text (elbow / silhouette / sqrt(n/2) / interpretability)
    num.update({"k_sil": k_sil, "k_elbow": k_elbow, "K": K, "sil_K": kt.loc[K, "silhouette"],
                "sil_max": kt["silhouette"].max(), "ari_K": kt.loc[K, "ARI"],
                "ari_max": kt["ARI"].max(), "k_ari_max": int(kt["ARI"].idxmax()),
                "sil_k2": kt.loc[2, "silhouette"]})

    km = KMeans(K, n_init=100, random_state=C.SEED).fit(X)
    lab = pd.Series(km.labels_, index=feats.index)
    # order clusters by mean market beta for readable tables
    order = feats.groupby(lab)["beta_mkt"].mean().sort_values().index
    remap = {old: new for new, old in enumerate(order)}
    lab = lab.map(remap)
    cent = pd.DataFrame(sc.inverse_transform(km.cluster_centers_), columns=FEATURES)
    cent.index = [remap[i] for i in range(K)]
    cent = cent.sort_index()
    cent.insert(0, "size", lab.value_counts().sort_index().to_numpy())
    cent.index = [f"C{i+1}" for i in cent.index]; cent.index.name = "Cluster"
    save_table("lab1_centroids", cent, nd={"size": 0, **{c: 2 for c in FEATURES}},
               header=["Cluster", "Size", "Return", "Vol.", "$\\beta_{\\mathrm{mkt}}$", "Idio.", "MaxDD", "Skew",
                       "$\\beta_{\\mathrm{rates}}$", "$\\beta_{\\mathrm{oil}}$", "$\\rho_{\\mathrm{mkt}}$"])
    members = {f"C{c+1}": ", ".join(sorted(lab.index[lab == c])) for c in range(K)}
    (C.GEN_DIR / "lab1_members.tex").write_text("\n".join(
        f"\\item[{k}:] {v.replace('&', '\\&')}" for k, v in members.items()) + "\n")

    # ------------------------------------------------------- PCA scatter
    pca = PCA(2).fit(X)
    Y = pca.transform(X)
    num.update({"pca2_var": 100 * pca.explained_variance_ratio_.sum(),
                "pca1_var": 100 * pca.explained_variance_ratio_[0]})
    f, (ax, ax2) = P.fig(3.9, ncols=2, gridspec_kw={"width_ratios": [1.6, 1]})
    for c in range(K):
        mmask = (lab == c).to_numpy()
        ax.scatter(Y[mmask, 0], Y[mmask, 1], s=26, color=P.CAT[c], edgecolor="white", lw=0.6, label=f"C{c+1}")
    rad = np.hypot(Y[:, 0] / Y[:, 0].std(), Y[:, 1] / Y[:, 1].std())
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.0f}%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), fontsize=9, ncol=6, markerscale=0.9, handletextpad=0, columnspacing=0.8)
    for j, fn in enumerate(FEATURES):
        v = pca.components_[:, j]
        ax2.annotate("", xy=(v[0], v[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=P.BLUE, lw=0.9))
    ax2.set_xlim(-0.75, 0.75); ax2.set_ylim(-0.75, 0.75); ax2.set_aspect("equal")
    ax2.set_xlabel("loading on PC1"); ax2.set_ylabel("loading on PC2")
    f.tight_layout()
    far = np.argsort(-rad)                                  # the points farthest from the centre choose first
    far = far[rad[far] > 1.6]
    P.label_points(ax, Y[far, 0], Y[far, 1], feats.index[far])
    P.label_points(ax2, pca.components_[0], pca.components_[1], FEATURES, color=P.INK)
    P.save(f, "lab1_pca_scatter")

    # ------------------------------------------------------ silhouette plot
    s = silhouette_samples(X, lab)
    f, ax = P.fig(3.0, w=4.6)
    y0 = 0
    for c in range(K):
        sc_ = np.sort(s[lab.to_numpy() == c])
        ax.fill_betweenx(np.arange(y0, y0 + len(sc_)), 0, sc_, color=P.CAT[c], lw=0)
        ax.text(-0.06, y0 + len(sc_) / 2, f"C{c+1}", fontsize=9, ha="right", va="center")
        y0 += len(sc_) + 3
    ax.axvline(s.mean(), color=P.INK2, ls="--", lw=0.8)
    ax.set_xlabel("silhouette coefficient $s(o)$"); ax.set_yticks([]); ax.set_xlim(-0.35, 0.8)
    P.save(f, "lab1_silhouette")
    num.update({"neg_sil": int((s < 0).sum())})

    # -------------------------------------- contingency: clusters vs sectors
    ct = pd.crosstab(sectors.map(SECTOR_SHORT), lab.map(lambda c: f"C{c+1}"))
    f, ax = P.fig(3.2, w=4.6)
    ax.imshow(ct.to_numpy(), cmap=P.SEQ_CELL, aspect="auto", vmin=0, vmax=10)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.iloc[i, j]
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=9, color=P.INK)
    ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels(ct.columns)
    ax.set_yticks(range(ct.shape[0])); ax.set_yticklabels(ct.index)
    ax.grid(False)
    P.save(f, "lab1_contingency")
    num.update({"purity_K": purity(sectors, lab), "nmi_K": normalized_mutual_info_score(sectors, lab)})

    # ------------------------------ hierarchical clustering on correlations
    D = corr_distance(R)
    Zl = linkage(squareform(D.to_numpy(), checks=False), method="average")
    Zw = linkage(X, method="ward")
    h11 = fcluster(Zl, 11, criterion="maxclust")
    hw = fcluster(Zw, K, criterion="maxclust")
    num.update({"ari_hier_corr11": adjusted_rand_score(sectors, h11),
                "purity_hier_corr11": purity(sectors, h11),
                "ari_ward_kmeans": adjusted_rand_score(lab, hw)})
    # 110 leaf labels are illegible in print: the sector of every leaf is shown as a colour strip instead
    f, (ax, ax_s) = P.fig(4.0, nrows=2, sharex=True, gridspec_kw={"height_ratios": [14, 1], "hspace": 0.03})
    dn = dendrogram(Zl, no_labels=True, ax=ax, color_threshold=0.0, above_threshold_color=P.BLUE,
                    link_color_func=lambda k: P.BLUE)
    ax.set_ylabel("distance $\\sqrt{2(1-\\rho)}$"); ax.grid(axis="x", visible=False)
    tab = plt.get_cmap("tab20")
    sec_col = dict(zip(SECTOR_SHORT, [tab(i) for i in range(0, 20, 2)] + [tab(1)]))
    for i, leaf in enumerate(dn["ivl"]):
        ax_s.bar(5 + 10 * i, 1, width=10, color=sec_col[sectors[R.columns[int(leaf)] if leaf.isdigit() else leaf]], lw=0)
    ax_s.set_yticks([]); ax_s.set_xticks([]); ax_s.grid(False)
    for sp in ax_s.spines.values():
        sp.set_visible(False)
    from matplotlib.patches import Patch
    ax_s.legend([Patch(color=c) for c in sec_col.values()], list(SECTOR_SHORT.values()), loc="upper center",
                bbox_to_anchor=(0.5, -0.4), ncol=6, fontsize=9, frameon=False, handlelength=1.2)
    f.subplots_adjust(bottom=0.2)
    P.save(f, "lab1_dendrogram")
    order_l = leaves_list(Zl)
    rho = R.corr().to_numpy()[np.ix_(order_l, order_l)]
    f, ax = P.fig(4.6, w=4.9)
    im = ax.imshow(rho, cmap=P.DIV, vmin=-1, vmax=1)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    f.colorbar(im, ax=ax, fraction=0.045, label="correlation of daily returns")
    P.save(f, "lab1_corr_ordered")
    num.update({"mean_corr": float(R.corr().to_numpy()[np.triu_indices(len(R.columns), 1)].mean())})

    # return-based (statistical-factor) clustering: K-means on correlation PCs
    Rs = (R - R.mean()) / R.std()
    pc = PCA(10).fit(Rs.fillna(0).to_numpy())
    load = pc.components_.T * np.sqrt(pc.explained_variance_)
    load = load / np.linalg.norm(load, axis=1, keepdims=True)
    kr = KMeans(11, n_init=100, random_state=C.SEED).fit(load)
    num.update({"ari_factor11": adjusted_rand_score(sectors, kr.labels_),
                "purity_factor11": purity(sectors, kr.labels_),
                "pc1_returns_var": 100 * pc.explained_variance_ratio_[0]})

    # -------------------------------------------------------------- DBSCAN
    nn = NearestNeighbors(n_neighbors=5).fit(X)
    kd = np.sort(nn.kneighbors(X)[0][:, -1])
    eps = float(np.quantile(kd, 0.9))
    db = DBSCAN(eps=eps, min_samples=5).fit(X)
    noise = list(feats.index[db.labels_ == -1])
    num.update({"db_eps": eps, "db_n_clusters": int(len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)),
                "db_n_noise": len(noise), "db_noise": ", ".join(noise)})

    # ---------------------------------------------- stability and persistence
    boot = bootstrap_stability(X, K, n_boot=100, seed=C.SEED)
    num.update({"boot_ari_mean": boot.mean(), "boot_ari_lo": np.quantile(boot, 0.05)})
    kmh = KMeans(K, n_init=100, random_state=C.SEED).fit(Xh)
    num["ari_fit_hold"] = adjusted_rand_score(lab, kmh.labels_)
    # assigning holdout feature vectors to the *old* centroids (fitted scaler)
    lab_h_old = km.predict(sc.transform(feats_h))
    num["ari_fit_hold_assign"] = adjusted_rand_score(km.labels_, lab_h_old)
    # within- vs between-cluster correlations in the holdout period
    rh = Rh.corr().to_numpy(); L = lab.to_numpy()
    same = (L[:, None] == L[None, :]) & ~np.eye(len(L), dtype=bool)
    diff = L[:, None] != L[None, :]
    sec_ = sectors.to_numpy()
    same_s = (sec_[:, None] == sec_[None, :]) & ~np.eye(len(L), dtype=bool)
    num.update({"hold_corr_within": rh[same].mean(), "hold_corr_between": rh[diff].mean(),
                "hold_corr_within_sector": rh[same_s].mean()})

    # factor (co-movement) clusters in the hold-out period
    Lf = kr.labels_
    same_f = (Lf[:, None] == Lf[None, :]) & ~np.eye(len(Lf), dtype=bool)
    num.update({"hold_corr_within_factor": rh[same_f].mean(),
                "hold_corr_between_factor": rh[Lf[:, None] != Lf[None, :]].mean()})
    ctf = pd.crosstab(sectors.map(SECTOR_SHORT), pd.Series(Lf, index=R.columns).map(lambda c: f"F{c+1}"))
    ctf = ctf[sorted(ctf.columns, key=lambda c: int(c[1:]))]
    f, ax = P.fig(3.2, w=5.6)
    ax.imshow(ctf.to_numpy(), cmap=P.SEQ_CELL, aspect="auto", vmin=0, vmax=10)
    for i in range(ctf.shape[0]):
        for j in range(ctf.shape[1]):
            v = ctf.iloc[i, j]
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=9, color=P.INK)
    ax.set_xticks(range(ctf.shape[1])); ax.set_xticklabels(ctf.columns)
    ax.set_yticks(range(ctf.shape[0])); ax.set_yticklabels(ctf.index); ax.grid(False)
    P.save(f, "lab1_contingency_factor")

    # --------------------------------- application: diversification test
    # equal weights, rebalanced daily: the return of the portfolio is the mean of the simple returns of its stocks
    vol_h = lambda cols: float(np.sqrt(252) * np.expm1(Rh[list(cols)]).mean(axis=1).std())
    rng = np.random.default_rng(C.SEED)
    res_div = {}
    for name, labels, k in [("risk", lab.to_numpy(), K), ("factor", Lf, 11)]:
        idx = np.array(R.columns)
        picks = [vol_h([rng.choice(idx[labels == c]) for c in range(k)]) for _ in range(2000)]
        rnd = [vol_h(rng.choice(idx, k, replace=False)) for _ in range(2000)]
        res_div[name] = (np.array(picks), np.array(rnd))
        num.update({f"div_{name}_cluster": 100 * np.mean(picks), f"div_{name}_random": 100 * np.mean(rnd),
                    f"div_{name}_share": 100 * np.mean(np.array(picks) < np.median(rnd))})
    f, axs = P.fig(2.6, ncols=2, sharey=True)
    for ax, name, ttl in [(axs[0], "risk", "6 clusters of risk features"),
                          (axs[1], "factor", "11 co-movement clusters")]:
        picks, rnd = res_div[name]
        bins = np.linspace(100 * min(picks.min(), rnd.min()), 100 * max(picks.max(), rnd.max()), 45)
        ax.hist(100 * rnd, bins=bins, color=P.ORANGE, alpha=0.6, label="random stocks")
        ax.hist(100 * picks, bins=bins, color=P.BLUE, alpha=0.6, label="one stock per cluster")
        ax.set_title(ttl, fontsize=9); ax.set_xlabel("portfolio volatility, %")
    axs[0].set_ylabel("number of portfolios"); axs[1].legend(fontsize=9)
    f.tight_layout(); P.save(f, "lab1_diversification")

    save_numbers("lab1", num)
    print(kt.round(3).to_string()); print(cent.round(2).to_string()); print(members); print(num)


if __name__ == "__main__":
    main()
