"""Didactic figures for Part 1 (theory) of the guide, generated from synthetic data.

Run:  python theory_figs.py
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

from dmfin import plotting as P

RNG = np.random.default_rng(7)


def fig_clustering():
    Xm, _ = make_moons(400, noise=0.06, random_state=1)
    Xb, _ = make_blobs([150, 150, 40], centers=[(0, 0), (3.2, 0.3), (1.6, 2.6)],
                       cluster_std=[0.6, 0.6, 0.25], random_state=3)
    f, axs = P.fig(4.6, nrows=2, ncols=3)
    panels = [
        (Xm, KMeans(2, n_init=10, random_state=0).fit_predict(Xm), "k-means, k = 2"),
        (Xm, fcluster(linkage(Xm, "single"), 2, "maxclust") - 1, "single linkage"),
        (Xm, DBSCAN(eps=0.18, min_samples=5).fit_predict(Xm), "DBSCAN"),
        (Xb, KMeans(3, n_init=10, random_state=0).fit_predict(Xb), "k-means, k = 3"),
        (Xb, fcluster(linkage(Xb, "single"), 3, "maxclust") - 1, "single linkage"),
        (Xb, fcluster(linkage(Xb, "average"), 3, "maxclust") - 1, "average linkage"),
    ]
    for ax, (X, lab, ttl) in zip(axs.ravel(), panels):
        for c in np.unique(lab):
            m = lab == c
            col = P.MUTED if c == -1 else P.CAT[int(c) % 8]
            ax.scatter(X[m, 0], X[m, 1], s=6, color=col, lw=0)
        ax.set_title(ttl, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    f.tight_layout(); P.save(f, "t05_clustering")


def fig_bias_variance():
    f_true = lambda x: np.sin(2.2 * x) + 0.3 * x
    n, reps, degs = 40, 400, range(1, 11)
    xt = np.linspace(-2, 2, 200)
    tr_err, te_err, bias2, var = [], [], [], []
    for d in degs:
        preds, tre, tee = [], [], []
        for _ in range(reps):
            x = RNG.uniform(-2, 2, n); y = f_true(x) + RNG.normal(0, 0.35, n)
            c = np.polynomial.Polynomial.fit(x, y, d, domain=[-2, 2])
            preds.append(c(xt))
            tre.append(np.mean((c(x) - y) ** 2))
            yt = f_true(xt) + RNG.normal(0, 0.35, xt.size)
            tee.append(np.mean((c(xt) - yt) ** 2))
        preds = np.array(preds)
        inner = (xt > -1.8) & (xt < 1.8)  # avoid the unstable boundary
        bias2.append(np.mean((preds.mean(0) - f_true(xt))[inner] ** 2)); var.append(np.mean(preds.var(0)[inner]))
        tr_err.append(np.mean(tre)); te_err.append(np.median(tee))
    f, axs = P.fig(2.8, ncols=2)
    axs[0].plot(degs, tr_err, "o-", ms=3, color=P.BLUE, label="training MSE")
    axs[0].plot(degs, te_err, "s-", ms=3, color=P.ORANGE, label="test MSE")
    axs[0].axhline(0.35 ** 2, color=P.MUTED, ls="--", lw=0.8, label="Var($\\varepsilon$)")
    axs[0].set_xlabel("flexibility (polynomial degree)"); axs[0].set_ylabel("mean squared error")
    axs[0].set_ylim(0, 0.6); axs[0].legend(fontsize=7.5)
    axs[1].plot(degs, bias2, "o-", ms=3, color=P.VIOLET, label="squared bias")
    axs[1].plot(degs, var, "s-", ms=3, color=P.AQUA, label="variance")
    axs[1].set_yscale("log"); axs[1].set_xlabel("flexibility (polynomial degree)")
    axs[1].legend(fontsize=7.5)
    f.tight_layout(); P.save(f, "t06_bias_variance")


def fig_activations():
    z = np.linspace(-4, 4, 300)
    f, axs = P.fig(2.4, ncols=2)
    axs[0].plot(z, 1 / (1 + np.exp(-z)), color=P.BLUE, label="sigmoid")
    axs[0].plot(z, np.tanh(z), color=P.ORANGE, label="tanh")
    axs[0].plot(z, np.maximum(z, 0), color=P.AQUA, label="ReLU")
    axs[0].step(z, (z > 0).astype(float), color=P.MUTED, ls="--", lw=0.9, label="step (perceptron)")
    axs[0].set_ylim(-1.2, 2.2); axs[0].set_xlabel("$z$"); axs[0].legend(fontsize=7.5)
    s = 1 / (1 + np.exp(-z))
    axs[1].plot(z, s * (1 - s), color=P.BLUE, label="sigmoid$'$")
    axs[1].plot(z, 1 - np.tanh(z) ** 2, color=P.ORANGE, label="tanh$'$")
    axs[1].plot(z, (z > 0).astype(float), color=P.AQUA, label="ReLU$'$")
    axs[1].set_xlabel("$z$"); axs[1].set_ylabel("derivative"); axs[1].legend(fontsize=7.5)
    f.tight_layout(); P.save(f, "t07_activations")


def fig_impurity():
    p = np.linspace(1e-4, 1 - 1e-4, 300)
    ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    gini = 1 - p ** 2 - (1 - p) ** 2
    err = 1 - np.maximum(p, 1 - p)
    f, ax = P.fig(2.4, w=3.8)
    ax.plot(p, ent, color=P.BLUE, label="entropy (bits)")
    ax.plot(p, 2 * gini, color=P.ORANGE, label="Gini index $\\times 2$")
    ax.plot(p, 2 * err, color=P.AQUA, label="error rate $\\times 2$")
    ax.set_xlabel("share $p$ of class 1 in a node"); ax.set_ylabel("impurity"); ax.legend(fontsize=7.5)
    P.save(f, "t08_impurity")


def fig_vanishing():
    """Gradient norm through T steps of a scalar RNN h_t = tanh(w h_{t-1} + x_t)."""
    T = 60
    f, ax = P.fig(2.4, w=4.2)
    for w, col in [(0.5, P.BLUE), (0.9, P.AQUA), (1.1, P.ORANGE), (2.0, P.RED)]:
        h, g = 0.0, [1.0]
        xs = RNG.normal(0, 0.5, T)
        hs = []
        for t in range(T):
            h = np.tanh(w * h + xs[t]); hs.append(h)
        prod = 1.0
        for t in range(T - 1, 0, -1):
            prod *= w * (1 - hs[t] ** 2); g.append(abs(prod))
        ax.plot(range(len(g)), g, color=col, label=f"$w={w}$")
    ax.set_yscale("log"); ax.set_xlabel("number of steps back in time $k$")
    ax.set_ylabel("$|\\partial h_T/\\partial h_{T-k}|$"); ax.legend(fontsize=7.5)
    P.save(f, "t09_vanishing")


def fig_ga():
    from lab8_genetic.ga import GeneticAlgorithm
    n = 40
    w = RNG.normal(size=n)
    fit = lambda g: -float(np.dot(w, g)) + 0.5 * abs(sum(g) - 15)
    f, ax = P.fig(2.4, w=4.4)
    for k, (pm, lab) in enumerate([(0.0, "no mutation"), (1 / n, "$p_m=1/L$"), (0.2, "$p_m=0.2$")]):
        res = GeneticAlgorithm([2] * n, fitness=fit, pop_size=30, generations=60, p_mut=pm, seed=1).run()
        h = np.array(res.history)
        ax.plot(h[:, 0], h[:, 1], color=P.CAT[k], label=lab)
    ax.set_xlabel("generation"); ax.set_ylabel("best fitness (minimized)"); ax.legend(fontsize=7.5)
    P.save(f, "t11_ga_mutation")


def fig_fuzzy():
    x = np.linspace(0, 60, 400)
    f, axs = P.fig(2.3, ncols=2, sharey=True)
    axs[0].plot(x, (x >= 25).astype(float), color=P.BLUE)
    axs[0].set_title("crisp set 'high VIX' (VIX $\\geq$ 25)", fontsize=9); axs[0].set_xlabel("VIX")
    axs[0].set_ylabel("membership")
    lo = np.clip((20 - x) / 8, 0, 1); hi = np.clip((x - 20) / 12, 0, 1); med = 1 - lo - hi
    for mu, col, lab in [(lo, P.BLUE, "low"), (med, P.ORANGE, "medium"), (hi, P.AQUA, "high")]:
        axs[1].plot(x, mu, color=col, label=lab)
    axs[1].set_title("fuzzy partition of VIX", fontsize=9); axs[1].set_xlabel("VIX"); axs[1].legend(fontsize=7.5)
    f.tight_layout(); P.save(f, "t12_fuzzy_sets")


if __name__ == "__main__":
    fig_clustering(); fig_bias_variance(); fig_activations(); fig_impurity(); fig_vanishing(); fig_ga(); fig_fuzzy()
    print("theory figures written")
