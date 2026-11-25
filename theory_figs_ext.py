"""Illustrations of Part 1 added in iteration 2 (synthetic data or published figures).

Run:  python theory_figs_ext.py
"""
from __future__ import annotations

import numpy as np

from dmfin import plotting as P

RNG = np.random.default_rng(2026)


def fig_kaggle():
    """Share of data scientists using a method family regularly (Kaggle 2021 report, p. 33)."""
    fam = ["Linear or logistic regression", "Decision trees or random forests", "Gradient boosting machines",
           "Convolutional neural networks", "Bayesian approaches", "Dense neural networks (MLP)",
           "Recurrent neural networks", "Transformer networks", "Generative adversarial networks", "Evolutionary approaches"]
    share = [80.3, 74.1, 59.5, 39.6, 28.7, 27.6, 26.7, 17.1, 7.6, 5.8]
    f, ax = P.fig(3.0)
    ax.barh(range(len(fam)), share, color=P.BLUE)
    for i, v in enumerate(share):
        ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=8)
    ax.set_yticks(range(len(fam))); ax.set_yticklabels(fam, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("data scientists using the method regularly, %"); ax.set_xlim(0, 92)
    P.save(f, "tx_kaggle")


def fig_gmm():
    from scipy.stats import norm
    x = np.r_[RNG.normal(-1.5, 0.6, 300), RNG.normal(1.2, 1.0, 500)]
    w, mu, sd = np.array([0.5, 0.5]), np.array([-0.5, 0.5]), np.array([1.0, 1.0])
    ll = []
    snaps = {}
    for it in range(40):
        dens = w * norm.pdf(x[:, None], mu, sd)
        r = dens / dens.sum(1, keepdims=True)                       # E-step: responsibilities
        ll.append(np.log(dens.sum(1)).sum())
        nk = r.sum(0); w = nk / len(x)                              # M-step
        mu = (r * x[:, None]).sum(0) / nk
        sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / nk)
        if it in (0, 39):
            snaps[it] = (w.copy(), mu.copy(), sd.copy())
    g = np.linspace(-4, 5, 400)
    f, axs = P.fig(2.6, ncols=2)
    axs[0].hist(x, bins=40, density=True, color="#d9d8d3")
    for it, ls in ((0, ":"), (39, "-")):
        ww, mm, ss = snaps[it]
        for k, col in enumerate((P.BLUE, P.ORANGE)):
            axs[0].plot(g, ww[k] * norm.pdf(g, mm[k], ss[k]), color=col, ls=ls, lw=1.2)
    axs[0].set_xlabel("$x$ (dotted: after 1 iteration, solid: after 40)"); axs[0].set_ylabel("density")
    axs[1].plot(np.arange(1, len(ll) + 1), ll, "o-", ms=2.5, color=P.BLUE)
    axs[1].set_xlabel("EM iteration"); axs[1].set_ylabel("log-likelihood")
    f.tight_layout(); P.save(f, "tx_gmm")


def fig_svm():
    from sklearn.svm import SVC
    X = np.r_[RNG.normal([-1.2, -1.0], 0.7, (40, 2)), RNG.normal([1.2, 1.0], 0.7, (40, 2))]
    y = np.r_[np.zeros(40), np.ones(40)]
    f, axs = P.fig(2.9, ncols=2, sharey=True)
    xx, yy = np.meshgrid(np.linspace(-3.5, 3.5, 300), np.linspace(-3.5, 3.5, 300))
    for ax, C in zip(axs, (100.0, 0.05)):
        m = SVC(kernel="linear", C=C).fit(X, y)
        z = m.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contour(xx, yy, z, levels=[-1, 0, 1], colors=[P.MUTED, P.INK, P.MUTED], linestyles=[":", "-", ":"], linewidths=1)
        ax.scatter(X[:, 0], X[:, 1], c=[P.BLUE if v == 0 else P.ORANGE for v in y], s=12)
        sv = m.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=60, facecolor="none", edgecolor=P.INK, lw=0.8)
        ax.set_title(f"$C={C:g}$: {len(sv)} support vectors", fontsize=9); ax.set_xlabel("$X_1$")
    axs[0].set_ylabel("$X_2$")
    f.tight_layout(); P.save(f, "tx_svm")


def fig_tsne():
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    n = 150
    t = RNG.uniform(0, 2 * np.pi, 3 * n)
    lab = np.repeat([0, 1, 2], n)
    r = np.array([1.0, 2.0, 3.0])[lab]
    X = np.c_[r * np.cos(t), r * np.sin(t), RNG.normal(0, 0.1, (3 * n, 8))]       # three nested rings in 10-D
    Z1 = PCA(2).fit_transform(X)
    Z2 = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(X)
    f, axs = P.fig(2.7, ncols=2)
    for ax, Z, title in ((axs[0], Z1, "PCA"), (axs[1], Z2, "t-SNE")):
        for k, col in enumerate((P.BLUE, P.ORANGE, P.AQUA)):
            ax.scatter(Z[lab == k, 0], Z[lab == k, 1], s=6, color=col)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    f.tight_layout(); P.save(f, "tx_tsne")


def fig_conformal():
    n = 2000
    x = RNG.uniform(0, 4, n)
    y = 1.0 + 0.8 * x + RNG.normal(0, 0.2 + 0.3 * x)
    tr, cal, te = slice(0, 500), slice(500, 1000), slice(1000, None)
    b = np.polyfit(x[tr], y[tr], 1)
    s = np.abs(y[cal] - np.polyval(b, x[cal]))
    q = np.quantile(s, np.ceil(501 * 0.9) / 500, method="higher")
    xt, yt = x[te], y[te]
    cov = np.abs(yt - np.polyval(b, xt)) <= q
    bins = np.linspace(0, 4, 9)
    cb = [cov[(xt >= a) & (xt < c)].mean() * 100 for a, c in zip(bins[:-1], bins[1:])]
    f, axs = P.fig(2.6, ncols=2)
    g = np.linspace(0, 4, 50)
    axs[0].scatter(xt[:400], yt[:400], s=3, color=P.MUTED)
    axs[0].plot(g, np.polyval(b, g), color=P.BLUE)
    axs[0].fill_between(g, np.polyval(b, g) - q, np.polyval(b, g) + q, color=P.BLUE, alpha=0.2, lw=0)
    axs[0].set_xlabel("$x$"); axs[0].set_ylabel("$y$"); axs[0].set_title(f"90 % split-conformal band, coverage {cov.mean()*100:.1f} %", fontsize=8.5)
    axs[1].bar((bins[:-1] + bins[1:]) / 2, cb, width=0.4, color=P.ORANGE)
    axs[1].axhline(90, color=P.INK2, ls=":", lw=0.9)
    axs[1].set_xlabel("$x$"); axs[1].set_ylabel("coverage in the bin, %"); axs[1].set_ylim(50, 102)
    f.tight_layout(); P.save(f, "tx_conformal")
    return cov.mean() * 100, min(cb), max(cb)


def fig_bh():
    from scipy.stats import norm
    m, m1 = 100, 10
    z = np.r_[RNG.normal(3.0, 1, m1), RNG.normal(0, 1, m - m1)]
    p = norm.sf(z)
    order = np.argsort(p); ps = p[order]; true = (order < m1)
    i = np.arange(1, m + 1)
    k = np.max(np.where(ps <= 0.1 * i / m)[0]) + 1 if np.any(ps <= 0.1 * i / m) else 0
    f, ax = P.fig(2.6)
    ax.scatter(i, ps, s=10, color=[P.ORANGE if t else P.BLUE for t in true], zorder=3)
    ax.plot(i, 0.1 * i / m, color=P.INK2, lw=1, label="Benjamini--Hochberg line $q\\,i/m$, $q=0.1$")
    ax.axhline(0.1 / m, color=P.RED, ls="--", lw=0.9, label="Bonferroni $\\alpha/m$, $\\alpha=0.1$")
    ax.axhline(0.1, color=P.MUTED, ls=":", lw=0.9, label="unadjusted 0.1")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("rank $i$ of the $p$-value"); ax.set_ylabel("$p$-value")
    ax.legend(fontsize=7.5, loc="lower right")
    P.save(f, "tx_bh")
    return int(k), int(true[:k].sum()), int((ps <= 0.1 / m).sum()), int((ps <= 0.1).sum()), int(true[ps <= 0.1].sum())


def fig_boosting():
    from sklearn.ensemble import GradientBoostingRegressor
    x = np.sort(RNG.uniform(0, 6, 200)); y = np.sin(x) + 0.3 * x + RNG.normal(0, 0.25, 200)
    g = np.linspace(0, 6, 400)[:, None]
    f, ax = P.fig(2.6)
    ax.scatter(x, y, s=5, color=P.MUTED)
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=1).fit(x[:, None], y)
    for k, col in ((1, P.RED), (10, P.ORANGE), (200, P.BLUE)):
        pred = list(gb.staged_predict(g))[k - 1]
        ax.plot(g[:, 0], pred, color=col, lw=1.2, label=f"{k} stump" + ("s" if k > 1 else ""))
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.legend(fontsize=7.5)
    P.save(f, "tx_boosting")


def fig_matrix_profile():
    """(a) a motif planted twice in noise; (b) a discord: one distorted cycle of a periodic signal."""
    import stumpy
    n, m = 2000, 60
    rng = np.random.default_rng(7)
    xa = np.cumsum(rng.normal(0, 0.05, n)) + rng.normal(0, 1.0, n)
    motif = 4 * np.sin(np.linspace(0, 3 * np.pi, m))
    for st in (300, 1400):
        xa[st:st + m] += motif
    t = np.arange(n)
    xb = np.sin(2 * np.pi * t / 50) + 0.5 * np.sin(4 * np.pi * t / 50) + rng.normal(0, 0.1, n)
    xb[1000:1050] = 0.4 * np.sin(2 * np.pi * t[1000:1050] / 25) + rng.normal(0, 0.1, 50)    # one distorted cycle
    pa = np.asarray(stumpy.stump(xa, m)[:, 0], dtype=float)
    mpa = stumpy.stump(xa, m)
    pb = np.asarray(stumpy.stump(xb, m)[:, 0], dtype=float)
    i1 = int(np.argmin(pa)); i2 = int(mpa[i1, 1]); d = int(np.argmax(pb))
    f, axs = P.fig(4.6, nrows=4, sharex=True, gridspec_kw={"height_ratios": [1.2, 0.8, 1.2, 0.8]})
    axs[0].plot(xa, color=P.INK2, lw=0.5)
    for i in (i1, i2):
        axs[0].plot(np.arange(i, i + m), xa[i:i + m], color=P.BLUE, lw=1.4)
    axs[0].set_ylabel("(a) series", fontsize=8)
    axs[1].plot(pa, color=P.BLUE, lw=0.8); axs[1].set_ylabel("profile", fontsize=8)
    axs[2].plot(xb, color=P.INK2, lw=0.5)
    axs[2].plot(np.arange(d, d + m), xb[d:d + m], color=P.RED, lw=1.4); axs[2].set_ylabel("(b) series", fontsize=8)
    axs[3].plot(pb, color=P.RED, lw=0.8); axs[3].set_ylabel("profile", fontsize=8); axs[3].set_xlabel("start of the window")
    f.tight_layout(); P.save(f, "tx_matrix_profile")
    return i1, i2, d


def fig_bandit():
    K, T, reps = 5, 2000, 200
    mu = np.array([0.2, 0.35, 0.5, 0.55, 0.6])
    reg = {"$\\varepsilon$-greedy, $\\varepsilon=0.1$": np.zeros(T), "UCB": np.zeros(T), "random": np.zeros(T)}
    for r in range(reps):
        rng = np.random.default_rng(r)
        for name in reg:
            n, s, cum = np.zeros(K), np.zeros(K), 0.0
            for t in range(T):
                if name == "random":
                    a = rng.integers(K)
                elif name == "UCB":
                    a = t if t < K else int(np.argmax(s / n + np.sqrt(2 * np.log(t + 1) / n)))
                else:
                    a = rng.integers(K) if (rng.uniform() < 0.1 or t < K) else int(np.argmax(s / np.maximum(n, 1)))
                rwd = float(rng.uniform() < mu[a]); n[a] += 1; s[a] += rwd
                cum += mu.max() - mu[a]
                reg[name][t] += cum / reps
    f, ax = P.fig(2.6)
    for (name, v), col in zip(reg.items(), (P.ORANGE, P.BLUE, P.MUTED)):
        ax.plot(np.arange(1, T + 1), v, color=col, label=name)
    ax.set_xlabel("round $t$"); ax.set_ylabel("cumulative regret"); ax.legend(fontsize=7.5)
    P.save(f, "tx_bandit")


def fig_halving():
    n0, eta, R = 27, 3, 81
    f, ax = P.fig(2.7)
    curves = []
    for i in range(n0):
        a, b = RNG.uniform(0.3, 0.7), RNG.uniform(0.02, 0.15)
        curves.append(lambda e, a=a, b=b: a + (1 - a) * np.exp(-b * e) + RNG.normal(0, 0.005))
    alive, r = list(range(n0)), R // eta ** 3
    while alive:
        e = np.arange(1, r + 1)
        vals = {i: [curves[i](t) for t in e] for i in alive}
        for i in alive:
            ax.plot(e, vals[i], color=P.BLUE if len(alive) > 1 else P.ORANGE, lw=0.6 if len(alive) > 3 else 1.3, alpha=0.8)
        if len(alive) == 1 or r >= R:
            break
        keep = max(1, len(alive) // eta)
        alive = sorted(alive, key=lambda i: vals[i][-1])[:keep]
        r *= eta
    ax.set_xscale("log"); ax.set_xlabel("resource (training epochs)"); ax.set_ylabel("validation loss")
    P.save(f, "tx_halving")


if __name__ == "__main__":
    fig_kaggle(); fig_gmm(); fig_svm(); fig_tsne()
    cov = fig_conformal(); bh = fig_bh(); fig_boosting(); mp = fig_matrix_profile(); fig_bandit(); fig_halving()
    from dmfin.reporting import save_numbers
    save_numbers("tx", {"cf_total": cov[0], "cf_min": cov[1], "cf_max": cov[2], "bh_rej": bh[0], "bh_true": bh[1],
                        "bonf_rej": bh[2], "raw_rej": bh[3], "raw_true": bh[4], "mp_a": mp[0], "mp_b": mp[1], "mp_discord": mp[2]})
    print("conformal coverage (total, min bin, max bin):", cov)
    print("BH: rejected, of which true; Bonferroni rejected; unadjusted rejected, of which true:", bh)
    print("matrix profile motif pair and discord:", mp)
