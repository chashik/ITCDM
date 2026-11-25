"""Part 1, chapter 2 (mathematical foundations), section M3. Semirings and tropical algebra on the guide's data.

1. Exact segmentation of a series into k segments of constant mean (Lab 10, regimes) is a shortest path: with the cost
   matrix C[i, j] = squared error of the segment y[i:j], the best cost with k segments ending at j is the (min, +)
   matrix product D_k = D_{k-1} (x) C, and D_k[n] is the optimum.  The product is computed here and compared with the
   dynamic programming of lab10_economies.patterns.segments for the inflation of the four countries.
2. Mamdani inference (Lab 5, market-stress index) is a max-min composition: the aggregated output fuzzy set is
   agg(u) = max_r min(s_r, R[r, u]), with s the firing strengths of the rules and R[r, u] the membership of u in the
   consequent of rule r.  Compared with the Mamdani class of Lab 5 on its own data.
3. A ReLU network with one hidden layer is a difference of two convex piecewise-linear functions, i.e. a tropical
   rational function (Zhang, Naitzat, Lim 2018): f(x) = sum_{a_i > 0} a_i max(0, w_i x + b_i) - sum_{a_i < 0} |a_i|
   max(0, w_i x + b_i) + c.  A network predicting inflation three months ahead from current inflation (Lab 10 panels)
   is split this way; the number of linear pieces is counted.
Outputs: paper/generated/math3_numbers.tex; figure paper/figures/math3_relu.pdf.
Run (from code/):  python -m math_foundations.m3_semirings
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from dmfin import config as C
from dmfin import plotting as P
from dmfin.reporting import save_numbers

INF = np.inf


def cost_matrix(y: np.ndarray, minlen: int) -> np.ndarray:
    """C[i, j] = squared error of y[i:j] around its mean (j - i >= minlen), else +inf: an edge i -> j of the path."""
    n = len(y)
    cs, cs2 = np.r_[0, np.cumsum(y)], np.r_[0, np.cumsum(y ** 2)]
    Cm = np.full((n + 1, n + 1), INF)
    for i in range(n + 1):
        for j in range(i + minlen, n + 1):
            Cm[i, j] = cs2[j] - cs2[i] - (cs[j] - cs[i]) ** 2 / (j - i)
    return Cm


def min_plus(v: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(min, +) product of a row vector and a matrix: (v (x) M)[j] = min_i (v[i] + M[i, j]); also the argmin."""
    S = v[:, None] + M
    return S.min(0), S.argmin(0)


def segment_min_plus(y: np.ndarray, kmax: int = 6, minlen: int = 4) -> tuple[list[int], dict]:
    """Best segmentation with the number of segments chosen by BIC, computed by (min, +) matrix products."""
    n, Cm = len(y), cost_matrix(y, minlen)
    D, back = [Cm[0]], [None]                                # D_1[j] = cost of one segment y[0:j]
    for _ in range(2, kmax + 1):
        d, a = min_plus(D[-1], Cm)
        D.append(d); back.append(a)
    bic = {k: n * np.log(D[k - 1][n] / n) + 2 * k * np.log(n) for k in range(1, kmax + 1) if np.isfinite(D[k - 1][n])}
    k = min(bic, key=bic.get)
    cuts, j = [], n
    for kk in range(k, 1, -1):
        j = int(back[kk - 1][j]); cuts.append(j)
    return sorted(cuts), {"k": k, "products": kmax - 1, "size": n + 1}


def mamdani_as_composition(fis, data: dict) -> float:
    """Largest difference between the Mamdani class's aggregated output and the max-min composition s o R."""
    strengths = fis.firing(data)
    s = np.stack([np.atleast_1d(st) for st, _ in strengths], 1)                     # records x rules
    R = np.stack([fis.out_mf[cons] for _, cons in strengths], 0)                      # rules x universe
    comp = np.max(np.minimum(s[:, :, None], R[None, :, :]), axis=1)                  # max-min composition
    agg = np.zeros((s.shape[0], len(fis.universe)))
    for st, cons in strengths:                                                        # as in lab5_fuzzy.fuzzy.Mamdani
        agg = np.maximum(agg, np.minimum(np.atleast_1d(st)[:, None], fis.out_mf[cons][None]))
    return float(np.abs(comp - agg).max())


def relu_tropical(x: np.ndarray, y: np.ndarray, hidden: int = 12, seed: int = 0):
    net = MLPRegressor(hidden_layer_sizes=(hidden,), activation="relu", max_iter=5000, random_state=seed,
                       alpha=1e-3).fit(x[:, None], y)
    w, b = net.coefs_[0][0], net.intercepts_[0]
    a, c = net.coefs_[1][:, 0], net.intercepts_[1][0]
    g = lambda t: (np.maximum(0, np.outer(t, w) + b) * np.clip(a, 0, None)).sum(1)             # convex, PL
    h = lambda t: (np.maximum(0, np.outer(t, w) + b) * np.clip(-a, 0, None)).sum(1)            # convex, PL
    grid = np.linspace(x.min(), x.max(), 20001)
    f = net.predict(grid[:, None])
    gap = float(np.abs(f - (g(grid) - h(grid) + c)).max())
    kinks = np.sort(-b / w)                                  # where a hidden unit switches: breakpoints of f
    kinks = kinks[(kinks > x.min()) & (kinks < x.max())]
    edges = np.r_[x.min(), kinks, x.max()]                    # slope of f on each interval between breakpoints
    mids = (edges[:-1] + edges[1:]) / 2
    eps = 1e-6 * (x.max() - x.min())
    slopes = (net.predict((mids + eps)[:, None]) - net.predict((mids - eps)[:, None])) / (2 * eps)
    pieces = 1 + int((np.abs(np.diff(slopes)) > 1e-9).sum())
    return net, grid, f, g(grid) + c, h(grid), gap, len(kinks), pieces


def main():
    num = {}
    # 1. segmentation: (min, +) products against the dynamic programming of Lab 10
    from lab10_economies.common import COUNTRIES, panel
    from lab10_economies.patterns import segments
    same = 0
    for c in COUNTRIES:
        y = panel(c)["cpi_yoy"].to_numpy()
        cuts, info = segment_min_plus(y)
        same += int(cuts == segments(y))
        num[f"seg_{c}_k"] = info["k"]
    num.update({"seg_countries_same": same, "seg_matrix_size": info["size"], "seg_products": info["products"]})
    # 2. Mamdani inference of Lab 5 as a max-min composition
    from dmfin.data import load_market_panel
    from dmfin.features import add_targets, market_features
    from lab5_fuzzy.run import stress_system
    feats = add_targets(market_features(load_market_panel()), horizons=(5, 22))
    d = feats[["vix", "credit_baa_aaa", "ret_22", "y_rv_5", "y_rv_22", "recession"]].dropna()   # the sample of Lab 5
    fis, rules = stress_system(d.loc[:C.TRAIN_END])
    num.update({"mamdani_rules": len(rules), "mamdani_days": len(d), "mamdani_universe": len(fis.universe),
                "mamdani_max_diff": mamdani_as_composition(fis, {"vix": d["vix"], "spread": d["credit_baa_aaa"],
                                                                  "mom": d["ret_22"]})})
    # 3. a ReLU network as a tropical rational function (inflation in three months from inflation now, 4 countries)
    xs, ys = [], []
    for c in COUNTRIES:
        p = panel(c)
        z = pd.DataFrame({"x": p["cpi_yoy"], "y": p["cpi_yoy"].shift(-3)}).dropna()
        xs.append(z["x"].to_numpy()); ys.append(z["y"].to_numpy())
    x, y = np.concatenate(xs), np.concatenate(ys)
    net, grid, f, gpart, hpart, gap, kinks, pieces = relu_tropical(x, y)
    num.update({"relu_hidden": 12, "relu_points": len(x), "relu_gap": gap, "relu_kinks": kinks, "relu_pieces": pieces})
    fig, ax = P.fig(2.8)
    ax.scatter(x, y, s=6, color=P.MUTED, lw=0, label="months (four countries)")
    ax.plot(grid, f, color=P.INK, lw=1.6, label="ReLU network f = g - h")
    ax.plot(grid, gpart, color=P.BLUE, lw=1.1, ls="--", label="g (convex, piecewise linear)")
    ax.plot(grid, hpart, color=P.ORANGE, lw=1.1, ls="--", label="h (convex, piecewise linear)")
    ax.set_xlabel("inflation now, %"); ax.set_ylabel("inflation in three months, %")
    ax.legend(fontsize=7)
    fig.tight_layout(); P.save(fig, "math3_relu")
    save_numbers("math3", num)
    print(pd.Series(num).to_string())


if __name__ == "__main__":
    main()
