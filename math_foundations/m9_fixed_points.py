"""Part 1, chapter 2 (mathematical foundations), section M9. Fixed points, Markov chains and floating-point arithmetic on the guide's data.

1. Markov chain of inflation regimes (Lab 10, fuzzy c-means regimes of the four economies, the regime of largest
   membership): the transition matrix P is stochastic; by Perron-Frobenius it has eigenvalue 1 with a positive
   stationary distribution pi = pi P, and powers converge at the rate of the second-largest eigenvalue |lambda_2|
   (Levin and Peres).
2. The Bellman operator (T V)(s) = r(s) + gamma sum_s' P(s, s') V(s') is a gamma-contraction in the maximum norm, so
   iteration converges to the unique fixed point from any start (Banach; Szepesvari 2010).  Checked on random pairs
   (V, W) with the regime chain and r(s) = -|inflation at the centre of s - 4|; error of value iteration after k steps.
3. k-means (Lab 1 stock features): every Lloyd step does not increase the within-cluster sum of squares, and since
   the number of partitions is finite the algorithm stops after finitely many steps.
4. A genetic algorithm as a Markov chain (Mitchell 1996, ch. 4): two strings of two bits, fitness 1 + number of ones,
   fitness-proportional selection and bit mutation; states = populations.  Without mutation the chain is absorbed;
   with mutation it is irreducible and aperiodic and has a unique stationary distribution (Perron-Frobenius).
5. Floating point (Goldberg 1991): the one-pass variance E[x^2] - E[x]^2 cancels catastrophically when the data have a
   large mean (Russian milk prices in kopecks shifted by 10^6 ... 10^10), the two-pass formula does not.
Outputs: paper/generated/math9_numbers.tex.  Run (from code/):  python -m math_foundations.m9_fixed_points
"""
from __future__ import annotations

from itertools import combinations_with_replacement, product

import numpy as np
import pandas as pd

from dmfin import config as C
from dmfin.reporting import grouped, save_numbers


def regime_chain() -> tuple[np.ndarray, np.ndarray]:
    from lab10_economies.paths import RESULTS
    from lab10_economies.fuzzy_tools import f4_fcm
    from lab10_economies.common import COUNTRIES, panel, PERIOD
    D, _, centres, _, _ = f4_fcm({c: panel(c, base=True).loc[:PERIOD[1]] for c in COUNTRIES})
    reg = D[[f"membership, regime {k + 1}" for k in range(3)]].to_numpy().argmax(1)
    N = np.zeros((3, 3))
    for c in COUNTRIES:
        r = reg[(D["country"] == c).to_numpy()]
        for a, b in zip(r[:-1], r[1:]):
            N[a, b] += 1
    return N / N.sum(1, keepdims=True), centres["inflation"].to_numpy()


def ga_chain(pm: float) -> np.ndarray:
    strings = ["00", "01", "10", "11"]
    fit = {s: 1 + s.count("1") for s in strings}
    states = list(combinations_with_replacement(strings, 2))           # populations of two strings (multisets)
    index = {tuple(sorted(s)): i for i, s in enumerate(states)}
    def mutate(s, t):                                                   # probability that s becomes t
        return np.prod([pm if a != b else 1 - pm for a, b in zip(s, t)])
    P = np.zeros((len(states), len(states)))
    for i, pop in enumerate(states):
        w = np.array([fit[s] for s in pop], float); w /= w.sum()
        for (a, wa), (b, wb) in product(zip(pop, w), repeat=2):        # select two parents with replacement
            for ta, tb in product(strings, repeat=2):
                P[i, index[tuple(sorted((ta, tb)))]] += wa * wb * mutate(a, ta) * mutate(b, tb)
    return P, states


def stationary(P: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(P.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return v / v.sum()


def main():
    num = {}
    # 1. regime chain
    P, infl = regime_chain()
    ev = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    pi = stationary(P)
    Pk = np.linalg.matrix_power(P, 24)
    num.update({"chain_lambda1": ev[0], "chain_lambda2": ev[1], "chain_rowsum_err": float(np.abs(P.sum(1) - 1).max()),
                **{f"chain_pi{k + 1}": 100 * pi[k] for k in range(3)},
                "chain_dist24": float(np.abs(Pk - pi[None]).sum(1).max() / 2), "chain_stay_shock": 100 * P[2, 2],
                "chain_mixing_months": int(np.ceil(np.log(0.01) / np.log(ev[1])))})
    # 2. Bellman operator: contraction and value iteration
    gamma, r = 0.9, -np.abs(infl - 4)
    T = lambda V: r + gamma * P @ V
    rng = np.random.default_rng(C.SEED)
    ratios = []
    for _ in range(1000):
        V, W = rng.normal(0, 10, 3), rng.normal(0, 10, 3)
        ratios.append(np.abs(T(V) - T(W)).max() / np.abs(V - W).max())
    Vstar = np.linalg.solve(np.eye(3) - gamma * P, r)
    V = np.zeros(3); err0 = np.abs(V - Vstar).max()
    for _ in range(50):
        V = T(V)
    num.update({"bellman_gamma": gamma, "bellman_max_ratio": float(max(ratios)), "bellman_err50": float(np.abs(V - Vstar).max()),
                "bellman_bound50": float(gamma ** 50 * err0)})
    # 3. k-means on the Lab 1 features: monotone objective, finite termination
    from dmfin.data import load_market_panel, load_stock_prices
    from lab1_clustering.model import stock_features
    px, mk = load_stock_prices(), load_market_panel()
    fit = slice(C.STOCKS_START, C.STOCKS_SPLIT)                          # the fit window of Lab 1
    feats = stock_features(np.log(px).diff().iloc[1:].loc[fit], np.log(mk["adj_close"]).diff().loc[fit],
                           mk["DGS10"].diff().loc[fit],
                           mk["DCOILWTICO"].pct_change(fill_method=None).clip(-0.5, 0.5).loc[fit])
    X = ((feats - feats.mean()) / feats.std(ddof=0)).to_numpy()
    starts, its, monotone = 20, [], True
    for seed in range(starts):
        cen = X[np.random.default_rng(seed).choice(len(X), 6, replace=False)]
        obj, lab = [], None
        for it in range(1000):
            new = ((X[:, None] - cen[None]) ** 2).sum(2).argmin(1)
            if lab is not None and (new == lab).all():
                break
            lab = new
            cen = np.array([X[lab == k].mean(0) if (lab == k).any() else cen[k] for k in range(6)])
            obj.append(float(((X - cen[lab]) ** 2).sum()))
        its.append(it); monotone &= all(b <= a + 1e-9 for a, b in zip(obj[:-1], obj[1:]))
    num.update({"kmeans_starts": starts, "kmeans_monotone": int(monotone), "kmeans_iter_max": max(its),
                "kmeans_iter_median": float(np.median(its)), "kmeans_stocks": len(X)})
    # 4. genetic algorithm as a Markov chain
    P0, states = ga_chain(0.0)
    absorbing = [states[i] for i in range(len(states)) if P0[i, i] == 1]
    P1, _ = ga_chain(0.05)
    pi1 = stationary(P1)
    best = sum(p for s, p in zip(states, pi1) if "11" in s)
    ev1 = np.sort(np.abs(np.linalg.eigvals(P1)))[::-1]
    num.update({"ga_states": len(states), "ga_absorbing": len(absorbing), "ga_pm": 0.05,
                "ga_pi_with_optimum": 100 * best, "ga_lambda2": ev1[1], "ga_positive": int((np.linalg.matrix_power(P1, 2) > 0).all())})
    # 5. floating point: one-pass against two-pass variance
    from lab10_economies.common import panel
    base = 100 * panel("RU")["price_milk"].dropna().to_numpy()        # kopecks
    exact = float(np.var(base - base.mean()))
    num["fp_var"] = grouped(exact)
    for e in (0, 6, 8, 10):                                            # the same data shifted by 10^e
        x = base + (10.0 ** e if e else 0.0)
        two = np.mean((x - x.mean()) ** 2)                             # two-pass formula
        one = np.mean(x * x) - np.mean(x) ** 2                         # one-pass ("textbook") formula
        num[f"fp_rel_err_two_e{e}"] = float(abs(two - exact) / exact)
        num[f"fp_rel_err_one_e{e}"] = float(abs(one - exact) / exact)
        num[f"fp_one_e{e}"] = grouped(one)
    save_numbers("math9", num)
    print(pd.Series(num).to_string())


if __name__ == "__main__":
    main()
