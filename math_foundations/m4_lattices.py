"""Part 1, chapter 2 (mathematical foundations), section M4. Lattices, closure operators and ultrametrics on the guide's data.

1. Itemsets and transactions (Lab 5, association rules on credit data): the maps f(X) = transactions containing all
   items of X and g(T) = items common to all transactions of T form a Galois connection between the subset lattices
   (X <= g(T)  <=>  T <= f(X)); c = g o f is a closure operator (extensive, monotone, idempotent) and preserves
   support, so the closed itemsets (c(X) = X) summarize the frequent ones without loss (Zaki and Hsiao 2002).
   Support is antitone on the lattice — the property that Apriori prunes with (Han, Kamber, Pei).
2. Dendrograms are ultrametrics (Carlsson and Memoli 2010): the cophenetic distance of the average-linkage dendrogram
   of Lab 1 (stocks, correlation distance) satisfies d(x, z) <= max(d(x, y), d(y, z)) for every triple, the
   correlation distance itself does not, and cutting the ultrametric at height h gives the clusters of the dendrogram.
Outputs: paper/generated/math4_numbers.tex.  Run (from code/):  python -m math_foundations.m4_lattices
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform

from dmfin import config as C
from dmfin.reporting import save_numbers


def closure_stats(B: pd.DataFrame, freq: dict, seed: int = 0) -> dict:
    M = B.to_numpy(bool)
    names = np.array(B.columns)
    idx = {n: j for j, n in enumerate(names)}
    f = lambda X: np.logical_and.reduce([M[:, idx[i]] for i in X]) if X else np.ones(len(M), bool)
    g = lambda T: frozenset(names[M[T].all(0)]) if T.any() else frozenset(names)
    c = lambda X: g(f(X))
    sets = list(freq)
    clo = {X: c(X) for X in sets}
    closed = [X for X in sets if clo[X] == X]
    maximal = [X for X in sets if not any(X < Y for Y in sets)]
    extensive = all(X <= clo[X] for X in sets)
    idempotent = all(c(clo[X]) == clo[X] for X in sets)
    same_support = all(f(X).sum() == f(clo[X]).sum() for X in sets)
    rng = np.random.default_rng(seed)
    have = set(sets)                                         # every pair X < Y with |Y| = |X| + 1 among the frequent sets
    sub = [(frozenset(Xs), Y) for Y in sets if len(Y) > 1 for Xs in combinations(Y, len(Y) - 1) if frozenset(Xs) in have]
    monotone = all(clo[X] <= clo[Y] for X, Y in sub)
    antitone = all(freq[Y] <= freq[X] for X, Y in sub)
    galois = True                                            # X <= g(T)  <=>  T <= f(X), random transaction sets
    for _ in range(2000):
        X = sets[rng.integers(len(sets))]
        T = rng.random(len(M)) < rng.uniform(0.001, 0.05)
        galois &= (X <= g(T)) == bool(np.all(~T | f(X)))
    return {"items": len(names), "transactions": len(M), "frequent": len(sets), "closed": len(closed),
            "maximal": len(maximal), "extensive": int(extensive), "idempotent": int(idempotent),
            "monotone": int(monotone), "support_preserved": int(same_support), "antitone": int(antitone),
            "subset_pairs": len(sub), "galois": int(galois)}


def ultrametric_stats(D: np.ndarray, method: str = "average") -> dict:
    Z = linkage(squareform(D, checks=False), method=method)
    coph_r, coph = cophenet(Z, squareform(D, checks=False))
    U = squareform(coph)
    n = len(D)
    worst_u = worst_d = 0.0
    viol_d = 0
    for i in range(n):                                       # all triples (i, j, k): d(i, k) <= max(d(i, j), d(j, k))
        mu = np.maximum(U[i][:, None], U)                    # max(d(i, j), d(j, k)) over j, k
        md = np.maximum(D[i][:, None], D)
        worst_u = max(worst_u, float((U[i][None, :] - mu).max()))
        dd = D[i][None, :] - md
        worst_d = max(worst_d, float(dd.max()))
        viol_d += int((dd > 1e-12).sum())
    same_cut = True
    for h in np.quantile(Z[:, 2], [0.5, 0.8, 0.95]):        # cutting the ultrametric = cutting the dendrogram
        _, lab_u = connected_components(U <= h)
        lab_z = fcluster(Z, h, criterion="distance")
        same_cut &= pd.crosstab(lab_u, lab_z).gt(0).sum(axis=1).eq(1).all() and len(set(lab_u)) == len(set(lab_z))
    return {"stocks": n, "coph_corr": float(coph_r), "ultra_worst": worst_u, "corr_worst": worst_d,
            "corr_violation_share": 100 * viol_d / n ** 3, "same_cut": int(same_cut)}


# the transactional data of Han, Kamber and Pei (Table 6.1): nine transactions, items I1..I5
HAN = [["I1", "I2", "I5"], ["I2", "I4"], ["I2", "I3"], ["I1", "I2", "I4"], ["I1", "I3"], ["I2", "I3"], ["I1", "I3"],
       ["I1", "I2", "I3", "I5"], ["I1", "I2", "I3"]]


def main():
    num = {}
    from lab5_fuzzy.apriori import apriori
    items = ["I1", "I2", "I3", "I4", "I5"]
    Bh = pd.DataFrame([[i in t for i in items] for t in HAN], columns=items)
    fh, _ = apriori(Bh, 2 / 9, max_len=5)
    num.update({f"han_{k}": v for k, v in closure_stats(Bh, fh).items()})
    # 1. closed itemsets on the transactions of Lab 5 (training part of the credit data, as in the lab)
    from sklearn.model_selection import train_test_split
    from dmfin.data import load_credit
    from lab3_classification.model import prepare
    from lab5_fuzzy.apriori import apriori, one_hot
    from lab5_fuzzy.run_ext import MIN_SUP, items_taiwan
    X, y = prepare(load_credit())
    Xtrva, _, ytrva, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=C.SEED)
    Xtr, _, ytr, _ = train_test_split(Xtrva, ytrva, test_size=0.25, stratify=ytrva, random_state=C.SEED)
    it = items_taiwan(Xtr, Xtr, full=True)
    it["default"] = np.where(ytr == 1, "yes", "no")
    B = one_hot(it)
    freq, _ = apriori(B, MIN_SUP, max_len=4)
    num.update({f"items_{k}": v for k, v in closure_stats(B, freq).items()})
    num["min_support_pct"] = 100 * MIN_SUP
    # 2. the dendrogram of Lab 1 as an ultrametric
    from dmfin.data import load_stock_prices
    from lab1_clustering.model import corr_distance
    px = load_stock_prices()
    R = np.log(px).diff().iloc[1:].loc[C.STOCKS_START:C.STOCKS_SPLIT]
    num.update({f"dendro_{k}": v for k, v in ultrametric_stats(corr_distance(R).to_numpy()).items()})
    save_numbers("math4", num)
    print(pd.Series(num).to_string())


if __name__ == "__main__":
    main()
