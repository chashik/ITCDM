"""k-medoids: PAM (Partitioning Around Medoids), PAMk and CLARA (Han et al., p. 454-457).

A medoid is the object of a cluster with the smallest total distance to the other objects of the cluster. PAM minimizes
the total distance of the objects to their nearest medoid: BUILD chooses k medoids greedily, SWAP exchanges a medoid with a
non-medoid while the exchange lowers the cost. One SWAP pass costs O(k(n - k)^2) distance evaluations. PAMk runs PAM
for several k and keeps the k with the largest average silhouette. CLARA runs PAM on random samples of s objects,
assigns all n objects to the medoids of each sample and keeps the medoids with the lowest cost on all objects.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples


def cost(D: np.ndarray, medoids) -> float:
    """Total distance of all objects to their nearest medoid (D: n x n distances, or n x m to candidate objects)."""
    return float(D[:, list(medoids)].min(axis=1).sum())


def pam(D: np.ndarray, k: int, max_swaps: int = 1000) -> tuple[np.ndarray, np.ndarray, float, int]:
    """PAM on a distance matrix D (n x n). Returns (medoids, labels, cost, number of swaps)."""
    n = len(D)
    med = [int(np.argmin(D.sum(axis=1)))]                       # BUILD: the most central object first ...
    while len(med) < k:                                          # ... then the object that lowers the cost most
        near = D[:, med].min(axis=1)
        gain = np.array([np.maximum(near - D[:, j], 0).sum() if j not in med else -1 for j in range(n)])
        med.append(int(np.argmax(gain)))
    swaps, best = 0, cost(D, med)
    while swaps < max_swaps:                                     # SWAP: the best exchange, while it improves
        top = (0.0, None, None)
        for i in range(k):
            others = med[:i] + med[i + 1:]
            near_o = D[:, others].min(axis=1) if others else np.full(n, np.inf)
            for h in range(n):
                if h in med:
                    continue
                c = float(np.minimum(near_o, D[:, h]).sum())
                if best - c > top[0] + 1e-12:
                    top = (best - c, i, h)
        if top[1] is None:
            break
        med[top[1]] = top[2]
        best -= top[0]
        swaps += 1
    med = np.array(med)
    return med, np.argmin(D[:, med], axis=1), cost(D, med), swaps


def pamk(D: np.ndarray, ks=range(2, 7)) -> tuple[int, dict]:
    """PAM for every k in ks; the k with the largest average silhouette width. Returns (k, {k: average silhouette})."""
    asw = {}
    for k in ks:
        _, lab, _, _ = pam(D, k)
        asw[k] = float(silhouette_samples(D, lab, metric="precomputed").mean())
    return max(asw, key=asw.get), asw


def clara(X: np.ndarray, k: int, n_samples: int = 5, sample_size: int | None = None, seed: int = 0):
    """CLARA on the objects X (n x p, Euclidean distance). Returns (medoids as indices of X, labels, cost on all objects,
    costs of the samples' medoids on all objects)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    s = min(n, sample_size or 40 + 2 * k)
    best, trials = None, []
    for _ in range(n_samples):
        idx = np.sort(rng.choice(n, size=s, replace=False))
        m, _, _, _ = pam(cdist(X[idx], X[idx]), k)
        med = idx[m]
        c = float(cdist(X, X[med]).min(axis=1).sum())
        trials.append(c)
        if best is None or c < best[1]:
            best = (med, c)
    med, c = best
    return med, np.argmin(cdist(X, X[med]), axis=1), c, trials
