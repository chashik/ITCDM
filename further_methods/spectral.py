"""Normalized spectral clustering of Ng, Jordan and Weiss (Han et al., p. 519-522; von Luxburg 2007, p. 7).

From an affinity matrix W (w_ij >= 0, w_ii = 0) the matrix A = D^(-1/2) W D^(-1/2) is formed, D the diagonal matrix of
the row sums of W; the k eigenvectors of A with the largest eigenvalues are the columns of a new data matrix, its rows are
normalized to length one, and k-means clusters the rows. The objects are clustered by their position in the eigenvector
space, which separates groups that are connected in W even when they are not convex in the original space. The
eigengap heuristic chooses k where the gap between the k-th and (k+1)-th eigenvalue of A is largest.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def gaussian_affinity(X: np.ndarray, sigma: float) -> np.ndarray:
    W = np.exp(-cdist(X, X, "sqeuclidean") / (2 * sigma ** 2))
    np.fill_diagonal(W, 0.0)
    return W


def eigen(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and eigenvectors of D^(-1/2) W D^(-1/2)."""
    d = W.sum(axis=1)
    s = 1 / np.sqrt(np.where(d > 0, d, 1))
    A = s[:, None] * W * s[None, :]
    vals, vecs = np.linalg.eigh(A)
    return vals[::-1], vecs[:, ::-1]


def eigengap(W: np.ndarray, kmax: int = 6) -> int:
    vals, _ = eigen(W)
    gaps = vals[: kmax] - vals[1: kmax + 1]
    return int(np.argmax(gaps[1:]) + 2)                          # at least two clusters


def njw(W: np.ndarray, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Cluster labels and the eigenvalues of A."""
    vals, vecs = eigen(W)
    Y = vecs[:, :k]
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    lab = KMeans(k, n_init=20, random_state=seed).fit_predict(Y)
    return lab, vals
