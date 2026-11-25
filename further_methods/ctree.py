"""Conditional inference trees (Hothorn, Hornik and Zeileis 2006) for a numeric or a class response and numeric covariates.

In every node: (1) for each covariate X_j the association with the response is measured by the linear statistic
T_j = sum_i g(X_ji) h(Y_i)^T with g(x) = x and h(y) = y (numeric response) or the indicator vector of the class; its
conditional mean and covariance over all permutations of the responses are those of Strasser and Weber (preprint, p. 4-5,
eq. 2), the quadratic statistic c = (t - mu) Sigma^+ (t - mu)^T is asymptotically chi-squared with rank(Sigma) degrees of
freedom (p. 6); the p-values are Bonferroni-adjusted over the covariates, and the recursion stops when the smallest
adjusted p-value is not below alpha. (2) The covariate with the smallest p-value is split at the cut point that maximizes
the same statistic for the two-sample indicator g(x) = I(x <= cut) (p. 6, eq. 3). Variable selection on the p-value scale
does not favour covariates with many possible splits, which exhaustive search (CART) does.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2


def _influence(y: np.ndarray, classes) -> np.ndarray:
    if classes is None:
        return y.reshape(-1, 1).astype(float)
    return (y[:, None] == np.asarray(classes)[None, :]).astype(float)


def linear_test(g: np.ndarray, h: np.ndarray) -> tuple[float, int]:
    """Quadratic statistic of T = sum g_i h_i^T (g: n x p, h: n x q) and its degrees of freedom (all weights one)."""
    n = len(h)
    t = (g.T @ h).ravel()                                        # the p x q matrix stacked by rows ...
    Eh = h.mean(axis=0)
    Vh = (h - Eh).T @ (h - Eh) / n
    sg = g.sum(axis=0)
    mu = np.outer(sg, Eh).ravel()
    # ... so the Kronecker factors of eq. 2 appear in the order (g part) x (h part); the quadratic form does not depend
    # on the order in which the elements are stacked
    S = n / (n - 1) * np.kron(g.T @ g, Vh) - 1 / (n - 1) * np.kron(np.outer(sg, sg), Vh)
    d = t - mu
    Sp = np.linalg.pinv(S, rcond=1e-10)
    df = int(np.linalg.matrix_rank(S, tol=1e-10 * max(1.0, np.abs(S).max())))
    return float(d @ Sp @ d), df


@dataclass
class Node:
    n: int
    prediction: np.ndarray
    pvalue: float = 1.0
    var: int | None = None
    cut: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None
    tests: list = field(default_factory=list)


class CTree:
    def __init__(self, alpha: float = 0.05, minsplit: int = 20, minbucket: int = 7, maxdepth: int = 10):
        self.alpha, self.minsplit, self.minbucket, self.maxdepth = alpha, minsplit, minbucket, maxdepth

    def fit(self, X: np.ndarray, y: np.ndarray, names=None, classify: bool = False) -> "CTree":
        X, y = np.asarray(X, float), np.asarray(y)
        self.names = list(names) if names is not None else [f"x{j}" for j in range(X.shape[1])]
        self.classes = np.unique(y) if classify else None
        self.root = self._grow(X, y, 0)
        return self

    def _pred(self, y):
        if self.classes is None:
            return np.array([y.mean()])
        return np.array([(y == c).mean() for c in self.classes])

    def _grow(self, X, y, depth) -> Node:
        node = Node(len(y), self._pred(y))
        h = _influence(y, self.classes)
        if len(y) < self.minsplit or depth >= self.maxdepth or np.allclose(h.std(axis=0), 0):
            return node
        m = X.shape[1]
        pv = []
        for j in range(m):
            x = X[:, [j]]
            if np.allclose(x.std(), 0):
                pv.append(1.0); continue
            c, df = linear_test(x, h)
            pv.append(float(chi2.sf(c, df)) if df > 0 else 1.0)
        adj = np.minimum(np.array(pv) * m, 1.0)                  # Bonferroni
        node.tests = list(zip(self.names, pv, adj))
        j = int(np.argmin(adj))
        node.pvalue = float(adj[j])
        if adj[j] >= self.alpha:
            return node
        x = X[:, j]
        best = (-1.0, None)
        for cut in np.unique(x)[:-1]:
            left = x <= cut
            if left.sum() < self.minbucket or (~left).sum() < self.minbucket:
                continue
            c, _ = linear_test(left[:, None].astype(float), h)
            if c > best[0]:
                best = (c, cut)
        if best[1] is None:
            return node
        node.var, node.cut = j, float(best[1])
        left = x <= node.cut
        node.left = self._grow(X[left], y[left], depth + 1)
        node.right = self._grow(X[~left], y[~left], depth + 1)
        return node

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        out = []
        for row in X:
            node = self.root
            while node.var is not None:
                node = node.left if row[node.var] <= node.cut else node.right
            out.append(node.prediction)
        P = np.array(out)
        return P[:, 0] if self.classes is None else self.classes[P.argmax(axis=1)]

    def rules(self, fmt=lambda v: f"{v:.2f}") -> list[tuple[str, int, np.ndarray]]:
        """Leaves as (conditions, size, prediction)."""
        out = []

        def walk(node, conds):
            if node.var is None:
                out.append((" AND ".join(conds) or "(all)", node.n, node.prediction)); return
            nm = self.names[node.var]
            walk(node.left, conds + [f"{nm} <= {fmt(node.cut)}"])
            walk(node.right, conds + [f"{nm} > {fmt(node.cut)}"])
        walk(self.root, [])
        return out

    def splits(self) -> list[str]:
        out = []

        def walk(node):
            if node.var is not None:
                out.append(self.names[node.var]); walk(node.left); walk(node.right)
        walk(self.root)
        return out
