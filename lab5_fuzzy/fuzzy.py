"""Lab 5. Fuzzy-logic tools implemented with NumPy.

* membership functions: triangular and trapezoidal;
* fuzzy c-means (Bezdek): objective  J_m = sum_i sum_j u_ij^m ||x_i - c_j||^2;
  validity indices: partition coefficient, partition entropy, Xie-Beni;
* Mamdani inference: min for AND, min-implication, max-aggregation,
  centroid defuzzification;
* Wang-Mendel rule generation for fuzzy classification.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd


def trimf(x, a, b, c):
    x = np.asarray(x, dtype=float)
    left = np.where(b > a, (x - a) / max(b - a, 1e-12), (x >= b).astype(float))
    right = np.where(c > b, (c - x) / max(c - b, 1e-12), (x <= b).astype(float))
    return np.clip(np.minimum(left, right), 0, 1)


def trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    left = np.ones_like(x) if b <= a else (x - a) / (b - a)
    right = np.ones_like(x) if d <= c else (d - x) / (d - c)
    return np.clip(np.minimum(np.minimum(left, 1.0), right), 0, 1)


def _low(x, lo, mid):
    x = np.asarray(x, dtype=float)
    return np.clip((mid - x) / (mid - lo), 0, 1)


def _high(x, mid, hi):
    x = np.asarray(x, dtype=float)
    return np.clip((x - mid) / (hi - mid), 0, 1)


class Partition:
    """Three overlapping fuzzy sets defined by three break-points lo < mid < hi."""

    def __init__(self, lo, mid, hi, names=("low", "medium", "high")):
        self.lo, self.mid, self.hi, self.names = lo, mid, hi, names

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return {self.names[0]: _low(x, self.lo, self.mid),
                self.names[1]: trimf(x, self.lo, self.mid, self.hi),
                self.names[2]: _high(x, self.mid, self.hi)}


# ------------------------------------ fuzzy c-means
def fcm(X, c, m=2.0, max_iter=300, tol=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    U = rng.dirichlet(np.ones(c), size=n)
    for it in range(max_iter):
        W = U ** m
        V = (W.T @ X) / W.sum(0)[:, None]
        D = np.linalg.norm(X[:, None, :] - V[None], axis=2) + 1e-12
        inv = D ** (-2.0 / (m - 1.0))
        U_new = inv / inv.sum(1, keepdims=True)
        if np.abs(U_new - U).max() < tol:
            U = U_new
            break
        U = U_new
    J = float(((U ** m) * D ** 2).sum())
    return U, V, J, it + 1


def partition_coefficient(U):
    return float((U ** 2).sum() / len(U))


def partition_entropy(U):
    return float(-(U * np.log(np.clip(U, 1e-12, 1))).sum() / len(U))


def xie_beni(X, U, V, m=2.0):
    D2 = ((X[:, None, :] - V[None]) ** 2).sum(2)
    sep = min(((V[i] - V[j]) ** 2).sum() for i in range(len(V)) for j in range(len(V)) if i != j)
    return float(((U ** m) * D2).sum() / (len(X) * sep))


# ----------------------------------- Mamdani system
class Mamdani:
    """Rules: list of (dict input -> set name, output set name)."""

    def __init__(self, inputs: dict, output: dict, universe: np.ndarray, rules: list):
        self.inputs, self.output, self.universe, self.rules = inputs, output, universe, rules
        self.out_mf = {k: f(universe) for k, f in output.items()}

    def firing(self, data: dict):
        mu = {v: self.inputs[v](np.asarray(data[v], dtype=float)) for v in self.inputs}
        strengths = []
        for ante, cons in self.rules:
            s = None
            for v, setname in ante.items():
                s = mu[v][setname] if s is None else np.minimum(s, mu[v][setname])
            strengths.append((s, cons))
        return strengths

    def __call__(self, data: dict):
        strengths = self.firing(data)
        n = len(np.atleast_1d(strengths[0][0]))
        agg = np.zeros((n, len(self.universe)))
        for s, cons in strengths:
            agg = np.maximum(agg, np.minimum(np.atleast_1d(s)[:, None], self.out_mf[cons][None]))
        den = agg.sum(1)
        out = (agg * self.universe[None]).sum(1) / np.where(den > 0, den, 1)
        return np.where(den > 0, out, np.nan)


# --------------------------- Wang-Mendel classifier
class WangMendel:
    """Fuzzy rule-based classifier.

    Every training example generates a candidate rule whose antecedent takes,
    for each input, the fuzzy set with maximal membership; the rule's degree
    is the product of those memberships.  Rules with the same antecedent are
    merged; for every antecedent the class weights are the summed degrees.
    Prediction: score = sum_r tau_r(x) * w_r(1) / sum_r tau_r(x) * (w_r(0)+w_r(1)).
    """

    def __init__(self, partitions: dict):
        self.partitions = partitions
        self.vars = list(partitions)

    def _mu(self, X: pd.DataFrame):
        return {v: self.partitions[v](X[v].to_numpy()) for v in self.vars}

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        mu = self._mu(X)
        names = {v: list(mu[v]) for v in self.vars}
        best = {v: np.stack([mu[v][s] for s in names[v]], 1) for v in self.vars}
        arg = {v: best[v].argmax(1) for v in self.vars}
        deg = np.prod([best[v].max(1) for v in self.vars], axis=0)
        table = {}
        for i in range(len(y)):
            key = tuple(names[v][arg[v][i]] for v in self.vars)
            w = table.setdefault(key, np.zeros(2))
            w[y[i]] += deg[i]
        self.rules_ = table
        return self

    def predict_proba(self, X: pd.DataFrame):
        mu = self._mu(X)
        num = np.zeros(len(X)); den = np.zeros(len(X))
        for key, w in self.rules_.items():
            tau = np.prod([mu[v][s] for v, s in zip(self.vars, key)], axis=0)
            num += tau * w[1]; den += tau * w.sum()
        p = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
        return np.c_[1 - p, p]

    def rule_table(self, prior: float):
        rows = []
        for key, w in self.rules_.items():
            rows.append((*key, w.sum(), w[1] / w.sum() if w.sum() else np.nan))
        df = pd.DataFrame(rows, columns=self.vars + ["support", "conf_default"])
        df["lift"] = df["conf_default"] / prior
        return df.sort_values("support", ascending=False)
