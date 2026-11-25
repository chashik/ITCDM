"""Lab 3. Binary classification: logistic regression and single-layer perceptrons.

Contents
  * ``prepare``            - cleaning and feature engineering of the credit data;
  * ``Perceptron``         - Rosenblatt's perceptron with the classical learning
                             rule  w <- w + eta (y - yhat) x  (pocket + averaging);
  * ``LogisticNeuron``     - a single sigmoid neuron trained by mini-batch SGD on
                             the (class-weighted) cross-entropy; mathematically
                             the same model as logistic regression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PAY = [f"PAY_{i}" for i in range(1, 7)]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT = [f"PAY_AMT{i}" for i in range(1, 7)]


def slog(x):
    """Sign-preserving log transform: sign(x) * ln(1 + |x|)."""
    return np.sign(x) * np.log1p(np.abs(x))


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean undocumented codes and engineer features; returns (X, y)."""
    d = df.copy()
    # undocumented categories -> 'other' (data-quality step, see Lab 0)
    d["EDUCATION"] = d["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    d["MARRIAGE"] = d["MARRIAGE"].replace({0: 3})
    X = pd.DataFrame(index=d.index)
    X["log_limit"] = np.log(d["LIMIT_BAL"])
    X["age"] = d["AGE"]
    X["female"] = (d["SEX"] == 2).astype(float)
    for k, name in [(1, "edu_graduate"), (2, "edu_university"), (3, "edu_high_school")]:
        X[name] = (d["EDUCATION"] == k).astype(float)
    X["married"] = (d["MARRIAGE"] == 1).astype(float)
    for c in PAY:
        X[c.lower()] = d[c].clip(lower=-2)                  # ordinal repayment status
    X["n_delinquent"] = (d[PAY] > 0).sum(axis=1)
    X["max_delay"] = d[PAY].max(axis=1).clip(lower=0)
    for c in BILL + PAYAMT:
        X[c.lower()] = slog(d[c])
    X["utilisation"] = (d["BILL_AMT1"] / d["LIMIT_BAL"]).clip(-1, 3)
    X["pay_ratio"] = (d["PAY_AMT1"] / d["BILL_AMT1"].where(d["BILL_AMT1"] > 0)).fillna(1).clip(0, 2)
    return X, d["default"].astype(int)


class Perceptron:
    """Rosenblatt perceptron, labels in {0, 1}.

    With ``pocket=True`` the weights with the lowest training error seen so
    far are kept (Gallant's pocket algorithm); with ``average=True`` the
    returned weights are the average over all updates (Freund-Schapire).
    These variants matter because the plain perceptron never converges on
    data that are not linearly separable.
    """

    def __init__(self, eta=1.0, epochs=20, shuffle=True, pocket=False, average=False,
                 class_weight=None, seed=0):
        self.eta, self.epochs, self.shuffle = eta, epochs, shuffle
        self.pocket, self.average, self.class_weight, self.seed = pocket, average, class_weight, seed

    def _err(self, Xb, y, w):
        return np.mean((Xb @ w > 0).astype(int) != y)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        Xb = np.c_[X, np.ones(len(X))]
        y = np.asarray(y).astype(int)
        cw = {0: 1.0, 1: 1.0}
        if self.class_weight == "balanced":
            p = y.mean(); cw = {0: 0.5 / (1 - p), 1: 0.5 / p}
        w = np.zeros(Xb.shape[1]); w_sum = np.zeros_like(w); n_upd = 0
        best_w, best_err = w.copy(), 1.0
        self.history_ = []
        for _ in range(self.epochs):
            order = rng.permutation(len(y)) if self.shuffle else np.arange(len(y))
            mistakes = 0
            for i in order:
                yhat = int(Xb[i] @ w > 0)
                if yhat != y[i]:
                    w = w + self.eta * cw[y[i]] * (y[i] - yhat) * Xb[i]
                    mistakes += 1
                w_sum += w; n_upd += 1
            err = self._err(Xb, y, w)
            if err < best_err:
                best_err, best_w = err, w.copy()
            # (mistakes, error of last iterate, of pocket weights, of averaged weights)
            self.history_.append((mistakes, err, best_err, self._err(Xb, y, w_sum / n_upd)))
            if mistakes == 0:
                break
        if self.average:
            w = w_sum / n_upd
        elif self.pocket:
            w = best_w
        self.w_ = w
        return self

    def decision_function(self, X):
        return np.c_[X, np.ones(len(X))] @ self.w_

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class LogisticNeuron:
    """Single sigmoid unit, cross-entropy loss, mini-batch SGD, L2 penalty."""

    def __init__(self, lr=0.1, epochs=60, batch=256, l2=1e-4, class_weight=None, seed=0):
        self.lr, self.epochs, self.batch, self.l2 = lr, epochs, batch, l2
        self.class_weight, self.seed = class_weight, seed

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.seed)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.w_, self.b_ = np.zeros(p), 0.0
        sw = np.ones(n)
        if self.class_weight == "balanced":
            pi = y.mean(); sw = np.where(y == 1, 0.5 / pi, 0.5 / (1 - pi))
        self.history_ = []
        for ep in range(self.epochs):
            lr = self.lr / (1 + 0.05 * ep)                       # decaying step size
            for idx in np.array_split(rng.permutation(n), max(1, n // self.batch)):
                pr = sigmoid(X[idx] @ self.w_ + self.b_)
                g = (pr - y[idx]) * sw[idx]
                self.w_ -= lr * (X[idx].T @ g / len(idx) + self.l2 * self.w_)
                self.b_ -= lr * g.mean()
            rec = [self.loss(X, y)]
            if X_val is not None:
                rec.append(self.loss(X_val, np.asarray(y_val, dtype=float)))
            self.history_.append(rec)
        return self

    def predict_proba(self, X):
        p = sigmoid(X @ self.w_ + self.b_)
        return np.c_[1 - p, p]

    def loss(self, X, y):
        p = np.clip(self.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def expected_cost(y, score, thr, c_fn=5.0, c_fp=1.0):
    pred = score >= thr
    fn = np.sum((y == 1) & ~pred); fp = np.sum((y == 0) & pred)
    return (c_fn * fn + c_fp * fp) / len(y)


def best_threshold(y, score, c_fn=5.0, c_fp=1.0, grid=None):
    grid = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 197))) if grid is None else grid
    costs = np.array([expected_cost(y, score, t, c_fn, c_fp) for t in grid])
    return float(grid[costs.argmin()]), grid, costs
