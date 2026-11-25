"""Lab 3 (iteration 2) tests: the multilayer perceptron trainer."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from lab3_classification.mlp import fit_mlp, predict_proba


def data(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    y = ((X[:, 0] ** 2 + X[:, 1] > 1.0) ^ (rng.uniform(size=n) < 0.1)).astype(np.float32)   # non-linear, 10 % noise
    return X[:800], y[:800], X[800:], y[800:]


def test_mlp_learns_a_nonlinear_boundary_and_is_reproducible():
    torch.set_num_threads(1)
    Xtr, ytr, Xva, yva = data()
    m1, _ = fit_mlp(Xtr, ytr, Xva, yva, width=32, depth=1, max_epochs=60, seed=3)
    m2, _ = fit_mlp(Xtr, ytr, Xva, yva, width=32, depth=1, max_epochs=60, seed=3)
    p1, p2 = predict_proba(m1, Xva), predict_proba(m2, Xva)
    lr = LogisticRegression().fit(Xtr, ytr).predict_proba(Xva)[:, 1]
    assert roc_auc_score(yva, p1) > roc_auc_score(yva, lr) + 0.05       # the linear model cannot draw the boundary
    assert np.allclose(p1, p2)


def test_early_stopping_returns_the_best_epoch():
    torch.set_num_threads(1)
    Xtr, ytr, Xva, yva = data(seed=1)
    m, hist = fit_mlp(Xtr, ytr, Xva, yva, width=256, depth=2, lr=1e-2, max_epochs=80, patience=5, seed=0)
    p = np.clip(predict_proba(m, Xva), 1e-7, 1 - 1e-7)
    loss = -np.mean(yva * np.log(p) + (1 - yva) * np.log(1 - p))
    assert len(hist) < 80 and abs(loss - hist[:, 1].min()) < 1e-4
