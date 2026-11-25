"""Part 1, chapter 2 (mathematical foundations), section M0. White, grey and black boxes: what the parameters of a model mean.

1. Parameter symmetries of a neural network (credit data of Lab 3).  A multilayer perceptron with one hidden layer of
   h tanh units computes the same function after any permutation of its hidden units and after flipping the signs of
   the weights into and out of a unit: 2^h h! parameter vectors give one function.  Individual weights are therefore
   not identifiable — the formal reason why they carry no meaning (black box).  Two trainings from different seeds give
   very different weights but nearly the same predictions.  A logistic regression, by contrast, has a unique optimum
   (strictly convex loss) and each coefficient is a log-odds ratio (grey box).
2. Shapley values (Shapley 1951; Lundberg and Lee 2017): for a coalition game v the value phi_j is the only one that is
   efficient (sum phi_j = v(N) - v(empty)), symmetric, gives 0 to a dummy and is additive.  With the interventional game
   v(S) = E_background f(x_S, X_rest) the values are computed exactly by enumerating the 2^p coalitions: for a linear
   score they are beta_j (x_j - E x_j); for gradient-boosted trees they are compared with SHAP's TreeExplainer.
Outputs: paper/generated/math0_numbers.tex.  Run (from code/):  python -m math_foundations.m0_boxes
"""
from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

from dmfin import config as C
from dmfin.reporting import save_numbers


def mlp_forward(net: MLPClassifier, X: np.ndarray) -> np.ndarray:
    H = np.tanh(X @ net.coefs_[0] + net.intercepts_[0])
    return H @ net.coefs_[1][:, 0] + net.intercepts_[1][0]                 # logit of default


def symmetric_copy(net: MLPClassifier, perm: np.ndarray, signs: np.ndarray):
    W1, b1, W2 = net.coefs_[0][:, perm] * signs, net.intercepts_[0][perm] * signs, net.coefs_[1][perm] * signs[:, None]
    return W1, b1, W2


def exact_shapley(f, x: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Interventional Shapley values of f at x by enumerating all coalitions."""
    p = len(x)
    def v(S):
        Z = background.copy()
        Z[:, list(S)] = x[list(S)]
        return float(f(Z).mean())
    cache = {S: v(S) for k in range(p + 1) for S in combinations(range(p), k)}
    phi = np.zeros(p)
    for j in range(p):
        others = [i for i in range(p) if i != j]
        for k in range(p):
            w = factorial(k) * factorial(p - k - 1) / factorial(p)
            for S in combinations(others, k):
                phi[j] += w * (cache[tuple(sorted(S + (j,)))] - cache[S])
    return phi, cache[()], cache[tuple(range(p))]


def main():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from dmfin.data import load_credit
    from lab3_classification.model import prepare
    Xall, yall = prepare(load_credit())
    Xtrva, Xte, ytrva, yte = train_test_split(Xall, yall, test_size=0.2, stratify=yall, random_state=C.SEED)
    Xtr, _, ytr, _ = train_test_split(Xtrva, ytrva, test_size=0.25, stratify=ytrva, random_state=C.SEED)
    sc = StandardScaler().fit(Xtr)
    Xs, Xt = sc.transform(Xtr), sc.transform(Xte)
    num = {"features": Xs.shape[1], "train": len(Xs)}
    # 1. symmetries of a tanh MLP; two seeds
    h = 8
    nets = [MLPClassifier((h,), activation="tanh", alpha=1e-3, max_iter=400, random_state=s).fit(Xs, ytr) for s in (0, 1)]
    rng = np.random.default_rng(0)
    perm, signs = rng.permutation(h), rng.choice([-1.0, 1.0], h)
    W1, b1, W2 = symmetric_copy(nets[0], perm, signs)
    z0 = mlp_forward(nets[0], Xt)
    z1 = np.tanh(Xt @ W1 + b1) @ W2[:, 0] + nets[0].intercepts_[1][0]
    num.update({"mlp_hidden": h, "mlp_equivalent": int(2 ** h * factorial(h)), "mlp_symmetry_maxdiff": float(np.abs(z0 - z1).max()),
                "mlp_weight_reldist": float(np.linalg.norm(nets[0].coefs_[0] - nets[1].coefs_[0]) / np.linalg.norm(nets[0].coefs_[0])),
                "mlp_auc_seed0": roc_auc_score(yte, nets[0].predict_proba(Xt)[:, 1]),
                "mlp_auc_seed1": roc_auc_score(yte, nets[1].predict_proba(Xt)[:, 1]),
                "mlp_pred_corr": float(np.corrcoef(nets[0].predict_proba(Xt)[:, 1], nets[1].predict_proba(Xt)[:, 1])[0, 1])})
    lr = [LogisticRegression(C=1.0, solver=sv, max_iter=10000, tol=1e-10).fit(Xs, ytr) for sv in ("lbfgs", "newton-cg", "newton-cholesky")]
    num["lr_coef_maxdiff"] = float(max(np.abs(a.coef_ - b.coef_).max() for a in lr for b in lr))   # one optimum
    # 2. exact Shapley values: linear score and boosted trees on six features
    cols = ["log_limit", "age", "female", "pay_1", "pay_2", "pay_3"]
    A, At = Xtr[cols].to_numpy(float), Xte[cols].to_numpy(float)
    bg = A[np.random.default_rng(1).choice(len(A), 100, replace=False)]   # TreeExplainer uses at most 100 rows
    x = At[0]
    lin = LogisticRegression(max_iter=5000).fit(A, ytr)
    score = lambda Z: Z @ lin.coef_[0] + lin.intercept_[0]
    phi, v0, vN = exact_shapley(score, x, bg)
    num.update({"shap_features": len(cols), "shap_coalitions": 2 ** len(cols),
                "shap_linear_vs_formula": float(np.abs(phi - lin.coef_[0] * (x - bg.mean(0))).max()),
                "shap_linear_efficiency": float(abs(phi.sum() - (vN - v0)))})
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=C.SEED).fit(A, ytr)
    raw = lambda Z: gb.decision_function(Z)
    phi_gb, g0, gN = exact_shapley(raw, x, bg)
    import shap
    ex = shap.TreeExplainer(gb, data=bg, feature_perturbation="interventional", model_output="raw")
    sv = np.asarray(ex.shap_values(x[None]))[0]
    num.update({"shap_gb_efficiency": float(abs(phi_gb.sum() - (gN - g0))), "shap_gb_vs_treeexplainer": float(np.abs(phi_gb - sv).max()),
                "shap_gb_top": cols[int(np.argmax(np.abs(phi_gb)))].replace("_", "\\_")})
    save_numbers("math0", num)
    print(pd.Series(num).to_string())


if __name__ == "__main__":
    main()
