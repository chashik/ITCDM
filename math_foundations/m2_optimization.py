"""Part 1, chapter 2 (mathematical foundations), section M2. Gradient descent, stochastic gradient descent and Adam on the credit data of Lab 3.

The objective is the L2-regularized logistic loss L(w) = mean log(1 + exp(-y x^T w)) + (lambda / 2) |w|^2, y in {-1, 1},
on the standardized features of Lab 3 (training part).  It is smooth with constant L = lambda_max(X^T X) / (4 n) + lambda
and strongly convex with mu >= lambda, so gradient descent with step 1/L converges linearly: L(w_k) - L* <=
(1 - mu/L)^k (L(w_0) - L*) (Bottou, Curtis, Nocedal 2018; Goodfellow et al., ch. 4, 8).  SGD with steps that satisfy the
Robbins-Monro conditions (sum = infinity, sum of squares < infinity; here 1/(L + mu t)) converges sublinearly,
a constant step only to a neighbourhood of the optimum whose size grows with the step; Adam (Kingma and Ba 2015)
adapts the step per coordinate.  The optimum is computed to 1e-12 by L-BFGS; the loss gap is recorded per epoch (gradient descent: one step per
epoch; SGD: one step per example; minibatches of 64).
Outputs: paper/generated/math2_numbers.tex; figure paper/figures/math2_convergence.pdf.
Run (from code/):  python -m math_foundations.m2_optimization
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from dmfin import config as C
from dmfin import plotting as P
from dmfin.reporting import save_numbers

LAM = 1e-2
EPOCHS = 100


def loss_grad(w, X, y, lam=LAM):
    z = y * (X @ w)
    loss = np.mean(np.logaddexp(0, -z)) + lam / 2 * w @ w
    g = -(X.T @ (y / (1 + np.exp(z)))) / len(y) + lam * w
    return loss, g


def run_gd(X, y, L, epochs):
    w = np.zeros(X.shape[1]); out = []
    for _ in range(epochs):
        w = w - loss_grad(w, X, y)[1] / L
        out.append(loss_grad(w, X, y)[0])
    return out


def run_sgd(X, y, epochs, batch, step, rng, adam=False):
    n, p = X.shape
    w = np.zeros(p); m = np.zeros(p); v = np.zeros(p); t = 0; out = []
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(n), n // batch):
            t += 1
            g = loss_grad(w, X[idx], y[idx])[1]
            if adam:                                         # Kingma and Ba (2015), default betas
                m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
                w = w - step(t) * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            else:
                w = w - step(t) * g
        out.append(loss_grad(w, X, y)[0])
    return out


def main():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from dmfin.data import load_credit
    from lab3_classification.model import prepare
    Xall, yall = prepare(load_credit())
    Xtrva, _, ytrva, _ = train_test_split(Xall, yall, test_size=0.2, stratify=yall, random_state=C.SEED)
    Xtr, _, ytr, _ = train_test_split(Xtrva, ytrva, test_size=0.25, stratify=ytrva, random_state=C.SEED)
    X = np.c_[StandardScaler().fit_transform(Xtr), np.ones(len(Xtr))]
    y = np.where(np.asarray(ytr) == 1, 1.0, -1.0)
    n, p = X.shape
    eig = np.linalg.eigvalsh(X.T @ X / n)
    L, mu = eig.max() / 4 + LAM, LAM
    opt = minimize(lambda w: loss_grad(w, X, y), np.zeros(p), jac=True, method="L-BFGS-B",
                   options={"gtol": 1e-12, "ftol": 1e-15, "maxiter": 10000})
    Lstar = opt.fun
    rng = np.random.default_rng(C.SEED)
    curves = {"gradient descent, step 1/L": run_gd(X, y, L, EPOCHS),
              "SGD, steps 1/(L + mu t)": run_sgd(X, y, EPOCHS, 1, lambda t: 1.0 / (L + mu * t), rng),
              "minibatch SGD (64), constant step": run_sgd(X, y, EPOCHS, 64, lambda t: 0.5 / L, rng),
              "Adam (64), step 0.01": run_sgd(X, y, EPOCHS, 64, lambda t: 0.01, rng, adam=True)}
    gaps = {k: np.maximum(np.array(v) - Lstar, 1e-16) for k, v in curves.items()}
    first = lambda g, tol: int(np.argmax(g < tol) + 1) if (g < tol).any() else np.nan
    num = {"n": n, "p": p, "lambda": LAM, "L": L, "mu": mu, "kappa": L / mu, "loss_star": Lstar,
           "rate_bound": 1 - mu / L}
    for k, g in gaps.items():
        tag = {"gradient": "gd", "SGD,": "sgd", "minibatch": "mb", "Adam": "adam"}[k.split()[0]]
        num.update({f"{tag}_gap1": float(g[0]), f"{tag}_gap10": float(g[9]), f"{tag}_gap100": float(g[-1]),
                    f"{tag}_ep1e3": first(g, 1e-3), f"{tag}_ep1e5": first(g, 1e-5)})
    gd = gaps["gradient descent, step 1/L"]
    num["gd_observed_rate"] = float((gd[-1] / gd[9]) ** (1 / (EPOCHS - 10)))   # per iteration, epochs 10-100
    num["gd_bound_after100"] = float((1 - mu / L) ** EPOCHS * (loss_grad(np.zeros(p), X, y)[0] - Lstar))
    f, ax = P.fig(2.8)
    for (k, g), col in zip(gaps.items(), P.CAT):
        ax.semilogy(range(1, EPOCHS + 1), g, color=col, lw=1.4, label=k)
    ax.set_xlabel("passes through the data (epochs)"); ax.set_ylabel("loss above the optimum")
    ax.legend(fontsize=7)
    f.tight_layout(); P.save(f, "math2_convergence")
    save_numbers("math2", num)
    print(pd.Series(num).to_string())


if __name__ == "__main__":
    main()
